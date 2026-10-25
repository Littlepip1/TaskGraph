"""Concurrent multi-target closed loop, reusing the upstream KF and network.

Truth is used by the sensor generator and evaluation only. Selection receives
estimates, roles, current positions and routes. Receiver context is benchmark
provided; its acquisition/signaling cost is explicitly outside payload results.
"""
from __future__ import annotations
from collections import defaultdict, deque
from dataclasses import dataclass
import copy
import importlib
import math
import time
import numpy as np

from airsim_ext.belief import TargetBelief2D
from airsim_ext.controller import clipped_velocity
from airsim_ext.sensing import sample_observation
from r4.estimation import ReplayEstimator
from r6f.selector import LocalGateConfig, innovation_nis
from r7.selector import PressureConfig, LocalTelemetry, ReceivedAgeHistory, decide_pressure_local
from r8a.network import HopByHopDynamicNetwork
from final_airsim.common import base_config, protocol
from .scenario import Scenario, PreparationIndex
from .selection import select
from simple_selectors import select_simple
from snapshot_capture import SnapshotCapture
from intuitive_policies import choose_receivers, IntuitivePolicy
from dynamic_environment import DynamicDemandEnvironment

METHODS = ['dense_global', 'dense_task', 'full', 'demand_only', 'legacy_additive',
           'uncertainty', 'innovation', 'no_group', 'no_prepare', 'no_shared',
           'forward_only', 'no_guard', 'intuitive']


class VectorizedNetwork(HopByHopDynamicNetwork):
    """Same radius graph and sorted adjacency; faster geometry construction."""
    def rebuild(self, positions_xy):
        self._tree_cache.clear()
        self.positions = np.asarray(positions_xy, float).reshape(self.n, 2).copy()
        d = self.positions[:, None, :] - self.positions[None, :, :]
        connected = np.sum(d * d, axis=-1) <= float(self.cfg['comm_range_m']) ** 2
        np.fill_diagonal(connected, False)
        self.adj = [np.flatnonzero(row).tolist() for row in connected]


@dataclass
class TaskPayload:
    task: int
    observation: object


def summarize(xs):
    a = np.asarray(xs, float)
    return dict(n=len(a), mean=float(a.mean()) if len(a) else None,
                p95=float(np.quantile(a, .95)) if len(a) else None,
                max=float(a.max()) if len(a) else None)


def _tracking_action(position, estimate, offset, ctrl):
    return clipped_velocity(position, estimate + offset, ctrl['kp'], ctrl['max_speed_mps'])


def _values(candidates, q, observation, beliefs, positions, references, weights, scenario,
            ctrl, dt, mode, context_snapshot=None, guard=True):
    values = {}
    scale = ctrl['reward_scale_m']
    for r in candidates:
        # A global-candidate ablation really visits all candidates. Unrelated
        # tasks get zero value rather than invented control relevance.
        w = float(weights[r, q])
        if q not in scenario.ever[r]:
            values[r] = 0.
            continue
        prior = beliefs[r, q]
        ref = references[r]
        if context_snapshot is not None and (r, q) in context_snapshot:
            prior = context_snapshot[r, q]
            ref = sum(weights[r, k] * context_snapshot.get((r, k), beliefs[r, k]).mean[:2]
                      for k in scenario.ever[r])
        post = prior.copy()
        post.update_position(observation.z_xy, observation.r_xy)
        old = _tracking_action(positions[r], ref, scenario.offsets[r], ctrl)
        new = _tracking_action(positions[r], ref + w * (post.mean[:2] - prior.mean[:2]), scenario.offsets[r], ctrl)
        action = dt * float(np.linalg.norm(new - old)) / scale
        decrease = max(0., float(np.trace(prior.cov[:2, :2] - post.cov[:2, :2])))
        u2 = max(0., float(np.trace(prior.cov[:2, :2]) / 2))
        protection = max(0., u2 / 36. - 1.) * (dt * ctrl['kp'] * w / scale) ** 2 * decrease if guard else 0.
        if mode == 'uncertainty':
            value = w * w * decrease / (scale * scale)
        elif mode == 'innovation':
            value = w * w * innovation_nis(prior.mean, prior.cov, observation.z_xy, observation.r_xy)
        elif mode == 'legacy_additive':
            value = action + w * math.sqrt(decrease) / scale
        else:
            value = action * action + protection
        values[r] = max(0., float(value / scenario.n))
    return values


def run_episode(job):
    start = time.perf_counter()
    capture = SnapshotCapture(job)
    n, tasks = job['n'], job['tasks']
    method = job['method']
    intuitive = method == 'intuitive'
    if method not in METHODS and method != 'external':
        raise ValueError(method)
    steps, seed, dt = job['steps'], job['seed'], .5
    cfg, prot = base_config(), protocol()
    scenario = Scenario(n, tasks, steps, seed, job['family'], job['scenario'], job['prep'])
    demand_env = DynamicDemandEnvironment(job, scenario) if job.get('dynamic_demand') else None
    intuitive_policy = IntuitivePolicy(job) if intuitive and demand_env is not None else None
    prep = 0 if method == 'no_prepare' else job['prep']
    groups = None if intuitive else PreparationIndex(scenario, prep)
    ctrl = cfg['controller']
    netcfg = dict(cfg['network'], bits_per_link_tick=job['capacity'], drop_probability=0.)
    net = VectorizedNetwork(n, netcfg, seed + 37, deadline_ticks=job['deadline'])
    gate_cfg = LocalGateConfig(**prot['frozen_r8_gate'])
    pressure_cfg = PressureConfig(opportunity_period=job['period'], replay_lag=job['deadline'],
                                  age_window=job['deadline'], pressure_strength=1.)
    initial = scenario.target_states(0)[:, :2]
    positions = scenario.before @ initial + scenario.offsets
    estimators = {}
    for r, qs in enumerate(scenario.ever):
        for q in qs:
            # Same broad source-position prior as the upstream single-task model.
            estimators[r, q] = ReplayEstimator(TargetBelief2D.broad(positions[r], 35., 7., 1.2), dt, job['deadline'])
    last_sent = {}
    ages = {(r, q): ReceivedAgeHistory(job['deadline']) for r, q in estimators}
    histories = deque(maxlen=job.get('context_lag', 0) + 1)
    disconnected = []
    rewards, errors, task_rewards = [], [], []
    trace, event_log = [], []
    source_opportunities = task_opportunities = admitted = 0
    candidate_visits = group_members_queried = 0
    selection_times, group_times, topology_times = [], [], []
    selected_counts, ages_at_delivery, queue_sizes = [], [], []
    protected_events = 0
    requested_snapshot = snapshot_upper_sum = 0
    max_group = max_hops = 0
    maximum_opportunity_recipients = 0
    event_feasibility_failures = bound_failures = 0
    plugin = None
    if method == 'external':
        module, name = job['external_selector'].split(':')
        # Factory creates fresh episode state even when worker processes are reused.
        plugin = getattr(importlib.import_module(module), name)(dict(job))
        if not callable(plugin):
            raise TypeError('External factory must return a callable episode policy')
    for tick in range(steps):
        common_warmup = demand_env is not None and tick < job['eval_start']
        if demand_env is not None:
            demand_env.begin_tick(tick, net)
        weights = scenario.weights(tick)
        targets = scenario.target_states(tick)
        clock = time.perf_counter()
        if groups is not None:
            groups.advance(tick)
        group_times.append(time.perf_counter() - clock)
        clock = time.perf_counter()
        net.rebuild(positions)
        topology_times.append(time.perf_counter() - clock)
        disconnected.append(len(net.reachable_from(0)) + 1 != n)
        remote = defaultdict(list)
        for delivery in net.receive(tick):
            if demand_env is not None:
                demand_env.delivery(tick, delivery)
            key = delivery.receiver, delivery.payload.task
            if key in estimators:
                remote[key].append(delivery.payload.observation)
                ages[key].observe(tick, delivery.created_tick)
            ages_at_delivery.append(tick - delivery.created_tick)
        due = demand_env.due(tick) if demand_env is not None else {r for r in range(n) if tick % job['period'] == r % job['period']}
        source_opportunities += len(due)
        obs, priors, beliefs = {}, {}, {}
        for key, est in estimators.items():
            r, q = key
            observation = None
            if weights[r, q] > 0:
                rng = np.random.default_rng(np.random.SeedSequence([seed, tick, r, q, 779]))
                observation = sample_observation(r, tick, positions[r], targets[q, :2], rng, cfg['sensor'])
                if job['family'] == 'MIS' and 16 <= tick < 48 and (r + q) % 2 == 0:
                    observation = None
            if demand_env is not None:
                observation = demand_env.apply_observation(tick, r, observation)
            obs[key] = observation
            if r in due and observation is not None and not intuitive:
                shadow = copy.deepcopy(est)
                priors[key] = shadow.advance(tick, remote[key]).copy()
            incoming = list(remote[key])
            if observation is not None:
                incoming.append(observation)
            beliefs[key] = est.advance(tick, incoming)
        references = np.array([sum(weights[r, q] * beliefs[r, q].mean[:2] for q in scenario.ever[r]) for r in range(n)])
        if demand_env is not None:
            demand_env.capture_warmup(tick, positions, beliefs, net, last_sent)
        if plugin is not None and hasattr(plugin, 'observe_step'):
            # Structure maintenance continues during silence and shared handover
            # protection; it is not delayed until a packet happens to be sent.
            plugin.observe_step(dict(tick=tick, positions=positions.copy(), references=references.copy(),
                                     current_roles=weights.copy(),
                                     future_roles=scenario.weights(min(steps-1,tick+prep)).copy()))
        actions = np.array([_tracking_action(positions[r], references[r], scenario.offsets[r], ctrl) for r in range(n)])
        desired = weights @ targets[:, :2] + scenario.offsets
        error = np.linalg.norm(positions - desired, axis=1)
        reward = np.exp(-error / ctrl['reward_scale_m'])
        if demand_env is not None:
            estimate_error = np.linalg.norm(references-weights @ targets[:, :2], axis=1)
            covariance = np.array([sum(weights[r, q]*float(np.trace(beliefs[r, q].cov[:2, :2]))
                                       for q in scenario.ever[r]) for r in range(n)])
            demand_env.observe(tick, reward, error, estimate_error, covariance)
        per_task = [float(np.average(reward, weights=weights[:, q])) for q in range(tasks)]
        if tick >= job['eval_start']:
            rewards.append(float(reward.mean()))
            errors.append(float(error.mean()))
            task_rewards.append(per_task)
        # Benchmark context-lag sensitivity: old beliefs predicted to the present.
        lagged = None
        lag = job.get('context_lag', 0)
        if lag:
            histories.append({key: b.copy() for key, b in beliefs.items()})
            old = histories[0]
            elapsed = len(histories) - 1
            lagged = {key: b.copy() for key, b in old.items()}
            for b in lagged.values():
                for _ in range(elapsed):
                    b.predict(dt)
        tick_selected = tick_events = 0
        for r in sorted(due):
            opportunity_recipients = 0
            for q in scenario.ever[r]:
                if weights[r, q] <= 0:
                    continue
                task_opportunities += 1
                o = obs[r, q]
                if demand_env is not None:
                    demand_env.source_opportunity(tick, o is not None)
                if o is None:
                    continue
                reachable = set(net.reachable_from(r))
                group = (set(np.flatnonzero(weights[:, q] > 0)) - {r}) if intuitive else groups.groups[q] - {r}
                max_group = max(max_group, len(group))
                protected = False if intuitive else groups.protected(q, tick)
                dense = method in {'dense_global', 'dense_task'}
                if intuitive:
                    # Common physical feasibility only. No TaskGraph source gate,
                    # preparation window, role protection, demand score or quota.
                    physical = bool(net.adj[r]) and net.min_first_hop_arrival_ticks(r, netcfg['payload_bits']) <= job['deadline']
                    gd = None
                else:
                    age, count = ages[r, q].summary(tick)
                    telemetry = LocalTelemetry(len(net.adj[r]), net.min_first_hop_arrival_ticks(r, netcfg['payload_bits']),
                                               net.incident_backlog_ticks(r), age, count)
                    gd = decide_pressure_local(priors[r, q], o, last_sent.get((r, q)), tick, gate_cfg,
                                               telemetry, pressure_cfg, use_received_age=True)
                    physical = bool(net.adj[r]) and telemetry.min_first_hop_arrival_ticks <= job['deadline']
                gate_admitted = common_warmup or intuitive or dense or protected or gd['send']
                if demand_env is not None:
                    demand_env.gate(tick, physical, gate_admitted)
                    if physical and demand_env.sample_audit(tick, r, q):
                        diagnostic_start = time.perf_counter()
                        audit_candidates = sorted(reachable & group)
                        audit_values = _values(audit_candidates, q, o, beliefs, positions, references, weights,
                                               scenario, ctrl, dt, 'full', guard=True)
                        demand_env.value_audit(tick, r, q, audit_values, gate_admitted,
                                               time.perf_counter()-diagnostic_start)
                if not physical or not gate_admitted:
                    continue
                global_external = method == 'external' and job.get('candidate_universe') == 'global'
                global_intuitive = intuitive and job['candidate_scope'] == 'global'
                candidates = sorted(reachable if method in {'dense_global', 'no_group'} or global_external or global_intuitive else reachable & group)
                if global_external and protected:
                    # Identical role-handover safeguard for all compared structures.
                    candidates = sorted(reachable & group)
                group_members_queried += len(group) if method not in {'dense_global', 'no_group'} else n - 1
                candidate_visits += len(candidates)
                if not candidates:
                    continue
                clock = time.perf_counter()
                routes = {} if intuitive else {receiver: {net._edge_key(a, b) for a, b in zip(net.path(r, receiver), net.path(r, receiver)[1:])}
                          for receiver in candidates}
                max_hops = max(max_hops, max(map(len, routes.values()), default=0))
                if intuitive or (plugin is not None and getattr(plugin, 'requires_controller_score', True) is False):
                    # Paper-inspired alternatives make independent decisions;
                    # they do not inherit TaskGraph's value score or quota.
                    values = {x: 0. for x in candidates}
                else:
                    values = _values(candidates, q, o, beliefs, positions, references, weights, scenario,
                                     ctrl, dt, method, lagged, guard=method != 'no_guard')
                fraction, absolute = job['fraction'], job.get('absolute', 0.)
                if common_warmup:
                    result = dict(selected=candidates, construction='common_dense_warmup', feasible=None)
                elif intuitive:
                    selected = intuitive_policy.choose(tick, r, q, candidates, positions, int(demand_env.phases[r])) if intuitive_policy is not None else choose_receivers(job, tick, r, q, candidates, positions)
                    result = dict(selected=selected, construction='intuitive', feasible=None)
                elif dense or protected:
                    result = select(values, routes, netcfg['payload_bits'], fraction, absolute, 'dense')
                    protected_events += int(protected)
                elif plugin is not None:
                    # Plugins get no target truth or future network.
                    chosen = plugin(dict(source=r, task=q, tick=tick, candidates=candidates,
                                         values=values, routes=routes, fraction=fraction,
                                         absolute=absolute, positions=positions.copy(),
                                         observation=dict(source=o.source, tick=o.tick, z_xy=o.z_xy.copy(), r_xy=o.r_xy.copy()),
                                         beliefs={x: beliefs[x, q].copy() for x in candidates if (x, q) in beliefs},
                                         source_belief=beliefs[r, q].copy(), references=references.copy(),
                                         current_roles=weights.copy(), future_roles=scenario.weights(min(steps-1,tick+prep)).copy(),
                                         offsets=scenario.offsets.copy(), controller=dict(ctrl),
                                         network=net, gate_decision=gd, gate_cfg=gate_cfg,
                                         payload_bits=netcfg['payload_bits'], period=job['period'], deadline=job['deadline']))
                    chosen = sorted(set(int(x) for x in chosen))
                    if not set(chosen) <= set(candidates):
                        raise ValueError('External selector returned noncandidate receivers')
                    result = dict(selected=chosen, construction='external', feasible=None)
                else:
                    strategy = 'additive' if method in {'demand_only', 'legacy_additive', 'uncertainty', 'innovation'} else 'shared'
                    if method == 'forward_only':
                        strategy = 'forward'
                    choice_routes = routes
                    if method == 'no_shared':
                        # Only selection cost changes. Physical forwarding still shares.
                        choice_routes = {x: {(x, e) for e in es} for x, es in routes.items()}
                    if job.get('simple_strategy'):
                        result = select_simple(values, choice_routes, netcfg['payload_bits'], fraction, absolute, job['simple_strategy'])
                    else:
                        result = select(values, choice_routes, netcfg['payload_bits'], fraction, absolute, strategy)
                selected = result['selected']
                elapsed = time.perf_counter() - clock
                selection_times.append(elapsed)
                if demand_env is not None:
                    demand_env.choice(tick, candidates, selected, elapsed)
                capture.add(tick, r, q, values, routes, netcfg['payload_bits'], protected)
                if result.get('feasible') is False:
                    event_feasibility_failures += 1
                actual_snapshot = net.multicast_cost(r, selected, netcfg['payload_bits'])
                if method in {'full', 'no_group', 'no_prepare', 'forward_only', 'no_guard'} and not protected:
                    if actual_snapshot > result.get('witness_cost', math.inf) + 1e-8:
                        bound_failures += 1
                snapshot_upper_sum += result.get('witness_cost', actual_snapshot)
                requested_snapshot += actual_snapshot
                if selected:
                    net.schedule_multicast(r, selected, netcfg['payload_bits'], tick, TaskPayload(q, o),
                                           deadline_ticks=job['deadline'])
                    last_sent[r, q] = tick
                    admitted += 1
                    selected_counts.append(len(selected))
                    opportunity_recipients += len(selected)
                    tick_selected += len(selected)
                    tick_events += 1
                if job.get('event_log', False):
                    event_log.append(dict(tick=tick, source=r, task=q, candidates=len(candidates), selected=len(selected),
                                          protected=protected, snapshot_cost=actual_snapshot,
                                          selection_seconds=elapsed, **{k: v for k, v in result.items() if k not in {'selected', 'snapshot_cost'}}))
            maximum_opportunity_recipients = max(maximum_opportunity_recipients, opportunity_recipients)
        net.service_tick(tick)
        if demand_env is not None:
            demand_env.after_service(tick, net)
        queue = net.queue_snapshot()['max_link_queue_work_ticks']
        queue_sizes.append(queue)
        trace.append(dict(tick=tick, reward=float(reward.mean()), error_m=float(error.mean()),
                          task_rewards=per_task, payload_bit_hop=net.payload_bit_hop,
                          queue_work=queue, selected=tick_selected, events=tick_events))
        positions += actions * dt
    ns = net.stats()
    accounting = ns['requested_recipient_copies'] - ns['delivered_packets'] - ns['expired_recipient_copies'] - ns['pending_deliveries']
    if accounting or event_feasibility_failures or bound_failures:
        raise AssertionError(f'Accounting/feasibility/bound failure: {accounting}, {event_feasibility_failures}, {bound_failures}')
    # Conservative attribution bound for this deadline network: propagation >=1,
    # no retransmission, and every service branch retains its recipient labels.
    H_safe = job['deadline'] + 1
    count_bound = netcfg['payload_bits'] * H_safe * ns['requested_recipient_copies']
    opportunity_bound = netcfg['payload_bits'] * n * math.ceil(steps / job['period']) * maximum_opportunity_recipients * H_safe
    if ns['payload_bit_hop'] > min(count_bound, opportunity_bound):
        raise AssertionError('Finite-horizon service accounting bound violated')
    window = [x for x in trace if scenario.tau - 8 <= x['tick'] < scenario.tau + 32]
    dynamic_result = demand_env.finish(net) if demand_env is not None else None
    return dict(job=job, implementation='multitask-extension-v1', is_smoke=job.get('smoke', False),
                is_AirSim=False, receiver_context='benchmark_provided', receiver_context_overhead_accounted=False,
                evaluation=dict(reward=summarize(rewards), tracking_error_m=summarize(errors),
                                per_task_mean=np.mean(task_rewards, axis=0).tolist(),
                                worst_task_reward=float(np.min(np.mean(task_rewards, axis=0))),
                                switch_window_reward=summarize([x['reward'] for x in window]),
                                switch_window_error_m=summarize([x['error_m'] for x in window])),
                network=ns, recipient_accounting_difference=accounting,
                cost_per_agent_tick=ns['payload_bit_hop'] / (n * steps),
                source_opportunities=source_opportunities, task_opportunities=task_opportunities,
                admitted_events=admitted, candidate_visits=candidate_visits,
                preparation_query_member_visits=group_members_queried,
                selected_count=summarize(selected_counts), delivered_age=summarize(ages_at_delivery),
                selection_seconds=summarize(selection_times), preparation_seconds=summarize(group_times),
                topology_seconds=summarize(topology_times), max_queue_work=max(queue_sizes, default=0.),
                disconnected_tick_fraction=float(np.mean(disconnected)), preparation_updates=0 if groups is None else groups.updates,
                maintained_task_filters=len(estimators), max_group=max_group, max_snapshot_hops=max_hops,
                protected_events=protected_events, snapshot_offered_sum=requested_snapshot,
                theory_audit=dict(snapshot_bound_failures=bound_failures, quota_failures=event_feasibility_failures,
                                  offered_snapshot_sum=requested_snapshot, snapshot_feasible_upper_sum=snapshot_upper_sum,
                                  counted_service_upper=count_bound, opportunity_service_upper=opportunity_bound,
                                  observed_K=maximum_opportunity_recipients, deadline_service_H=H_safe,
                                  notice='observed K audits this run; it is not an a-priori scalability certificate'),
                structure_audit=plugin.summary() if plugin is not None and hasattr(plugin, 'summary') else {},
                trace=trace, events=event_log, snapshots=capture.export(),
                snapshot_capture=dict(eligible_events=capture.visited, retained=len(capture.heap), seconds=capture.seconds),
                dynamic_demand=dynamic_result,
                elapsed_seconds=time.perf_counter() - start)

