from __future__ import annotations

import copy
import numpy as np

from airsim_ext.belief import TargetBelief2D
from airsim_ext.controller import clipped_velocity, normalized_tracking_reward
from airsim_ext.sensing import sample_observation
from r4.estimation import ReplayEstimator
from r6f.selector import LocalGateConfig
from r7.selector import PressureConfig, ReceivedAgeHistory, LocalTelemetry, decide_pressure_local
from r8a.network import HopByHopDynamicNetwork
from r8b.scope import ScopeConfig, adaptive_diverse_scope, fixed_diverse_scope
from dynamic_set import DynamicConfig, select_dynamic
from service_opt import ServiceConfig, select_service_optimized

METHODS = ('uniform_global', 'pressure_global', 'fixed_diverse40', 'r8_final', 'taskgraph_dynamic', 'taskgraph_service_opt')
GATE_METHODS = {'pressure_global', 'fixed_diverse40', 'r8_final', 'taskgraph_dynamic', 'taskgraph_service_opt'}


def due_sources(tick: int, n: int):
    """Frozen R7/R8 opportunity clock: one opportunity/source/16 ticks."""
    phase = int(tick) % 16
    if phase % 2 or phase > 14:
        return ()
    cls = phase // 2
    return tuple(j for j in range(int(n)) if j % 8 == cls)


def static_keep(source: int) -> bool:
    """Frozen 7/8 simple comparator."""
    return int(source) % 8 != 2


def _summary(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return {'n': 0, 'mean': None, 'min': None, 'max': None}
    return {'n': int(len(a)), 'mean': float(a.mean()), 'min': float(a.min()), 'max': float(a.max())}


def _network_cfg(base_cfg, capacity):
    cfg = copy.deepcopy(base_cfg)
    cfg['bits_per_link_tick'] = int(capacity)
    cfg['drop_probability'] = 0.0
    return cfg


def _max_distance(source, recipients, positions):
    if not recipients:
        return 0.0
    p = np.asarray(positions, dtype=float)
    return float(max(np.linalg.norm(p[int(r)] - p[int(source)]) for r in recipients))


def scope_descriptor(method: str):
    if method in {'uniform_global', 'pressure_global'}:
        return None
    if method == 'fixed_diverse40':
        return {'kind': 'diverse_fixed', 'fanout': 40}
    if method == 'taskgraph_service_opt':
        return {'kind': 'service_optimized_dynamic', 'recipient_cap': None}
    if method == 'taskgraph_dynamic':
        return {'kind': 'dynamic_task_value', 'recipient_cap': None, 'stopping': 'event_value_omission_budget'}
    if method == 'r8_final':
        return {
            'kind': 'diverse_adaptive',
            'routine_fanout': 32,
            'critical_fanout': 48,
            'critical_reasons': ['innovation_guard', 'uncertainty_guard'],
        }
    raise ValueError(method)


def select_scope(method, source, reachable, positions, gate_reason):
    desc = scope_descriptor(method)
    if desc is None:
        return {'scope_kind': 'global', 'scope_fanout_cap': None,
                'selected_recipients': list(reachable)}
    if desc['kind'] == 'diverse_fixed':
        return fixed_diverse_scope(source, reachable, positions, int(desc['fanout']))
    cfg = ScopeConfig(desc['routine_fanout'], desc['critical_fanout'])
    return adaptive_diverse_scope(source, reachable, positions, gate_reason, cfg)


class FrozenR8CAirSimExperiment:
    """AirSim confirmation of the already-frozen R8-C communication rule.

    AirSim supplies vehicle motion. Sensing remains the same truth-driven noisy local
    observation adapter used in earlier external-validation work; it is not a vision
    detector. The communication/network layer is the final hop-by-hop R8 model.
    """

    def __init__(self, config, client, offsets_xy, capacity,
                 gate_cfg: LocalGateConfig, pressure_cfg: PressureConfig,
                 replay_lag_ticks: int = 32, layer1=None, dynamic_cfg=None, task_factors=(), service_cfg=None):
        self.cfg = copy.deepcopy(config)
        self.client = client
        self.layer1 = layer1
        self.dynamic_cfg = dynamic_cfg or DynamicConfig()
        self.dynamic_cfg.validate()
        self.task_factors = tuple(task_factors)
        self.service_cfg = service_cfg or ServiceConfig()
        self.service_cfg.validate()
        if layer1 is not None:
            layer1.validate_agent_count(len(config['airsim']['trackers']))
        self.offsets = np.asarray(offsets_xy, dtype=float).copy()
        self.n = len(config['airsim']['trackers'])
        if self.n != 64:
            raise ValueError('AirSim targeted validation is frozen at N=64')
        if self.offsets.shape != (self.n, 2):
            raise ValueError('Offset shape differs from agent count')
        self.dt = float(config['experiment'].get('dt_s', 0.5))
        self.eval_start = int(config['experiment'].get('eval_start_step', 20))
        self.sensor_cfg = config['sensor']
        self.planner_cfg = config['planner']
        self.ctrl_cfg = config['controller']
        self.net_cfg = _network_cfg(config['network'], capacity)
        self.gate_cfg = gate_cfg
        self.pressure_cfg = pressure_cfg
        self.replay_lag = int(replay_lag_ticks)
        if int(pressure_cfg.replay_lag) != self.replay_lag:
            raise ValueError('Pressure deadline must equal estimator replay lag')

    def run(self, method: str, seed: int, steps: int) -> dict:
        if method not in METHODS:
            raise ValueError(method)
        if self.layer1 is not None:
            self.layer1.reset_audit()
        self.client.connect()
        self.client.reset_and_prepare()
        if hasattr(self.client, 'mark_collision_baseline'):
            self.client.mark_collision_baseline()
        if getattr(self.client, 'fixed_step_control', False):
            self.client.pause(True)
        self.client.start_target_path()

        xy0 = self.client.positions_xy()
        target0 = self.client.target_state_xyv()[:2]
        spawn_err = float(np.mean(np.linalg.norm((xy0 - target0) - self.offsets, axis=1)))
        estimators = [
            ReplayEstimator(
                TargetBelief2D.broad(
                    xy0[i],
                    self.planner_cfg['initial_pos_std_m'],
                    self.planner_cfg['initial_vel_std_mps'],
                    self.planner_cfg['process_accel_std_mps2'],
                ),
                self.dt,
                self.replay_lag,
            )
            for i in range(self.n)
        ]
        net = HopByHopDynamicNetwork(
            self.n, self.net_cfg, int(seed) + 37, deadline_ticks=self.replay_lag
        )
        last_sent = [None] * self.n
        age_history = [ReceivedAgeHistory(self.pressure_cfg.age_window) for _ in range(self.n)]

        rewards, formation, esterr, actionerr = [], [], [], []
        ages, eval_ages, decisions = [], [], []
        sent = opp = suppressed = disconnected_ticks = 0
        queue_trace, reachable_fraction = [], []
        selected_counts, selected_distances = [], []
        received_per_agent = np.zeros(self.n, dtype=int)
        payload_bits = int(self.net_cfg['payload_bits'])

        try:
            for tick in range(int(steps)):
                if tick == 0 or (tick + 1) % 10 == 0:
                    print(f'[AirSim episode] method={method} seed={seed} tick={tick+1}/{steps}', flush=True)
                self.client.pause(True)
                positions = self.client.positions_xy()
                target = self.client.target_state_xyv()[:2]
                net.rebuild(positions)
                reachable0 = len(net.reachable_from(0)) + 1
                reachable_fraction.append(reachable0 / self.n)
                if reachable0 != self.n:
                    disconnected_ticks += 1

                remote = [[] for _ in range(self.n)]
                for d in net.receive(tick):
                    remote[d.receiver].append(d.payload)
                    received_per_agent[int(d.receiver)] += 1
                    age = tick - int(d.payload.tick)
                    ages.append(age)
                    age_history[d.receiver].observe(tick, int(d.payload.tick))
                    if tick >= self.eval_start:
                        eval_ages.append(age)

                obs = []
                for j in range(self.n):
                    rng = np.random.default_rng(np.random.SeedSequence([int(seed), tick, j]))
                    o = sample_observation(j, tick, positions[j], target, rng, self.sensor_cfg)
                    if hasattr(self.client, 'sensor_available') and not self.client.sensor_available(j, tick):
                        o = None
                    obs.append(o)

                due = due_sources(tick, self.n)
                priors = {}
                if method in GATE_METHODS:
                    for j in due:
                        shadow = copy.deepcopy(estimators[j])
                        priors[j] = shadow.advance(tick, list(remote[j])).copy()

                beliefs = []
                for i, est in enumerate(estimators):
                    rec = list(remote[i])
                    if obs[i] is not None:
                        rec.append(obs[i])
                    beliefs.append(est.advance(tick, rec))

                actions = np.stack([
                    clipped_velocity(
                        positions[i], beliefs[i].mean[:2] + self.offsets[i],
                        self.ctrl_cfg['kp'], self.ctrl_cfg['max_speed_mps']
                    )
                    for i in range(self.n)
                ])
                oracle = np.stack([
                    clipped_velocity(
                        positions[i], target + self.offsets[i],
                        self.ctrl_cfg['kp'], self.ctrl_cfg['max_speed_mps']
                    )
                    for i in range(self.n)
                ])

                if tick >= self.eval_start:
                    rewards.append(float(np.mean([
                        normalized_tracking_reward(
                            positions[i], target, self.offsets[i], self.ctrl_cfg['reward_scale_m']
                        ) for i in range(self.n)
                    ])))
                    formation.append(float(np.mean([
                        np.linalg.norm((positions[i] - target) - self.offsets[i])
                        for i in range(self.n)
                    ])))
                    esterr.append(float(np.mean([
                        np.linalg.norm(beliefs[i].mean[:2] - target) for i in range(self.n)
                    ])))
                    actionerr.append(float(np.mean(np.linalg.norm(actions - oracle, axis=1))))

                for j in due:
                    j = int(j)
                    opp += 1
                    o = obs[j]
                    reachable = net.reachable_from(j)
                    raw_reachable_count = len(reachable)
                    if self.layer1 is not None:
                        reachable = self.layer1.filter(j, reachable, tick)
                    should_send = (o is not None)
                    gate_reason = 'baseline_send' if should_send else 'no_measurement'
                    gd = {}

                    if method in GATE_METHODS:
                        age, count = age_history[j].summary(tick)
                        telemetry = LocalTelemetry(
                            neighbor_count=len(net.adj[j]),
                            min_first_hop_arrival_ticks=net.min_first_hop_arrival_ticks(j, payload_bits),
                            max_incident_wait_ticks=net.incident_backlog_ticks(j),
                            recent_received_mean_age_ticks=float(age),
                            received_age_samples=int(count),
                        )
                        gd = decide_pressure_local(
                            priors[j], o, last_sent[j], tick, self.gate_cfg,
                            telemetry, self.pressure_cfg, use_received_age=True
                        )
                        should_send = bool(gd['send'])
                        gate_reason = str(gd['reason'])

                    if method == 'taskgraph_service_opt':
                        sd = select_service_optimized(j, reachable, positions, beliefs, o if should_send else None,
                                            self.offsets, self.ctrl_cfg, net, self.dt, self.replay_lag,
                                            cfg=self.service_cfg, factors=self.task_factors)
                    elif method == 'taskgraph_dynamic':
                        sd = select_dynamic(j, reachable, positions, beliefs, o if should_send else None,
                                            self.offsets, self.ctrl_cfg, net, self.dt, self.replay_lag,
                                            cfg=self.dynamic_cfg, factors=self.task_factors)
                    else:
                        sd = select_scope(method, j, reachable, positions, gate_reason)
                    recipients = sd['selected_recipients']
                    decision = {
                        'tick': int(tick), 'source': j, 'method': method,
                        'had_measurement': o is not None,
                        'reachable_recipients': int(len(reachable)),
                        'send': bool(should_send), 'reason': gate_reason,
                        **gd, **sd,
                    }
                    if self.layer1 is not None:
                        decision.update(layer1_raw_reachable_count=raw_reachable_count,
                                        layer1_candidate_count=len(reachable),
                                        layer1_group_version=self.layer1.version)
                    scheduled = bool(should_send and recipients)
                    if scheduled:
                        net.schedule_multicast(
                            j, recipients, payload_bits, tick, o,
                            deadline_ticks=self.replay_lag
                        )
                        last_sent[j] = tick
                        sent += 1
                        selected_counts.append(len(recipients))
                        selected_distances.append(_max_distance(j, recipients, positions))
                    else:
                        suppressed += int(o is not None)
                    decision['selected_recipient_count'] = int(len(recipients) if should_send else 0)
                    decision['actually_scheduled'] = scheduled
                    decisions.append(decision)

                net.service_tick(tick)
                queue_trace.append(float(net.queue_snapshot()['max_link_queue_work_ticks']))
                if not getattr(self.client, 'fixed_step_control', False):
                    self.client.pause(False)
                self.client.command_tracker_velocities(actions, self.dt)

            self.client.hover()
        finally:
            self.client.pause(False)

        ns = net.stats()
        accounted = ns['delivered_packets'] + ns['expired_recipient_copies'] + ns['pending_deliveries']
        ns['recipient_accounting_difference'] = int(ns['requested_recipient_copies'] - accounted)
        ns['pending_after_horizon_fraction'] = float(ns['pending_deliveries'] / max(1, ns['requested_recipient_copies']))
        ns['expired_recipient_fraction'] = float(ns['expired_recipient_copies'] / max(1, ns['requested_recipient_copies']))
        ns['delivered_recipient_fraction'] = float(ns['delivered_packets'] / max(1, ns['requested_recipient_copies']))
        ns['mean_max_link_queue_work_ticks'] = float(np.mean(queue_trace)) if queue_trace else 0.0
        ns['estimator_too_old_observations'] = int(sum(e.too_old for e in estimators))
        collisions = self.client.new_collision_events() if hasattr(self.client, 'new_collision_events') else []

        result = {
            'implementation_version': 'FinalAirSim-v1',
            'clock_audit': getattr(self.client, 'clock_audit', []),
            'method': method,
            'agents': self.n,
            'seed': int(seed),
            'steps': int(steps),
            'eval_start_step': self.eval_start,
            'gate_config': self.gate_cfg.__dict__ if method in GATE_METHODS else None,
            'pressure_config': self.pressure_cfg.__dict__ if method in GATE_METHODS else None,
            'scope_config': scope_descriptor(method),
            'evaluation': {
                'team_reward': _summary(rewards),
                'formation_error_m': _summary(formation),
                'estimation_error_m': _summary(esterr),
                'oracle_action_error_mps': _summary(actionerr),
            },
            'network': ns,
            'scheduled_opportunities': int(opp),
            'sent_payloads': int(sent),
            'suppressed_payloads': int(suppressed),
            'send_fraction_of_opportunities': float(sent / opp) if opp else 0.0,
            'selected_recipient_count': _summary(selected_counts),
            'selected_max_distance_m': _summary(selected_distances),
            'received_packets_per_agent': _summary(received_per_agent),
            'delivered_age_ticks': _summary(ages),
            'eval_delivered_age_ticks': _summary(eval_ages),
            'disconnected_control_ticks': int(disconnected_ticks),
            'mean_reachable_fraction_from_node0': float(np.mean(reachable_fraction)) if reachable_fraction else 0.0,
            'bit_hop_per_agent_control_tick': float(ns['total_bit_hop']) / max(1.0, float(self.n * steps)),
            'spawn_formation_error_m': spawn_err,
            'max_formation_offset_m': float(np.linalg.norm(self.offsets, axis=1).max()),
            'new_collision_events': collisions,
            'new_collision_count': int(len(collisions)),
            'send_decisions': decisions,
            'claim_scope': (
                'Final frozen TaskGraph-MSC AirSim external validation at N=64. The held-out comparison uses Uniform-global and the frozen adaptive 32/48 TaskGraph-MSC rule only. No algorithm parameter is retuned.'
            ),
        }
        if self.layer1 is not None:
            result['layer1_audit'] = self.layer1.summary()
            result['claim_scope'] = ('Two-layer integration using frozen R3 second layer; '
                                     'not the original frozen AirSim confirmation protocol. '
                                     'Static group snapshot is preloaded, with no online maintenance traffic.')
        if method == 'taskgraph_dynamic':
            result['implementation_version'] = 'DynamicSet-v2-development'
            result['claim_scope'] = ('New dynamic receiver-set algorithm with no fixed recipient count. '
                'Current receiver beliefs are benchmark-provided; their acquisition cost is not included. '
                'Event surrogate feasibility does not certify closed-loop MSC. '
                'Not a reproduction of historical dhg_set results.')
        if method == 'taskgraph_service_opt':
            result['implementation_version'] = 'ServiceOpt-v3-development'
            result['claim_scope'] = ('No fixed recipient count; controller-impact proxy and shared-branch pruning. '
                'Benchmark receiver context acquisition is unaccounted. Event proxy is not a return certificate.')
        return result
