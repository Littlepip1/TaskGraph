"""Exogenous missing-observation schedule and evaluation-only measurements.

No schedule or measurement here is passed to the receiver-selection policy.
Ground truth is used by evaluation only. Full method scores remain unchanged.
"""
from __future__ import annotations
from collections import deque
from dataclasses import fields, is_dataclass
import hashlib
import json
import numpy as np

PROFILES = ('normal', 'spread', 'fixed', 'rotating')


def canonical(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return {f.name: canonical(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return [[canonical(k), canonical(v)] for k, v in sorted(value.items(), key=lambda kv: repr(kv[0]))]
    if isinstance(value, (list, tuple, deque)):
        return [canonical(v) for v in value]
    if isinstance(value, set):
        return [canonical(v) for v in sorted(value)]
    return value


def fingerprint(value):
    return hashlib.sha256(json.dumps(canonical(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


class DynamicDemandEnvironment:
    def __init__(self, job, scenario):
        self.job, self.scenario = job, scenario
        self.n, self.steps = scenario.n, scenario.steps
        self.warmup = int(job['eval_start'])
        self.profile = job['demand_profile']
        self.starts = list(map(int, job['blackout_starts']))
        self.duration = int(job['blackout_duration'])
        self.tail = int(job['recovery_tail'])
        if self.profile not in PROFILES or scenario.changed_tasks:
            raise ValueError('Dynamic-demand experiment requires a valid profile and fixed roles')
        if self.warmup < 1 or not self.starts or self.starts[0] < self.warmup:
            raise ValueError('Invalid warmup/event schedule')
        if any(s < 0 or s+self.duration+self.tail > self.steps for s in self.starts):
            raise ValueError('Event window outside horizon')
        if any(a+self.duration+self.tail > b for a, b in zip(self.starts, self.starts[1:])):
            raise ValueError('Overlapping evaluation event windows are not allowed')
        self.phases = np.empty(self.n, dtype=int)
        self.counts = {}
        rotations = [set() for _ in self.starts]
        fixed = set()
        for q in range(scenario.tasks):
            ids = np.flatnonzero(scenario.home == q)
            h = max(1, int(np.floor(len(ids)*float(job['blackout_fraction'])+.5)))
            if h >= len(ids):
                raise ValueError('Blackout must leave some local observers')
            self.counts[str(q)] = dict(members=len(ids), blocked=h)
            rng = np.random.default_rng(np.random.SeedSequence([job['seed'], q, 91001]))
            order = rng.permutation(ids).tolist()
            fixed.update(order[:h])
            for e in range(len(self.starts)):
                rotations[e].update(order[(e*h+i) % len(order)] for i in range(h))
            # Balanced phases remove ID/task-period aliasing. A separate stream
            # makes phase assignment independent of blackout assignment.
            rng_phase = np.random.default_rng(np.random.SeedSequence([job['seed'], q, 91002]))
            phase_ids = rng_phase.permutation(ids)
            offset = int(rng_phase.integers(job['period']))
            self.phases[phase_ids] = (np.arange(len(ids))*job['period']//len(ids)+offset) % job['period']
        self.cohorts = [sorted(fixed if self.profile == 'fixed' else x) for x in rotations]
        self.mask = np.zeros((self.steps, self.n), dtype=bool)
        for e, start in enumerate(self.starts):
            for tick in range(start, start+self.duration):
                if self.profile in {'fixed', 'rotating'}:
                    self.mask[tick, self.cohorts[e]] = True
                elif self.profile == 'spread':
                    for q in range(scenario.tasks):
                        ids = np.flatnonzero(scenario.home == q)
                        rng = np.random.default_rng(np.random.SeedSequence([job['seed'], q, tick, 91003]))
                        self.mask[tick, rng.permutation(ids)[:self.counts[str(q)]['blocked']]] = True
        self.schedule_hash = fingerprint(dict(profile=self.profile, starts=self.starts, duration=self.duration,
                                              cohorts=self.cohorts, mask=self.mask))
        self.phase_hash = fingerprint(self.phases)
        shape = self.steps, self.n
        self.reward = np.zeros(shape)
        self.error = np.zeros(shape)
        self.estimate_error = np.zeros(shape)
        self.covariance = np.zeros(shape)
        self.pre_observations = np.zeros(self.steps, dtype=int)
        self.suppressed_observations = np.zeros(self.steps, dtype=int)
        self.first = [{r: None for r in cohort} for cohort in self.cohorts]
        self.fresh_counts = [{r: 0 for r in cohort} for cohort in self.cohorts]
        self.seen_deliveries = set()
        self.tick_trace = []
        self.stage_counts = dict(source_task_opportunities=0, valid_observation_opportunities=0,
                                 physical_feasible_opportunities=0, gate_admitted_opportunities=0,
                                 nonempty_multicasts=0, candidate_copies=0, selected_copies=0,
                                 blocked_candidate_copies=0, blocked_selected_copies=0,
                                 nonblocked_selected_copies=0)
        self.value_audits = []
        self.diagnostic_seconds = 0.
        self.selection_seconds = []
        self.selected_tick = np.zeros(self.steps, dtype=int)
        self.blocked_selected_tick = np.zeros(self.steps, dtype=int)
        self.warmup_cost = None
        self.warmup_state_hash = None

    def due(self, tick):
        return set(np.flatnonzero(tick % self.job['period'] == self.phases).tolist())

    def begin_tick(self, tick, network):
        if tick == self.warmup:
            self.warmup_cost = int(network.payload_bit_hop)

    def capture_warmup(self, tick, positions, beliefs, network, last_sent):
        if tick != self.warmup:
            return
        self.warmup_state_hash = fingerprint(dict(positions=positions, last_sent=last_sent,
            beliefs={k: dict(mean=b.mean, cov=b.cov) for k, b in beliefs.items()},
            queues=network.link_queues, arrivals=network.arrivals, stalled=network.stalled,
            counter=network._message_counter, rng=network.rng.bit_generator.state, stats=network.stats()))

    def apply_observation(self, tick, receiver, observation):
        if observation is not None:
            self.pre_observations[tick] += 1
            if self.mask[tick, receiver]:
                self.suppressed_observations[tick] += 1
                return None
        return observation

    def delivery(self, tick, delivery):
        r, task = delivery.receiver, delivery.payload.task
        if task not in self.scenario.ever[r] or self.scenario.before[r, task] <= 0:
            return
        identity = r, task, delivery.payload.observation.source, delivery.payload.observation.tick
        if identity in self.seen_deliveries:
            return
        self.seen_deliveries.add(identity)
        for e, start in enumerate(self.starts):
            if start <= tick < start+self.duration and r in self.first[e] and delivery.created_tick >= start:
                if self.first[e][r] is None:
                    self.first[e][r] = tick-start
                self.fresh_counts[e][r] += 1

    def observe(self, tick, reward, error, estimate_error, covariance):
        self.reward[tick] = reward
        self.error[tick] = error
        self.estimate_error[tick] = estimate_error
        self.covariance[tick] = covariance

    def source_opportunity(self, tick, has_observation):
        if tick >= self.warmup:
            self.stage_counts['source_task_opportunities'] += 1
            self.stage_counts['valid_observation_opportunities'] += int(has_observation)

    def gate(self, tick, physical, admitted):
        if tick >= self.warmup:
            self.stage_counts['physical_feasible_opportunities'] += int(physical)
            self.stage_counts['gate_admitted_opportunities'] += int(physical and admitted)

    def choice(self, tick, candidates, selected, seconds):
        if tick < self.warmup:
            return
        c = self.stage_counts
        self.selection_seconds.append(seconds)
        c['nonempty_multicasts'] += int(bool(selected))
        c['candidate_copies'] += len(candidates)
        c['selected_copies'] += len(selected)
        c['blocked_candidate_copies'] += int(self.mask[tick, candidates].sum())
        blocked = int(self.mask[tick, selected].sum()) if selected else 0
        c['blocked_selected_copies'] += blocked
        c['nonblocked_selected_copies'] += len(selected)-blocked
        self.selected_tick[tick] += len(selected)
        self.blocked_selected_tick[tick] += blocked

    def sample_audit(self, tick, source, task):
        if tick < self.warmup or not self.job.get('value_audit', True):
            return False
        b = f'{self.job["seed"]}:{tick}:{source}:{task}:91004'.encode()
        return int.from_bytes(hashlib.sha256(b).digest()[:4], 'big') % 16 == 0

    def value_audit(self, tick, source, task, values, gate_admitted, seconds):
        total = float(sum(values.values()))
        blocked = float(sum(v for r, v in values.items() if self.mask[tick, r]))
        self.value_audits.append(dict(tick=int(tick), source=int(source), task=int(task), gate_admitted=bool(gate_admitted),
                                     total_proxy_value=total, blocked_proxy_value=blocked,
                                     blocked_proxy_fraction=blocked/total if total > 0 else None))
        self.diagnostic_seconds += seconds

    def after_service(self, tick, network):
        blocked = self.mask[tick]
        row = dict(tick=tick, blocked_count=int(blocked.sum()),
                   served_payload_bit_hop=int(network.payload_bit_hop),
                   local_valid_before_mask=int(self.pre_observations[tick]),
                   local_suppressed=int(self.suppressed_observations[tick]),
                   selected_copies=int(self.selected_tick[tick]), blocked_selected_copies=int(self.blocked_selected_tick[tick]))
        for name, array in [('reward', self.reward), ('tracking_error_m', self.error),
                            ('estimate_error_m', self.estimate_error), ('weighted_cov_trace', self.covariance)]:
            row['all_'+name] = float(array[tick].mean())
            row['blocked_'+name] = float(array[tick, blocked].mean()) if blocked.any() else None
            row['unblocked_'+name] = float(array[tick, ~blocked].mean()) if (~blocked).any() else None
        for e, start in enumerate(self.starts):
            if start <= tick < start+self.duration+self.tail:
                cohort = self.cohorts[e]
                row.update(event=e, relative_tick=tick-start,
                           cohort_reward=float(self.reward[tick, cohort].mean()),
                           cohort_tracking_error_m=float(self.error[tick, cohort].mean()),
                           cohort_estimate_error_m=float(self.estimate_error[tick, cohort].mean()),
                           cohort_weighted_cov_trace=float(self.covariance[tick, cohort].mean()),
                           cohort_fresh_update_fraction=float(np.mean([v is not None for v in self.first[e].values()])))
        self.tick_trace.append(row)

    def finish(self, network):
        if self.warmup_cost is None or self.warmup_state_hash is None:
            raise AssertionError('Warmup checkpoint missing')
        events, individual = [], []
        for e, start in enumerate(self.starts):
            cohort = self.cohorts[e]
            end = start+self.duration
            stop = end+self.tail
            matched_reward = float(self.reward[start:stop, cohort].mean())
            # Spread uses actual blocked member-times. Others use a persistent
            # cohort plus recovery. Both measures are saved, never conflated.
            active_mask = self.mask[start:end]
            instant_reward = float(self.reward[start:end][active_mask].mean()) if active_mask.any() else None
            primary = instant_reward if self.profile == 'spread' else matched_reward
            event = dict(event=e, start=start, end=end, evaluation_end=stop, cohort=cohort,
                         primary_reward=primary, matched_cohort_reward=matched_reward,
                         instantaneous_blocked_reward=instant_reward,
                         cohort_tracking_error_m=float(self.error[start:stop, cohort].mean()),
                         cohort_estimate_error_m=float(self.estimate_error[start:stop, cohort].mean()),
                         cohort_weighted_cov_trace=float(self.covariance[start:stop, cohort].mean()))
            events.append(event)
            for r in cohort:
                latency = self.first[e][r]
                individual.append(dict(event=e, receiver=r, first_fresh_update_ticks=latency,
                                       wait_capped_ticks=self.duration if latency is None else latency,
                                       no_fresh_update=latency is None, fresh_observations=self.fresh_counts[e][r],
                                       **{f'updated_within_{h}': latency is not None and latency < h for h in [4, 8, 16]}))
        waits = [r['wait_capped_ticks'] for r in individual]
        result = dict(profile=self.profile, schedule_hash=self.schedule_hash, phase_hash=self.phase_hash,
                      warmup_state_hash=self.warmup_state_hash, source_phases=self.phases.tolist(),
                      primary_event_basis='instantaneous_blocked_member_times' if self.profile == 'spread' else 'cohort_with_recovery',
                      update_cohort_basis='virtual_matched_cohort' if self.profile in {'normal', 'spread'} else 'persistent_blocked_cohort',
                      home_group_counts=self.counts,
                      actual_blocked_fraction=sum(x['blocked'] for x in self.counts.values())/self.n if self.profile != 'normal' else 0.,
                      injected_blocked_agent_ticks=int(self.mask.sum()),
                      suppressed_valid_local_observations=int(self.suppressed_observations.sum()),
                      events=events, fresh_update_individual=individual,
                      fresh_update=dict(no_update_fraction=float(np.mean([r['no_fresh_update'] for r in individual])),
                                        mean_capped_wait_ticks=float(np.mean(waits)),
                                        **{f'within_{h}_fraction': float(np.mean([r[f'updated_within_{h}'] for r in individual])) for h in [4, 8, 16]}),
                      communication=dict(warmup_payload_bit_hop=self.warmup_cost,
                                         evaluation_payload_bit_hop=int(network.payload_bit_hop)-self.warmup_cost,
                                         evaluation_agent_ticks=self.n*(self.steps-self.warmup),
                                         pending_recipients_end=network.pending(),
                                         endpoint_queue=network.queue_snapshot()),
                      stage_counts=self.stage_counts, value_audits=self.value_audits,
                      online_selector_mean_ms=1000*float(np.mean(self.selection_seconds)) if self.selection_seconds else 0.,
                      diagnostic_seconds=self.diagnostic_seconds, time_series=self.tick_trace)
        if self.job.get('save_agent_arrays', False):
            result['_agent_arrays'] = dict(reward=self.reward.astype('float32'), tracking_error_m=self.error.astype('float32'),
                                           estimate_error_m=self.estimate_error.astype('float32'),
                                           weighted_cov_trace=self.covariance.astype('float32'), blocked=self.mask)
        return result
