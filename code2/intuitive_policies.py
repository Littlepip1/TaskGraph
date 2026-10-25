"""Intuitive communication rules; no beliefs, controller scores or task hypergraph.

The simulator supplies only candidate IDs and (for the nearest rule) geometry.
Random subsets are uniform at a fixed count, reproducible and nested across rates.
"""
from __future__ import annotations
import math
import numpy as np


class IntuitivePolicy:
    """Episode-local policy state. No receiver beliefs or blackout labels."""
    def __init__(self, job):
        self.seed, self.period = job['seed'], job['period']
        self.rule = job['intuitive_rule']
        self.fraction = float(job.get('keep_fraction', 1.))
        self.multiplier = int(job.get('period_multiplier', 1))
        self.warmup = int(job['eval_start'])
        self.cursors = {}

    def choose(self, tick, source, task, candidates, positions, phase):
        ids = sorted(candidates)
        if self.rule == 'none':
            return []
        if self.rule == 'periodic':
            first = (self.warmup-phase+self.period-1)//self.period
            index = (tick-phase)//self.period-first
            return ids if index % self.multiplier == 0 else []
        k = min(len(ids), math.ceil(self.fraction*len(ids)))
        if self.rule == 'round_robin':
            if not ids:
                return []
            key = source, task
            if key not in self.cursors:
                rng = np.random.default_rng(np.random.SeedSequence([self.seed, source, task, 91005]))
                self.cursors[key] = int(rng.integers(len(ids)))
            start = self.cursors[key]
            chosen = [ids[(start+i) % len(ids)] for i in range(k)]
            self.cursors[key] = (start+k) % len(ids)
            return sorted(chosen)
        return choose_receivers(dict(seed=self.seed, period=self.period, intuitive_rule=self.rule,
                                     keep_fraction=self.fraction), tick, source, task, ids, positions)


def choose_receivers(job, tick, source, task, candidates, positions):
    ids = sorted(candidates)
    rule = job['intuitive_rule']
    if rule == 'none':
        return []
    if rule == 'periodic':
        opportunity = (tick - source % job['period']) // job['period']
        return ids if opportunity % int(job['period_multiplier']) == 0 else []
    fraction = float(job.get('keep_fraction', 1.))
    if not 0 <= fraction <= 1:
        raise ValueError('keep_fraction must be between zero and one')
    count = min(len(ids), math.ceil(fraction * len(ids)))
    if rule == 'random':
        rng = np.random.default_rng(np.random.SeedSequence([job['seed'], tick, source, task, 632917]))
        return sorted(rng.permutation(ids)[:count].tolist())
    if rule == 'nearest':
        ordered = sorted(ids, key=lambda r: (float(np.sum((positions[r]-positions[source])**2)), r))
        return sorted(ordered[:count])
    raise ValueError('Unknown intuitive rule: ' + str(rule))
