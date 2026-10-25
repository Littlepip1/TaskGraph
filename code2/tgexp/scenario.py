"""Prespecified target trajectories and task roles, independent of method.

Role switching is real: it changes the target estimate used by the controller.
Coordinator weights give genuine overlapping control dependencies.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np


def hex_offsets(n, spacing=10.715136106222515):
    radius = max(1, math.ceil(math.sqrt(n)))
    lattice = [(a, b) for a in range(-radius, radius + 1) for b in range(-radius, radius + 1)]
    lattice.sort(key=lambda ab: (ab[0] ** 2 + ab[0] * ab[1] + ab[1] ** 2, ab))
    xy = np.array([[spacing * (a + b / 2), spacing * math.sqrt(3) * b / 2] for a, b in lattice[:n]])
    return xy - xy.mean(axis=0)


@dataclass
class Scenario:
    n: int
    tasks: int
    steps: int
    seed: int
    family: str
    mode: str
    prep: int = 32
    spacing: float = 70.

    def __post_init__(self):
        if self.n < 2 or not 1 <= self.tasks <= self.n or self.mode not in {'static', 'overlap', 'switch'}:
            raise ValueError('Invalid scenario')
        if self.family not in {'SP', 'MIS'} or self.steps < 2 or self.prep < 0:
            raise ValueError('Invalid trajectory/time settings')
        self.home = np.arange(self.n) * self.tasks // self.n
        self.before = np.zeros((self.n, self.tasks))
        self.before[np.arange(self.n), self.home] = 1.
        self.after = self.before.copy()
        self.tau = self.steps // 2
        width = math.ceil(math.sqrt(self.tasks))
        self.centers = np.array([[q % width * self.spacing, q // width * self.spacing] for q in range(self.tasks)])
        self.offsets = np.zeros((self.n, 2))
        for q in range(self.tasks):
            ids = np.flatnonzero(self.home == q)
            self.offsets[ids] = hex_offsets(len(ids))
        if self.tasks > 1 and self.mode in {'overlap', 'switch'}:
            for q in range(0, self.tasks - 1, 2):
                for task, neighbor in [(q, q + 1), (q + 1, q)]:
                    ids = np.flatnonzero(self.home == task)
                    count = max(1, len(ids) // 10)
                    # Boundary members really use both target estimates.
                    direction = self.centers[neighbor] - self.centers[task]
                    ids = ids[np.argsort(self.offsets[ids] @ direction)[-count:]]
                    self.before[ids, task] = .75
                    self.before[ids, neighbor] = .25
                    self.after[ids] = self.before[ids]
                    if self.mode == 'switch':
                        # Separate handover agents acquire a previously unused
                        # task. Their dependency set genuinely changes at tau.
                        others = np.array([r for r in np.flatnonzero(self.home == task) if r not in ids], int)
                        if len(others):
                            handover = others[np.argsort(self.offsets[others] @ direction)[-count:]]
                            self.after[handover, task] = 0.
                            self.after[handover, neighbor] = 1.
        self.changed_tasks = set(np.flatnonzero(np.any(self.before != self.after, axis=0)))
        # At most two task filters per agent in these generated scenarios.
        self.ever = [tuple(np.flatnonzero((self.before[r] + self.after[r]) > 0)) for r in range(self.n)]
        self.heading = float(np.random.default_rng(self.seed).uniform(-math.pi, math.pi))

    def weights(self, tick):
        return self.before if tick < self.tau else self.after

    def target_states(self, tick, dt=.5):
        t = tick * dt
        c, s = math.cos(self.heading), math.sin(self.heading)
        rotation = np.array([[c, -s], [s, c]])
        rows = []
        for q in range(self.tasks):
            if self.family == 'SP':
                a = t * 5 / 30 + .18 * (q % 4)
                local = np.array([30 * math.cos(a), 30 * math.sin(a)])
                velocity = np.array([-5 * math.sin(a), 5 * math.cos(a)])
            else:
                local = np.array([30., 3 * t]) if t < 12 else np.array([30 + 3 * (t - 12), 36.])
                local += np.array([0., 2. * (q % 2)])
                velocity = np.array([0., 3.]) if t < 12 else np.array([3., 0.])
            rows.append(np.r_[self.centers[q] + rotation @ local, rotation @ velocity])
        return np.asarray(rows)


class PreparationIndex:
    """Rolling union from known role intervals; changes applied atomically.

    Membership support counts avoid deleting a task association still supported
    by another interval. Query is by task, not a scan over the whole team.
    """
    def __init__(self, scenario, prep):
        self.scenario, self.prep = scenario, int(prep)
        self.groups = [set() for _ in range(scenario.tasks)]
        self.counts = {}
        self.events = {}
        for r in range(scenario.n):
            for start, end, weights in [(0, scenario.tau, scenario.before),
                                        (scenario.tau, scenario.steps, scenario.after)]:
                for q in np.flatnonzero(weights[r] > 0):
                    key = r, int(q)
                    self.events.setdefault(max(0, start - self.prep), []).append((key, 1))
                    self.events.setdefault(end, []).append((key, -1))
        self.tick = -1
        self.updates = 0

    def advance(self, tick):
        if tick != self.tick + 1:
            raise ValueError('Sequential preparation updates required')
        changes = {}
        for key, delta in self.events.get(tick, []):
            changes[key] = changes.get(key, 0) + delta
        for (r, q), delta in changes.items():
            count = self.counts.get((r, q), 0) + delta
            if count < 0:
                raise AssertionError('Negative association count')
            self.counts[r, q] = count
            if count:
                self.groups[q].add(r)
            else:
                self.groups[q].discard(r)
            self.updates += 1
        self.tick = tick

    def protected(self, q, tick):
        return q in self.scenario.changed_tasks and self.scenario.tau - self.prep <= tick < self.scenario.tau + self.prep

