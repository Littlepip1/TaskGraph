from __future__ import annotations
import copy
import math
import numpy as np
from r1.test_support import KinematicTestClient
from r8a.layout import SUPPORTED_N


def _rot(theta):
    c, s = math.cos(float(theta)), math.sin(float(theta))
    return np.asarray([[c, -s], [s, c]], dtype=float)


def make_r8a_task_spec(seed: int, family: str, n: int) -> dict:
    seed = int(seed); n = int(n)
    if family not in {'circle_shifted', 'turn_dropout_randomized'}:
        raise ValueError(family)
    if n not in SUPPORTED_N:
        raise ValueError(n)
    tag = 811 if family == 'circle_shifted' else 821
    # Geometry and time structure are paired across N for a given seed.
    rg = np.random.default_rng(np.random.SeedSequence([seed, tag, 1]))
    # Only source identities/count depend on N.
    rd = np.random.default_rng(np.random.SeedSequence([seed, tag, 2, n]))
    spec = {
        'family': family,
        'agents': n,
        'formation_rotation_rad': float(rg.uniform(0.0, 2.0 * math.pi)),
        'target_path_rotation_rad': float(rg.uniform(0.0, 2.0 * math.pi)),
        'target_phase_rad': float(rg.uniform(0.0, 2.0 * math.pi)),
    }
    if family == 'circle_shifted':
        k = max(1, n // 4)
        spec.update(
            dropout_sources=sorted(int(x) for x in rd.choice(n, size=k, replace=False)),
            dropout_start_tick=int(rg.integers(50, 81)),
            dropout_duration_ticks=int(rg.integers(20, 41)),
            turn_direction=0,
        )
    else:
        k = max(1, n // 2)
        spec.update(
            dropout_sources=sorted(int(x) for x in rd.choice(n, size=k, replace=False)),
            dropout_start_tick=int(rg.integers(45, 76)),
            dropout_duration_ticks=int(rg.integers(30, 51)),
            turn_direction=int(rg.choice([-1, 1])),
        )
    return spec


def target_state_for_spec(spec: dict, time_s: float) -> np.ndarray:
    t = float(time_s)
    world = _rot(float(spec.get('target_path_rotation_rad', 0.0)))
    if spec['family'] == 'circle_shifted':
        radius = 30.0; speed = 5.0; omega = speed / radius
        a = omega * t + float(spec.get('target_phase_rad', 0.0))
        pos = np.asarray([radius * math.cos(a), radius * math.sin(a)], dtype=float)
        vel = np.asarray([-speed * math.sin(a), speed * math.cos(a)], dtype=float)
        return np.r_[world @ pos, world @ vel]
    direction = int(spec.get('turn_direction', 1))
    if t < 30.0:
        pos = np.asarray([30.0, 3.0 * t], dtype=float)
        vel = np.asarray([0.0, 3.0], dtype=float)
    else:
        pos = np.asarray([30.0 + direction * 3.0 * (t - 30.0), 90.0], dtype=float)
        vel = np.asarray([direction * 3.0, 0.0], dtype=float)
    # target_phase gives a second, seed-paired orientation perturbation rather than an
    # initial formation displacement. The same transform applies to position and velocity.
    world2 = _rot(float(spec.get('target_phase_rad', 0.0))) @ world
    return np.r_[world2 @ pos, world2 @ vel]


class R8AKinematicClient(KinematicTestClient):
    """Clean software benchmark: spawn exactly on the rotated desired formation."""
    def __init__(self, airsim_cfg: dict, task_spec: dict):
        self.task_spec = copy.deepcopy(task_spec)
        super().__init__(airsim_cfg)

    def target_state_xyv(self):
        return target_state_for_spec(self.task_spec, self.time)

    def sensor_available(self, agent, tick):
        s = self.task_spec
        start = int(s['dropout_start_tick'])
        end = start + int(s['dropout_duration_ticks'])
        return not (start <= int(tick) < end and int(agent) in set(s['dropout_sources']))
