from __future__ import annotations

import math
import numpy as np


def formation_offsets(n: int, radius_m: float) -> np.ndarray:
    out = np.zeros((n, 2), dtype=np.float64)
    for i in range(n):
        a = 2.0 * math.pi * i / max(1, n)
        out[i] = [radius_m * math.cos(a), radius_m * math.sin(a)]
    return out


def clipped_velocity(current_xy: np.ndarray, desired_xy: np.ndarray, kp: float, max_speed: float) -> np.ndarray:
    v = float(kp) * (np.asarray(desired_xy, dtype=np.float64) - np.asarray(current_xy, dtype=np.float64))
    s = float(np.linalg.norm(v))
    if s > max_speed > 0:
        v = v * (max_speed / s)
    return v


def normalized_tracking_reward(drone_xy: np.ndarray, true_target_xy: np.ndarray,
                               offset_xy: np.ndarray, scale_m: float) -> float:
    desired = np.asarray(true_target_xy) + np.asarray(offset_xy)
    err = float(np.linalg.norm(np.asarray(drone_xy) - desired))
    return float(math.exp(-err / max(float(scale_m), 1e-6)))
