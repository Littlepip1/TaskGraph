from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass
class SensorObservation:
    source: int
    tick: int
    z_xy: np.ndarray
    r_xy: np.ndarray
    source_xy: np.ndarray
    target_distance: float


def anisotropic_covariance(source_xy: np.ndarray, target_xy: np.ndarray,
                           sigma_perp: float, sigma_parallel: float,
                           distance_scale: float = 0.0) -> np.ndarray:
    """Position covariance with large uncertainty along line of sight.

    Two views from different directions can therefore be jointly more informative than
    either alone, giving the AirSim experiment a natural source-complementarity case.
    """
    d = np.asarray(target_xy, dtype=np.float64) - np.asarray(source_xy, dtype=np.float64)
    dist = float(np.linalg.norm(d))
    if dist < 1e-9:
        u = np.array([1.0, 0.0], dtype=np.float64)
    else:
        u = d / dist
    v = np.array([-u[1], u[0]], dtype=np.float64)
    scale = 1.0 + float(distance_scale) * dist
    sp = float(sigma_parallel) * scale
    st = float(sigma_perp) * scale
    # Columns are principal directions; diagonal is variance in that frame.
    rot = np.column_stack([u, v])
    return rot @ np.diag([sp**2, st**2]) @ rot.T


def expected_detection_probability(source_xy: np.ndarray, target_xy: np.ndarray,
                                   sensor_range_m: float, base_dropout: float) -> float:
    dist = float(np.linalg.norm(np.asarray(target_xy) - np.asarray(source_xy)))
    if dist >= sensor_range_m:
        return 0.0
    x = dist / max(sensor_range_m, 1e-6)
    # Smoothly make edge-of-range measurements less reliable without requiring a
    # scene-specific vision detector in the first AirSim integration.
    p = (1.0 - base_dropout) * (1.0 - 0.55 * x * x)
    return float(np.clip(p, 0.05, 1.0))


def sample_observation(source: int, tick: int, source_xy: np.ndarray, target_xy: np.ndarray,
                       rng: np.random.Generator, sensor_cfg: dict) -> SensorObservation | None:
    source_xy = np.asarray(source_xy, dtype=np.float64).reshape(2)
    target_xy = np.asarray(target_xy, dtype=np.float64).reshape(2)
    p_det = expected_detection_probability(
        source_xy, target_xy,
        float(sensor_cfg['range_m']),
        float(sensor_cfg.get('base_dropout', 0.05)),
    )
    if rng.random() > p_det:
        return None
    r = anisotropic_covariance(
        source_xy, target_xy,
        float(sensor_cfg['sigma_perp_m']),
        float(sensor_cfg['sigma_parallel_m']),
        float(sensor_cfg.get('distance_scale', 0.0)),
    )
    z = rng.multivariate_normal(target_xy, r)
    return SensorObservation(source, tick, z, r, source_xy,
                             float(np.linalg.norm(target_xy - source_xy)))
