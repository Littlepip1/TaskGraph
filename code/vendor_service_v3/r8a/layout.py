from __future__ import annotations
import copy
import math
import numpy as np

BASE_N8_RING_RADIUS_M = 14.0
BASE_NEIGHBOR_SPACING_M = 2.0 * BASE_N8_RING_RADIUS_M * math.sin(math.pi / 8.0)
SUPPORTED_N = (8, 16, 32, 64)


def _rot(theta: float) -> np.ndarray:
    c, s = math.cos(float(theta)), math.sin(float(theta))
    return np.asarray([[c, -s], [s, c]], dtype=float)


def _hex_candidates(spacing: float, span: int = 16):
    rows = []
    for q in range(-span, span + 1):
        for r in range(-span, span + 1):
            x = spacing * (q + 0.5 * r)
            y = spacing * (math.sqrt(3.0) / 2.0) * r
            rr = math.hypot(x, y)
            a = math.atan2(y, x)
            rows.append((rr, a, q, r, x, y))
    return rows


def hex_disk_offsets(n: int, spacing_m: float = BASE_NEIGHBOR_SPACING_M) -> np.ndarray:
    """Nearest points of a centered hexagonal lattice with deterministic balanced ties.

    Density is held fixed while physical radius grows approximately with sqrt(N).  At a
    partially filled outer shell, points are selected approximately uniformly in angle;
    the final cloud is recentered exactly so the target remains its centroid.
    """
    n = int(n)
    if n not in SUPPORTED_N:
        raise ValueError(f'Unsupported R8-A N={n}')
    rows = _hex_candidates(float(spacing_m), span=20)
    # Group equal radii robustly; a lattice shell can contain many angularly distinct points.
    by_radius = {}
    for row in rows:
        key = round(row[0], 10)
        by_radius.setdefault(key, []).append(row)
    selected = []
    for key in sorted(by_radius):
        shell = sorted(by_radius[key], key=lambda z: z[1])
        need = n - len(selected)
        if need <= 0:
            break
        if len(shell) <= need:
            selected.extend(shell)
        else:
            # Choose evenly spaced angular positions from the last shell.
            ids = []
            for k in range(need):
                idx = int(math.floor((k + 0.5) * len(shell) / need)) % len(shell)
                while idx in ids:
                    idx = (idx + 1) % len(shell)
                ids.append(idx)
            selected.extend(shell[i] for i in sorted(ids))
            break
    pts = np.asarray([[z[4], z[5]] for z in selected], dtype=float)
    if len(pts) != n:
        raise RuntimeError('Could not construct requested hex layout')
    pts -= pts.mean(axis=0, keepdims=True)
    # Stable source IDs: sweep by polar angle, then radius. This spreads modulo-8 clock
    # classes around the geometry instead of grouping them by lattice construction order.
    angle = np.arctan2(pts[:, 1], pts[:, 0])
    radius = np.linalg.norm(pts, axis=1)
    order = np.lexsort((radius, angle))
    return pts[order]


def rotated_offsets(n: int, theta: float) -> np.ndarray:
    return hex_disk_offsets(int(n)) @ _rot(theta).T


def layout_diagnostics(n: int, theta: float = 0.0) -> dict:
    pts = rotated_offsets(n, theta)
    dists = []
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dists.append(float(np.linalg.norm(pts[i] - pts[j])))
    nearest = []
    for i in range(len(pts)):
        nearest.append(min(float(np.linalg.norm(pts[i] - pts[j])) for j in range(len(pts)) if j != i))
    return {
        'agents': int(n),
        'spacing_reference_m': float(BASE_NEIGHBOR_SPACING_M),
        'max_offset_radius_m': float(np.linalg.norm(pts, axis=1).max()),
        'mean_nearest_neighbor_m': float(np.mean(nearest)),
        'min_nearest_neighbor_m': float(np.min(nearest)),
        'max_pair_distance_m': float(max(dists, default=0.0)),
        'centroid_xy': [float(x) for x in pts.mean(axis=0)],
    }


def make_r8a_config(base_cfg: dict, n: int, task_spec: dict) -> tuple[dict, np.ndarray]:
    n = int(n)
    cfg = copy.deepcopy(base_cfg)
    names = [f'Drone{i+1}' for i in range(n)]
    cfg['airsim']['trackers'] = names
    offsets = rotated_offsets(n, float(task_spec['formation_rotation_rad']))
    # Import lazily to avoid a circular dependency at module import time.
    from r8a.tasks import target_state_for_spec
    target0 = np.asarray(target_state_for_spec(task_spec, 0.0)[:2], dtype=float)
    origins = {name: [float(x) for x in target0 + offsets[i]] for i, name in enumerate(names)}
    origins[cfg['airsim']['target']] = [float(x) for x in target0]
    cfg['airsim']['vehicle_origin_xy'] = origins
    cfg['controller']['formation_radius_m'] = float(np.linalg.norm(offsets, axis=1).max())
    cfg['controller']['formation_layout'] = 'hex_disk_density_preserving'
    cfg['controller']['formation_spacing_m'] = float(BASE_NEIGHBOR_SPACING_M)
    return cfg, offsets
