from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import numpy as np

from airsim_ext.controller import clipped_velocity
from r6f.selector import expected_position_gain, innovation_nis


def _sym(a):
    a = np.asarray(a, dtype=float)
    return 0.5 * (a + a.T)


@dataclass(frozen=True)
class DynamicHyperedgeConfig:
    """Dimensionless set objective for development-stage dynamic hyperedges.

    The objective normalizes total receiver-side task value and the full reachable
    multicast-tree cost to one.  ``lambda0`` therefore has a direct interpretation:
    at pressure=0 and source_strength=1, one unit of normalized service costs
    ``lambda0`` units of normalized task value.

    This selector intentionally has *no fixed recipient count*.  Cardinality is the
    result of greedy positive marginal utility under the current receiver beliefs,
    task/controller sensitivity, route sharing and network pressure.
    """
    lambda0: float = 0.50
    redundancy_strength: float = 0.65
    min_uniqueness: float = 0.15
    max_metric_strength: float = 4.0
    pressure_cost_gain: float = 1.0
    source_value_relief: float = 0.00
    critical_reason_relief: float = 0.20
    delay_discount_floor: float = 0.05


CRITICAL_REASONS = {'innovation_guard', 'uncertainty_guard'}


def _position_uncertainty(belief) -> float:
    return float(math.sqrt(max(float(np.trace(_sym(belief.cov[:2, :2]))) / 2.0, 0.0)))


def _updated_copy(belief, observation):
    b = belief.copy()
    b.update_position(observation.z_xy, observation.r_xy)
    return b


def _path_edges(net, source: int, receiver: int):
    path = net.path(int(source), int(receiver))
    if not path or len(path) < 2:
        return frozenset()
    return frozenset(net._edge_key(int(a), int(b)) for a, b in zip(path[:-1], path[1:]))


def _path_delay_ticks(net, source: int, receiver: int, bits: int) -> float:
    """Current-state delivery-time proxy using current shortest path and queues.

    It is a ranking diagnostic, not a future-route guarantee.  The real experiment
    still performs hop-by-hop dynamic forwarding and expiration exactly as before.
    """
    path = net.path(int(source), int(receiver))
    if not path or len(path) < 2:
        return float('inf')
    cap = max(1, int(net.cfg['bits_per_link_tick']))
    prop = max(0, int(net.cfg.get('prop_delay_ticks', 1)))
    total = 0.0
    for a, b in zip(path[:-1], path[1:]):
        key = net._edge_key(int(a), int(b))
        backlog = int(net.queued_bits_by_edge.get(key, 0))
        total += math.ceil((backlog + int(bits)) / cap - 1e-12) + prop
    return float(total)


def _safe_standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd = np.where(sd > 1e-8, sd, 1.0)
    return (x - mu) / sd


def _similarity_matrix(features: np.ndarray) -> np.ndarray:
    z = _safe_standardize(features)
    d = max(1, z.shape[1])
    diff = z[:, None, :] - z[None, :, :]
    d2 = np.sum(diff * diff, axis=2) / float(d)
    s = np.exp(-0.5 * d2)
    np.fill_diagonal(s, 1.0)
    return s


def _source_strength(gate_decision: dict, gate_cfg) -> float:
    gd = gate_decision or {}
    gain = float(gd.get('gain_ratio', 0.0)) / max(float(gd.get('effective_gain_threshold', gate_cfg.gain_threshold)), 1e-12)
    nis = float(gd.get('innovation_nis', 0.0)) / max(float(gd.get('effective_nis_threshold', gate_cfg.nis_threshold)), 1e-12)
    unc = float(gd.get('uncertainty_std_m', 0.0)) / max(float(gate_cfg.uncertainty_guard_m), 1e-12)
    silence = float(gd.get('silence_ticks', 0.0)) / max(float(gate_cfg.max_silence_ticks), 1.0)
    return max(1.0, gain, nis, unc, silence)


def _receiver_rows(source, reachable, beliefs, positions, offsets, observation,
                   net, payload_bits, gate_cfg, ctrl_cfg, deadline_ticks,
                   cfg: DynamicHyperedgeConfig):
    source = int(source)
    candidates = sorted(set(int(r) for r in reachable if int(r) != source))
    if observation is None or not candidates:
        return candidates, [], np.zeros((0, 0)), set()

    src_belief = beliefs[source]
    max_speed = max(float(ctrl_cfg['max_speed_mps']), 1e-9)
    action_scale = max(0.10 * max_speed, 1e-6)
    u0 = max(float(gate_cfg.uncertainty_guard_m), 1e-9)
    gamma0 = max(float(gate_cfg.gain_threshold), 1e-12)
    eta0 = max(float(gate_cfg.nis_threshold), 1e-12)
    src_unc = _position_uncertainty(src_belief)

    rows = []
    feat = []
    all_edges = set()
    for r in candidates:
        b = beliefs[r]
        bp = _updated_copy(b, observation)
        gain = expected_position_gain(b.cov, observation.r_xy)
        nis = innovation_nis(b.mean, b.cov, observation.z_xy, observation.r_xy)
        unc = _position_uncertainty(b)
        mismatch = float(np.linalg.norm(np.asarray(src_belief.mean[:2]) - np.asarray(b.mean[:2]))) / max(src_unc + unc, 1e-6)

        a0 = clipped_velocity(positions[r], b.mean[:2] + offsets[r], ctrl_cfg['kp'], ctrl_cfg['max_speed_mps'])
        a1 = clipped_velocity(positions[r], bp.mean[:2] + offsets[r], ctrl_cfg['kp'], ctrl_cfg['max_speed_mps'])
        action_delta = float(np.linalg.norm(a1 - a0))

        metrics = {
            'gain': gain / gamma0,
            'innovation': math.sqrt(max(nis / eta0, 0.0)),
            'uncertainty': unc / u0,
            'belief_mismatch': mismatch,
            'action_relevance': action_delta / action_scale,
        }
        raw = max(metrics.values())
        raw = min(float(cfg.max_metric_strength), max(0.0, float(raw)))

        delay = _path_delay_ticks(net, source, r, payload_bits)
        if math.isfinite(delay):
            delay_frac = delay / max(1.0, float(deadline_ticks))
            discount = max(float(cfg.delay_discount_floor), 1.0 - min(1.0, delay_frac))
        else:
            delay_frac = float('inf')
            discount = 0.0
        value = raw * discount
        edges = _path_edges(net, source, r)
        all_edges.update(edges)
        rows.append({
            'receiver': int(r), 'raw_need': float(raw), 'discounted_need': float(value),
            'gain_strength': float(metrics['gain']), 'innovation_strength': float(metrics['innovation']),
            'uncertainty_strength': float(metrics['uncertainty']), 'belief_mismatch_strength': float(metrics['belief_mismatch']),
            'action_relevance_strength': float(metrics['action_relevance']), 'delay_ticks_proxy': float(delay),
            'delay_fraction': float(delay_frac) if math.isfinite(delay_frac) else None,
            'route_edges': edges,
        })
        # Set redundancy is defined over receiver task state, not only geometry.
        feat.append(np.r_[
            np.asarray(b.mean[:2], float) - np.asarray(src_belief.mean[:2], float),
            np.sqrt(np.maximum(np.diag(_sym(b.cov[:2, :2])), 1e-12)),
            np.asarray(a0, float) / max_speed,
            np.asarray(offsets[r], float),
        ])

    sim = _similarity_matrix(np.asarray(feat, dtype=float))
    return candidates, rows, sim, all_edges


def select_dynamic_hyperedge(source: int, reachable, beliefs, positions, offsets,
                             observation, net, payload_bits: int, gate_reason: str,
                             gate_decision: dict, gate_cfg, ctrl_cfg,
                             deadline_ticks: int, mode: str = 'set_conditioned',
                             cfg: DynamicHyperedgeConfig = DynamicHyperedgeConfig()) -> dict:
    """Greedily synthesize a variable-cardinality task hyperedge.

    ``additive``: receiver-specific value + set-conditioned multicast cost.
    ``set_conditioned``: additionally discounts the marginal value of receivers whose
    belief/control-role state is redundant with the already selected set.

    No target truth, future random realization, oracle action or future route is used.
    Current receiver beliefs *are* used; therefore this stage is a scientific
    development diagnostic for genuine hyperedge formation, not yet a claim that the
    same context can be obtained at zero communication cost in a deployed system.
    """
    if mode not in {'additive', 'set_conditioned'}:
        raise ValueError(mode)
    candidates, rows, sim, global_edges = _receiver_rows(
        source, reachable, beliefs, positions, offsets, observation, net,
        int(payload_bits), gate_cfg, ctrl_cfg, int(deadline_ticks), cfg)
    if not rows:
        return {
            'scope_kind': f'dynamic_{mode}', 'scope_fanout_cap': None,
            'selected_recipients': [], 'dynamic_selector': {'mode': mode, 'cardinality': 0}
        }

    needs = np.asarray([max(0.0, x['discounted_need']) for x in rows], dtype=float)
    total_need = float(needs.sum())
    if total_need <= 1e-15:
        shares = np.ones(len(rows), dtype=float) / len(rows)
    else:
        shares = needs / total_need

    edge_denom = max(1, len(global_edges))
    pressure = float((gate_decision or {}).get('pressure', 0.0))
    src_strength = _source_strength(gate_decision or {}, gate_cfg)
    reason = str(gate_reason)
    denom = 1.0 + float(cfg.source_value_relief) * max(0.0, src_strength - 1.0)
    if reason in CRITICAL_REASONS:
        denom += float(cfg.critical_reason_relief)
    lambda_eff = float(cfg.lambda0) * (1.0 + float(cfg.pressure_cost_gain) * pressure) / denom

    selected_idx = []
    selected = []
    current_edges = set()
    trace = []
    available = set(range(len(rows)))

    while available:
        best = None
        for idx in sorted(available, key=lambda k: rows[k]['receiver']):
            if selected_idx and mode == 'set_conditioned':
                redundancy = max(float(sim[idx, q]) for q in selected_idx)
                uniqueness = max(float(cfg.min_uniqueness), 1.0 - float(cfg.redundancy_strength) * redundancy)
            else:
                redundancy = 0.0
                uniqueness = 1.0
            marginal_value = float(shares[idx]) * uniqueness
            new_edges = set(rows[idx]['route_edges']) - current_edges
            marginal_cost = len(new_edges) / float(edge_denom)
            score = marginal_value - lambda_eff * marginal_cost
            density = marginal_value / max(marginal_cost, 1.0 / (1000.0 * edge_denom))
            cand = (score, density, marginal_value, -marginal_cost, -rows[idx]['receiver'], idx,
                    redundancy, uniqueness, new_edges)
            if best is None or cand[:5] > best[:5]:
                best = cand
        if best is None or best[0] <= 1e-12:
            break
        score, density, marginal_value, neg_cost, neg_receiver, idx, redundancy, uniqueness, new_edges = best
        selected_idx.append(idx)
        available.remove(idx)
        selected.append(int(rows[idx]['receiver']))
        current_edges.update(new_edges)
        trace.append({
            'receiver': int(rows[idx]['receiver']), 'score': float(score), 'value_density': float(density),
            'marginal_value': float(marginal_value), 'marginal_cost_fraction': float(-neg_cost),
            'max_similarity_to_selected_before': float(redundancy), 'uniqueness_factor': float(uniqueness),
            'cumulative_tree_edge_fraction': float(len(current_edges) / edge_denom),
        })

    selected_set = set(selected)
    selected_need = float(sum(shares[i] for i in selected_idx))
    selected_raw = [rows[i] for i in selected_idx]
    diag = {
        'mode': mode,
        'cardinality': int(len(selected)),
        'reachable_count': int(len(rows)),
        'lambda_effective': float(lambda_eff),
        'source_strength': float(src_strength),
        'pressure': float(pressure),
        'selected_need_share_before_redundancy': float(selected_need),
        'tree_edge_fraction': float(len(current_edges) / edge_denom),
        'global_tree_edges': int(edge_denom),
        'selected_mean_discounted_need': float(np.mean([r['discounted_need'] for r in selected_raw])) if selected_raw else 0.0,
        'all_mean_discounted_need': float(np.mean(needs)) if len(needs) else 0.0,
        'trace': trace,
        'selected_receiver_metrics': [
            {k: v for k, v in r.items() if k != 'route_edges'} for r in selected_raw
        ],
        'config': asdict(cfg),
    }
    return {
        'scope_kind': f'dynamic_{mode}', 'scope_fanout_cap': None,
        'selected_recipients': selected, 'dynamic_selector': diag,
    }
