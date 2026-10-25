from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import numpy as np


def _sym(a):
    return 0.5 * (np.asarray(a, float) + np.asarray(a, float).T)


@dataclass(frozen=True)
class LocalGateConfig:
    gain_threshold: float
    nis_threshold: float
    uncertainty_guard_m: float
    max_silence_ticks: int
    efficiency_threshold: float = 0.0


@dataclass
class GateDecision:
    send: bool
    reason: str
    gain_ratio: float
    innovation_nis: float
    uncertainty_std_m: float
    multicast_edges: int
    value_per_edge: float
    silence_ticks: int

    def to_dict(self):
        return asdict(self)


def expected_position_gain(prior_cov4, measurement_cov2):
    p = _sym(np.asarray(prior_cov4, float)[:2, :2])
    r = _sym(np.asarray(measurement_cov2, float).reshape(2, 2))
    eye = np.eye(2)
    p = p + eye * 1e-10
    r = r + eye * 1e-10
    post = np.linalg.inv(np.linalg.inv(p) + np.linalg.inv(r))
    denom = max(float(np.trace(p)), 1e-12)
    gain = max(0.0, float(np.trace(p) - np.trace(post)))
    return gain / denom


def innovation_nis(prior_mean4, prior_cov4, z_xy, r_xy):
    p = _sym(np.asarray(prior_cov4, float)[:2, :2])
    r = _sym(np.asarray(r_xy, float).reshape(2, 2))
    e = np.asarray(z_xy, float).reshape(2) - np.asarray(prior_mean4, float)[:2]
    s = p + r + np.eye(2) * 1e-10
    return float(e.T @ np.linalg.solve(s, e))


def decide_sender_local(prior_belief, observation, multicast_cost_bits, payload_bits,
                        last_sent_tick, tick, cfg: LocalGateConfig):
    """Pure sender-local deletion gate.

    There is deliberately no team-global suppression counter. Each sender decides only
    from information locally available before its already scheduled slot. The local
    max-silence guard prevents repeated suppression of the same source without requiring
    coordination among sources.
    """
    if observation is None:
        return GateDecision(False, 'no_measurement', 0.0, 0.0,
                            float(math.sqrt(max(np.trace(prior_belief.cov[:2, :2]) / 2.0, 0.0))),
                            0, 0.0, int(tick - last_sent_tick) if last_sent_tick is not None else 10**9)
    gain = expected_position_gain(prior_belief.cov, observation.r_xy)
    nis = innovation_nis(prior_belief.mean, prior_belief.cov, observation.z_xy, observation.r_xy)
    unc = float(math.sqrt(max(np.trace(prior_belief.cov[:2, :2]) / 2.0, 0.0)))
    edges = max(1, int(round(float(multicast_cost_bits) / max(1, int(payload_bits)))))
    eff = float(gain / edges)
    silence = int(tick - last_sent_tick) if last_sent_tick is not None else 10**9

    if silence >= int(cfg.max_silence_ticks):
        return GateDecision(True, 'refresh_guard', gain, nis, unc, edges, eff, silence)
    if unc >= float(cfg.uncertainty_guard_m):
        return GateDecision(True, 'uncertainty_guard', gain, nis, unc, edges, eff, silence)
    if nis >= float(cfg.nis_threshold):
        return GateDecision(True, 'innovation_guard', gain, nis, unc, edges, eff, silence)
    if gain >= float(cfg.gain_threshold) and eff >= float(cfg.efficiency_threshold):
        return GateDecision(True, 'expected_gain', gain, nis, unc, edges, eff, silence)
    return GateDecision(False, 'safe_suppress', gain, nis, unc, edges, eff, silence)
