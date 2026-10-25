"""Known-timestamp replay of received raw observations; no future data."""
import numpy as np
from airsim_ext.belief import delayed_measurement_to_now


class LegacyEstimator:
    def __init__(self, initial, dt, lag=32):
        self.belief = initial.copy()
        self.dt = dt
        self.tick = -1
        self.too_old = self.duplicates = self.replayed_ticks = 0
        self.seen = set()

    def advance(self, tick, observations):
        if tick != self.tick + 1:
            raise ValueError('Sequential control ticks required')
        if tick:
            self.belief.predict(self.dt)
        for o in observations:
            if o.tick > tick or o.tick < 0:
                raise ValueError('Invalid or future observation')
            key = (o.source, o.tick)
            if key in self.seen:
                self.duplicates += 1
                continue
            self.seen.add(key)
            if o.tick == tick:
                self.belief.update_position(o.z_xy, o.r_xy)
            else:
                z, r = delayed_measurement_to_now(o.z_xy, o.tick, tick, self.dt,
                    self.belief.mean[2:], o.r_xy, self.belief.process_accel_std)
                self.belief.update_position(z, r)
        self.tick = tick
        return self.belief


class ReplayEstimator:
    """Matches chronological KF on the admitted observations under its model.

    This is not the exact posterior conditional on event-triggered silences:
    selection information and the nonlinear/mismatched sensor model are not
    incorporated. Covariance coverage is measured, never presumed.
    """
    def __init__(self, initial, dt, lag=32):
        self.initial = initial.copy()
        self.belief = initial.copy()
        self.dt, self.lag, self.tick = float(dt), int(lag), -1
        if self.lag < 1:
            raise ValueError('Positive replay lag required')
        self.observations, self.posteriors, self.fused = {}, {}, {}
        self.too_old = self.duplicates = self.replayed_ticks = 0

    def advance(self, tick, observations):
        if tick != self.tick + 1:
            raise ValueError('Sequential control ticks required')
        cutoff = max(0, tick - self.lag)
        earliest = tick
        for o in observations:
            if o.tick > tick or o.tick < 0:
                raise ValueError('Invalid or future observation')
            if o.tick < cutoff:
                self.too_old += 1
                continue
            group = self.observations.setdefault(o.tick, {})
            if o.source in group:
                self.duplicates += 1
                continue
            # Explicit whitelist; no source truth/target-distance fields retained.
            group[o.source] = (np.array(o.z_xy, float, copy=True), np.array(o.r_xy, float, copy=True))
            self.fused.pop(o.tick, None)
            earliest = min(earliest, o.tick)
        b = self.initial.copy() if earliest == 0 else self.posteriors[earliest - 1].copy()
        for t in range(earliest, tick + 1):
            if t:
                b.predict(self.dt)
            group = self.observations.get(t, {})
            if group:
                if t not in self.fused:
                    precision = np.zeros((2, 2))
                    information = np.zeros(2)
                    for source in sorted(group):
                        z, r = group[source]
                        inv = np.linalg.inv(r)
                        precision += inv
                        information += inv @ z
                    covariance = np.linalg.inv(precision)
                    self.fused[t] = (covariance @ information, covariance)
                b.update_position(*self.fused[t])
            self.posteriors[t] = b.copy()
        self.replayed_ticks += tick - earliest
        self.belief, self.tick = b, tick
        # Retain the posterior immediately before the earliest eligible timestamp.
        for t in list(self.posteriors):
            if t < cutoff - 1:
                del self.posteriors[t]
        for d in [self.observations, self.fused]:
            for t in list(d):
                if t < cutoff:
                    del d[t]
        return self.belief
