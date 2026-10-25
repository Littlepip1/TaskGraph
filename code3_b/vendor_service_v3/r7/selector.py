from dataclasses import dataclass, asdict, replace
from collections import deque
import math
from r6f.selector import decide_sender_local


@dataclass(frozen=True)
class PressureConfig:
    opportunity_period: int = 16
    replay_lag: int = 32
    age_window: int = 32
    pressure_strength: float = 1.0


@dataclass(frozen=True)
class LocalTelemetry:
    neighbor_count: int
    min_first_hop_arrival_ticks: int
    max_incident_wait_ticks: float
    recent_received_mean_age_ticks: float
    received_age_samples: int


class ReceivedAgeHistory:
    """Only packets already received at this receiver; silence is not a delay sample."""
    def __init__(self, window=32):
        self.window = int(window)
        self.rows = deque()

    def observe(self, tick, created_tick):
        if created_tick > tick:
            raise ValueError('Future packet cannot be observed')
        self.rows.append((int(tick), int(tick-created_tick)))

    def summary(self, tick):
        while self.rows and self.rows[0][0] < tick-self.window+1:
            self.rows.popleft()
        return (sum(x[1] for x in self.rows)/len(self.rows), len(self.rows)) if self.rows else (0.0, 0)


def local_telemetry(source, neighbors, incident_calendar, tick, bits, capacity,
                    prop_delay, received_history):
    """The adapter supplies only current neighbors and incident link service calendars.

    No full graph, recipient beliefs, future arrivals or remote link queues enter the gate.
    Calendars are the inherited simulator's idealized shared-link reservations, not
    measured real-radio queue telemetry. No extra control packets are assumed.
    """
    neighbors = tuple(neighbors)
    waits = [max(0.0, float(incident_calendar.get(v, tick))-tick) for v in neighbors]
    arrivals = [int(math.ceil(w+bits/max(1, capacity)+prop_delay-1e-12)) for w in waits]
    age, count = received_history.summary(tick)
    return LocalTelemetry(len(neighbors), min(arrivals) if arrivals else 0,
                          max(waits, default=0.0), float(age), int(count))


def decide_pressure_local(prior, observation, last_sent, tick, base_cfg, telemetry,
                          cfg=PressureConfig(), use_received_age=True):
    # Baseline Safe has efficiency_threshold=0: tree size never affects its decision.
    if base_cfg.efficiency_threshold != 0:
        raise ValueError('R7 requires the frozen Safe profile without route-cost input')
    original = decide_sender_local(prior, observation, 0, 352, last_sent, tick, base_cfg)
    q = telemetry.max_incident_wait_ticks/cfg.opportunity_period
    a = max(0.0, telemetry.recent_received_mean_age_ticks/cfg.opportunity_period-1.0)
    pressure = min(1.0, max(q, a if use_received_age else 0.0))
    effective = replace(base_cfg,
                        gain_threshold=base_cfg.gain_threshold*(1+cfg.pressure_strength*pressure),
                        nis_threshold=base_cfg.nis_threshold*(1+cfg.pressure_strength*pressure))
    decision = decide_sender_local(prior, observation, 0, 352, last_sent, tick, effective)
    if observation is not None:
        if not telemetry.neighbor_count:
            decision.send, decision.reason = False, 'no_local_link'
        elif telemetry.min_first_hop_arrival_ticks > cfg.replay_lag:
            # Even the earliest first hop cannot enter the replay window. This also
            # overrides refresh/uncertainty guards: enqueuing is not delivery.
            decision.send, decision.reason = False, 'first_hop_expired'
        elif original.send and not decision.send:
            decision.reason = 'pressure_suppress'
    row = decision.to_dict()
    row.update(pressure=float(pressure), base_send=original.send, base_reason=original.reason,
               effective_gain_threshold=effective.gain_threshold,
               effective_nis_threshold=effective.nis_threshold,
               telemetry=asdict(telemetry), use_received_age=bool(use_received_age))
    return row
