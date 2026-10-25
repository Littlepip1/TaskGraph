from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import heapq
import math
from typing import Iterable
import numpy as np


@dataclass
class Delivery:
    receiver: int
    source: int
    created_tick: int
    arrival_tick: int
    payload: object
    bits: int
    hops: int


@dataclass
class _Tx:
    message_id: int
    source: int
    u: int
    v: int
    recipients: tuple[int, ...]
    created_tick: int
    deadline_tick: int
    payload: object
    bits_total: int
    remaining_bits: int
    hops: int
    kind: str = 'payload'


@dataclass
class _Arrival:
    message_id: int
    source: int
    node: int
    recipients: tuple[int, ...]
    created_tick: int
    deadline_tick: int
    payload: object
    bits: int
    hops: int


@dataclass
class _Stalled:
    message_id: int
    source: int
    node: int
    recipients: tuple[int, ...]
    created_tick: int
    deadline_tick: int
    payload: object
    bits: int
    hops: int


class HopByHopDynamicNetwork:
    """Dynamic geometric multicast network with per-hop forwarding.

    Unlike the inherited network, a packet does *not* reserve its complete future
    multicast tree at creation time.  Only a current first-hop transmission enters a
    physical link queue.  After that hop arrives, the next hop is chosen from the graph
    that exists at that future tick.  Link service occurs only while the physical edge
    exists, so later motion can pause an already queued first hop instead of silently
    guaranteeing a stale future path.

    Cost is actual serviced bit-hop, including bits spent on a transmission that later
    expires.  A recipient copy expires after ``deadline_ticks`` because the frozen
    replay estimator cannot use older observations anyway.
    """

    def __init__(self, n: int, cfg: dict, seed: int, deadline_ticks: int = 32):
        self.n = int(n)
        self.cfg = dict(cfg)
        self.rng = np.random.default_rng(int(seed))
        self.deadline_ticks = int(deadline_ticks)
        self.positions = np.zeros((self.n, 2), dtype=float)
        self.adj: list[list[int]] = [[] for _ in range(self.n)]
        self.link_queues: dict[tuple[int, int], deque[_Tx]] = defaultdict(deque)
        self.queued_bits_by_edge: dict[tuple[int, int], int] = defaultdict(int)
        self._tree_cache = {}
        self.arrivals: dict[int, list[_Arrival]] = defaultdict(list)
        self.stalled: list[_Stalled] = []
        self._message_counter = 0

        self.total_bit_hop = 0
        self.offered_current_tree_bit_hop = 0
        self.payload_bit_hop = 0
        self.control_bit_hop = 0
        self.sent_packets = 0
        self.requested_recipient_copies = 0
        self.delivered_packets = 0
        self.dropped_packets = 0
        self.expired_recipient_copies = 0
        self.expired_transmissions = 0
        self.stalled_recipient_ticks = 0
        self.delay_samples: list[int] = []
        self.total_link_capacity_bits = 0
        self.used_link_capacity_bits = 0
        self.max_queued_bits = 0
        self.max_queued_work_ticks = 0.0
        self.forwarded_hops = 0
        self.recomputed_forwarding_events = 0

    @staticmethod
    def _edge_key(a: int, b: int) -> tuple[int, int]:
        return (a, b) if a < b else (b, a)

    def rebuild(self, positions_xy: np.ndarray) -> None:
        # Cached BFS trees belong to the previous topology, even if the source is unchanged.
        self._tree_cache.clear()
        self.positions = np.asarray(positions_xy, dtype=float).reshape(self.n, 2).copy()
        radius = float(self.cfg['comm_range_m'])
        self.adj = [[] for _ in range(self.n)]
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if float(np.linalg.norm(self.positions[i] - self.positions[j])) <= radius:
                    self.adj[i].append(j)
                    self.adj[j].append(i)
        for x in self.adj:
            x.sort()

    def _shortest_tree(self, source: int):
        source = int(source)
        cached = self._tree_cache.get(source)
        if cached is not None:
            return cached
        inf = float('inf')
        dist = [inf] * self.n
        parent: list[int | None] = [None] * self.n
        dist[source] = 0.0
        q = deque([source])
        while q:
            u = q.popleft()
            nd = dist[u] + 1.0
            for v in self.adj[u]:
                if not math.isfinite(dist[v]):
                    dist[v] = nd
                    parent[v] = u
                    q.append(v)
        self._tree_cache[source] = (dist, parent)
        return dist, parent

    def path(self, source: int, receiver: int):
        source = int(source); receiver = int(receiver)
        if source == receiver:
            return [source]
        dist, parent = self._shortest_tree(source)
        if not math.isfinite(dist[receiver]):
            return None
        rev = [receiver]
        cur = receiver
        while cur != source:
            cur = parent[cur]
            if cur is None:
                return None
            rev.append(cur)
        return list(reversed(rev))

    def reachable_from(self, source: int) -> list[int]:
        dist, _ = self._shortest_tree(int(source))
        return [i for i, d in enumerate(dist) if i != int(source) and math.isfinite(d)]

    def multicast_cost(self, source: int, recipients: Iterable[int], bits: int) -> int:
        """Current-tree diagnostic cost only; it does not reserve future hops."""
        recipients = sorted(set(int(x) for x in recipients if int(x) != int(source)))
        if not recipients:
            return 0
        _, parent = self._shortest_tree(int(source))
        edges = set()
        for r in recipients:
            cur = r
            while cur != int(source):
                p = parent[cur]
                if p is None:
                    return math.inf
                edges.add(self._edge_key(p, cur))
                cur = p
        return int(bits) * len(edges)

    def _queue_bits(self, key: tuple[int, int]) -> int:
        return int(self.queued_bits_by_edge.get(key, 0))

    def incident_backlog_ticks(self, node: int) -> float:
        cap = max(1, int(self.cfg['bits_per_link_tick']))
        rows = [self._queue_bits(self._edge_key(int(node), v))/cap for v in self.adj[int(node)]]
        return float(max(rows, default=0.0))

    def min_first_hop_arrival_ticks(self, node: int, bits: int) -> int:
        cap = max(1, int(self.cfg['bits_per_link_tick']))
        prop = max(0, int(self.cfg.get('prop_delay_ticks', 1)))
        rows = []
        for v in self.adj[int(node)]:
            backlog = self._queue_bits(self._edge_key(int(node), v))
            rows.append(int(math.ceil((backlog + int(bits))/cap + prop - 1e-12)))
        return min(rows) if rows else 0

    def queue_snapshot(self) -> dict:
        cap = max(1, int(self.cfg['bits_per_link_tick']))
        bits = [self._queue_bits(k) for k in self.link_queues]
        return {
            'queued_links': int(sum(1 for x in bits if x > 0)),
            'queued_bits': int(sum(bits)),
            'max_link_queue_bits': int(max(bits, default=0)),
            'max_link_queue_work_ticks': float(max(bits, default=0)/cap),
        }

    def _expire_recipients(self, recipients: Iterable[int], transmissions: int = 0):
        k = len(tuple(recipients))
        self.expired_recipient_copies += int(k)
        self.dropped_packets += int(k)
        self.expired_transmissions += int(transmissions)

    def _enqueue(self, tx: _Tx):
        if not tx.recipients:
            return
        key = self._edge_key(tx.u, tx.v)
        self.link_queues[key].append(tx)
        self.queued_bits_by_edge[key] += int(tx.remaining_bits)

    def _route_branch(self, branch: _Arrival | _Stalled, tick: int):
        if int(tick) > int(branch.deadline_tick):
            self._expire_recipients(branch.recipients)
            return
        remaining = tuple(sorted(set(int(x) for x in branch.recipients if int(x) != int(branch.node))))
        if not remaining:
            return
        dist, parent = self._shortest_tree(int(branch.node))
        groups: dict[int, list[int]] = defaultdict(list)
        unreachable = []
        for r in remaining:
            if not math.isfinite(dist[r]):
                unreachable.append(r)
                continue
            cur = r
            prev = None
            while cur != int(branch.node):
                prev = cur
                cur = parent[cur]
                if cur is None:
                    break
            if cur != int(branch.node) or prev is None:
                unreachable.append(r)
            else:
                groups[int(prev)].append(r)
        for v, recips in sorted(groups.items()):
            self._enqueue(_Tx(
                int(branch.message_id), int(branch.source), int(branch.node), int(v), tuple(sorted(recips)),
                int(branch.created_tick), int(branch.deadline_tick), branch.payload, int(branch.bits), int(branch.bits),
                int(branch.hops) + 1, 'payload'))
        if unreachable:
            self.stalled.append(_Stalled(
                int(branch.message_id), int(branch.source), int(branch.node), tuple(sorted(unreachable)),
                int(branch.created_tick), int(branch.deadline_tick), branch.payload, int(branch.bits), int(branch.hops)))
        self.recomputed_forwarding_events += 1

    def _retry_stalled(self, tick: int):
        old = self.stalled
        self.stalled = []
        for b in old:
            if int(tick) > int(b.deadline_tick):
                self._expire_recipients(b.recipients)
            else:
                before = len(self.stalled)
                self._route_branch(b, tick)
                # If the branch was still unreachable, _route_branch has re-added it.
                if len(self.stalled) > before:
                    self.stalled_recipient_ticks += len(b.recipients)

    def schedule_multicast(self, source: int, recipients, bits: int, created_tick: int,
                           payload, kind: str = 'payload', deadline_ticks: int | None = None):
        source = int(source); created_tick = int(created_tick); bits = int(bits)
        recipients = tuple(sorted(set(int(x) for x in recipients if int(x) != source)))
        if not recipients:
            return {}
        dlag = self.deadline_ticks if deadline_ticks is None else int(deadline_ticks)
        deadline = created_tick + dlag
        self._message_counter += 1
        mid = self._message_counter
        self.sent_packets += 1
        self.requested_recipient_copies += len(recipients)
        offered = self.multicast_cost(source, recipients, bits)
        if math.isfinite(offered):
            self.offered_current_tree_bit_hop += int(offered)
        root = _Arrival(mid, source, source, recipients, created_tick, deadline, payload, bits, 0)
        self._route_branch(root, created_tick)
        # Exact future arrival is intentionally unknown in the hop-by-hop model.
        return {r: None for r in recipients}

    def receive(self, tick: int) -> list[Delivery]:
        tick = int(tick)
        self._retry_stalled(tick)
        rows = self.arrivals.pop(tick, [])
        out: list[Delivery] = []
        for b in rows:
            if tick > int(b.deadline_tick):
                self._expire_recipients(b.recipients)
                continue
            recips = set(int(x) for x in b.recipients)
            if int(b.node) in recips:
                d = Delivery(int(b.node), int(b.source), int(b.created_tick), tick,
                             b.payload, int(b.bits), int(b.hops))
                out.append(d)
                recips.remove(int(b.node))
            if recips:
                self._route_branch(_Arrival(
                    b.message_id, b.source, b.node, tuple(sorted(recips)), b.created_tick,
                    b.deadline_tick, b.payload, b.bits, b.hops), tick)
        self.delivered_packets += len(out)
        self.delay_samples.extend([d.arrival_tick-d.created_tick for d in out])
        return out

    def service_tick(self, tick: int):
        """Serve one control tick of all currently existing physical links."""
        tick = int(tick)
        cap = max(1, int(self.cfg['bits_per_link_tick']))
        prop = max(0, int(self.cfg.get('prop_delay_ticks', 1)))
        active_edges = {self._edge_key(i, j) for i in range(self.n) for j in self.adj[i] if i < j}
        self.total_link_capacity_bits += cap * len(active_edges)
        for key in sorted(list(self.link_queues.keys())):
            q = self.link_queues[key]
            if not q:
                continue
            # Expiry is a clock event, not a link-service event. A disconnected
            # edge must not retain expired recipient copies forever.
            live = deque()
            for tx in q:
                if tick > tx.deadline_tick:
                    self.queued_bits_by_edge[key] -= int(tx.remaining_bits)
                    self._expire_recipients(tx.recipients, transmissions=1)
                else:
                    live.append(tx)
            q = self.link_queues[key] = live
            if not q:
                self.link_queues.pop(key, None)
                self.queued_bits_by_edge.pop(key, None)
                continue
            if key not in active_edges:
                continue
            budget = cap
            used = 0
            while q and budget > 0:
                tx = q[0]
                # If even the earliest possible completion of this hop cannot arrive by
                # the estimator deadline, carrying more bits is control-useless.
                earliest_completion = tick + (used + tx.remaining_bits) / cap
                earliest_arrival = int(math.ceil(earliest_completion + prop - 1e-12))
                if tick > tx.deadline_tick or earliest_arrival > tx.deadline_tick:
                    q.popleft()
                    self.queued_bits_by_edge[key] -= int(tx.remaining_bits)
                    self._expire_recipients(tx.recipients, transmissions=1)
                    continue
                served = min(int(budget), int(tx.remaining_bits))
                tx.remaining_bits -= served
                self.queued_bits_by_edge[key] -= served
                budget -= served
                used += served
                self.used_link_capacity_bits += served
                self.total_bit_hop += served
                if tx.kind == 'payload':
                    self.payload_bit_hop += served
                else:
                    self.control_bit_hop += served
                if tx.remaining_bits == 0:
                    q.popleft()
                    completion = tick + used / cap
                    arrival = int(math.ceil(completion + prop - 1e-12))
                    self.arrivals[arrival].append(_Arrival(
                        tx.message_id, tx.source, tx.v, tx.recipients, tx.created_tick,
                        tx.deadline_tick, tx.payload, tx.bits_total, tx.hops))
                    self.forwarded_hops += 1
            if not q:
                self.link_queues.pop(key, None)
                self.queued_bits_by_edge.pop(key, None)
        snap = self.queue_snapshot()
        self.max_queued_bits = max(self.max_queued_bits, int(snap['max_link_queue_bits']))
        self.max_queued_work_ticks = max(self.max_queued_work_ticks, float(snap['max_link_queue_work_ticks']))

    def pending(self) -> int:
        # Every recipient exists in exactly one queued transmission, arrival branch, or
        # stalled branch because routing partitions the recipient set at every hop.
        total = 0
        for q in self.link_queues.values():
            total += sum(len(tx.recipients) for tx in q)
        for rows in self.arrivals.values():
            total += sum(len(x.recipients) for x in rows)
        total += sum(len(x.recipients) for x in self.stalled)
        return int(total)

    def stats(self) -> dict:
        delays = np.asarray(self.delay_samples, dtype=float)
        snap = self.queue_snapshot()
        return {
            'total_bit_hop': int(self.total_bit_hop),
            'offered_current_tree_bit_hop': int(self.offered_current_tree_bit_hop),
            'payload_bit_hop': int(self.payload_bit_hop),
            'control_bit_hop': int(self.control_bit_hop),
            'sent_packets': int(self.sent_packets),
            'requested_recipient_copies': int(self.requested_recipient_copies),
            'delivered_packets': int(self.delivered_packets),
            'dropped_packets': int(self.dropped_packets),
            'expired_recipient_copies': int(self.expired_recipient_copies),
            'expired_transmissions': int(self.expired_transmissions),
            'mean_delay_ticks': float(delays.mean()) if len(delays) else 0.0,
            'max_delay_ticks': int(delays.max()) if len(delays) else 0,
            'pending_deliveries': int(self.pending()),
            'forwarded_hops': int(self.forwarded_hops),
            'recomputed_forwarding_events': int(self.recomputed_forwarding_events),
            'stalled_recipient_ticks': int(self.stalled_recipient_ticks),
            'link_capacity_utilization': float(self.used_link_capacity_bits/max(1, self.total_link_capacity_bits)),
            'used_link_capacity_bits': int(self.used_link_capacity_bits),
            'available_link_capacity_bits': int(self.total_link_capacity_bits),
            'max_queued_bits': int(self.max_queued_bits),
            'max_queued_work_ticks': float(self.max_queued_work_ticks),
            **snap,
        }
