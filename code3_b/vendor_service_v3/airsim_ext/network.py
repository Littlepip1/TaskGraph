from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import heapq
import math
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


class DynamicNetwork:
    """Dynamic geometric multi-hop network with shared multicast accounting.

    Paths are recomputed when a plan is formed. A multicast packet is charged once on
    each edge of the union of shortest paths. Per-link transmission calendars add a
    deterministic queueing delay under finite link capacity.
    """

    def __init__(self, n: int, cfg: dict, seed: int):
        self.n = int(n)
        self.cfg = cfg
        self.rng = np.random.default_rng(int(seed))
        self.positions = np.zeros((n, 2), dtype=np.float64)
        self.adj: list[list[int]] = [[] for _ in range(n)]
        # Link-service calendars are stored in continuous tick units.  A link with
        # capacity C bits/tick can therefore carry multiple small packets inside one
        # physical tick instead of incorrectly reserving a full tick per packet.
        self.next_free: dict[tuple[int, int], float] = defaultdict(float)
        self.events: dict[int, list[Delivery]] = defaultdict(list)
        self.total_bit_hop = 0
        self.payload_bit_hop = 0
        self.control_bit_hop = 0
        self.sent_packets = 0
        self.delivered_packets = 0
        self.dropped_packets = 0
        self.delay_samples: list[int] = []

    def rebuild(self, positions_xy: np.ndarray) -> None:
        self.positions = np.asarray(positions_xy, dtype=np.float64).reshape(self.n, 2).copy()
        r = float(self.cfg['comm_range_m'])
        self.adj = [[] for _ in range(self.n)]
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if float(np.linalg.norm(self.positions[i] - self.positions[j])) <= r:
                    self.adj[i].append(j)
                    self.adj[j].append(i)
        for a in self.adj:
            a.sort()

    @staticmethod
    def _edge_key(a: int, b: int) -> tuple[int, int]:
        return (a, b) if a < b else (b, a)

    def _shortest_tree(self, source: int) -> tuple[list[float], list[int | None]]:
        # Weight edges by one propagation step; Dijkstra is used so future
        # distance-dependent weights can be added without changing the interface.
        inf = float('inf')
        dist = [inf] * self.n
        parent: list[int | None] = [None] * self.n
        dist[source] = 0.0
        pq = [(0.0, source)]
        while pq:
            d, u = heapq.heappop(pq)
            if d != dist[u]:
                continue
            for v in self.adj[u]:
                nd = d + 1.0
                if nd < dist[v] - 1e-12 or (abs(nd - dist[v]) < 1e-12 and (parent[v] is None or u < parent[v])):
                    dist[v] = nd
                    parent[v] = u
                    heapq.heappush(pq, (nd, v))
        return dist, parent

    def path(self, source: int, receiver: int) -> list[int] | None:
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

    def hops(self, source: int, receiver: int) -> int | None:
        p = self.path(source, receiver)
        return None if p is None else max(0, len(p) - 1)

    def multicast_cost(self, source: int, recipients: list[int] | tuple[int, ...] | set[int], bits: int) -> int:
        recipients = sorted(set(int(x) for x in recipients if int(x) != source))
        if not recipients:
            return 0
        _, parent = self._shortest_tree(source)
        edges: set[tuple[int, int]] = set()
        for r in recipients:
            cur = r
            while cur != source:
                p = parent[cur]
                if p is None:
                    return math.inf
                edges.add(self._edge_key(p, cur))
                cur = p
        return int(bits) * len(edges)

    def schedule_multicast(self, source: int, recipients, bits: int, created_tick: int,
                           payload, kind: str = 'payload') -> dict[int, int]:
        recipients = sorted(set(int(x) for x in recipients if int(x) != source))
        if not recipients:
            return {}
        dist, parent = self._shortest_tree(source)
        reachable = [r for r in recipients if math.isfinite(dist[r])]
        if not reachable:
            self.dropped_packets += len(recipients)
            return {}

        # Needed nodes are the union of source->recipient tree paths.
        needed = {source}
        for r in reachable:
            cur = r
            needed.add(cur)
            while cur != source:
                p = parent[cur]
                if p is None:
                    break
                needed.add(p)
                cur = p

        children: dict[int, list[int]] = defaultdict(list)
        for v in needed:
            if v == source:
                continue
            p = parent[v]
            if p is not None and p in needed:
                children[p].append(v)
        for c in children.values():
            c.sort()

        capacity = max(1, int(self.cfg['bits_per_link_tick']))
        tx_duration = float(int(bits)) / float(capacity)
        prop = max(0, int(self.cfg.get('prop_delay_ticks', 1)))
        drop_p = float(self.cfg.get('drop_probability', 0.0))
        node_arrival = {source: int(created_tick)}
        dropped_subtree: set[int] = set()
        q = deque([source])
        edge_count = 0
        while q:
            u = q.popleft()
            for v in children.get(u, []):
                edge_count += 1
                key = self._edge_key(u, v)
                # Serialize bits on the shared undirected link, then add propagation.
                # Arrival is quantized to the next control tick because beliefs are
                # updated only at integer experiment ticks.
                start = max(float(node_arrival[u]), float(self.next_free[key]))
                finish_tx = start + tx_duration
                self.next_free[key] = finish_tx
                finish = int(math.ceil(finish_tx + float(prop) - 1e-12))
                if u in dropped_subtree or self.rng.random() < drop_p:
                    dropped_subtree.add(v)
                    node_arrival[v] = finish
                else:
                    node_arrival[v] = finish
                q.append(v)

        bit_hop = int(bits) * edge_count
        self.total_bit_hop += bit_hop
        if kind == 'payload':
            self.payload_bit_hop += bit_hop
        else:
            self.control_bit_hop += bit_hop
        self.sent_packets += 1

        out = {}
        for r in recipients:
            if r not in node_arrival or r in dropped_subtree:
                self.dropped_packets += 1
                continue
            # If any ancestor was dropped, r cannot receive. Walk back to check.
            cur = r
            ok = True
            while cur != source:
                if cur in dropped_subtree:
                    ok = False
                    break
                cur = parent[cur]
                if cur is None:
                    ok = False
                    break
            if not ok:
                self.dropped_packets += 1
                continue
            arrival = int(node_arrival[r])
            d = Delivery(r, source, int(created_tick), arrival, payload, int(bits), int(dist[r]))
            self.events[arrival].append(d)
            out[r] = arrival
        return out

    def account_unicast(self, source: int, receiver: int, bits: int, kind: str = 'control') -> int | None:
        p = self.path(source, receiver)
        if p is None:
            return None
        hops = len(p) - 1
        bh = int(bits) * hops
        self.total_bit_hop += bh
        if kind == 'payload':
            self.payload_bit_hop += bh
        else:
            self.control_bit_hop += bh
        return hops

    def receive(self, tick: int) -> list[Delivery]:
        rows = self.events.pop(int(tick), [])
        self.delivered_packets += len(rows)
        self.delay_samples.extend([d.arrival_tick - d.created_tick for d in rows])
        return rows

    def pending(self) -> int:
        return sum(len(v) for v in self.events.values())

    def stats(self) -> dict:
        delays = np.asarray(self.delay_samples, dtype=np.float64)
        return {
            'total_bit_hop': int(self.total_bit_hop),
            'payload_bit_hop': int(self.payload_bit_hop),
            'control_bit_hop': int(self.control_bit_hop),
            'sent_packets': int(self.sent_packets),
            'delivered_packets': int(self.delivered_packets),
            'dropped_packets': int(self.dropped_packets),
            'mean_delay_ticks': float(delays.mean()) if len(delays) else 0.0,
            'max_delay_ticks': int(delays.max()) if len(delays) else 0,
            'pending_deliveries': int(self.pending()),
        }
