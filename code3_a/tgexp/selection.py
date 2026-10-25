"""Adaptive quota selection; no fixed receiver budget and no target truth.

Costs are snapshot bit-hops. Actual service is measured by the upstream network.
Python integer edge masks work above 64 receivers without truncation.
"""
from __future__ import annotations
import math


def select(values, routes, bits, fraction=.1, absolute=0., strategy='shared', tol=1e-12):
    if not 0 <= fraction <= 1 or absolute < 0:
        raise ValueError('Invalid omission settings')
    if set(values) != set(routes) or any(not math.isfinite(v) or v < 0 for v in values.values()):
        raise ValueError('Invalid values/routes')
    edge_ids = {e: k for k, e in enumerate(sorted(set().union(*routes.values()) if routes else set()))}
    masks = {r: sum(1 << edge_ids[e] for e in routes[r]) for r in values}
    total = math.fsum(values.values())
    budget = min(total, fraction * total + absolute)
    target = total - budget

    def union(rs):
        out = 0
        for r in rs:
            out |= masks[r]
        return out

    def record(rs):
        ids = sorted(rs)
        captured = math.fsum(values[r] for r in ids)
        omitted = max(0., total - captured)
        return dict(selected=ids, snapshot_cost=bits * union(ids).bit_count(),
                    total_value=total, omitted_value=omitted, omission_budget=budget,
                    feasible=omitted <= budget + tol)

    if strategy == 'dense':
        answer = record(values)
        answer['construction'] = 'dense'
        return answer
    # This feasible witness also certifies a snapshot upper bound.
    witness = []
    captured = 0.
    for r in sorted(values, key=lambda r: (-values[r], r)):
        if captured + tol >= target:
            break
        witness.append(r)
        captured = math.fsum(values[x] for x in witness)
    if strategy in {'additive', 'uncertainty', 'innovation'}:
        answer = record(witness)
        answer.update(construction='value_prefix', witness_cost=answer['snapshot_cost'])
        return answer

    selected = set()
    covered = 0
    captured = 0.
    while captured + tol < target:
        choices = [r for r in values if r not in selected and values[r] > 0]
        if not choices:
            break
        def key(r):
            incremental = (masks[r] & ~covered).bit_count()
            return (math.inf if incremental == 0 else values[r] / incremental,
                    values[r], -incremental, -r)
        r = max(choices, key=key)
        selected.add(r)
        covered |= masks[r]
        captured = math.fsum(values[x] for x in selected)
    alternatives = [('forward', record(selected)), ('witness', record(witness))]

    if strategy != 'forward':
        # Removing zero-value receivers is always feasible and cannot increase cost.
        remaining = {r for r in values if values[r] > 0}
        omitted = max(0., total - math.fsum(values[r] for r in remaining))
        users = {}
        for r in remaining:
            for e in routes[r]:
                users.setdefault(e, set()).add(r)
        while remaining:
            blocks = {frozenset([r]) for r in remaining}
            blocks.update(frozenset(u & remaining) for u in users.values() if u & remaining)
            best = None
            current_mask = union(remaining)
            for block in sorted(blocks, key=lambda b: tuple(sorted(b))):
                lost = math.fsum(values[r] for r in block)
                if omitted + lost > budget + tol:
                    continue
                freed = current_mask.bit_count() - union(remaining - block).bit_count()
                if freed <= 0:
                    continue
                key = (math.inf if lost == 0 else freed / lost, freed, -lost,
                       tuple(-r for r in sorted(block)))
                if best is None or key > best[0]:
                    best = key, block, lost
            if best is None:
                break
            _, block, lost = best
            remaining.difference_update(block)
            omitted += lost
        alternatives.append(('backward', record(remaining)))
    name, answer = min(alternatives, key=lambda x: (x[1]['snapshot_cost'], len(x[1]['selected']),
                                                   x[1]['omitted_value'], x[1]['selected']))
    answer.update(construction=name, witness_cost=record(witness)['snapshot_cost'],
                  alternative_costs={name: r['snapshot_cost'] for name, r in alternatives})
    # Independent invariants, not claims of global optimality.
    if not answer['feasible'] or answer['snapshot_cost'] > answer['witness_cost']:
        raise AssertionError('Quota or witness guarantee violated')
    positive = sorted(values, key=lambda r: (-values[r], r))
    quota_k = min(len(values), math.ceil((1 - fraction) * len(values)))
    H = max((len(e) for e in routes.values()), default=0)
    answer['witness_receiver_count'] = len(witness)
    answer['snapshot_H'] = H
    answer['count_quota_bound'] = bits * H * quota_k
    return answer

