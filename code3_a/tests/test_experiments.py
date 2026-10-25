from __future__ import annotations
import itertools
import math
import unittest
import numpy as np

import tgexp
from tgexp.selection import select
from tgexp.scenario import Scenario, PreparationIndex
from tgexp.engine import VectorizedNetwork, run_episode
from tgexp.reporting import interval, comparison_rows
from r8a.network import HopByHopDynamicNetwork
from final_airsim.common import base_config


def job(method='full', **changes):
    j = dict(n=8, tasks=2, scenario='switch', family='MIS', capacity=256, seed=2710790000,
             method=method, fraction=.1, absolute=0., steps=40, eval_start=8, prep=8,
             period=16, deadline=16, context_lag=0, event_log=True, smoke=True,
             external_selector=None)
    j.update(changes)
    return j


class SelectionTests(unittest.TestCase):
    def test_random_trees_quota_and_witness(self):
        rng = np.random.default_rng(17)
        for n in [2, 8, 70]:
            routes = {}
            for r in range(1, n + 1):
                parent = int(rng.integers(r))
                routes[r] = set(routes.get(parent, set())) | {(parent, r)}
            for theta in [0., .1, .5, 1.]:
                values = {r: float(rng.random()) for r in routes}
                for strategy in ['shared', 'forward', 'additive']:
                    answer = select(values, routes, 352, theta, strategy=strategy)
                    self.assertTrue(answer['feasible'])
                    self.assertLessEqual(answer['snapshot_cost'], answer.get('witness_cost', math.inf))
                    self.assertLessEqual(answer['omitted_value'], theta * sum(values.values()) + 1e-10)
                    cost = 352 * len(set().union(*(routes[r] for r in answer['selected']))) if answer['selected'] else 0
                    self.assertEqual(cost, answer['snapshot_cost'])
                    if strategy == 'shared':
                        self.assertLessEqual(cost, answer['count_quota_bound'])

    def test_zero_empty_absolute_and_tiny(self):
        self.assertEqual(select({}, {}, 352)['selected'], [])
        for value in [0., 1e-16]:
            self.assertTrue(select({1: value}, {1: {(0, 1)}}, 352)['feasible'])
        self.assertEqual(select({1: 1.}, {1: {(0, 1)}}, 352, absolute=1.)['selected'], [])

    def test_shared_branch_cost(self):
        routes = {1: {(0, 9), (9, 1)}, 2: {(0, 9), (9, 2)}}
        self.assertEqual(select({1: 1., 2: 1.}, routes, 100, 0.)['snapshot_cost'], 300)

    def test_optimum_is_below_returned_feasible_set(self):
        routes = {1: {(0, 1)}, 2: {(0, 1), (1, 2)}, 3: {(0, 3)}}
        values = {1: .1, 2: .8, 3: .1}
        result = select(values, routes, 1, .2)
        feasible = []
        for k in range(4):
            for subset in itertools.combinations(values, k):
                if sum(values[r] for r in subset) >= .8 - 1e-12:
                    feasible.append(len(set().union(*(routes[r] for r in subset))) if subset else 0)
        self.assertGreaterEqual(result['snapshot_cost'], min(feasible))


class GroupTests(unittest.TestCase):
    def test_incremental_matches_explicit_window_union(self):
        for mode in ['static', 'overlap', 'switch']:
            s = Scenario(32, 4, 80, 3, 'SP', mode, 8)
            index = PreparationIndex(s, 8)
            for tick in range(80):
                index.advance(tick)
                for q in range(4):
                    expected = {r for v in range(tick, min(79, tick + 8) + 1)
                                for r in range(32) if s.weights(v)[r, q] > 0}
                    self.assertEqual(index.groups[q], expected)

    def test_handover_changes_dependencies(self):
        s = Scenario(32, 4, 80, 3, 'SP', 'switch', 8)
        self.assertTrue(np.any((s.before > 0) != (s.after > 0)))
        self.assertTrue(np.any(np.count_nonzero(s.before, axis=1) > 1))
        self.assertTrue(np.allclose(s.before.sum(axis=1), 1))
        self.assertTrue(np.allclose(s.after.sum(axis=1), 1))


class NetworkTests(unittest.TestCase):
    def test_vectorized_graph_preserves_upstream_routes(self):
        cfg = base_config()['network']
        rng = np.random.default_rng(2)
        positions = rng.normal(size=(50, 2)) * 50
        old = HopByHopDynamicNetwork(50, cfg, 1)
        new = VectorizedNetwork(50, cfg, 1)
        old.rebuild(positions)
        new.rebuild(positions)
        self.assertEqual(old.adj, new.adj)
        for source in [0, 14, 49]:
            for receiver in range(50):
                self.assertEqual(old.path(source, receiver), new.path(source, receiver))

    def test_shared_service_and_recipient_accounting(self):
        cfg = dict(base_config()['network'], comm_range_m=1.1, bits_per_link_tick=256)
        net = VectorizedNetwork(4, cfg, 2, 10)
        net.rebuild(np.array([[0, 0], [1, 0], [2, 0], [3, 0]]))
        net.schedule_multicast(0, [2, 3], 352, 0, 'message', deadline_ticks=10)
        received = []
        for tick in range(14):
            received.extend(net.receive(tick))
            net.service_tick(tick)
        self.assertEqual(sorted(d.receiver for d in received), [2, 3])
        self.assertEqual(net.payload_bit_hop, 3 * 352)
        self.assertEqual(net.requested_recipient_copies, net.delivered_packets + net.expired_recipient_copies + net.pending())

    def test_disconnected_expiry_and_partial_service(self):
        cfg = dict(base_config()['network'], comm_range_m=1.1, bits_per_link_tick=256)
        net = VectorizedNetwork(3, cfg, 2, 4)
        net.rebuild(np.array([[0, 0], [1, 0], [2, 0]]))
        net.schedule_multicast(0, [2], 352, 0, 'message', deadline_ticks=4)
        net.service_tick(0)
        net.rebuild(np.array([[0, 0], [100, 0], [200, 0]]))
        for tick in range(1, 8):
            net.receive(tick)
            net.service_tick(tick)
        self.assertEqual(net.pending(), 0)
        self.assertEqual(net.expired_recipient_copies, 1)
        self.assertEqual(net.payload_bit_hop, 256)


class ClosedLoopTests(unittest.TestCase):
    def test_multitask_actual_closure_and_repeatability(self):
        a, b = run_episode(job()), run_episode(job())
        self.assertEqual(a['network'], b['network'])
        self.assertEqual(a['evaluation'], b['evaluation'])
        self.assertEqual(len(a['evaluation']['per_task_mean']), 2)
        self.assertEqual(a['maintained_task_filters'], b['maintained_task_filters'])
        self.assertEqual(a['recipient_accounting_difference'], 0)
        self.assertEqual(a['theory_audit']['quota_failures'], 0)
        self.assertEqual(len(a['trace']), 40)

    def test_all_methods_and_context_lag_execute(self):
        for method in ['dense_global', 'dense_task', 'demand_only', 'legacy_additive', 'uncertainty',
                       'innovation', 'no_group', 'no_prepare', 'no_shared', 'forward_only', 'no_guard']:
            r = run_episode(job(method))
            self.assertEqual(r['recipient_accounting_difference'], 0)
        r = run_episode(job(context_lag=8))
        self.assertEqual(r['theory_audit']['snapshot_bound_failures'], 0)


class StatisticsTests(unittest.TestCase):
    def test_pairing_uses_seed_not_ticks(self):
        rows = []
        for seed in [1, 2, 3]:
            for method, gap in [('dense_task', 0.), ('full', .005)]:
                j = job(method, seed=seed)
                rows.append(dict(job=j, network={'payload_bit_hop': 100 if method == 'dense_task' else 50},
                                 evaluation=dict(reward={'mean': .8 - gap}, per_task_mean=[.8 - gap, .8 - gap],
                                                 switch_window_error_m={'mean': 2.}),
                                 cost_per_agent_tick=1., candidate_visits=10,
                                 selection_seconds={'mean': .001, 'p95': .002}, selected_count={'mean': 3.}))
        result = comparison_rows(rows)
        self.assertEqual(result[0]['seed_count'], 3)
        self.assertAlmostEqual(result[0]['gap_mean'], .005)
        self.assertAlmostEqual(result[0]['saving_mean'], .5)
        self.assertTrue(result[0]['margin_pass'])
        self.assertIsNone(interval([1.])['upper'])


if __name__ == '__main__':
    unittest.main()

