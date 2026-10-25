import json
import sys
import types
import unittest
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_dynamic_demand import effective_config, jobs, variants, digest
from dynamic_environment import DynamicDemandEnvironment
from dynamic_reporting import calibrate, paired_losses, cost
from intuitive_policies import IntuitivePolicy
from tgexp.scenario import Scenario
from tgexp.engine import run_episode


def configuration():
    return effective_config(json.loads((ROOT/'dynamic_protocol.json').read_text(encoding='utf-8')), 'smoke')


def job(label='full_0.1', profile='rotating'):
    return next(j for j in jobs(configuration(), 'smoke') if j['method_label'] == label and j['demand_profile'] == profile)


def environment(j):
    return DynamicDemandEnvironment(j, Scenario(j['n'], j['tasks'], j['steps'], j['seed'], j['family'], j['scenario'], j['prep']))


class DynamicDemandTests(unittest.TestCase):
    def test_formal_grid_and_independent_seeds(self):
        c = effective_config(json.loads((ROOT/'dynamic_protocol.json').read_text()), 'full')
        jj = jobs(c, 'full')
        self.assertEqual(len(c['conditions']), 12)
        self.assertEqual(len(variants(c)), 20)
        self.assertEqual(len(jj), 3840)
        self.assertEqual(len({digest(j) for j in jj}), len(jj))
        self.assertFalse(set(c['development_seeds']) & set(c['evaluation_seeds']))

    def test_masks_equal_budget_rotation_and_common_first_event(self):
        envs = {p: environment(job(profile=p)) for p in ['normal', 'spread', 'fixed', 'rotating']}
        self.assertEqual(envs['normal'].mask.sum(), 0)
        self.assertEqual(envs['spread'].mask.sum(), envs['fixed'].mask.sum())
        self.assertEqual(envs['fixed'].mask.sum(), envs['rotating'].mask.sum())
        np.testing.assert_array_equal(envs['fixed'].mask[16:24], envs['rotating'].mask[16:24])
        self.assertFalse(set(envs['rotating'].cohorts[0]) & set(envs['rotating'].cohorts[1]))
        self.assertEqual(envs['fixed'].cohorts[0], envs['fixed'].cohorts[1])
        self.assertTrue(all(not e.mask[:16].any() for e in envs.values()))

    def test_schedule_independent_of_policy_and_balanced_phases(self):
        a, b = environment(job('full_0.1')), environment(job('random_0.5'))
        self.assertEqual(a.schedule_hash, b.schedule_hash)
        self.assertEqual(a.phase_hash, b.phase_hash)
        for q in range(a.scenario.tasks):
            phases = a.phases[a.scenario.home == q]
            counts = np.bincount(phases, minlength=a.job['period'])
            self.assertLessEqual(counts.max()-counts.min(), 1)

    def test_round_robin_covers_everyone_and_has_fresh_episode_state(self):
        j = job('round_robin_0.25')
        policy = IntuitivePolicy(j)
        positions = np.zeros((16, 2))
        seen = set()
        first_choice = None
        for tick in [16, 24, 32, 40]:
            chosen = policy.choose(tick, 0, 0, list(range(1, 9)), positions, 0)
            if first_choice is None:
                first_choice = chosen
            self.assertEqual(len(chosen), 2)
            self.assertFalse(seen & set(chosen))
            seen.update(chosen)
        self.assertEqual(seen, set(range(1, 9)))
        self.assertEqual(IntuitivePolicy(j).choose(16, 0, 0, list(range(1, 9)), positions, 0), first_choice)

    def test_fresh_update_rejects_old_and_deduplicates(self):
        e = environment(job())
        r = e.cohorts[0][0]
        q = int(e.scenario.home[r])
        def delivery(created, source=99):
            return types.SimpleNamespace(receiver=r, payload=types.SimpleNamespace(task=q,
                   observation=types.SimpleNamespace(source=source, tick=created)), created_tick=created)
        e.delivery(18, delivery(15))
        self.assertIsNone(e.first[0][r])
        e.delivery(20, delivery(17))
        e.delivery(21, delivery(17))
        self.assertEqual(e.first[0][r], 4)
        self.assertEqual(e.fresh_counts[0][r], 1)

    def test_baseline_does_not_call_taskgraph_decisions(self):
        for label in ['random_0.5', 'nearest_0.5', 'round_robin_0.5', 'periodic_2', 'no_new_communication']:
            with self.subTest(label=label):
                j = dict(job(label), value_audit=False)
                with patch('tgexp.engine.PreparationIndex', side_effect=AssertionError('group index')), \
                     patch('tgexp.engine._values', side_effect=AssertionError('control score')), \
                     patch('tgexp.engine.decide_pressure_local', side_effect=AssertionError('source gate')), \
                     patch('tgexp.engine.select', side_effect=AssertionError('set solver')):
                    r = run_episode(j)
                self.assertEqual(r['protected_events'], 0)
                self.assertEqual(r['preparation_updates'], 0)
                self.assertEqual(r['recipient_accounting_difference'], 0)

    def test_plan_only_reports_full_workload_without_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)/'must_not_exist'
            result = subprocess.run([sys.executable, str(ROOT/'run_dynamic_demand.py'),
                                     '--mode', 'full', '--plan-only', '--out', str(out)],
                                    check=True, capture_output=True, text=True)
            plan = json.loads(result.stdout)
            self.assertEqual(plan['total_episodes'], 3840)
            self.assertEqual(plan['development_episodes'], 960)
            self.assertEqual(plan['evaluation_episodes'], 2880)
            self.assertFalse(out.exists())

    def test_common_warmup_and_service_subtraction(self):
        records = [run_episode(dict(job(label), value_audit=False)) for label in
                   ['full_0.1', 'random_0.5', 'round_robin_0.5', 'dense_task_reference', 'no_new_communication']]
        self.assertEqual(len({r['dynamic_demand']['warmup_state_hash'] for r in records}), 1)
        for r in records:
            c = r['dynamic_demand']['communication']
            self.assertEqual(c['evaluation_payload_bit_hop'], r['network']['payload_bit_hop']-c['warmup_payload_bit_hop'])
        silent = records[-1]
        self.assertEqual(silent['dynamic_demand']['stage_counts']['nonempty_multicasts'], 0)
        self.assertEqual(silent['dynamic_demand']['fresh_update']['no_update_fraction'], 1.)
        self.assertGreater(silent['dynamic_demand']['communication']['warmup_payload_bit_hop'], 0)

    def test_diagnostics_do_not_change_closed_loop(self):
        for label in ['full_0.1', 'random_0.5']:
            a = run_episode(dict(job(label), value_audit=True))
            b = run_episode(dict(job(label), value_audit=False))
            self.assertEqual(a['network'], b['network'])
            self.assertEqual(a['evaluation'], b['evaluation'])
            self.assertEqual(a['trace'], b['trace'])
            json.loads(json.dumps(a, allow_nan=False))

    def test_spread_primary_is_actual_blocked_member_times(self):
        e = environment(job(profile='spread'))
        e.reward[:] = .8
        e.reward[e.mask] = .2
        e.warmup_cost, e.warmup_state_hash = 0, 'test'
        net = types.SimpleNamespace(payload_bit_hop=0, pending=lambda: 0, queue_snapshot=lambda: {})
        result = e.finish(net)
        self.assertTrue(all(abs(x['primary_reward']-.2) < 1e-12 for x in result['events']))

    def test_no_qualified_full_does_not_select_test_best(self):
        c = configuration()
        c['conditions'] = [c['conditions'][0]]
        rows = []
        for v in variants(c):
            rows.append(dict(condition_id=c['conditions'][0]['condition_id'], method=v['method_label'],
                             quality_pass=v['method_label'] == 'dense_task_reference', cost_mean=100,
                             team_loss_upper=.02, worst_task_loss_upper=.02, worst_event_loss_upper=.02))
        choice = calibrate(rows, c, 'test')
        full = next(x for x in choice['selected'] if x['family'] == 'full')
        self.assertIsNone(full['selected_label'])
        self.assertFalse(full['calibration_pass'])
        self.assertTrue(all(x['selected_label'] == 'dense_task_reference' for x in choice['selected'] if x['family'] != 'full'))

    def test_worst_event_is_maximum_of_paired_event_losses(self):
        def record(event_rewards, team, tasks):
            return dict(dynamic_demand=dict(schedule_hash='s', phase_hash='p', warmup_state_hash='w',
                        events=[dict(primary_reward=x) for x in event_rewards]),
                        evaluation=dict(reward=dict(mean=team), per_task_mean=tasks))
        ours = record([.8, .8, .8, .8], .79, [.7, .9])
        ref = record([.9, .7, .8, .8], .8, [.9, .8])
        loss = paired_losses(ours, ref)
        self.assertAlmostEqual(loss['worst_event_loss'], .1)
        self.assertAlmostEqual(loss['mean_event_loss'], 0.)
        self.assertAlmostEqual(loss['worst_task_loss'], .2)
        self.assertAlmostEqual(loss['team_loss'], .01)

    def test_legacy_full_pipeline_unchanged_without_environment_hook(self):
        module = types.ModuleType('tgexp._reference_engine')
        module.__package__ = 'tgexp'
        sys.modules[module.__name__] = module
        exec(compile((ROOT/'reference/engine_before_dynamic.py').read_text(), 'reference_engine.py', 'exec'), module.__dict__)
        j = dict(job(), dynamic_demand=False)
        a, b = run_episode(j), module.run_episode(j)
        for k in ['network', 'evaluation', 'trace', 'theory_audit']:
            self.assertEqual(a[k], b[k], k)


if __name__ == '__main__':
    unittest.main()
