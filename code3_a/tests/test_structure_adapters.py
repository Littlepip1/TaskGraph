import unittest
import numpy as np
import tgexp
from structure_adapters import load_exponential_function,load_clustering_class,NumpyHistory
from tgexp.engine import run_episode


class OfficialAdapterTests(unittest.TestCase):
    def test_exponential_neighbors_match_equation(self):
        fn=load_exponential_function()
        for n in [12,64,128,256]:
            degree=int(np.ceil(np.log2(n)))+1
            actual=np.asarray(fn(2,n,degree))
            offsets=[0]+[2**k for k in range(degree-1)]
            expected=np.array([[(r+o)%n for o in offsets] for r in range(n)])
            np.testing.assert_array_equal(actual[0],expected)
            np.testing.assert_array_equal(actual[1],expected)

    def test_official_clustering_complete_partition(self):
        data=np.random.default_rng(4).normal(size=(16,3,4))
        cls=load_clustering_class();c=cls(2,3,16)
        state=np.random.get_state()
        try:
            np.random.seed(4)
            updated,groups,moved=c.update_groups(NumpyHistory(data),.6)
        finally:np.random.set_state(state)
        self.assertTrue(updated)
        self.assertEqual(sorted(r for g in groups for r in g),list(range(16)))

    def test_each_adapter_actual_closed_loop(self):
        for name in ['hygma_group','expocomm_static','expocomm_one_peer','legacy_dhg_set','legacy_dhg_additive']:
            q=1 if name.startswith('legacy') else 2
            job=dict(n=12,tasks=q,scenario='overlap',family='MIS',capacity=256,seed=2710799000,
                     method='external',method_label=name,baseline_name=name,fraction=0.,absolute=0.,
                     steps=40,eval_start=8,prep=8,period=16,deadline=32,context_lag=0,
                     event_log=False,smoke=True,candidate_universe='global',history_length=8,
                     max_clusters=4,external_selector='structure_adapters:create_policy')
            result=run_episode(job)
            self.assertEqual(result['recipient_accounting_difference'],0)
            self.assertEqual(result['structure_audit']['baseline'],name)
            self.assertGreater(result['structure_audit']['policy_calls'],0)


if __name__=='__main__':unittest.main()
