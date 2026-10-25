import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import tgexp
from paper_adapters import opinion, combine_opinions, PaperPolicy
from run_recent_paper_baselines import base_job, train_graph, ROOT
from tgexp.engine import run_episode


class PaperMechanismTests(unittest.TestCase):
    def test_evidence_mass_and_fusion(self):
        ctrl=dict(kp=.4,max_speed_mps=3.,reward_scale_m=6.)
        args=(np.zeros(2),np.zeros(2),np.ones(2),1.,ctrl)
        b,u=opinion(args[0],np.eye(2)*4,*args[1:])
        b1,u1=opinion(args[0],np.eye(2),*args[1:])
        self.assertAlmostEqual(float(b.sum()+u),1.)
        self.assertLess(u1,u)
        combined,uncertainty,conflict=combine_opinions(b,u,b1,u1)
        self.assertAlmostEqual(float(combined.sum()+uncertainty),1.)
        self.assertTrue(np.isfinite(combined).all())
        self.assertGreaterEqual(conflict,0.)
        self.assertLessEqual(uncertainty,u+1e-12)

    def test_independent_policies_in_actual_closed_loop(self):
        config=json.loads((ROOT/'recent_paper_config.json').read_text())
        for name in ['commformer_idea','t2mac_idea','hygma_group','expocomm_static','expocomm_one_peer']:
            job=base_job(config,12,2,'overlap','MIS',256,2710799500,name,True,
                         baseline_name=name,graph_weights=[-.1,.05,.05,.05,.05,.8,.8,.05],
                         evidence_threshold=.2,max_clusters=4)
            result=run_episode(job)
            self.assertEqual(result['recipient_accounting_difference'],0)
            audit=result['structure_audit']
            self.assertFalse(audit['uses_taskgraph_receiver_score'])
            self.assertFalse(audit['uses_taskgraph_omission_quota'])
            self.assertGreater(audit['policy_calls'],0)

    def test_learned_graph_checkpoint_resume(self):
        config=json.loads((ROOT/'recent_paper_config.json').read_text())
        spec=dict(service_weight=.05,cost_index=0,training_seed=2710799600,
                  training_index=0,smoke=True)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.json'
            train_graph(spec,config,'test-source',path)
            first=path.read_bytes();data=json.loads(first)
            self.assertTrue(data['complete'])
            self.assertEqual(len(data['history']),2)
            self.assertTrue(all(len(h['utility'])==2 for h in data['history']))
            self.assertEqual(len(data['weights']),8)
            train_graph(spec,config,'test-source',path)
            self.assertEqual(path.read_bytes(),first)
            with self.assertRaises(RuntimeError):
                train_graph(spec,config,'different-source',path)

    def test_graph_parameters_control_receiver_decisions(self):
        policy=PaperPolicy(dict(n=4,baseline_name='commformer_idea',graph_weights=[0.]*8))
        # The intercept alone creates either an empty or full feasible mask.
        # Patch features to isolate the graph decision from belief construction.
        from unittest.mock import patch
        context=dict(source=0,task=0,candidates=[1,2,3])
        with patch('paper_adapters.graph_features',return_value=np.array([1.,0,0,0,0,0,0,0])):
            policy.job['graph_weights'][0]=-1.
            self.assertEqual(policy(context),[])
            policy.job['graph_weights'][0]=1.
            self.assertEqual(policy(context),[1,2,3])


if __name__=='__main__':unittest.main()
