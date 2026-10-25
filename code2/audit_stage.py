"""Independent output audit; reads raw episodes, not the analysis functions."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import t


def audit_stage(directory):
    out = Path(directory).resolve()
    freeze = json.loads((out/'protocol_freeze.json').read_text(encoding='utf-8'))
    ids, index, hashes = set(), {}, {}
    for phase in ['development', 'evaluation']:
        for p in (out/(phase+'_episodes')).glob('*.json'):
            assert '.error.' not in p.name, str(p)
            r = json.loads(p.read_text(encoding='utf-8'))
            j, d = r['job'], r['dynamic_demand']
            identity = hashlib.sha256(json.dumps(j, sort_keys=True).encode()).hexdigest()[:24]
            assert r['job_id'] == identity and r['source_hash'] == freeze['source_hash']
            assert r['recipient_accounting_difference'] == 0
            assert r['theory_audit']['quota_failures'] == 0
            assert d['communication']['evaluation_payload_bit_hop'] == r['network']['payload_bit_hop']-d['communication']['warmup_payload_bit_hop']
            assert d['communication']['evaluation_payload_bit_hop'] >= 0
            hkey = phase, j['condition_id'], j['seed']
            hashes.setdefault(hkey, set()).add((d['warmup_state_hash'], d['schedule_hash'], d['phase_hash']))
            ids.add(identity)
            if phase == 'evaluation':
                index[j['condition_id'], j['seed'], j['method_label']] = dict(
                    reward=r['evaluation']['reward']['mean'], tasks=r['evaluation']['per_task_mean'],
                    cost=d['communication']['evaluation_payload_bit_hop'], events=[x['primary_reward'] for x in d['events']])
    assert ids == set(freeze['expected_job_ids'])
    assert all(len(values) == 1 for values in hashes.values())
    with (out/'evaluation_curves.csv').open(encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    checked = 0
    for row in rows:
        arrays = {k: [] for k in ['team_loss', 'worst_task_loss', 'worst_event_loss', 'cost']}
        for seed in freeze['config']['evaluation_seeds']:
            a = index[row['condition_id'], seed, row['method']]
            b = index[row['condition_id'], seed, 'dense_task_reference']
            arrays['team_loss'].append(b['reward']-a['reward'])
            arrays['worst_task_loss'].append(max(np.asarray(b['tasks'])-a['tasks']))
            arrays['worst_event_loss'].append(max(y-x for x, y in zip(a['events'], b['events'])))
            arrays['cost'].append(a['cost'])
        for name, values in arrays.items():
            values = np.asarray(values)
            mu = values.mean()
            half = t.ppf(.975, len(values)-1)*values.std(ddof=1)/np.sqrt(len(values))
            for suffix, expected in [('mean', mu), ('lower', mu-half), ('upper', mu+half)]:
                assert abs(float(row[name+'_'+suffix])-expected) < max(1e-12, abs(expected)*1e-12)
            checked += 1
    summary = dict(audited_episodes=len(ids), independently_recomputed_intervals=checked,
                   source_hash=freeze['source_hash'], all_expected_jobs_present=True,
                   paired_warmup_and_schedule_match=True, service_subtraction_correct=True, accounting_failures=0)
    (out/'independent_audit.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit_stage(args.directory), indent=2))
