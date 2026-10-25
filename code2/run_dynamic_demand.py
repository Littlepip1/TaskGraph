#!/usr/bin/env python3
"""One command: frozen dynamic-demand protocol, development, evaluation, ZIP."""
from __future__ import annotations
import os
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[name] = '1'
import argparse
import copy
import hashlib
import json
import multiprocessing
import platform
import sys
import time
import traceback
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import scipy
from tgexp.engine import run_episode
from tgexp.reporting import atomic_json, write_csv
from dynamic_reporting import curves, calibrate, export_results
from audit_stage import audit_stage

ROOT = Path(__file__).resolve().parent


def digest(job):
    return hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()[:24]


def source_files():
    files = [p for p in ROOT.iterdir() if p.suffix == '.py' or p.name == 'dynamic_protocol.json']
    for folder in ['tgexp', 'vendor_service_v3']:
        files += [p for p in (ROOT/folder).rglob('*') if p.suffix in {'.py', '.json'}]
    return sorted(files)


def source_hash():
    h = hashlib.sha256()
    for p in source_files():
        h.update(p.relative_to(ROOT).as_posix().encode())
        h.update(p.read_text(encoding='utf-8').encode())
    return h.hexdigest()


def variants(c):
    result = [dict(method='full', method_label=f'full_{f:g}', fraction=f, simple_strategy='full') for f in c['full_fractions']]
    result.append(dict(method='dense_task', method_label='dense_task_reference', fraction=0.))
    for rule in ['random', 'nearest', 'round_robin']:
        for p in c['keep_fractions']:
            result.append(dict(method='intuitive', method_label=f'{rule}_{p:g}', fraction=0.,
                               intuitive_rule=rule, candidate_scope='current_task', keep_fraction=p))
    for k in c['period_multipliers']:
        result.append(dict(method='intuitive', method_label=f'periodic_{k}', fraction=0.,
                           intuitive_rule='periodic', candidate_scope='current_task', period_multiplier=k))
    result.append(dict(method='intuitive', method_label='no_new_communication', fraction=0.,
                       intuitive_rule='none', candidate_scope='current_task'))
    return result


def effective_config(c, mode):
    c = copy.deepcopy(c)
    if mode == 'smoke':
        c.update(n=16, tasks=2, steps=96, eval_start=16, prep=8, deadline=16,
                 blackout_starts=[16, 36, 56, 76], blackout_duration=8, recovery_tail=4,
                 capacities=[256], main_period=8, sparse_period=16,
                 development_seeds=[2710909900, 2710909901], evaluation_seeds=[2710909902, 2710909903])
    elif mode == 'pilot':
        c.update(n=64, tasks=2, development_seeds=[2710909000, 2710909001],
                 evaluation_seeds=[2710909002, 2710909003])
    conditions = []
    if mode == 'pilot':
        tuples = [('rotating', 256, c['main_period']), ('rotating', 256, c['sparse_period'])]
    else:
        tuples = [(p, cap, c['main_period']) for p in c['profiles'] for cap in c['capacities']]
        tuples += [(p, cap, c['sparse_period']) for p in ['normal', 'rotating'] for cap in c['capacities']]
    for profile, capacity, period in tuples:
        conditions.append(dict(condition_id=f'{profile}_c{capacity}_p{period}', demand_profile=profile,
                               capacity=capacity, period=period))
    c['conditions'] = conditions
    c['variant_labels'] = [v['method_label'] for v in variants(c)]
    return c


def jobs(c, mode):
    result = []
    for phase in ['development', 'evaluation']:
        for condition in c['conditions']:
            for seed in c[phase+'_seeds']:
                for v in variants(c):
                    job = {k: c[k] for k in ['n', 'tasks', 'scenario', 'family', 'steps', 'eval_start', 'prep', 'deadline',
                                             'blackout_starts', 'blackout_duration', 'recovery_tail', 'blackout_fraction',
                                             'value_audit', 'save_agent_arrays']}
                    job.update(condition)
                    job.update(v)
                    job.update(seed=seed, phase=phase, experiment_mode=mode, dynamic_demand=True,
                               smoke=mode == 'smoke', absolute=0., context_lag=0, snapshot_count=0, event_log=False)
                    result.append(job)
    return result


def worker(job, sha, path):
    path = Path(path)
    record = run_episode(job)
    arrays = record['dynamic_demand'].pop('_agent_arrays', None)
    if arrays is not None:
        array_path = path.with_suffix('.npz')
        temporary = array_path.with_suffix('.npz.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(array_path)
        record['agent_arrays_file'] = array_path.name
    record.update(source_hash=sha, job_id=digest(job), receiver_context_used_for_selection=job['method'] == 'full')
    atomic_json(path, record)
    return record['elapsed_seconds']


def execute(specs, folder, sha, workers, summary_only=False):
    folder.mkdir(parents=True, exist_ok=True)
    pending = []
    for job in specs:
        path = folder/(digest(job)+'.json')
        if path.exists():
            r = json.loads(path.read_text(encoding='utf-8'))
            if r.get('source_hash') != sha or r['job'] != job:
                raise RuntimeError('Result/source mismatch. Use a new output directory: '+str(path))
            if job['save_agent_arrays'] and not path.with_suffix('.npz').exists():
                raise RuntimeError('Missing companion arrays: '+str(path))
        else:
            pending.append((job, path))
    print(f'{specs[0]["phase"]}: {len(specs)-len(pending)} cached; {len(pending)} pending; workers={workers}', flush=True)
    if pending and summary_only:
        raise RuntimeError('Stage incomplete; rerun without --summary-only')
    errors = []
    if pending:
        start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = {pool.submit(worker, j, sha, str(p)): (j, p) for j, p in pending}
            for count, future in enumerate(as_completed(futures), 1):
                job, path = futures[future]
                try:
                    future.result()
                    path.with_suffix('.error.json').unlink(missing_ok=True)
                except Exception:
                    error = dict(job=job, error=traceback.format_exc())
                    atomic_json(path.with_suffix('.error.json'), error)
                    errors.append(error)
                    print('FAILED: '+job['condition_id']+' '+job['method_label'], flush=True)
                    for other in futures:
                        other.cancel()
                    break
                if count == 1 or count % 8 == 0 or count == len(pending):
                    elapsed = time.perf_counter()-start
                    print(f'[{count}/{len(pending)}] elapsed={elapsed/60:.1f} min; ETA={(len(pending)-count)*elapsed/count/60:.1f} min; failures={len(errors)}', flush=True)
    if errors:
        raise RuntimeError(f'{len(errors)} failures. Inspect *.error.json and resume with the identical command.')
    return [json.loads((folder/(digest(j)+'.json')).read_text(encoding='utf-8')) for j in specs]


def pack(out, sha, mode, workers):
    dest = out/'DYNAMIC_DEMAND_RETURN.zip'
    temporary = dest.with_suffix('.zip.tmp')
    atomic_json(out/'manifest.json', dict(mode=mode, source_hash=sha, python=platform.python_version(),
                numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform(), workers=workers))
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p not in {dest, temporary} and '.tmp' not in p.name:
                z.write(p, p.relative_to(out))
        for p in source_files():
            z.write(p, 'source/'+p.relative_to(ROOT).as_posix())
    temporary.replace(dest)
    return dest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['smoke', 'pilot', 'full'], default='full')
    parser.add_argument('--workers', type=int, default=min(6, max(1, (os.cpu_count() or 2)-1)))
    parser.add_argument('--out', type=Path)
    parser.add_argument('--summary-only', action='store_true')
    parser.add_argument('--plan-only', action='store_true', help='Print frozen workload without running or writing results')
    parser.add_argument('--agent-arrays', action='store_true', help='Save optional per-agent NPZ diagnostics; larger return file')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    c = effective_config(json.loads((ROOT/'dynamic_protocol.json').read_text(encoding='utf-8')), args.mode)
    c['save_agent_arrays'] = args.agent_arrays or c['save_agent_arrays']
    specs = jobs(c, args.mode)
    sha = source_hash()
    if args.plan_only:
        print(json.dumps(dict(mode=args.mode, conditions=len(c['conditions']),
                              configurations=len(variants(c)),
                              development_episodes=sum(j['phase'] == 'development' for j in specs),
                              evaluation_episodes=sum(j['phase'] == 'evaluation' for j in specs),
                              total_episodes=len(specs), source_hash=sha,
                              labels=c['variant_labels'], condition_list=c['conditions']),
                         ensure_ascii=False, indent=2))
        return
    out = (args.out or ROOT/'runs'/args.mode).resolve()
    out.mkdir(parents=True, exist_ok=True)
    freeze = dict(source_hash=sha, config=c, expected_job_ids=[digest(j) for j in specs])
    p = out/'protocol_freeze.json'
    if p.exists() and json.loads(p.read_text(encoding='utf-8')) != freeze:
        raise RuntimeError('Frozen source/protocol mismatch. Resume original code or use a new output directory.')
    atomic_json(p, freeze)
    atomic_json(out/'acceptance.json', dict(execution_complete=False, mode=args.mode, source_hash=sha,
                                          expected_total_episodes=len(specs)))
    atomic_json(out/'progress.json', dict(execution_complete=False, phase='development', expected_episodes=len(specs)))
    print(f'Mode={args.mode}; conditions={len(c["conditions"])}; configurations={len(variants(c))}; total episodes={len(specs)}', flush=True)
    dev_specs = [j for j in specs if j['phase'] == 'development']
    dev = execute(dev_specs, out/'development_episodes', sha, args.workers, args.summary_only)
    dev_rows = curves(dev, c['tolerance'])
    write_csv(out/'development_curves.csv', dev_rows)
    choices = calibrate(dev_rows, c, sha)
    selected_file = out/'selected_on_development.json'
    if selected_file.exists() and json.loads(selected_file.read_text(encoding='utf-8')) != choices:
        raise RuntimeError('Frozen development choices differ')
    atomic_json(selected_file, choices)
    # Validate common warmup for development too, then release large records.
    common = {}
    for r in dev:
        k = r['job']['condition_id'], r['job']['seed']
        common.setdefault(k, set()).add(r['dynamic_demand']['warmup_state_hash'])
    if any(len(v) != 1 for v in common.values()):
        raise AssertionError('Development warmup mismatch')
    del dev
    atomic_json(out/'progress.json', dict(execution_complete=False, phase='evaluation', expected_episodes=len(specs)))
    print('Development choices frozen. Starting independent evaluation.', flush=True)
    ev_specs = [j for j in specs if j['phase'] == 'evaluation']
    records = execute(ev_specs, out/'evaluation_episodes', sha, args.workers, args.summary_only)
    status = export_results(out, records, c, choices, args.mode, sha)
    atomic_json(out/'acceptance.json', dict(status, execution_complete=False,
                                           independent_audit_passed=False,
                                           phase='independent_audit'))
    print('Checking raw episodes and paired intervals independently.', flush=True)
    audit = audit_stage(out)
    status['independent_audit_passed'] = True
    status['independently_audited_episodes'] = audit['audited_episodes']
    status.update(completed_total_episodes=len(specs), expected_total_episodes=len(specs))
    atomic_json(out/'acceptance.json', status)
    atomic_json(out/'progress.json', dict(execution_complete=True, phase='complete', expected_episodes=len(specs)))
    dest = pack(out, sha, args.mode, args.workers)
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    print('RETURN FILE: '+str(dest), flush=True)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit('Interrupted. Rerun the same command to resume completed episodes.')
