#!/usr/bin/env python3
"""One command: validate -> development -> freeze -> held-out -> paired report.

No figures, no fixed-K policies, no fabricated learned baselines.
"""
from __future__ import annotations
import os
# Keep process parallelism from multiplying BLAS threads; user can override.
for _name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ.setdefault(_name, '1')
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import hashlib
import importlib.util
import json
import multiprocessing
from pathlib import Path
import platform
import sys
import time
import traceback
import unittest
import zipfile

import tgexp
from tgexp.engine import run_episode
from tgexp.reporting import atomic_json, comparison_rows, export_results, write_csv

ROOT = Path(__file__).resolve().parent
VERSION = 'TaskGraph-AllExperiments-v1.0'


def source_hash():
    h = hashlib.sha256()
    for folder in ['tgexp', 'tests', 'vendor_service_v3']:
        for f in sorted((ROOT / folder).rglob('*')):
            if f.is_file() and f.suffix in {'.py', '.json'} and '__pycache__' not in f.parts:
                h.update(f.relative_to(ROOT).as_posix().encode())
                h.update(f.read_bytes())
    for f in [ROOT / 'run_all_experiments.py', ROOT / 'experiment_config.json']:
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def identity(job):
    return hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()[:24]


def worker(job, code_hash, destination):
    result = run_episode(job)
    result.update(source_hash=code_hash, job_id=identity(job))
    atomic_json(destination, result)
    return dict(job_id=result['job_id'], elapsed=result['elapsed_seconds'])


def execute(jobs, folder, workers, code_hash):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    unique = {identity(j): j for j in jobs}
    pending = []
    for jid, job in unique.items():
        path = folder / (jid + '.json')
        if path.exists():
            try:
                r = json.loads(path.read_text(encoding='utf-8'))
                if r.get('source_hash') == code_hash and r.get('job_id') == jid and r.get('job') == job:
                    continue
            except (ValueError, OSError):
                pass
            path.rename(path.with_suffix(f'.invalid-{time.time_ns()}.json'))
        pending.append((jid, job, str(path)))
    print(f'{folder.name}: cached={len(unique)-len(pending)}, pending={len(pending)}', flush=True)
    failures = []
    started = time.perf_counter()
    # spawn behaves consistently on Windows and Linux; each job has independent RNG.
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'))
    try:
        futures = {pool.submit(worker, j, code_hash, p): (jid, j) for jid, j, p in pending}
        for done, future in enumerate(as_completed(futures), 1):
            jid, job = futures[future]
            try:
                result = future.result()
                error_file = folder / (jid + '.error.json')
                if error_file.exists():
                    error_file.unlink()
                status = f'ok {result["elapsed"]:.1f}s'
            except Exception:
                detail = dict(job_id=jid, job=job, traceback=traceback.format_exc())
                failures.append(detail)
                atomic_json(folder / (jid + '.error.json'), detail)
                status = 'FAILED (see .error.json)'
            elapsed = time.perf_counter() - started
            eta = elapsed * (len(pending) - done) / done
            print(f'[{done}/{len(pending)}] N={job["n"]} Q={job["tasks"]} {job["scenario"]} '
                  f'{job["family"]} C={job["capacity"]} {job["method"]} theta={job["fraction"]} '
                  f'seed={job["seed"]}: {status}; ETA~{eta/60:.1f} min', flush=True)
            atomic_json(folder.parent / (folder.name + '_progress.json'),
                        dict(expected=len(unique), initial_cached=len(unique)-len(pending), processed=done,
                             failed=len(failures), pending=len(pending)-done, elapsed_seconds=elapsed))
    except KeyboardInterrupt:
        print('Interrupted: canceling queued jobs; waiting for in-flight jobs to save. Re-run to resume.', flush=True)
        for future in futures:
            future.cancel()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    results = []
    for jid in unique:
        path = folder / (jid + '.json')
        if path.exists():
            r = json.loads(path.read_text(encoding='utf-8'))
            if r.get('source_hash') == code_hash and r.get('job_id') == jid:
                results.append(r)
    return results, failures


def templates(config, smoke=False, extended=False):
    sizes = [12, 24, 48] if smoke else config['sizes'] + ([512] if extended else [])
    local_group = 6 if smoke else config['local_group_size']
    # label is planning metadata; duplicate jobs are physically run once.
    cases = []
    single_n = sizes[0]
    cases.append(('single_task', single_n, 1, 'static',
                  ['dense_global', 'dense_task', 'full', 'demand_only', 'legacy_additive', 'uncertainty', 'innovation'], 0))
    for n in sizes:
        cases.append(('local_scaling', n, max(2, n // local_group), 'overlap',
                      ['dense_task', 'full', 'demand_only', 'dense_global'] if n <= sizes[1] else ['dense_task', 'full', 'demand_only'], 0))
    cases.append(('ablation_handover', sizes[1], max(2, sizes[1] // local_group), 'switch',
                  ['dense_task', 'full', 'no_group', 'no_prepare', 'no_shared', 'no_guard', 'forward_only', 'demand_only'], 0))
    for n in sizes:
        cases.append(('growing_group_pressure', n, 2, 'overlap', ['dense_task', 'full'], 0))
    for lag in config['context_lags']:
        if lag:
            cases.append(('context_sensitivity', sizes[1], max(2, sizes[1] // local_group), 'overlap', ['full'], lag))
    return cases


def make_job(config, n, tasks, scenario, family, capacity, seed, method, fraction=.1, lag=0, smoke=False, external=None):
    return dict(n=int(n), tasks=int(tasks), scenario=scenario, family=family, capacity=int(capacity), seed=int(seed),
                method=method, fraction=float(fraction), absolute=float(config['absolute_event_budget']),
                steps=config['smoke_steps'] if smoke else config['steps'], eval_start=8 if smoke else config['eval_start'],
                prep=8 if smoke else config['preparation_steps'], period=config['opportunity_period'],
                deadline=config['deadline_steps'], context_lag=int(lag), smoke=smoke,
                external_selector=external, event_log=bool(config['save_event_logs']))


def build_development(config, smoke):
    seeds = config['development_seeds'][:1] if smoke else config['development_seeds']
    n = 12 if smoke else 64
    methods = ['full', 'demand_only', 'legacy_additive', 'uncertainty', 'innovation']
    jobs = []
    for family in config['families']:
        for capacity in config['capacities']:
            for seed in seeds:
                jobs.append(make_job(config, n, 2, 'overlap', family, capacity, seed, 'dense_task', 0., smoke=smoke))
                for method in methods:
                    for theta in config['development_fractions']:
                        jobs.append(make_job(config, n, 2, 'overlap', family, capacity, seed, method, theta, smoke=smoke))
    return jobs, methods


def freeze(rows, methods, config, smoke):
    table = comparison_rows(rows, config['development_margin'])
    selected = {}
    for method in methods:
        profiles = []
        for theta in config['development_fractions']:
            stats = [r for r in table if r['method'] == method and r['fraction'] == theta]
            valid = len(stats) == len(config['families']) * len(config['capacities']) and all(
                r['gap_upper'] is not None and r['gap_upper'] <= config['development_margin'] for r in stats)
            saving = sum(r['saving_mean'] for r in stats) / len(stats) if stats else float('-inf')
            profiles.append(dict(fraction=theta, eligible=valid, mean_saving=saving if stats else None))
        eligible = [p for p in profiles if p['eligible']]
        chosen = max(eligible, key=lambda p: (p['mean_saving'], -p['fraction'])) if eligible else profiles[0]
        selected[method] = dict(fraction=chosen['fraction'], eligible=chosen['eligible'], profiles=profiles,
                                rule='all development paired95 upper gap <= margin, then greatest mean saving',
                                fallback=not bool(eligible), smoke=smoke)
    return selected, table


def build_evaluation(config, selected, smoke, extended, external):
    seeds = config['evaluation_seeds'][:1] if smoke else config['evaluation_seeds']
    jobs = {}
    case_map = []
    for label, n, tasks, scenario, methods, lag in templates(config, smoke, extended):
        methods = list(methods)
        if external and label == 'single_task':
            methods.append('external')
        for family in config['families']:
            for capacity in config['capacities']:
                for seed in seeds:
                    # Every comparison has a task-dense paired reference.
                    ref = make_job(config, n, tasks, scenario, family, capacity, seed, 'dense_task', 0., smoke=smoke)
                    jobs[identity(ref)] = ref
                    case_map.append(dict(module=label, job_id=identity(ref)))
                    for method in methods:
                        if method == 'dense_task':
                            continue
                        profile = selected.get(method, selected['full'])
                        theta = 0. if method == 'dense_global' else profile['fraction']
                        j = make_job(config, n, tasks, scenario, family, capacity, seed, method, theta, lag, smoke, external)
                        jobs[identity(j)] = j
                        case_map.append(dict(module=label, job_id=identity(j)))
    return list(jobs.values()), case_map


def verify_vendor():
    root = ROOT / 'vendor_service_v3'
    expected = json.loads((root / 'UPSTREAM_SHA256.json').read_text())
    for rel, digest in expected.items():
        if hashlib.sha256((root / rel).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Upstream source changed: ' + rel)


def validate_config(c):
    if set(c['development_seeds']) & set(c['evaluation_seeds']):
        raise ValueError('Development and evaluation seeds overlap')
    if len(set(c['evaluation_seeds'])) != len(c['evaluation_seeds']) or len(c['evaluation_seeds']) < 2:
        raise ValueError('At least two distinct evaluation seeds required')
    if not 0 <= c['eval_start'] < c['steps'] or c['smoke_steps'] <= 8:
        raise ValueError('Invalid evaluation horizon')
    if c['opportunity_period'] < 1 or c['deadline_steps'] < 1 or c['preparation_steps'] < 0:
        raise ValueError('Invalid time settings')
    if any(x < 2 for x in c['sizes']) or len(set(c['sizes'])) != len(c['sizes']):
        raise ValueError('Invalid sizes')
    if not c['development_fractions'] or c['development_fractions'][0] != 0 or any(not 0 <= x <= 1 for x in c['development_fractions']):
        raise ValueError('Fractions must begin with a zero-omission fallback')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'experiment_config.json')
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--workers', type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument('--smoke', action='store_true', help='integrity checks, not performance evidence')
    parser.add_argument('--extended', action='store_true', help='also evaluate N=512')
    parser.add_argument('--plan', action='store_true', help='print workload without running')
    parser.add_argument('--resume', action='store_true', help='accepted for clarity; exact-job resume is always enabled')
    parser.add_argument('--summarize-only', action='store_true')
    parser.add_argument('--external-selector', help='optional real external adapter: module:function')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    config = json.loads(args.config.read_text(encoding='utf-8'))
    validate_config(config)
    external_hash = None
    if args.external_selector:
        module, _ = args.external_selector.split(':')
        spec = importlib.util.find_spec(module)
        if spec is None or not spec.origin:
            raise SystemExit('External baseline module unavailable')
        external_hash = hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest()
    verify_vendor()
    dev_jobs, methods = build_development(config, args.smoke)
    illustrative = {m: dict(fraction=.1) for m in methods}
    eval_jobs, _ = build_evaluation(config, illustrative, args.smoke, args.extended, args.external_selector)
    workload = dict(development_episodes=len(dev_jobs), evaluation_episodes=len(eval_jobs),
                    tasks=templates(config, args.smoke, args.extended), smoke=args.smoke,
                    independent_evaluation_seeds=1 if args.smoke else len(config['evaluation_seeds']))
    if args.plan:
        print(json.dumps(workload, ensure_ascii=False, indent=2))
        return 0
    out = (args.output or ROOT / 'runs' / ('SMOKE' if args.smoke else 'ALL_EXPERIMENTS')).resolve()
    out.mkdir(parents=True, exist_ok=True)
    code_hash = source_hash()
    manifest = dict(version=VERSION, source_hash=code_hash, config=config, smoke=args.smoke,
                    extended=args.extended, external_selector=args.external_selector, external_sha256=external_hash,
                    workload=workload, statistical_protocol='development freeze before evaluation; seed-paired two-sided95 CI',
                    baseline_scope='adaptive rules; no learned recent method is reproduced by renaming a heuristic')
    # Normalize tuple-valued planning metadata before comparing to stored JSON.
    manifest = json.loads(json.dumps(manifest, sort_keys=True))
    existing = out / 'manifest.json'
    if existing.exists() and json.loads(existing.read_text(encoding='utf-8')) != manifest:
        raise SystemExit('Configuration/source changed. Use a NEW output directory; old results are preserved.')
    atomic_json(existing, manifest)
    if not args.summarize_only:
        with (out / 'tests.log').open('w', encoding='utf-8') as log:
            suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'))
            with contextlib.redirect_stdout(log):
                check = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
        if not check.wasSuccessful():
            raise SystemExit('Preflight checks failed; see tests.log')
        dev_rows, failures = execute(dev_jobs, out / 'development', args.workers, code_hash)
        if failures:
            raise SystemExit('Development jobs failed. Re-run the same command to resume.')
        chosen, development_stats = freeze(dev_rows, methods, config, args.smoke)
        selection = dict(selected=chosen, development_source_hash=code_hash, selected_before_evaluation=True)
        selection_path = out / 'FROZEN_SELECTION.json'
        if selection_path.exists() and json.loads(selection_path.read_text(encoding='utf-8')) != selection:
            raise SystemExit('Frozen profile mismatch. Do not overwrite; use a new output directory.')
        atomic_json(selection_path, selection)
        write_csv(out / 'development_frontier.csv', development_stats)
    else:
        chosen = json.loads((out / 'FROZEN_SELECTION.json').read_text(encoding='utf-8'))['selected']
    eval_jobs, case_map = build_evaluation(config, chosen, args.smoke, args.extended, args.external_selector)
    atomic_json(out / 'evaluation_plan.json', dict(jobs=eval_jobs, case_map=case_map))
    print('Profiles frozen before held-out evaluation:', {m: p['fraction'] for m, p in chosen.items()}, flush=True)
    if args.summarize_only:
        evaluated, failures = [], []
        for j in eval_jobs:
            p = out / 'evaluation' / (identity(j) + '.json')
            if p.exists():
                r = json.loads(p.read_text(encoding='utf-8'))
                if r.get('source_hash') != code_hash or r.get('job') != j:
                    raise SystemExit('Stale result: ' + str(p))
                evaluated.append(r)
            else:
                failures.append(dict(job_id=identity(j), error='missing result'))
    else:
        evaluated, failures = execute(eval_jobs, out / 'evaluation', args.workers, code_hash)
    audit = export_results(out, evaluated, chosen, [identity(j) for j in eval_jobs], failures,
                           args.smoke, config['reward_tolerance'])
    atomic_json(out / 'runtime.json', dict(python=platform.python_version(), platform=platform.platform(), workers=args.workers))
    archive = out.parent / (out.name + '_RETURN.zip')
    tmp_archive = archive.with_suffix('.zip.tmp')
    with zipfile.ZipFile(tmp_archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.rglob('*')):
            if f.is_file() and not f.name.endswith('.tmp'):
                z.write(f, f.relative_to(out))
    tmp_archive.replace(archive)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print('RETURN PACKAGE:', archive, flush=True)
    return 0 if audit['execution_complete'] else 2


if __name__ == '__main__':
    multiprocessing.freeze_support()
    raise SystemExit(main())

