#!/usr/bin/env python3
"""Controlled official structure-module comparison; no full-MARL claim."""
from __future__ import annotations
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ.setdefault(key,'1')
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import platform
import time
import traceback
import unittest
import zipfile
import numpy as np
import scipy
import sklearn
import tgexp
from tgexp.engine import run_episode
from tgexp.reporting import atomic_json,comparison_rows,write_csv,interval,environment_key

ROOT=Path(__file__).resolve().parent


def source_hash():
    h=hashlib.sha256()
    paths=[]
    for folder in ['tgexp','vendor_service_v3','official_modules','legacy_dynamic','tests']:
        paths.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in {'.py','.json'})
    paths.extend([ROOT/'run_baseline_comparison.py',ROOT/'structure_adapters.py',ROOT/'baseline_config.json'])
    for p in sorted(paths,key=lambda p:p.relative_to(ROOT).as_posix()):
        h.update(p.relative_to(ROOT).as_posix().encode());h.update(p.read_bytes())
    return h.hexdigest()


def identity(job):return hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest()[:24]


def jobs(config,smoke):
    seeds=config['seeds'][:1] if smoke else config['seeds']
    cases=[(12,1,'static'),(24,4,'overlap'),(48,8,'overlap'),(24,4,'switch')] if smoke else config['cases']
    result=[]
    for n,q,scenario in cases:
        names=['dense_task','ours_parent','ours_theta010_secondary','demand_only','hygma_group','expocomm_static','expocomm_one_peer']
        if q==1:names+=['legacy_dhg_additive','legacy_dhg_set']
        for family in config['families']:
            for capacity in config['capacities']:
                for seed in seeds:
                    for name in names:
                        method='dense_task' if name=='dense_task' else 'full' if name.startswith('ours_') else 'demand_only' if name=='demand_only' else 'external'
                        j=dict(n=n,tasks=q,scenario=scenario,family=family,capacity=capacity,seed=seed,
                               method=method,method_label=name,baseline_name=name,
                               fraction=.1 if name in {'ours_theta010_secondary','demand_only'} else 0.,absolute=0.,
                               steps=48 if smoke else config['steps'],eval_start=8 if smoke else config['eval_start'],
                               prep=8 if smoke else config['prep'],period=16,deadline=32,context_lag=0,
                               event_log=False,smoke=smoke,candidate_universe='global',
                               history_length=8,max_clusters=min(max(4,2*q),8),
                               external_selector='structure_adapters:create_policy' if method=='external' else None)
                        result.append(j)
    return result


def worker(job,sha,path):
    r=run_episode(job);r.update(job_id=identity(job),source_hash=sha)
    atomic_json(path,r)
    return r['elapsed_seconds']


def display_results(rows):
    return [dict(r,job=dict(r['job'],method=r['job']['method_label'])) for r in rows]


def summarize(out,rows,expected,failures,config,smoke):
    display=display_results(rows)
    # Parent name maps to engine's full for summary, then report explicit label.
    stats=comparison_rows(display,config['tolerance'])
    write_csv(out/'paired_vs_task_dense.csv',stats)
    lookup={environment_key(r['job']):r for r in rows if r['job']['method_label']=='ours_parent'}
    pairs={}
    for r in rows:
        name=r['job']['method_label']
        if name in {'ours_parent','dense_task'}:continue
        ours=lookup.get(environment_key(r['job']))
        if ours is None:continue
        j=r['job'];key=tuple(j[k] for k in ['n','tasks','scenario','family','capacity'])+(name,)
        pairs.setdefault(key,[]).append((ours,r))
    direct=[]
    for key,ps in sorted(pairs.items()):
        n,q,scenario,family,cap,name=key
        delta=interval([a['evaluation']['reward']['mean']-b['evaluation']['reward']['mean'] for a,b in ps])
        saving=[1-a['network']['payload_bit_hop']/b['network']['payload_bit_hop'] for a,b in ps if b['network']['payload_bit_hop']>0]
        cs=interval(saving) if saving else dict(mean=None,lower=None,upper=None)
        direct.append(dict(n=n,tasks=q,scenario=scenario,family=family,capacity=cap,baseline=name,seeds=len(ps),
                           ours_minus_baseline_reward=delta['mean'],reward_lower=delta['lower'],reward_upper=delta['upper'],
                           ours_service_saving_vs_baseline=cs['mean'],saving_lower=cs['lower'],saving_upper=cs['upper'],
                           zero_service_baseline_seeds=len(ps)-len(saving)))
    write_csv(out/'paired_vs_ours.csv',direct)
    episode=[]
    for r in rows:
        j=r['job'];ns=r['network']
        episode.append(dict(**{k:j[k] for k in ['n','tasks','scenario','family','capacity','seed','method_label']},
                            reward=r['evaluation']['reward']['mean'],worst_task_reward=r['evaluation']['worst_task_reward'],
                            payload=ns['payload_bit_hop'],per_agent_tick=r['cost_per_agent_tick'],
                            selected_mean=r['selected_count']['mean'],delivered=ns['delivered_packets'],
                            requested=ns['requested_recipient_copies'],expired=ns['expired_recipient_copies'],pending=ns['pending_deliveries'],
                            selector_ms=1000*(r['selection_seconds']['mean'] or 0.),
                            grouping_seconds=r.get('structure_audit',{}).get('grouping_seconds',0.),elapsed=r['elapsed_seconds']))
    write_csv(out/'episode_metrics.csv',episode)
    audit=dict(execution_complete=not failures and len(rows)==len(expected),completed=len(rows),expected=len(expected),
               failures=len(failures),smoke=smoke,full_trained_MARL_reproductions=False,
               official_structure_module_adaptations=True,
               primary_ours_fraction=0.,secondary_ours_fraction=.1,
               automatic_winner_selection=False,receiver_context_acquisition_accounted=False,
               hgc_message_encoding_reproduced=False,
               statistical_unit='environment seed; paired Student-t two-sided95 marginal intervals')
    atomic_json(out/'acceptance.json',audit)
    atomic_json(out/'failures.json',failures)
    (out/'ACCEPTANCE_CN.md').write_text('\n'.join([
        '# 通信结构受控对比验收','',f'执行完整：{audit["execution_complete"]}；完成{len(rows)}/{len(expected)}回合。',
        '比较的是相同估计器、控制器、原始观测载荷和物理网络上的通信组织结构。',
        'HYGMA仅接入官方动态谱聚类；ExpoComm仅接入官方指数拓扑。不得表述为完整训练方法复现。',
        '主方法保持父批次theta=0；theta=0.1为预先列出的次要配置，全部报告，不按评价表现切换主方法。',
        '无额外训练或参数选择，不使用这批独立种子调参；结构模块的适配参数见manifest。',
        '角色切换保护窗口对全部方法使用相同任务准备范围，窗口内暂不施加结构掩码。',
        '旧DHG原样复用，只适用于单任务。所有方法无固定32/48接收人数上限。',
        'paired_vs_task_dense.csv核对各方法是否满足0.01控制容差；paired_vs_ours.csv报告直接配对差异。',
        '服务更低且控制未通过的方法不能据此称为优胜；组内共享转发、上下文供给条件一致。',
        '上下文获取、结构维护信令和学习隐消息字节未纳入此结构适配的载荷统计。',
        '正式学习基线CommFormer/T2MAC及HYGMA/ExpoComm完整策略尚未在本包复现。',
        '冒烟结果仅验证程序执行；低维规则/结构方法对比不自动证明原论文完整系统优劣。',
    ]),encoding='utf-8')
    return audit


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workers',type=int,default=4);ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--output',type=Path);ap.add_argument('--resume',action='store_true')
    ap.add_argument('--plan',action='store_true');ap.add_argument('--summarize-only',action='store_true')
    args=ap.parse_args()
    if args.workers<1:ap.error('workers must be positive')
    config=json.loads((ROOT/'baseline_config.json').read_text());expected=jobs(config,args.smoke)
    if args.plan:
        print(json.dumps(dict(episodes=len(expected),seeds=1 if args.smoke else len(config['seeds']),
                              cases=config['cases'],scope='official structure-module adaptation'),indent=2));return 0
    out=(args.output or ROOT/'runs'/('SMOKE' if args.smoke else 'BASELINE_STRUCTURES')).resolve();out.mkdir(parents=True,exist_ok=True)
    sha=source_hash()
    official=json.loads((ROOT/'OFFICIAL_SOURCES.json').read_text())
    for rec in official:
        if hashlib.sha256((ROOT/rec['local_path']).read_bytes()).hexdigest()!=rec['sha256']:
            raise SystemExit('Official source changed: '+rec['local_path'])
    manifest=dict(source_hash=sha,config=config,smoke=args.smoke,expected=len(expected),official=official,
                  parameter_selection='none; parent primary frozen; all comparison variants prespecified',
                  runtime=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__))
    mf=out/'manifest.json'
    if mf.exists() and json.loads(mf.read_text())!=manifest:raise SystemExit('Different config/source/runtime; use a NEW output directory')
    atomic_json(mf,manifest)
    todo=[]
    folder=out/'episodes';folder.mkdir(exist_ok=True)
    for j in expected:
        p=folder/(identity(j)+'.json')
        if p.exists():
            try:
                r=json.loads(p.read_text())
                if r['source_hash']==sha and r['job']==j:continue
            except (ValueError,KeyError):pass
            p.rename(p.with_suffix('.invalid-'+str(time.time_ns())+'.json'))
        todo.append((j,p))
    failures=[]
    if not args.summarize_only:
        with (out/'tests.log').open('w',encoding='utf-8') as log:
            result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT/'tests')))
        if not result.wasSuccessful():raise SystemExit('Tests failed; inspect tests.log')
        print(f'Cached {len(expected)-len(todo)}; pending {len(todo)}',flush=True)
        pool=ProcessPoolExecutor(args.workers,mp_context=multiprocessing.get_context('spawn'))
        start=time.perf_counter();futures={}
        try:
            futures={pool.submit(worker,j,sha,str(p)):(j,p) for j,p in todo}
            for k,f in enumerate(as_completed(futures),1):
                j,p=futures[f]
                try:
                    duration=f.result();message=f'ok {duration:.1f}s'
                    error=p.with_suffix('.error.json')
                    if error.exists():error.unlink()
                except Exception:
                    detail=dict(job=j,error=traceback.format_exc());failures.append(detail)
                    atomic_json(p.with_suffix('.error.json'),detail);message='FAILED'
                elapsed=time.perf_counter()-start;eta=elapsed*(len(todo)-k)/k
                print(f'[{k}/{len(todo)}] N{j["n"]} Q{j["tasks"]} {j["family"]} C{j["capacity"]} '
                      f'{j["method_label"]} s{j["seed"]}: {message}; ETA {eta/60:.1f} min',flush=True)
                atomic_json(out/'progress.json',dict(processed=k,pending=len(todo)-k,failures=len(failures)))
        except KeyboardInterrupt:
            for f in futures:f.cancel()
            print('Canceling queued jobs. In-flight results will save; run again to resume.',flush=True)
            raise
        finally:pool.shutdown(wait=True,cancel_futures=True)
    rows=[]
    for j in expected:
        p=folder/(identity(j)+'.json')
        if p.exists():
            r=json.loads(p.read_text())
            if r.get('source_hash')!=sha or r['job']!=j:raise SystemExit('Stale data: '+str(p))
            rows.append(r)
        elif not any(f.get('job')==j for f in failures):failures.append(dict(job=j,error='missing'))
    audit=summarize(out,rows,expected,failures,config,args.smoke)
    archive=out.parent/(out.name+'_RETURN.zip');temporary=archive.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(out))
    temporary.replace(archive)
    print(json.dumps(audit,ensure_ascii=False,indent=2));print('RETURN:',archive)
    return 0 if audit['execution_complete'] else 2


if __name__=='__main__':
    multiprocessing.freeze_support();raise SystemExit(main())

