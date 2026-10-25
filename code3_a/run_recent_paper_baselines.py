#!/usr/bin/env python3
"""Train graph -> calibrate paper ideas -> freeze -> fresh paired evaluation."""
from __future__ import annotations
import os
for _key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ.setdefault(_key,'1')
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
from tgexp.reporting import atomic_json,comparison_rows,write_csv
from run_baseline_comparison import summarize,display_results

ROOT=Path(__file__).resolve().parent


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()[:24]


def source_hash():
    h=hashlib.sha256()
    files=[p for p in ROOT.rglob('*') if p.is_file() and p.suffix in {'.py','.json'}
           and '__pycache__' not in p.parts and 'runs' not in p.relative_to(ROOT).parts
           and p.name not in {'BUILD_VERIFICATION.json'}]
    for p in sorted(files,key=lambda p:p.relative_to(ROOT).as_posix()):
        h.update(p.relative_to(ROOT).as_posix().encode());h.update(p.read_bytes())
    return h.hexdigest()


def base_job(c,n,q,mode,family,cap,seed,label,smoke=False,**extra):
    method='dense_task' if label=='dense_task' else 'full' if label.startswith('ours_') else 'external'
    j=dict(n=n,tasks=q,scenario=mode,family=family,capacity=cap,seed=seed,method=method,
           method_label=label,baseline_name=extra.pop('baseline_name',label),
           fraction=.1 if label=='ours_theta010_secondary' else 0.,absolute=0.,
           steps=48 if smoke else c['steps'],eval_start=8 if smoke else c['eval_start'],
           prep=8 if smoke else c['prep'],period=16,deadline=32,context_lag=0,event_log=False,
           smoke=smoke,candidate_universe='global',history_length=8,
           external_selector='paper_adapters:create_policy' if method=='external' else None)
    j.update(extra)
    return j


def episode_worker(job,sha,path):
    r=run_episode(job);r.update(source_hash=sha,job_id=digest(job));atomic_json(path,r)
    return r['elapsed_seconds']


def execute(jobs,folder,workers,sha,summary_only=False):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    unique={digest(j):j for j in jobs};todo=[];failed=[]
    for ident,j in unique.items():
        p=folder/(ident+'.json')
        if p.exists():
            try:
                r=json.loads(p.read_text(encoding='utf-8'))
                if r['source_hash']==sha and r['job']==j:continue
            except (ValueError,KeyError):pass
            if summary_only:raise RuntimeError('Stale result: '+str(p))
            p.rename(p.with_suffix('.invalid-'+str(time.time_ns())+'.json'))
        todo.append((j,p))
    print(f'{folder.name}: cached={len(unique)-len(todo)} pending={len(todo)}',flush=True)
    if not summary_only:
        pool=ProcessPoolExecutor(workers,mp_context=multiprocessing.get_context('spawn'))
        futures={};start=time.perf_counter()
        try:
            futures={pool.submit(episode_worker,j,sha,str(p)):(j,p) for j,p in todo}
            for k,f in enumerate(as_completed(futures),1):
                j,p=futures[f]
                try:
                    seconds=f.result();status=f'ok {seconds:.1f}s'
                    if p.with_suffix('.error.json').exists():p.with_suffix('.error.json').unlink()
                except Exception:
                    e=dict(job=j,error=traceback.format_exc());failed.append(e);atomic_json(p.with_suffix('.error.json'),e);status='FAILED'
                elapsed=time.perf_counter()-start
                print(f'[{k}/{len(todo)}] {j["method_label"]} N{j["n"]} Q{j["tasks"]} {j["family"]} C{j["capacity"]}: {status}; ETA~{elapsed*(len(todo)-k)/k/60:.1f} min',flush=True)
        except KeyboardInterrupt:
            for f in futures:f.cancel()
            raise
        finally:pool.shutdown(wait=True,cancel_futures=True)
    rows=[]
    for ident,j in unique.items():
        p=folder/(ident+'.json')
        if p.exists():
            r=json.loads(p.read_text(encoding='utf-8'))
            if r.get('source_hash')!=sha or r.get('job')!=j:raise RuntimeError('Stale result')
            rows.append(r)
        elif not any(x['job']==j for x in failed):failed.append(dict(job=j,error='missing'))
    return rows,failed


def train_graph(spec,config,sha,path):
    """Actual closed-loop SPSA learning of observable sparse-graph edge logits.

    Fixed controller means graph parameters are the learned decision variables.
    This replaces original CommFormer's actor/graph bi-level PPO, not a claim
    of reproducing that complete architecture or its optimizer.
    """
    path=Path(path);iterations=2 if spec['smoke'] else config['graph_iterations']
    init=np.array([-.1,.05,.05,.05,.05,.8,.8,.05])
    initial_weights=init+np.random.default_rng(spec['training_seed']).normal(0,.03,8)
    if path.exists():
        ck=json.loads(path.read_text(encoding='utf-8'))
        if ck['source_hash']!=sha or ck['spec']!=spec:raise RuntimeError('Graph checkpoint changed')
        weights=np.asarray(ck['weights']);history=ck['history'];begin=len(history)
    else:
        weights=initial_weights.copy()
        history=[];begin=0
    for step in range(begin,iterations):
        rng=np.random.default_rng(np.random.SeedSequence([spec['training_seed'],step,911]))
        direction=rng.choice([-1.,1.],size=8)
        sigma=.35/(step+1)**.101;rate=1.0/(step+1)**.602
        family=config['families'][(step//2)%len(config['families'])]
        cap=config['capacities'][step%len(config['capacities'])]
        seed=3_000_000_000+spec['training_index']*100_000+spec['cost_index']*1_000+step
        scores=[];rewards=[];costs=[]
        for sign in [1.,-1.]:
            trial=np.clip(weights+sign*sigma*direction,-4.,4.)
            j=base_job(config,12 if spec['smoke'] else 64,2,'overlap',family,cap,seed,
                       'commformer_idea_training',spec['smoke'],baseline_name='commformer_idea',
                       graph_weights=trial.tolist(),graph_trained=False)
            if not spec['smoke']:j.update(steps=config['graph_training_steps'],eval_start=20)
            r=run_episode(j);reward=r['evaluation']['reward']['mean']
            normalized=r['network']['payload_bit_hop']/(352.*j['n']*j['steps'])
            scores.append(reward-spec['service_weight']*normalized);rewards.append(reward);costs.append(normalized)
        gradient=(scores[0]-scores[1])/(2*sigma)*direction
        weights=np.clip(weights+rate*np.clip(gradient,-1.,1.),-4.,4.)
        history.append(dict(step=step,seed=seed,family=family,capacity=cap,utility=scores,reward=rewards,
                            normalized_service=costs,gradient_norm=float(np.linalg.norm(gradient))))
        atomic_json(path,dict(source_hash=sha,spec=spec,weights=weights.tolist(),history=history,
                              iterations=iterations,complete=step+1==iterations,
                              initial_weights=initial_weights.tolist(),
                              parameter_change=float(np.linalg.norm(weights-initial_weights))))
    return str(path)


def train_models(c,smoke,out,workers,sha,summary_only):
    specs=[]
    seeds=c['graph_training_seeds'][:1] if smoke else c['graph_training_seeds']
    for cost_index,weight in enumerate(c['graph_service_weights']):
        for index,seed in enumerate(seeds):
            specs.append(dict(service_weight=weight,cost_index=cost_index,training_seed=seed,
                              training_index=index,smoke=smoke))
    folder=out/'graph_training';folder.mkdir(exist_ok=True)
    if not summary_only:
        pool=ProcessPoolExecutor(workers,mp_context=multiprocessing.get_context('spawn'));futures={}
        try:
            futures={pool.submit(train_graph,s,c,sha,str(folder/(digest(s)+'.json'))):s for s in specs}
            for k,f in enumerate(as_completed(futures),1):
                f.result();print(f'Graph training {k}/{len(specs)} complete',flush=True)
        except KeyboardInterrupt:
            for f in futures:f.cancel()
            raise
        finally:pool.shutdown(wait=True,cancel_futures=True)
    models=[]
    for weight in c['graph_service_weights']:
        checkpoints=[json.loads((folder/(digest(s)+'.json')).read_text(encoding='utf-8')) for s in specs if s['service_weight']==weight]
        if not all(x['complete'] and x['source_hash']==sha for x in checkpoints):raise RuntimeError('Incomplete graph training')
        # Prespecified ensemble, not a choice of the luckiest training seed.
        models.append(dict(service_weight=weight,graph_weights=np.mean([x['weights'] for x in checkpoints],axis=0).tolist(),
                           training_models=len(checkpoints),graph_trained=True,
                           nonzero_updates=sum(any(h['gradient_norm']>0 for h in x['history']) for x in checkpoints)))
    atomic_json(out/'TRAINED_GRAPH_MODELS.json',models)
    return models


def catalogue(c,models):
    entries=[]
    for k,m in enumerate(models):
        entries.append(dict(family='commformer_idea',label='commformer_idea__'+str(k),
                            parameters={key:m[key] for key in ['graph_weights','graph_trained','service_weight']}))
    for k,t in enumerate(c['evidence_thresholds']):
        entries.append(dict(family='t2mac_idea',label='t2mac_idea__'+str(k),parameters=dict(evidence_threshold=t)))
    for k,m in enumerate(c['group_multipliers']):
        entries.append(dict(family='hygma_group',label='hygma_group__'+str(k),parameters=dict(group_multiplier=m)))
    for name in ['expocomm_static','expocomm_one_peer']:
        entries.append(dict(family=name,label=name,parameters={}))
    return entries


def instantiate(c,n,q,mode,family,cap,seed,entry,smoke):
    params=dict(entry['parameters'])
    if entry['family']=='hygma_group':
        multiplier=params.pop('group_multiplier')
        params['max_clusters']=min(max(3,multiplier*q),n-1,16)
    return base_job(c,n,q,mode,family,cap,seed,entry['label'],smoke,
                    baseline_name=entry['family'],**params)


def development_jobs(c,entries,smoke):
    result=[];n=12 if smoke else 64
    seeds=c['development_seeds'][:1] if smoke else c['development_seeds']
    for family in c['families']:
        for cap in c['capacities']:
            for seed in seeds:
                result.append(base_job(c,n,2,'overlap',family,cap,seed,'dense_task',smoke))
                for e in entries:result.append(instantiate(c,n,2,'overlap',family,cap,seed,e,smoke))
    return result


def choose(rows,entries,c):
    stats=comparison_rows(display_results(rows),c['development_margin'])
    selected={};audit=[]
    for e in entries:
        rr=[r for r in stats if r['method']==e['label']]
        eligible=len(rr)==len(c['families'])*len(c['capacities']) and all(r['gap_upper'] is not None and r['gap_upper']<=c['development_margin'] for r in rr)
        saving=float(np.mean([r['saving_mean'] for r in rr])) if rr else -1.
        worst=max([r['gap_upper'] if r['gap_upper'] is not None else r['gap_mean'] for r in rr],default=1.)
        audit.append(dict(entry=e,eligible=eligible,mean_saving=saving,worst_upper=worst))
    for family in dict.fromkeys(e['family'] for e in entries):
        options=[r for r in audit if r['entry']['family']==family]
        feasible=[r for r in options if r['eligible']]
        winner=max(feasible,key=lambda r:r['mean_saving']) if feasible else min(options,key=lambda r:r['worst_upper'])
        selected[family]=dict(entry=winner['entry'],development_pass=winner['eligible'],
                              rule='all four upper paired95 gaps<=0.008, then best mean service saving; otherwise flagged smallest worst upper gap')
    return selected,audit,stats


def evaluation_jobs(c,selected,smoke):
    result=[];seeds=c['evaluation_seeds'][:1] if smoke else c['evaluation_seeds']
    cases=[[12,1,'static'],[24,4,'overlap'],[48,8,'overlap'],[24,4,'switch']] if smoke else c['cases']
    for n,q,mode in cases:
        for family in c['families']:
            for cap in c['capacities']:
                for seed in seeds:
                    for label in ['dense_task','ours_parent','ours_theta010_secondary']:
                        result.append(base_job(c,n,q,mode,family,cap,seed,label,smoke))
                    for name,selection in selected.items():
                        entry=dict(selection['entry'],label=name)
                        result.append(instantiate(c,n,q,mode,family,cap,seed,entry,smoke))
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workers',type=int,default=4);ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--resume',action='store_true');ap.add_argument('--output',type=Path)
    ap.add_argument('--plan',action='store_true');ap.add_argument('--summarize-only',action='store_true')
    args=ap.parse_args()
    if args.workers<1:ap.error('workers must be positive')
    c=json.loads((ROOT/'recent_paper_config.json').read_text(encoding='utf-8'))
    if set(c['development_seeds'])&set(c['evaluation_seeds']):raise ValueError('Seed split overlap')
    nmodels=len(c['graph_service_weights'])*(1 if args.smoke else len(c['graph_training_seeds']))
    variants=len(c['graph_service_weights'])+len(c['evidence_thresholds'])+len(c['group_multipliers'])+2
    counts=dict(graph_models=nmodels,training_episodes=2*(2 if args.smoke else c['graph_iterations'])*nmodels,
                development_episodes=4*(1 if args.smoke else len(c['development_seeds']))*(variants+1),
                evaluation_episodes=4*4*(1 if args.smoke else len(c['evaluation_seeds']))*8)
    if args.plan:print(json.dumps(counts,indent=2));return 0
    out=(args.output or ROOT/'runs'/('SMOKE' if args.smoke else 'RECENT_PAPER_BASELINES')).resolve();out.mkdir(parents=True,exist_ok=True)
    sha=source_hash()
    manifest=dict(version='recent-paper-core-ideas-v2',source_hash=sha,config=c,smoke=args.smoke,workload=counts,
                  runtime=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__),
                  primary_ours_fraction=0.,secondary_ours_fraction=.1,
                  primary_baselines=['CommFormer-inspired learned sparse graph','T2MAC-inspired evidence engagement',
                                     'HYGMA-inspired dynamic grouping','ExpoComm-inspired exponential topology'],
                  full_original_MARL_reproduction=False)
    mf=out/'manifest.json'
    if mf.exists() and json.loads(mf.read_text(encoding='utf-8'))!=manifest:raise SystemExit('Source/config/runtime differs; use NEW output folder')
    atomic_json(mf,manifest)
    if not args.summarize_only:
        with (out/'tests.log').open('w',encoding='utf-8') as log:
            tests=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT/'tests')))
        if not tests.wasSuccessful():raise SystemExit('Preflight tests failed')
    models=train_models(c,args.smoke,out,args.workers,sha,args.summarize_only)
    entries=catalogue(c,models);devjobs=development_jobs(c,entries,args.smoke)
    dev,fail=execute(devjobs,out/'development',args.workers,sha,args.summarize_only)
    if fail:raise SystemExit('Incomplete development; rerun to resume')
    selected,audit,stats=choose(dev,entries,c)
    selection=dict(selected=selected,candidate_audit=audit,source_hash=sha,selected_before_evaluation=True)
    frozen=out/'FROZEN_PAPER_BASELINES.json'
    # Floating point comparisons tolerate reserialization exactly on this runtime.
    if frozen.exists() and json.loads(frozen.read_text(encoding='utf-8'))!=selection:raise SystemExit('Frozen profile changed')
    atomic_json(frozen,selection);write_csv(out/'development_comparison.csv',stats)
    evaljobs=evaluation_jobs(c,selected,args.smoke)
    atomic_json(out/'evaluation_plan.json',evaljobs)
    rows,fail=execute(evaljobs,out/'evaluation',args.workers,sha,args.summarize_only)
    status=summarize(out,rows,evaljobs,fail,dict(tolerance=c['tolerance']),args.smoke)
    status.update(recent_paper_core_idea_baselines=True,
                  baseline_uses_TaskGraph_receiver_score=False,
                  learned_graph_training_completed=True,
                  baseline_development_pass={k:v['development_pass'] for k,v in selected.items()},
                  complete_original_MARL_reproduction=False)
    atomic_json(out/'acceptance.json',status)
    (out/'ACCEPTANCE_CN.md').write_text('\n'.join([
        '# 近期论文核心思想对比验收','',f'独立评价完成：{len(rows)}/{len(evaljobs)}；失败：{len(fail)}。',
        '主对比：可学习稀疏图（CommFormer思想）、证据选择（T2MAC思想）、动态分组（HYGMA思想）、指数拓扑（ExpoComm思想）。',
        '学习图经过真实闭环SPSA训练；三个训练初始化按预声明方式集成，之后仅用开发集选择资源权重。',
        '证据选择直接计算动作意见的不确定性下降；模型化证据编码替代原论文训练编码器。',
        '替代方法不使用TaskGraph的接收评分、遗漏额度或双向搜索；各自独立决定接收者。',
        '各方法仅在开发集校准；未满足开发门槛的情况如实标记，不在评价集继续调参。',
        'TaskGraph主配置保持父批次theta=0；theta=0.1为预先声明的次要配置，不能根据本批结果切换主结论。',
        '所有方法共享估计器、固定控制器、物理网络与交接保护；实际载荷比特跳与完整控制回报配对统计。',
        '这是核心思想适配，不是四篇论文完整MARL系统复现；保留/改变的部分见源码包PAPER_TO_CODE.md。',
        '旧TaskGraph和自写规则未作为主强基线。条件区间为边际95%区间，学习结果条件于冻结模型集成。',
        '本批不绘图；上下文及路由/分组信令的获取成本仍未包含。',
        '',json.dumps(status,ensure_ascii=False,indent=2)]),encoding='utf-8')
    archive=out.parent/(out.name+'_RETURN.zip');tmp=archive.with_suffix('.zip.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(out))
    tmp.replace(archive)
    print(json.dumps(status,ensure_ascii=False,indent=2));print('RETURN:',archive)
    return 0 if status['execution_complete'] else 2


if __name__=='__main__':
    multiprocessing.freeze_support();raise SystemExit(main())

