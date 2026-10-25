"""Seed-paired statistics. Steps and messages are never treated as replicates."""
from __future__ import annotations
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def interval(values):
    values = np.asarray(values, float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Statistics require finite nonempty samples')
    mean = float(values.mean())
    if len(values) < 2:
        return dict(n=len(values), mean=mean, lower=None, upper=None)
    half = float(student_t.ppf(.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return dict(n=len(values), mean=mean, lower=mean - half, upper=mean + half)


def environment_key(job):
    return tuple(job[k] for k in ['n', 'tasks', 'scenario', 'family', 'capacity', 'seed',
                                  'steps', 'eval_start', 'deadline', 'period', 'prep'])


def comparison_rows(results, tolerance=.01):
    refs = {environment_key(r['job']): r for r in results if r['job']['method'] == 'dense_task'}
    grouped = defaultdict(list)
    for r in results:
        j = r['job']
        if j['method'] == 'dense_task':
            continue
        key = tuple(j[k] for k in ['n', 'tasks', 'scenario', 'family', 'capacity', 'method', 'fraction', 'absolute', 'context_lag'])
        ref = refs.get(environment_key(j))
        if ref is None:
            continue
        baseline_cost = ref['network']['payload_bit_hop']
        grouped[key].append((r, ref, baseline_cost))
    out = []
    for key, pairs in sorted(grouped.items()):
        n, tasks, scenario, family, capacity, method, fraction, absolute, context_lag = key
        metrics = dict(
            gap=[b['evaluation']['reward']['mean'] - r['evaluation']['reward']['mean'] for r, b, _ in pairs],
            worst_task_gap=[max(np.asarray(b['evaluation']['per_task_mean']) - r['evaluation']['per_task_mean']) for r, b, _ in pairs],
            saving=[1 - r['network']['payload_bit_hop'] / c for r, b, c in pairs if c > 0],
            cost=[r['network']['payload_bit_hop'] for r, _, _ in pairs],
            per_agent_cost=[r['cost_per_agent_tick'] for r, _, _ in pairs],
            candidate_visits=[r['candidate_visits'] for r, _, _ in pairs],
            selector_ms=[1000 * (r['selection_seconds']['mean'] or 0.) for r, _, _ in pairs],
            selector_p95_ms=[1000 * (r['selection_seconds']['p95'] or 0.) for r, _, _ in pairs],
            switch_error=[r['evaluation']['switch_window_error_m']['mean'] for r, _, _ in pairs],
            selected=[r['selected_count']['mean'] or 0. for r, _, _ in pairs],
        )
        row = dict(n_agents=n, tasks=tasks, scenario=scenario, family=family, capacity=capacity,
                   method=method, fraction=fraction, absolute=absolute, context_lag=context_lag,
                   seed_count=len(pairs), zero_cost_reference_seeds=sum(c == 0 for _, _, c in pairs))
        for name, values in metrics.items():
            if not values:
                row.update({name + '_mean': None, name + '_lower': None, name + '_upper': None})
                continue
            stat = interval(values)
            row.update({name + '_' + k: stat[k] for k in ['mean', 'lower', 'upper']})
        row['margin_pass'] = row['gap_upper'] is not None and row['gap_upper'] <= tolerance
        row['worst_task_margin_pass'] = row['worst_task_gap_upper'] is not None and row['worst_task_gap_upper'] <= tolerance
        out.append(row)
    return out


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export_results(out, results, selected_profiles, expected_ids, failures, smoke, tolerance=.01):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = comparison_rows(results, tolerance)
    write_csv(out / 'paired_summary.csv', rows)
    episode_rows = []
    theory = []
    for r in results:
        j = r['job']
        episode_rows.append(dict(**j, reward=r['evaluation']['reward']['mean'],
                                 payload_bit_hop=r['network']['payload_bit_hop'],
                                 per_agent_tick=r['cost_per_agent_tick'], candidates=r['candidate_visits'],
                                 maintained_filters=r['maintained_task_filters'],
                                 selected_mean=r['selected_count']['mean'],
                                 disconnected_fraction=r['disconnected_tick_fraction'],
                                 elapsed_seconds=r['elapsed_seconds']))
        theory.append(dict(**{k: j[k] for k in ['method', 'n', 'tasks', 'family', 'capacity', 'seed']},
                           actual_payload=r['network']['payload_bit_hop'], max_group=r['max_group'],
                           max_snapshot_hops=r['max_snapshot_hops'], **r['theory_audit']))
    write_csv(out / 'episodes.csv', episode_rows)
    write_csv(out / 'theory_audit.csv', theory)
    # Scaling ratios use common seeds, scenario and method at consecutive sizes.
    index = {(j['n'], j['tasks'], j['scenario'], j['family'], j['capacity'], j['seed'], j['method'], j['fraction'], j['context_lag']): r
             for r in results for j in [r['job']]}
    scaling = []
    for row in rows:
        if row['method'] != 'full' or row['context_lag'] or row['scenario'] != 'overlap':
            continue
        n0 = row['n_agents']
        if n0 * 2 not in {r['job']['n'] for r in results}:
            continue
        ratios, cost_ratios, visit_ratios = [], [], []
        for r in results:
            j = r['job']
            if (j['n'], j['tasks'], j['scenario'], j['family'], j['capacity'], j['method'], j['fraction'], j['context_lag']) != (
                    n0, row['tasks'], 'overlap', row['family'], row['capacity'], 'full', row['fraction'], 0):
                continue
            nxt = index.get((2 * n0, 2 * j['tasks'], 'overlap', j['family'], j['capacity'], j['seed'], 'full', j['fraction'], 0))
            if nxt and r['network']['payload_bit_hop'] > 0:
                cost_ratios.append(nxt['network']['payload_bit_hop'] / r['network']['payload_bit_hop'])
                ratios.append(nxt['cost_per_agent_tick'] / r['cost_per_agent_tick'])
                if r['candidate_visits'] > 0:
                    visit_ratios.append(nxt['candidate_visits'] / r['candidate_visits'])
        if ratios:
            a, b = interval(ratios), interval(cost_ratios)
            scaling.append(dict(n_from=n0, n_to=2 * n0, family=row['family'], capacity=row['capacity'],
                                seeds=len(ratios), per_agent_ratio=a['mean'], lower=a['lower'], upper=a['upper'],
                                total_ratio=b['mean'], candidate_ratio=interval(visit_ratios)['mean'] if visit_ratios else None,
                                diagnostic='finite-scale observation, not an asymptotic proof'))
    write_csv(out / 'scaling_summary.csv', scaling)
    primary = [r for r in rows if r['method'] == 'full' and r['context_lag'] == 0]
    complete = not failures and len(results) == len(expected_ids)
    audit = dict(execution_complete=complete, is_smoke=smoke, completed=len(results), expected=len(expected_ids),
                 failed=len(failures), development_calibrated=all(x['eligible'] for x in selected_profiles.values()),
                 full_all_mean_reward_margins_pass=bool(primary) and all(r['margin_pass'] for r in primary),
                 full_all_worst_task_margins_pass=bool(primary) and all(r['worst_task_margin_pass'] for r in primary),
                 receiver_context_overhead_accounted=False, real_AirSim=False,
                 trained_recent_baselines_reproduced=False,
                 statistical_unit='independent seed, paired by common exogenous randomness',
                 ci='paired two-sided Student-t 95%; task/condition comparisons are marginal, not simultaneous',
                 notice='execution completion, empirical performance and publication readiness are separate')
    atomic_json(out / 'acceptance.json', audit)
    lines = ['# 本阶段统一验收', '', f'- 执行完整：{complete}；完成 {len(results)}/{len(expected_ids)} 回合。',
             f'- 冒烟模式：{smoke}。冒烟结果不用于论文性能结论。',
             f'- 开发集校准全部通过：{audit["development_calibrated"]}。',
             f'- 完整方法各评价条件平均回报容差通过：{audit["full_all_mean_reward_margins_pass"]}。',
             f'- 各条件最差任务差异判据通过：{audit["full_all_worst_task_margins_pass"]}。', '',
             '## 如何解读', '',
             'paired_summary.csv：同一环境种子相对于任务内密集通信的回报差、实际服务节省、耗时等。',
             'scaling_summary.csv：保持局部任务规模时，相邻规模的人均及总服务比值。',
             'theory_audit.csv：逐事件可行性、快照上界和完整运行计数核对。观测到的K不是先验规模保证。',
             'episodes.csv 与每回合 trace：后续画图的数据；本阶段不生成图。', '',
             '## 证据范围', '',
             '这是基于ServiceOpt-v3底层组件的新多任务运动学扩展，不能把新结果混称为旧版实验复现。',
             '主方法无固定接收人数。比较方法为明确标识的自适应规则，不是近期学习方法的完整训练复现。',
             '接收者估计由仿真基准提供，获取它们的通信成本未计入；拓扑发现和路由信令未计入。',
             '控制回报基于完整闭环；并不构成碰撞安全或全局最优保证。',
             '均值改善或某个条件通过不自动推导为全部任务、全部规模通过。',
             '上下文滞后实验仅检验陈旧信息敏感性，未实现分布式副本维护。',
             '选择耗时受到进程并发和CPU影响，只用于当前机器上的实现评价。', '',
             '## 失败回合', '', json.dumps(failures, ensure_ascii=False, indent=2)]
    (out / 'ACCEPTANCE_CN.md').write_text('\n'.join(lines), encoding='utf-8')
    return audit

