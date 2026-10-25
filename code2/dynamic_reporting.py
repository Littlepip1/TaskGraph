"""Paired seed-level analysis for dynamic-demand experiments."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import csv
import json
import numpy as np
from tgexp.reporting import interval, atomic_json, write_csv

KEYS = ['condition_id', 'n', 'tasks', 'demand_profile', 'capacity', 'period']
QUALITY = ['team_loss', 'worst_task_loss', 'worst_event_loss']


def stream_csv(path, fieldnames, rows):
    with Path(path).open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def info(job):
    return {k: job[k] for k in KEYS}


def fields(prefix, values):
    if not values:
        return {prefix+'_'+k: None for k in ['mean', 'lower', 'upper']}
    stat = interval(values)
    return {prefix+'_'+k: stat[k] for k in ['mean', 'lower', 'upper']}


def cost(record):
    return record['dynamic_demand']['communication']['evaluation_payload_bit_hop']


def paired_losses(record, ref):
    a, b = record['dynamic_demand'], ref['dynamic_demand']
    if a['schedule_hash'] != b['schedule_hash'] or a['phase_hash'] != b['phase_hash'] or a['warmup_state_hash'] != b['warmup_state_hash']:
        raise AssertionError('Paired schedule/phase/warmup state differs')
    event_loss = [y['primary_reward']-x['primary_reward'] for x, y in zip(a['events'], b['events'])]
    return dict(team_loss=ref['evaluation']['reward']['mean']-record['evaluation']['reward']['mean'],
                worst_task_loss=float(np.max(np.asarray(ref['evaluation']['per_task_mean'])-record['evaluation']['per_task_mean'])),
                worst_event_loss=float(max(event_loss)), mean_event_loss=float(np.mean(event_loss)))


def curves(records, tolerance):
    refs = {(r['job']['condition_id'], r['job']['seed']): r for r in records if r['job']['method_label'] == 'dense_task_reference'}
    groups = defaultdict(list)
    for r in records:
        ref = refs[r['job']['condition_id'], r['job']['seed']]
        groups[r['job']['condition_id'], r['job']['method_label']].append((r, ref))
    rows = []
    for (condition, label), pairs in sorted(groups.items()):
        row = dict(**info(pairs[0][0]['job']), method=label, seeds=len(pairs))
        losses = [paired_losses(r, ref) for r, ref in pairs]
        for metric in QUALITY+['mean_event_loss']:
            row.update(fields(metric, [x[metric] for x in losses]))
        metrics = dict(cost=[cost(r) for r, ref in pairs],
                       saving_vs_dense=[1-cost(r)/cost(ref) for r, ref in pairs if cost(ref) > 0],
                       reward=[r['evaluation']['reward']['mean'] for r, ref in pairs],
                       no_fresh_update_fraction=[r['dynamic_demand']['fresh_update']['no_update_fraction'] for r, ref in pairs],
                       mean_capped_wait_ticks=[r['dynamic_demand']['fresh_update']['mean_capped_wait_ticks'] for r, ref in pairs],
                       selector_ms=[r['dynamic_demand']['online_selector_mean_ms'] for r, ref in pairs])
        for name, values in metrics.items():
            row.update(fields(name, values))
        for metric in QUALITY:
            row[metric+'_pass'] = row[metric+'_upper'] is not None and row[metric+'_upper'] <= tolerance
        row['quality_pass'] = all(row[k+'_pass'] for k in QUALITY)
        rows.append(row)
    return rows


def family_labels(config):
    endpoint = 'dense_task_reference'
    return dict(full=[f'full_{x:g}' for x in config['full_fractions']],
                random=[f'random_{x:g}' for x in config['keep_fractions']]+[endpoint],
                nearest=[f'nearest_{x:g}' for x in config['keep_fractions']]+[endpoint],
                round_robin=[f'round_robin_{x:g}' for x in config['keep_fractions']]+[endpoint],
                periodic=[f'periodic_{x}' for x in config['period_multipliers']]+[endpoint])


def calibrate(rows, config, sha):
    selected = []
    for condition in config['conditions']:
        available = {r['method']: r for r in rows if r['condition_id'] == condition['condition_id']}
        for family, labels in family_labels(config).items():
            eligible = [available[label] for label in labels if available[label]['quality_pass']]
            best = min(eligible, key=lambda r: (r['cost_mean'], r['method'])) if eligible else None
            selected.append(dict(condition_id=condition['condition_id'], family=family,
                                 selected_label=best['method'] if best else None, calibration_pass=best is not None,
                                 non_dense_selected=best is not None and best['method'] != 'dense_task_reference',
                                 development_cost=best['cost_mean'] if best else None,
                                 **{k+'_upper': best[k+'_upper'] if best else None for k in QUALITY}))
    return dict(source_hash=sha, rule=config['calibration'], selected=selected)


def comparison(records, tolerance, choices, use_original_anchor=False):
    index = {(r['job']['condition_id'], r['job']['seed'], r['job']['method_label']): r for r in records}
    curve = {(r['condition_id'], r['method']): r for r in curves(records, tolerance)}
    configs = sorted({r['job']['condition_id'] for r in records})
    selected = {(r['condition_id'], r['family']): r for r in choices['selected']}
    rows = []
    for condition in configs:
        full_choice = selected[condition, 'full']
        full_label = 'full_0.1' if use_original_anchor else full_choice['selected_label']
        if use_original_anchor:
            baselines = [(label, label, None) for cid, label in sorted(curve) if cid == condition and not label.startswith('full_')]
        else:
            baselines = [(family, selected[condition, family]['selected_label'], selected[condition, family])
                         for family in ['random', 'nearest', 'round_robin', 'periodic']]
        for family, label, baseline_choice in baselines:
            baseline_curve = curve.get((condition, label))
            base_job = next(r['job'] for r in records if r['job']['condition_id'] == condition)
            row = dict(**info(base_job), baseline_family=family, baseline_label=label, full_label=full_label,
                       full_development_qualified=full_choice['calibration_pass'],
                       baseline_development_qualified=baseline_choice['calibration_pass'] if baseline_choice else None,
                       full_test_quality_pass=curve.get((condition, full_label), {}).get('quality_pass', False),
                       baseline_test_quality_pass=baseline_curve['quality_pass'] if baseline_curve else False)
            pairs = [(r, index[condition, r['job']['seed'], label], index[condition, r['job']['seed'], 'dense_task_reference'])
                     for r in records if r['job']['condition_id'] == condition and r['job']['method_label'] == full_label] if full_label and label else []
            row['seeds'] = len(pairs)
            row.update(fields('additional_saving', [1-cost(a)/cost(b) for a, b, ref in pairs if cost(b) > 0]))
            row.update(fields('baseline_minus_full_cost', [cost(b)-cost(a) for a, b, ref in pairs]))
            row.update(fields('full_minus_baseline_reward', [a['evaluation']['reward']['mean']-b['evaluation']['reward']['mean'] for a, b, ref in pairs]))
            row.update(fields('baseline_minus_full_wait_ticks', [b['dynamic_demand']['fresh_update']['mean_capped_wait_ticks']-a['dynamic_demand']['fresh_update']['mean_capped_wait_ticks'] for a, b, ref in pairs]))
            for prefix, at in [('full', 0), ('baseline', 1)]:
                losses = [paired_losses(p[at], p[2]) for p in pairs]
                for metric in QUALITY:
                    row.update(fields(prefix+'_'+metric, [x[metric] for x in losses]))
            row['zero_baseline_cost_seeds'] = sum(cost(b) == 0 for a, b, ref in pairs)
            row['common_quality_saving_advantage'] = bool(pairs) and row['full_test_quality_pass'] and row['baseline_test_quality_pass'] and row['additional_saving_lower'] is not None and row['additional_saving_lower'] > 0
            row['calibrated_common_quality_advantage'] = row['common_quality_saving_advantage'] and row['full_development_qualified'] and (row['baseline_development_qualified'] is not False)
            rows.append(row)
    return rows


def mechanism_effect(records):
    """No-new-send rotating vs normal, matched same virtual cohorts and seeds."""
    normal = {(r['job']['seed'], r['job']['capacity'], r['job']['period']): r for r in records
              if r['job']['method_label'] == 'no_new_communication' and r['job']['demand_profile'] == 'normal'}
    rows = []
    for r in records:
        j = r['job']
        if j['method_label'] != 'no_new_communication' or j['demand_profile'] != 'rotating':
            continue
        ref = normal.get((j['seed'], j['capacity'], j['period']))
        if ref is None:
            continue
        for a, b in zip(r['dynamic_demand']['events'], ref['dynamic_demand']['events']):
            if a['cohort'] != b['cohort']:
                raise AssertionError('Mechanism cohorts differ')
            rows.append(dict(**info(j), seed=j['seed'], event=a['event'],
                             added_tracking_error_m=a['cohort_tracking_error_m']-b['cohort_tracking_error_m'],
                             added_estimate_error_m=a['cohort_estimate_error_m']-b['cohort_estimate_error_m'],
                             added_covariance_trace=a['cohort_weighted_cov_trace']-b['cohort_weighted_cov_trace'],
                             added_reward_loss=b['matched_cohort_reward']-a['matched_cohort_reward']))
    return rows


def export_results(out, records, config, choices, mode, sha):
    out = Path(out)
    rows = curves(records, config['tolerance'])
    direct = comparison(records, config['tolerance'], choices)
    anchor = comparison(records, config['tolerance'], choices, True)
    write_csv(out/'evaluation_curves.csv', rows)
    write_csv(out/'heldout_selected_comparison.csv', direct)
    write_csv(out/'paired_vs_original_full_0p1.csv', anchor)
    write_csv(out/'mechanism_effect_no_new_communication.csv', mechanism_effect(records))
    episodes, events, stages = [], [], []
    warmups = defaultdict(set)
    for r in records:
        j, d = r['job'], r['dynamic_demand']
        base = dict(**info(j), seed=j['seed'], method=j['method_label'])
        warmups[j['condition_id'], j['seed']].add(d['warmup_state_hash'])
        episodes.append(dict(**base, reward=r['evaluation']['reward']['mean'],
                             evaluation_payload_bit_hop=cost(r), warmup_payload_bit_hop=d['communication']['warmup_payload_bit_hop'],
                             per_agent_tick=cost(r)/d['communication']['evaluation_agent_ticks'],
                             selector_ms=d['online_selector_mean_ms'],
                             no_fresh_update_fraction=d['fresh_update']['no_update_fraction'],
                             mean_capped_wait_ticks=d['fresh_update']['mean_capped_wait_ticks'],
                             suppressed_observations=d['suppressed_valid_local_observations'],
                             actual_blocked_fraction=d['actual_blocked_fraction'],
                             diagnostic_seconds=d['diagnostic_seconds'], elapsed_seconds=r['elapsed_seconds']))
        for e in d['events']:
            events.append(dict(**base, **{k: v for k, v in e.items() if k != 'cohort'}, cohort=','.join(map(str, e['cohort']))))
        stages.append(dict(**base, **d['stage_counts']))
    for filename, rr in [('episodes.csv', episodes), ('event_metrics.csv', events), ('stage_counts.csv', stages)]:
        write_csv(out/filename, rr)
    # Stream long tables to avoid duplicating all per-tick dictionaries in RAM.
    base_keys = KEYS+['seed', 'method']
    for name, field, extra in [('time_series.csv', 'time_series', []),
                               ('fresh_updates.csv', 'fresh_update_individual', ['cohort_basis']),
                               ('value_audits.csv', 'value_audits', [])]:
        columns = sorted({k for r in records for item in r['dynamic_demand'][field] for k in item})
        def iterator(field=field, extra=extra):
            for r in records:
                base = dict(**info(r['job']), seed=r['job']['seed'], method=r['job']['method_label'])
                for item in r['dynamic_demand'][field]:
                    row = dict(**base, **item)
                    if extra:
                        row['cohort_basis'] = r['dynamic_demand']['update_cohort_basis']
                    yield row
        stream_csv(out/name, base_keys+columns+extra, iterator())
    status = dict(execution_complete=True, mode=mode, is_formal=mode == 'full', source_hash=sha,
                  conditions=len(config['conditions']), configurations=len(config['variant_labels']),
                  development_seeds=len(config['development_seeds']), evaluation_seeds=len(config['evaluation_seeds']),
                  completed_evaluation_episodes=len(records),
                  warmup_mismatch_pairs=sum(len(x) != 1 for x in warmups.values()),
                  accounting_failures=sum(r['recipient_accounting_difference'] != 0 for r in records),
                  quota_failures=sum(r['theory_audit']['quota_failures'] for r in records),
                  selected_full_qualified_development=sum(x['family'] == 'full' and x['calibration_pass'] for x in choices['selected']),
                  selected_full_passed_evaluation=sum(r['method'] in {x['selected_label'] for x in choices['selected'] if x['family'] == 'full' and x['condition_id'] == r['condition_id']} and r['quality_pass'] for r in rows),
                  original_full_0p1_passed_evaluation=sum(r['method'] == 'full_0.1' and r['quality_pass'] for r in rows),
                  selected_comparisons=len(direct), calibrated_common_quality_advantages=sum(r['calibrated_common_quality_advantage'] for r in direct),
                  blackout_labels_given_to_policies=False, receiver_context_acquisition_cost_accounted=False,
                  statistical_unit=config['statistical_unit'])
    if status['warmup_mismatch_pairs'] or status['accounting_failures'] or status['quota_failures']:
        raise AssertionError('Stage invariant failure: '+json.dumps(status))
    atomic_json(out/'acceptance.json', status)
    report(out, rows, direct, status)
    return status


def report(out, rows, direct, status):
    def pct(v):
        return f'{100*v:.2f}%' if v is not None else '未定义'
    lines = ['# 动态局部观测中断实验：统一验收与比较', '',
             f'模式：{status["mode"]}；评价回合：{status["completed_evaluation_episodes"]}；条件：{status["conditions"]}；每条件评价种子：{status["evaluation_seeds"]}。', '']
    if not status['is_formal']:
        lines += ['这是程序检查或预运行，不能代替正式full阶段结论。', '']
    lines += [f'完整方法在开发集找到合格配置：{status["selected_full_qualified_development"]}/{status["conditions"]}；对应冻结配置在评价集通过：{status["selected_full_passed_evaluation"]}/{status["conditions"]}。',
              f'原完整方法0.10配置在评价集通过：{status["original_full_0p1_passed_evaluation"]}/{status["conditions"]}。',
              f'预热状态不一致配对：{status["warmup_mismatch_pairs"]}；计费核对失败：{status["accounting_failures"]}。', '',
              '| 条件 | 简单家族 | 冻结完整配置 | 冻结基线配置 | 额外节省 | 完整/基线控制通过 | 同质量节省优势 |',
              '|---|---|---|---|---:|---|---|']
    for r in direct:
        lines.append(f'| {r["condition_id"]} | {r["baseline_family"]} | {r["full_label"]} | {r["baseline_label"]} | {pct(r["additional_saving_mean"])} | {r["full_test_quality_pass"]}/{r["baseline_test_quality_pass"]} | {r["calibrated_common_quality_advantage"]} |')
    lines += ['', '控制通过要求团队平均、最差任务和最大受阻事件损失的配对95%区间上限分别不超过0.01。区间为逐条件逐指标区间，不是同时保证。',
              '完整方法开发集无合格配置时不挑选测试集最佳配置；比较标记未获得合格完整配置，原0.10配置仍在独立表中完整保留。',
              '额外节省以基线的评价阶段实际服务成本为分母；双方控制不通过时，不解释为同等控制质量优势。',
              'spread的主事件指标是实际缺测成员—时刻；其他条件是预设成员及恢复窗口。matched_cohort_reward另行保存，供一致口径的跨条件图使用。',
              'fresh_updates对normal和spread为虚拟匹配成员诊断；fixed和rotating才是持续受阻成员诊断。无更新者保留，并按窗口长度截断等待时间。',
              '预热统一密集通信，队列和状态不重置。no_new_communication不再生成新发送，但会消耗预热遗留包的服务；不能自动把其成本写成零。',
              '非受阻成员的更新也可能有用，nonblocked_selected_copies不等同于浪费。value_audits为固定采样的离线代理诊断，不进入决策。',
              '成本为有效载荷服务，接收者状态获取与拓扑发现信令未计入。', '',
              '优先查看 heldout_selected_comparison.csv、evaluation_curves.csv、mechanism_effect_no_new_communication.csv。',
              'stage_counts.csv区分观测机会、源端触发和接收者选择；fresh_updates.csv及time_series.csv展示更新与控制响应。',
              '全部预设配置和不利条件均保留。执行完成与方法性能是分别记录的事实。', '']
    (Path(out)/'COMPARISON_CN.md').write_text('\n'.join(lines), encoding='utf-8')
