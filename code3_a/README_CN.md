# 近期论文核心思想对比 v2

本版替代BaselineStructures_v1。主对比来自CommFormer (ICLR 2024)、T2MAC (AAAI 2024)、
HYGMA (ICML 2025)、ExpoComm (ICLR 2025)；旧TaskGraph、自写排序不再充当主强基线。
每项思想怎样保留、怎样适配、哪些未复现，详见PAPER_TO_CODE.md。

## 一次运行

解压后在Anaconda终端进入本文件夹：

```bat
python -m pip install -r requirements.txt
python -u run_recent_paper_baselines.py --workers 4 --resume
```

也可执行RUN_RECENT_BASELINES.bat；Linux/macOS用bash run_recent_baselines.sh --workers 4。
沿用已有环境；只需numpy、scipy、scikit-learn，不需要GPU、PyTorch或AirSim。
学习图采用实际闭环SPSA训练。自动完成训练、开发选择、冻结、独立评价、统计打包。
训练每次更新保存，回合逐一保存；中断后同一命令续跑。改变代码、配置、依赖版本时用新输出目录。

## 工作量

- 训练：3个资源权重×3个初始化×24次更新×2个匹配回合=432回合，每回合96步。
- 开发：12种配置（含共同密集参考）×四环境×3种子=144回合，每回合240步。
- 独立评价：4场景×四环境×12种子×8种配置=1536回合，每回合240步。
- 合计2112回合，一次运行，不画图；回合数以--plan为准。

```bat
python run_recent_paper_baselines.py --plan
```

四场景：N64单任务、N128四任务、N256八任务、N128四任务角色交接。
四环境：SP/MIS×常规/受限链路。所有任务共享真实逐跳队列网络，不拼接独立小实验。

## 主对比

| 标签 | 方法 |
|---|---|
| dense_task | 共同密集参考，不计作近期强基线 |
| ours_parent | 父批次冻结TaskGraph主配置theta=0 |
| ours_theta010_secondary | 预声明TaskGraph theta=0.1次要配置，全部报告，不按结果替换主配置 |
| commformer_idea | 实际训练的稀疏通信图 |
| t2mac_idea | 动作证据、意见组合与不确定性下降驱动选择 |
| hygma_group | 官方动态谱聚类，独立确定组内传播 |
| expocomm_static | 官方静态指数邻接 |
| expocomm_one_peer | 官方单邻居轮换指数邻接 |

外部思想基线不复用TaskGraph接收评分、遗漏约束或双向搜索。
源端准入、KF、连续控制器、物理网络和原始消息载荷保持一致。
公共角色切换保护窗口内统一使用任务准备范围；窗口外独立决定接收者。

## 训练与选参

学习图参数从专用训练轨迹学习；三个初始化按预声明规则集成。
独立开发种子用于选择学习图资源权重、证据门槛和分组范围。
要求四条件配对95%回报差区间上端<=0.008，再最大化平均实际服务节省。
无合格候选时标记未校准，报告控制差异最小候选，不把回退写成通过。
指数拓扑两个版本均报告，不在评价集挑选有利版本。
正式容差0.01。新独立种子不参与训练或选参；完整方法主配置仍为theta=0。
不会把不同来源、不同配置的旧批次混入新置信区间。

CommFormer思想适配使用参数化边logit与SPSA，未复现原Transformer/PPO。
T2MAC思想适配用模型化证据替代学习编码器，未复现完整证据策略。
HYGMA保留分组、ExpoComm保留拓扑，未复现消息网络和策略训练。
标签必须保留inspired/idea；核心思想比较与完整算法复现分开陈述。

## 短流程验证

```bat
python -u run_recent_paper_baselines.py --smoke --workers 2
```

短流程只验证训练到汇总能执行，单种子不能作性能证据。
只重新汇总加--summarize-only，保持原输出目录和smoke开关。

## 回传

正式完成后交回：

```text
runs/RECENT_PAPER_BASELINES_RETURN.zip
```

包含训练检查点、TRAINED_GRAPH_MODELS.json、FROZEN_PAPER_BASELINES.json、
development_comparison.csv、paired_vs_task_dense.csv、paired_vs_ours.csv、
episode_metrics.csv、逐步trace、acceptance.json与中文验收报告。

统计单位是独立环境种子；区间为逐条件95%区间，学习结果条件于冻结模型集成。
通信更少但控制未通过的方法不算优胜。上下文、路由及分组信令成本仍未纳入。
旧run_baseline_comparison.py只作为统计辅助；本版只运行run_recent_paper_baselines.py。
