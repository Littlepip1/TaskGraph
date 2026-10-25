# 论文—核心思想—代码对应表

主对比取自四篇2024—2025年外部论文。标签统一使用inspired/idea，不将核心思想适配写成原论文完整复现。

| 来源 | 保留的思想 | 代码 | 任务适配 |
|---|---|---|---|
| CommFormer，ICLR 2024 | 把通信图作为可学习决策，学习稀疏通信关系 | paper_adapters.graph_features；train_graph | 8维可观测边特征参数化logit替代自由邻接矩阵/Transformer；闭环SPSA替代原PPO与双层优化；固定共同控制器 |
| T2MAC，AAAI 2024 | 通信前后决策不确定性驱动伙伴选择，证据层组合意见 | opinion、combine_opinions、t2mac_idea分支 | 高斯状态经sigma点映射动作意见；模型化证据编码替代学习编码器；直接计算不确定性下降替代辅助选择网络 |
| HYGMA，ICML 2025 | 根据状态历史谱聚类，动态形成群组 | 官方DynamicSpectralClustering原类 | 输入和时钟适配当前任务；独立决定组内传播，不复现HGCN/QMIX |
| ExpoComm，ICLR 2025 | 指数邻接、单邻居轮换的稀疏通信 | 官方get_exp_neighbors原函数 | 相同物理网络上发送原始观测，不复现学习消息、记忆及辅助预测 |

## 学习图确实训练

边特征为偏置、相对距离、信念差异、接收者不确定性、预计协方差改善、当前角色、已知近期角色、路径跳数。
不包含TaskGraph动作变化分数，不读取目标真值。由学习logit正负决定是否保留边，不固定接收人数。
训练目标是实际闭环平均回报减去资源权重乘以实际载荷服务/(b*N*T)。
每种资源权重、每个初始化进行24次SPSA更新；每次两条相同外生条件的闭环轨迹。
保存两边效用、回报、服务、梯度范数和相对于初始化的参数变化；不训练随机网络后直接称作强基线。
三个初始化先按规则平均参数形成冻结集成，再在开发集选资源权重，不在评价集挑选幸运模型。
这是学习稀疏图思想的控制适配，不能将其性能写成原CommFormer网络性能。

## 证据选择独立于我们的评分

动作分为8个方向与停止，仅用于描述意见，不改变实际连续控制器。
五个正权sigma点获得类别支持度，控制相关位置方差确定证据总量，由此形成Dirichlet意见(b,u)。
采用Dempster-Shafer组合，组合不确定性u=u0*u1/(1-C)，C为不同动作类别支持的冲突。
当相对不确定性下降超过开发门槛时接收。它不使用TaskGraph动作平方分数、遗漏额度或共享路径搜索，
也不是将NIS或距离排序改名为T2MAC。意见只用于选择试算，到达后仍由共同KF融合。
所以保留证据驱动选择思想，不宣称复现原文学习证据消息与最终证据策略。
意见的模型化构造不是已校准真实动作概率，也不产生安全保证。

## 分组与指数拓扑也独立选接收者

HYGMA直接调用官方谱聚类和原0.6稳定阈值。每16步更新，用最近8步历史；
组数上界max(3,m*Q)，m=1或2在开发集选择，最大16组。单成员只在一组。
组内可达成员直接接收，不叠加TaskGraph评分。无消息与公共交接保护期间仍维护分组。

ExpoComm执行原拓扑函数，保持收发方向；静态指数邻居或单邻居轮换直接确定接收者。
自身轮次不产生远程事件；所有物理传输、排队、共享路径及过期服务计费。
不叠加TaskGraph裁剪，不假设免费全网信息扩散。没有消息记忆网络，结论限于拓扑思想。

## 论文来源

- CommFormer: https://proceedings.iclr.cc/paper_files/paper/2024/hash/37c6d0bc4d2917dcbea693b18504bd87-Abstract-Conference.html
- T2MAC: https://ojs.aaai.org/index.php/AAAI/article/view/29438 ; https://arxiv.org/html/2401.10973v1
- HYGMA: https://proceedings.mlr.press/v267/liu25af.html
- ExpoComm: https://proceedings.iclr.cc/paper_files/paper/2025/hash/3514dbacaebf0f38b25adfe59ed81a8a-Abstract-Conference.html

HYGMA/ExpoComm原模块、提交与许可证在OFFICIAL_SOURCES.json及official_modules。
CommFormer和T2MAC思想基线依据论文机制独立实现，没有拷贝未授权代码或使用原作者训练权重。

推荐论文名称为Learned-Graph (CommFormer-inspired)、Evidence-Selective (T2MAC-inspired)、
Dynamic-Grouping (HYGMA-inspired)、Exponential-Topology (ExpoComm-inspired)。
可比较同任务中的通信思想；不能直接写成全面优于四篇论文的完整端到端系统。
