# 480 图 selector group label-strata audit

`audit_targeted_selector_group_label_strata.py` 是只读诊断。它读取 visual-pool release 的 frozen group cache，并读取 quarantine `layers.json` 中的 atom centers 计算最近中心距离；不读取图片、文字记录、模型或 sealed 数据，也不产生训练数据。

结果：

- train confirmed-negative：28,495 个，其中 303 个（1.06%）距离某 atom center 不超过 10 px；
- holdout confirmed-negative：14,688 个，其中 199 个（1.35%）距离某 atom center 不超过 10 px；
- strict positive 的距离中位数为 train 3.41 px、holdout 3.11 px；confirmed-negative 中位数为 train 24.17 px、holdout 23.87 px；
- near-target 负例主要来自 raw proposal 的坐标近邻/重复候选，不能因为 raw matcher 未命中就直接当作“背景负例”。

因此本报告的距离分层只能用于审计，不能直接写回标签或用于训练。下一分支应把候选坐标精修/匹配质量与 existence 分开；在未完成独立标签协议前，不应继续训练相同的 existence BCE head。
