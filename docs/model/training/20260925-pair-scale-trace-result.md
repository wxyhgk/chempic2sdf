# pair-scale 退化只读 trace 结果

## 范围

本次运行只读比较 `initial` 与 `pair-015443` 两个已保存模型 arm，固定 15 张完整分子开发图：9 张已知 broken、6 张已知 rescued。

- 未读取 sealed test；
- 未进行训练、反向或优化器更新；
- 未修改 production evaluation 接口；
- GPU 为 RTX 5090 / CUDA 12.8；设备为 `cuda:0`；
- 两个 arm 的生产预测与保存 validation 结果均为 **15/15 parity**。

证据文件：

- [trace report](../../../outputs/audits/chemvector-pair-scale-trace-20260925-gpu/report.json)
- [trace plan](../../../outputs/audits/chemvector-pair-scale-trace-20260925-gpu/plan.json)
- [trace tool](../../../tools/run_pair_scale_trace_diagnostic.py)

## 结果

15 张图中：

- `broken=9`、`rescued=6` 与预注册样本一致；
- pair threshold 选择集发生变化的样本为 6 张；
- aromatic promotion 统计发生变化的样本为 14 张；
- compiler 选中 bond 数发生变化的样本为 6 张；
- 新增 fallback 的样本为 4 张。

更关键的是，按相同 atom pair 对齐后：

- 9 张 broken 图共有 **41 个已选 pair 的 chemical/final-chemical 变化**；
- 6 张 rescued 图共有 **10 个已选 pair 的 chemical/final-chemical 变化**；
- presence 只在 4 张 broken 图改变了选中集合，另有 2 张 rescued 图改变了集合。

因此本轮退化的主信号不是“所有键都漏检”，而是 pair head 在已选候选上的键型与芳香概率漂移。presence 的影响仍然存在，但只解释少数图：例如 `fresh_dev_000221_rdkit` 的 P=O 邻接边被阈值过滤，`fresh_dev_001091_rdkit` 则新增了一条边并触发价态风险。

在 9 张 broken 图中可以进一步分成两类：4/9 是 presence threshold 跨越，且 compiler 的 bond count 一一同步；另外 5/9 的 selected pair 数量不变，但 chemical/aromatic/promotion 漂移。4 个新增 fallback 全部属于后一类，都是 RDKit kekulization 失败，而不是 compiler 丢弃端点。`aromatic_consistency` 只对少量非环 aromatic 边做了 demotion，不能解释主退步。

聚合看，broken 组 selected/compiled bonds 为 `290 -> 288`，accepted aromatic cycles 为 `27 -> 16`；rescued 组 selected/compiled bonds 保持 `286 -> 286`，accepted cycles 保持 `21 -> 21`。这进一步把主风险定位到 broken 组的环级 chemical/promotion 稳定性，而不是整体候选覆盖崩溃。

## 芳香与 fallback

broken 图中最明显的变化是 aromatic promotion：

- `fresh_dev_000263_rdkit`：initial 接受 2 个环、pair arm 接受 0 个环，随后进入 kekulization fallback；
- `fresh_dev_000314_rdkit`：接受的环从 3 个降到 0 个，芳香边从 16 降到 5；
- `fresh_dev_000120_coordgen`：接受的环从 2 个降到 1 个并进入 fallback；
- `fresh_dev_000746_rdkit`、`fresh_dev_000953_stress` 也保留了较少的 aromatic promotion 并进入 fallback；
- 两张大图 `fresh_dev_001091_rdkit` 与 `fresh_dev_001900_coordgen` 出现额外边/价态风险，其中前者新增 pair，后者保持边数但化学类型发生漂移。

同时，rescued 图的改善也主要来自相同机制的另一侧：pair arm 恢复了原先缺失的芳香 promotion 或修复了少量边型，例如 `fresh_dev_000883_rdkit`、`fresh_dev_001800_coordgen`、`fresh_dev_001539_coordgen` 的 fallback 消失并达到 exact。

## 结论

1. CUDA、模型加载、生产路径和报告 parity 均正常；不能再把这次回退归因于环境故障。
2. `pair-015443` 不适合作为部署模型；继续保留 initial 为默认模型。
3. 单纯增加 pair 训练量或再跑第二遍没有被这次证据支持。训练目标在课程域获得收益，但对完整分子的 aromatic/chemical 边界不稳定。
4. 下一步应优先研究“已选 pair 的键型/芳香保护”与“presence/chemical 解耦”的最小实验，而不是扩大数据量、降低阈值或继续追加训练。

## 决策

- `pair-015443`：**NO-GO，shadow only**；
- initial：继续作为默认部署基线；
- 本次 trace 不包含新训练授权；任何保护训练或联合训练都需要单独预注册预算、验收集和授权。

补充边界：trace builder 的 `scene_summary.aromatic_atoms` 默认不反写 atom aromatic 标志，因此不使用该字段衡量芳香原子数量；芳香结论来自 promotion、pair chemical 输出以及已有 validation/RDKit 结果。
