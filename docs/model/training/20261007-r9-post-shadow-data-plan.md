# r9 direct AtomPair：GPU shadow 后的数据与损失计划

## 结论

本轮不直接改 graph loss，也不把现有 800-step sidecar 接入 production。r9
GPU residual 已在 95 张 fresh holdout 上将 edge recall 从 `0.934625` 提高到
`0.952785`，但 step 200 已达到 `0.956416`，到 step 800 没有继续提高；这
说明当前瓶颈更像训练分布/采样，而不是优化器步数不足。

## 当前训练分布不是代表性分布

数据清单中共有 `761` 张 train 图、`192` 个 train molecule identity。r9
runner 为了固定小试训只取每 profile 前 24 张，即 96 张。这个子集的类别分布为：

| 类别 | 图数 | pair 行数 | 正 pair |
|---|---:|---:|---:|
| AROMATIC | 20 | 904 | 199 |
| DOUBLE-only | 32 | 1,067 | 247 |
| mixed_multiple_bond | 20 | 975 | 209 |
| TRIPLE-only | 20 | 364 | 116 |
| hetero_N_O_multiple | 4 | 180 | 36 |

fresh holdout 为 `95` 张，全部属于 `hetero_N_O_multiple`。完整 train 中该类
只有 `27/761` 张。因此当前 sidecar 实际只看到了 4 张 hetero 图，holdout
存在明显 category shift。继续增加随机样式而不重平衡类别，无法说明模型是否
真的学会了 N/O 多重键。

## 推荐的下一轮顺序

1. **CPU 数据重放，不改模型**：用全部 761 张 train 做 frozen evidence cache；
   保持 95 张 fresh holdout 不变，仍按 molecule identity 隔离。先只统计每类
   pair 行、正负比例和 hard-negative 数量。
2. **分层采样 CPU A/B**：固定每 profile、每 category 的上限，至少让
   `hetero_N_O_multiple` 不低于 24 张（若不足则使用全部 27 张）；其余类别
   按比例采样。保持所有正 pair，负 pair 对每个正 pair 采样 1--2 个。
3. **hard negative**：优先同图非键、几何距离不超过 80 px 的 pair；96 图测得
   此池有 676 个负 pair（≤60 px 为 112），而 frozen base probability
   `>=0.8` 的负 pair 只有 9 个。故不能仅依赖 logit-hard mining，也不能把
   所有近距离负样本一次性灌入导致负类压倒正类。
4. **CPU 200--500 steps**：仅训练现有 8,973 参数 DirectPairResidual，
   train-only threshold，报告整体、四 profile 以及五 category 的 edge
   precision/recall、dangling、type、sanitize。只有相对当前 r9 baseline
   在 hetero holdout 上严格改善且总体 gate 不退化，才申请下一次 GPU shadow。
5. **graph-level loss 暂缓**：当前 residual 只有 presence 输出，chemical logits
   冻结；直接加入 chemistry loss 不会产生梯度。若要试，必须新建独立 sidecar
   并做 CPU-only A/B，先用 `lambda=0.05/0.1` 的可微 valence/edge-count
   penalty，不能把硬约束直接用于删除候选。任何 graph loss 都必须同时报告
   recall、dangling、type 和 sanitize，避免用化学约束换来假负。

## GPU 成本估算

当前 800-step GPU shadow 用时约 `253 s`（远低于 30 分钟上限）；扩大 train
主要增加一次 frozen model evidence cache，不显著增加 sidecar optimizer
步数。GPU 仍固定最多 800 steps，每 200 steps 检查 holdout，且只保存隔离
shadow state，不修改 production checkpoint。

相关证据：

- `outputs/audits/chemvector-identity-aligned-direct-graph-pair-scorer-gpu-shadow-r9/report.json`
- `outputs/audits/chemvector-identity-aligned-direct-graph-pair-scorer-cpu-r9/report.json`
- `outputs/audits/chemvector-identity-aligned-direct-graph-proposal-ceiling-r9/report.json`

## r10 coverage CPU 结果

按上述顺序已完成一次独立 CPU 试训（不使用 GPU、不写 production）：384 张
train 图（每 profile 96 张），保留全部正 pair，并以 2:1 比例采样同图几何
hard negatives（距离不超过 80 px 优先）。300 步、8,973 参数 sidecar 的
fresh95 结果为：

| 指标 | frozen baseline | r10 coverage CPU |
|---|---:|---:|
| edge precision | .994845 | **1.000000** |
| edge recall | .934625 | **.966102** |
| dangling rate | .003851 | .003745 |
| type accuracy | .963731 | .959900 |

四个 profile 的 edge recall 最低为 `.947368`，最低 profile type accuracy 为
`.917098`，整体及 profile gate 均通过。该结果支持“扩大且分层的数据 + 受控
hard negatives”优先于立即修改 graph loss；它仍是 sidecar 诊断结果，不能直接
部署或替换 production。报告：

`outputs/audits/chemvector-identity-aligned-direct-graph-pair-scorer-cpu-r10-coverage/report.json`

### Type-only CPU sidecar

随后做了独立的 type-only CPU sidecar：仅更新 `TypeResidual`（1,576 参数），
presence scorer、pair hidden、候选边集合全部冻结；使用 r10 的 384 张 train 图
上的 3,168 条正 type rows，300 步训练。fresh95 的 oracle sanitize 从 `.757895`
（72/95）升至 `.821053`（78/95），aromatic errors 从 20 降至 14，但 profile
sanitize 为 `.875/.875/.6087/.9167`，最低 profile 仍远低于 `.90`，因此不申请
GPU，也不进入 production。主要剩余问题集中在 `r6_s32_r30_w16_serif`。

报告：

`outputs/audits/chemvector-identity-aligned-direct-graph-type-sidecar-cpu-r10/report.json`

### r11 profile coverage A/B

将 `r6_s32_r30_w16_serif` 的全部 186 张 train 图加入，其余三个 profile 保持
96 张，并对每个 profile×type bucket 最多取 64 条正 type rows（共 846 条）后
做 300 步 CPU type residual。presence 边集合保持不变（801 selected/798
mapped）。fresh95 oracle sanitize 由 `.757895` 升至 `.831579`，但 profile
结果为 `.9167/.875/.6522/.875`；serif30 仍不达 `.90`，且 valence errors
由 3 增至 5。因此本轮仍 **NO-GO GPU**；单纯扩大 serif30 或 profile×type
采样不能解决剩余问题。

报告：

`outputs/audits/chemvector-identity-aligned-direct-graph-type-sidecar-cpu-r11-profile/report.json`

此外，r11_profile 的训练目标是 `CE(type_residual(features), y)` 后将输出
直接加到 frozen chemical logits，属于绝对类别先验注入，不是严格 residual。
若继续 CPU 探针，应改为 `CE(base_chemical_logits + residual, y)`，配合
profile×type 平方根逆频率权重和 residual scale 不超过 `.5`；在此 corrected
objective 通过整体及 profile chemistry gate 之前，不启动 GPU。

corrected r12 CPU（`CE(base_chemical_logits + 0.5 * residual, target)`）随后完成：
整体 oracle sanitize `.757895 → .852632`（72→81/95），aromatic errors
`20 → 6`，但 valence errors `3 → 8`；profile sanitize 为
`.9167/.9583/.6522/.875`，serif30 仍失败，且 `valence_not_increased=false`。
所以即使修正 residual 定义也不能进 GPU；下一步若继续只能加入严格的
valence-safe trust region/abstention，并针对 serif30 的视觉证据单独做 CPU
验证。

报告：

`outputs/audits/chemvector-identity-aligned-direct-graph-type-residual-cpu-r12-corrected/report.json`

同一 r10 sidecar 的只读 oracle-element sanitize replay 为 `73/95=.768421`
（aromatic errors 20、valence errors 2），低于 r9 shadow 的 `.789474`。因此
r10 只证明 presence operating point 的改善，不能被解释为化学图合法性改善；
presence 数据扩充后下一项应独立验证 type/aromatic sidecar 或 chemistry-aware
abstention。
r10 的额外只读诊断把每个已选 mapped edge 的 chemical type 临时替换为 GT type
（仅后验，不进入 scorer 或推理）后，fresh95 的 sanitize 变为 `95/95=1.0`
（dangling 仍为 `.003745`）。因此当前 `.768421` 的 sanitize 缺口几乎完全来自
frozen chemical type logits，而不是 presence/拓扑候选。后续最小实验应是独立
的 chemical/type residual sidecar，而不是对 presence 加硬 valence penalty。
