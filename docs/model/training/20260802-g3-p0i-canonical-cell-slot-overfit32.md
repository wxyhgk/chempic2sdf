# G3 P0-I Canonical Cell Slot overfit32

> 日期：2026-08-02
>
> 状态：已完成，fixed-final `GO`
>
> 性质：Atom cell 内稳定 slot identity 因果实验；不是正式模型，不创建 Git tag

## 背景与唯一假设

P0-H 在完全去掉 Bond/AtomPair objective 后，从相同 seed 和相同 2,400-step 预算训练
Atom-only head。fixed-final 结果为：

| 指标 | P0-H final |
|---|---:|
| deploy Atom count exact | `31/32` |
| all-dense 6px coverage | `1424/1424`，`32/32` |
| top-128 6px coverage | `1424/1424`，`32/32` |
| supervision-positive localization | `1424/1424`，median/p90 `0.52/1.07px` |
| GT-count top-K 6px coverage | `1369/1424 = 96.14%` |
| GT-count Atom whole-set exact | `5/32` |
| fixed-final selector/positive identity overlap | `1199/1424 = 84.20%` |
| supervision identity first-to-final churn | `1000/1424 = 70.22%` |
| supervision identity step 2000→2400 churn | `558/1424 = 39.19%` |

这些结果排除了 candidate capacity、top-128 截断和 anchor localization：每张图的完整
6px 解都已经位于 top-128 内，被监督为正的 anchor 也全部定位正确。失败只发生在从 128
个候选中按 presence 取正确 count `K` 的选择阶段。

当前 `_build_atom_slot_targets` 在每一步都读取 predicted anchor，并在同一 cell 内用 L1
Hungarian 决定哪个 slot 是 GT positive。三个 slot 的预测 offset 随训练变化，导致同一个
GT Atom 的 positive slot 持续切换；某个 sibling slot 可以在这一阶段为正、下一阶段为负，
但它们在几何上仍可能同时落入同一 Atom 的 6px 邻域。presence head 因而面对移动的正负
身份，并可能在 top-K 中选入重复 sibling，挤掉其他 Atom。

P0-I 只验证一个假设：

> 将 cell 内 GT-to-slot assignment 改成不依赖模型预测的固定局部坐标顺序后，稳定的
> Atom slot identity 能否让现有 presence/top-K selector 在相同 2,400-step 预算内过拟合？

## 冻结数据与 occupancy 事实

P0-I 完全复用 P0-H 的数据和视觉输入：

- manifest：`Data/sdf/chemtraverse_v0_1_capacity/overfit_32.jsonl`；
- manifest SHA256：
  `8fbb55b26ed170a64d9f258f702432d5ec626049976f6d9194bcf6751033b8a0`；
- 32 张图、8 个分子、每个分子 4 种布局；
- GT Atom 总数：`1424`；
- vision checkpoint：`outputs/checkpoints/chemvector-v4-clean-50k/last.pt`；
- vision checkpoint SHA256：
  `65a443d26bbffd41359604ac8147fa62a66ee18ca877693ebb3dea0b0a637095`；
- frozen feature shape：`32 x 96 x 192 x 192`；
- Atom slots/cell：`3`。

当前 overfit32 的固定 occupancy 审计为：

```text
occupancy == 1: 1424 Atom / 1424 occupied cells
occupancy >= 2: 0 cells
identical normalized anchor collisions: 0
```

因此本数据上每个 occupied cell 只有一个 GT Atom，canonical 规则会确定性地把它分配到
slot 0；slot 1 和 slot 2 始终是该 cell 的 negative sibling。该事实必须写入训练
`audit.json` 并在启动前重新核验。若实际 adapter 得到任何不同结果，正式运行必须停止，
不得静默更换数据或规则。

不得读取 fresh dev、sealed test、confirm、USPTO 或 MR-TADF。P0-I 仍只是 train-only
overfit32 容量实验。

## 唯一改动：Canonical Cell Slot Targets

P0-I 只替换 `_build_atom_slot_targets` 中 cell 内 target-to-slot assignment，其他模型、loss、
selector 和优化设置必须与 P0-H 完全相同。

对每张图的每个 GT Atom：

1. 使用与 P0-H 相同的规则计算 feature cell `(cell_y, cell_x)`；
2. 将同一 cell 内的 GT Atom 按 normalized anchor `(x, y)` lexicographic 升序排列；
3. 排序后的第 `n` 个 GT Atom 固定映射到 slot `n`，即 slot `0..occupancy-1`；
4. 若同一 cell 内出现完全相同的 normalized `(x,y)`，立即报错并拒绝样本；
5. 不使用 source atom index、target id、SDF 顺序或其他字段打破坐标 tie。

assignment 过程严禁读取以下任何模型输出：

- predicted anchor 或 anchor offset；
- presence logit、rank 或 top-K 结果；
- slot feature、count 或后续图预测。

cell 内不得调用 Hungarian 或其他 prediction-conditioned matching。canonical identity 只由
GT 的局部可观察坐标定义，并且只存在于 loss target 构造中；GT anchor 不进入 forward。

本规则不是 P0-E 的全图 canonical Bond prefix。P0-E 要求 learned Bond query 学习“本图
第 N 根键”这一没有跨图视觉含义的序号；P0-I 只固定同一 192×192 feature cell 内最多三个
对象的局部顺序。当前 occupancy 全为 1 时，identity 更简单地等价于“occupied cell 的
slot 0”。

## 保持不变的模型与损失

P0-I 继续只执行 `SpatialAtomSlotHead`：

- input projection 和 dilation `1/2/4/8` context 不变；
- 每 cell 3 个 presence logits 与 anchor offsets 不变；
- Atom count head 不变；
- `select_count_conditioned_atom_slots` 仍按 presence 精确选择 count 个对象；
- 不增加 NMS、阈值、去重、距离裁剪或后处理；
- loss 权重原样固定为：

```text
L_atom = focal_heatmap_loss
       + 8.0 * smooth_l1_anchor_loss
       + 0.1 * atom_count_cross_entropy
```

唯一差异是 focal heatmap 与 anchor regression 的 positive slot identity 由 canonical cell
规则确定，不再由 predicted-anchor Hungarian 确定。

训练不得构造或调用 Dense Bond、learned Bond query、AtomPair、P0-G pair head、Bond
count、ChemScene 或 SDF。optimizer 只能包含 P0-H 相同的 Atom head 参数。

## 初始化公平性

P0-I 必须从 `seed=20260713` 重新初始化，禁止 resume 或 warmstart P0、P0-H、P0-G。

训练前按 P0-H 相同的 canonical tensor hash 协议审计
`SpatialAtomSlotHead.state_dict()`：按 key 排序，将 key、dtype、shape 与 contiguous CPU
raw bytes 纳入 SHA256。P0-I step 0 Atom SHA 必须同时等于：

1. 原 P0 reference Atom step 0 SHA；
2. P0-H step 0 Atom SHA。

初始化 SHA、参数名、参数量、trainable 参数和 optimizer 参数必须写入 `audit.json`。普通
`torch.save` 容器 SHA 不能替代 tensor-level canonical SHA。任一不一致时禁止启动训练。

## 固定训练协议

| 项目 | 固定值 |
|---|---:|
| seed | `20260713` |
| batch | `4` |
| updates | `2400` |
| optimizer | AdamW |
| LR | `3e-4` |
| weight decay | `0.01` |
| max gradient norm | `1.0` |
| precision | BF16 |
| image | `768x768` |
| Atom slots/cell | `3` |
| maximum Atom | `128` |
| evaluation/save interval | `400` |

manifest 行顺序、cyclic batch、vision cache、autocast、gradient clipping 和 checkpoint
interval 全部与 P0-H 一致。P0-I 只按 fixed final step 2,400 判定，不生成或选择
`best.pt`。

预注册 CLI 与独立输出目录固定为：

```bash
source .venv-train/bin/activate
python -m chemvector.train_g3_hierarchical_graph \
  --sample-manifest Data/sdf/chemtraverse_v0_1_capacity/overfit_32.jsonl \
  --vision-checkpoint outputs/checkpoints/chemvector-v4-clean-50k/last.pt \
  --output-dir outputs/smoke/chemvector-g3-p0i-canonical-cell-slot-overfit32 \
  --device cuda \
  --precision bf16 \
  --max-steps 2400 \
  --batch-size 4 \
  --learning-rate 3e-4 \
  --weight-decay 0.01 \
  --evaluate-every 400 \
  --save-every 400 \
  --seed 20260713 \
  --atom-only-capacity \
  --canonical-cell-slot-targets
```

`--canonical-cell-slot-targets` 必须要求 `--atom-only-capacity`，并强制使用上述唯一输出
目录。它必须与所有 Bond protocol、GT Atom oracle 和其他 target assignment flag 互斥。
当前文档只预注册协议；实现、聚焦测试和审计完成前不得启动 GPU。

## Final GO/NO-GO Gate

### 协议 Gate

以下项目必须全部严格通过：

| 指标 | 要求 |
|---|---:|
| GT Atom / occupied cell | `1424 / 1424` |
| occupancy >= 2 | `0` |
| identical coordinate collision | `0` |
| canonical target coverage | `1424/1424` |
| supervision target identity churn | `0/1424 = 0%` |
| step 0 Atom canonical SHA | 与 P0/P0-H reference 完全相同 |

identity churn 必须在 step 400/800/1200/1600/2000/2400 的所有相邻组合和
first-to-final 上均为 0。由于 assignment 不读取模型输出，任何非零 churn 都是实现错误，
不能记作能力失败后继续训练。

### 能力 Gate

| 模式 | 指标 | GO 门槛 |
|---|---|---:|
| deploy-count | Atom count exact | `>=30/32` |
| GT-count | all-dense 6px set exact | `32/32` |
| GT-count | top-128 6px set exact | `32/32` |
| oracle fixed-positive | localization 6px set exact | `32/32` |
| GT-count | top-K / fixed-positive identity micro | `>=99%` |
| GT-count | top-K / fixed-positive identity set exact | `>=30/32` |
| GT-count | selected Atom 6px whole-set exact | `>=30/32` |

所有位置指标继续使用 768px 图像上的 L2 6px 半径和一对一 maximum-cardinality matching。
不得以 micro、median、最优中间 step 或 candidate oracle 替代 final whole-set Gate。

任一能力门失败即为 `NO-GO`；不追加 step、不换 seed、不调 loss、不选择 step 1,600 等
中间 checkpoint。

## 必须保留的诊断

六个固定 checkpoint 继续报告：

- deploy count exact、MAE 与逐图 count error；
- all-dense、top-128、GT-count top-K 的 matched Atom 和 whole-set exact；
- top-K 与 fixed-positive identity 的 intersection、selected-only、positive-only、micro 与
  set exact；
- fixed-positive localization median、p90 和 6px coverage；
- top-128 与 GT-count top-K selected set/rank churn。

P0-I 的 supervision identity 已由协议固定，因此 selected set 自身仍允许变化；只有 final
能力 Gate 判定它是否学会稳定选择。selected rank churn 不得误写成 supervision churn。

在比较 P0 与 P0-H/P0-I 时，必须使用相同的 maximum-cardinality-first 6px matcher。原 P0
历史审计使用普通最小距离 Hungarian 后再阈值，不能直接把 `92.98%` 与新 matcher 的 micro
作严格增量结论；如需量化，先只读重放原 P0 final。

## 禁止事项

- 禁止读取 predicted anchor 后决定 positive slot；
- 禁止在 cell 内调用 Hungarian 或使用 loss/logit/rank 做 target assignment；
- 禁止用 source index、target id 或 SDF atom order打破完全相同坐标 tie；
- 禁止 resume P0-H、追加 step、换 seed、搜索排序规则或选择中间 checkpoint；
- 禁止修改 Atom head、slot 数、loss 权重、count head、top-K selector 或 6px 指标；
- 禁止把 slot 1/2 从模型中删除；它们必须保留并由现有 focal heatmap 学为 negative；
- 禁止加入 NMS、duplicate suppression、距离/近邻裁剪、valence 或图 repair；
- 禁止调用或训练 P0-G pair head，或传入任何 GT Bond count、endpoint、refs、type；
- 禁止读取 MR-TADF、USPTO、fresh dev、sealed test 或 confirm；
- 禁止为 smoke/capacity 实验创建正式版本或 Git tag。

## 预期判读

- **GO**：说明 P0-H 的主要断点是 prediction-conditioned cell Hungarian 造成的移动 Atom
  slot identity；下一步才允许把稳定 predicted Atom set 接回冻结 P0-G relation 接口。
- **identity churn 非零**：实现违反协议，运行无效，不属于模型 NO-GO。
- **fixed identity 过门但 top-K identity 未过门**：moving target 不是唯一原因，现有 focal
  heatmap/global top-K objective 仍不能完成集合去重；下一轮研究 equivalence/group-level
  selector loss。
- **identity top-K 过门但 geometry 未过门**：anchor regression 被新 assignment 破坏，
  只处理 localization。
- **GT-count Atom 过门但 deploy count 未过门**：保持 candidate/selector 不变，单独处理
  count head。

所有结论只适用于固定 overfit32 容量，不代表公开集或真实分子图泛化。

## 实际运行结果

唯一一次固定运行已于 2026-08-02 完成。训练使用本文预注册的双 CLI flag、相同 seed、
冻结 vision、32 张图、BF16、batch 4 和 2,400 steps；未 warmstart、未追加 step、未调参，
也未读取任何外部集。训练耗时 `200.44s`。

初始化公平审计通过。原 P0 reference、P0-H 和 P0-I Atom head 的 canonical tensor
SHA256 均为：

```text
8bfd1e6d5d2763b99893085b1e3a3aefe76c73c5eb94ccd0fcfb241b45d82806
```

P0-I 仍只有 `2,983,562` 个 Atom head 参数、39 个 parameter tensor；optimizer 只包含
这 39 项。模型、数据、loss 权重、selector 和训练预算均未变化。

### Fixed-final Gate

| 指标 | 门槛 | step 2400 | 结论 |
|---|---:|---:|---|
| canonical target coverage | `1424/1424` | `1424/1424` | PASS |
| canonical identity churn | `0%` | `0/1424 = 0%` | PASS |
| deploy Atom count exact | `>=30/32` | `32/32` | PASS |
| deploy Atom count MAE | 仅诊断 | `0` | - |
| all-dense 6px set exact | `32/32` | `32/32` | PASS |
| top-128 6px set exact | `32/32` | `32/32` | PASS |
| fixed-positive localization | `32/32` | `1424/1424`，`32/32` | PASS |
| top-K / fixed-positive identity micro | `>=99%` | `1424/1424 = 100%` | PASS |
| top-K / fixed-positive identity set exact | `>=30/32` | `32/32` | PASS |
| GT-count Atom 6px whole-set exact | `>=30/32` | `32/32` | PASS |

机器可读 `p0i_audit_gate.status` 为 `passed`，全部 13 项 protocol/capability check 为
`true`。因此 P0-I fixed-final 正式 `GO`。

### 六 checkpoint 诊断

GT-count Atom whole-set 在 step 400 已达到 `32/32`；step 800 短暂降到 `27/32`，从
step 1200 起到 final 均保持 `32/32`。deploy count 从 `19/32 -> 29/32 -> 32/32`，并从
step 1200 起保持 `32/32`。

final 还有以下证据：

- selected duplicate candidate：`0`；
- missing target cell / Atom：`0 / 0`；
- fixed-positive localization median / p90：`0.85px / 1.64px`；
- cutoff crossing tie：`0/32`；
- cutoff margin min / median / p90：`6.875 / 8.6875 / 9.521875`；
- canonical identity first-to-final：`1424/1424` retained，churn `0%`；
- GT-count top-K 从 step 1200 到 final 的 selected set 保持 `1424/1424`。

这不是靠阈值 tie 或后处理得到的结果。与 P0-H 相比，唯一变量是移除
prediction-conditioned cell Hungarian，改为固定局部 slot identity；P0-H final 的
`5/32` 因而提升到 P0-I final 的 `32/32`。本轮直接验证：动态监督身份是现有 Atom
selector 无法过拟合的主要原因。

限制同样明确：overfit32 中 1,424 个 occupied cell 全部 occupancy=1，所以本轮只充分验证
了单 occupancy cell 的 canonical slot 规则。坐标排序下 occupancy=2/3 的学习能力、跨分子
泛化、真实图片和 Atom feature 到 P0-G relation 的接口都尚未验证，不能从本轮 GO 外推。

## Artifact

目录：`outputs/smoke/chemvector-g3-p0i-canonical-cell-slot-overfit32`

```text
last.pt SHA256:
452a95c84cbb5bfb4a0e185f9955a963e5c126f77663fc8888052fdae619fe60

step-002400.pt SHA256:
93c01230c0b2d0bf8d9a157666f6bfb84906695514d1508ca7b7c62c1ff5c3d5

metrics.json SHA256:
32f40084dabce603739a484105d505af0190dbbfcd8d453f6b096f08a0c4b9a7

audit.json SHA256:
d7ca3ea16647cb01ee8cd28297a26d5ebcc32e94ae580d5f37f01ce7b3748ca3

canonical_cell_slot_audit.json SHA256:
fe40583239a27c1a16f7f6e41445fbe2a09c4feeb133d54105690fb2857af8e4
```

该实验是容量 smoke，不创建正式模型 tag。
