# G3 P0-H Atom-only overfit32

> 日期：2026-08-02
>
> 状态：已完成，fixed-final `NO-GO`
>
> 性质：预测 Atom 集合容量门；不是正式模型，不创建 Git tag

## 背景与唯一问题

P0-G 已证明：在 GT Atom anchor 和稳定 Atom slot identity 已知时，枚举全部无序
AtomPair 足以在固定 overfit32 上把 GT-count 和 deploy-count 的 skeleton、typed graph
全部做到 `32/32`，pair identity churn 为 `0%`。因此 P0-H 不再修改 Bond relation
协议，只回答一个更靠前的问题：

> 在相同视觉特征、相同初始化、相同 32 张图和相同 2,400 次更新下，去掉旧 Bond
> objective 对 Atom head 的梯度后，现有 `SpatialAtomSlotHead`、
> `spatial_atom_slot_loss` 与 count-conditioned selector 能否独立记住完整 Atom 集合？

原始 P0 final 的已知基线是：

| 指标 | final |
|---|---:|
| GT-count Atom 6px micro | `1324/1424 = 92.98%` |
| GT-count Atom 6px whole-set exact | `17/32` |
| deploy Atom count exact | `28/32` |
| deploy Atom count MAE | `2.3125` |

六个评测点的 Atom set exact 为 `0, 2, 5, 14, 15, 17/32`，deploy Atom count exact
为 `8, 9, 11, 17, 21, 28/32`，说明原 P0 在 step 2,400 尚未明显平台。但直接续训会同时
改变训练预算，无法区分“旧 joint Bond loss 干扰”与“只需要更多更新”。P0-H 因此必须
从头运行相同的 2,400 steps，不能 resume 或延长原 P0。

## 冻结数据与视觉输入

P0-H 完全复用 G3 P0 的冻结输入：

- manifest：`Data/sdf/chemtraverse_v0_1_capacity/overfit_32.jsonl`；
- manifest SHA256：
  `8fbb55b26ed170a64d9f258f702432d5ec626049976f6d9194bcf6751033b8a0`；
- 32 张图、8 个分子、每个分子 4 种布局，严格按 manifest 行顺序循环，不 shuffle；
- vision checkpoint：`outputs/checkpoints/chemvector-v4-clean-50k/last.pt`；
- vision checkpoint SHA256：
  `65a443d26bbffd41359604ac8147fa62a66ee18ca877693ebb3dea0b0a637095`；
- 只加载并缓存冻结的 `forward_stroke_resolution` 特征，vision 始终为 `eval()` 且
  `requires_grad=False`；
- 不做图片增强，不更换布局、字体、分辨率或样本顺序。

不得读取 fresh dev、sealed test、confirm、USPTO 或 MR-TADF。P0-H 是 train-only
overfit 容量实验，不能外推泛化能力。

## 单变量模型与损失

P0-H 只实例化或只执行现有 `SpatialAtomSlotHead`。以下行为必须与原 P0 保持一致：

1. 每个空间 cell 保留 3 个 Atom slot；
2. presence、cell 内 anchor offset 和 Atom count head 的定义不变；
3. `select_count_conditioned_atom_slots` 仍按 presence 排名精确选择 count 个对象，不增加
   NMS、阈值、距离裁剪或后处理；
4. `_build_atom_slot_targets` 原有的 cell 内 L1 Hungarian 保持不变；它只定义同一 cell
   内多个 GT Atom 的监督 slot，不进入 forward 或部署选择；
5. 训练 loss 原样固定为：

```text
L_atom = focal_heatmap_loss
       + 8.0 * smooth_l1_anchor_loss
       + 0.1 * atom_count_cross_entropy
```

训练过程不得构造 Dense Bond、learned Bond query、AtomPair、Bond count 或 ChemScene；
不得计算任何 Bond、pointer、type、pair presence 或图级 loss。optimizer 只能包含 Atom
分支中可由上述 loss 到达的参数。没有 Atom-only 监督的 `slot_features` 投影不得借其他
目标暗中训练。

## 初始化公平性

P0-H 必须使用 `seed=20260713` 从头初始化，禁止加载 P0、P0-G 或其他 Atom checkpoint。
仅写“相同 seed”还不够：新建轻量模型可能改变模块构造顺序和随机数消耗。因此训练前必须
完成以下审计：

1. 在 CPU 上设置相同 seed，并用原 P0 的模型构造顺序重建 reference Atom head；
2. 对 `SpatialAtomSlotHead.state_dict()` 按 key 排序，将每个 tensor 的 key、dtype、shape
   和 contiguous CPU raw bytes 纳入确定性哈希；不得用含容器元数据的 `torch.save` 文件
   SHA 代替；
3. P0-H step 0 Atom head 的 canonical tensor SHA256 必须与 reference 完全一致；
4. 初始 SHA、参数名、参数量、trainable 参数名和 optimizer 参数名必须写入 `audit.json`；
5. 任一项不一致时不得启动正式 2,400-step 运行。

这项检查用于保证 P0 与 P0-H 的差异只有训练 objective，而不是初始化变化。

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

32 张图按原 P0 的 cyclic manifest 顺序进入每一步。P0-H 只按 fixed final step 2,400
判定，不生成或选择 `best.pt`。若 final 仍在上升但未过门，结论仍是“2,400-step
Atom-only NO-GO”；后续如需验证训练预算，必须另行预注册并从头运行，不能续当前 run。

预留接口与输出目录：

```text
CLI:        --atom-only-capacity
output-dir: outputs/smoke/chemvector-g3-p0h-atom-only-overfit32
```

该 flag 必须与 P0-C、P0-D、P0-E、P0-F、P0-G 及所有 Bond protocol flag 互斥。当前文档
只预注册协议；在训练入口、审计和聚焦测试完成前不得启动 GPU。

## Final GO/NO-GO Gate

P0-H 的能力 Gate 只有两个主指标：

| 模式 | 指标 | GO 门槛 |
|---|---|---:|
| deploy-count | Atom count exact | `>=30/32` |
| GT-count | selected Atom 6px whole-set exact | `>=30/32` |

同时必须通过以下行政与协议检查：

- deploy-count 与 GT-count 均实际评测 32 张固定训练图；
- GT Atom 总数为 1,424，样本 key 和顺序与 manifest 完全一致；
- step 0 Atom canonical SHA 与 reference 相同；
- vision、manifest、源码、配置、optimizer 参数和 fixed-final 状态均已审计；
- 训练期间没有调用 Bond/AtomPair forward，没有 Bond 参数进入 optimizer，也没有读取外部集。

任一主 Gate 或协议检查失败即为 `NO-GO`。micro recall、median error 或后述 oracle
诊断不能替代 whole-set exact。

## 四层 Atom 诊断

所有位置指标统一使用 768px 图像上的 L2 距离和 6px 半径。所有集合覆盖必须采用一对一
最大二部匹配；禁止让多个 GT Atom 重复占用同一个 candidate。

### 1. Count

- deploy Atom count exact、MAE 和逐图误差；
- GT-count selection 与 deploy-count selection 分开报告；
- GT-count 只移除 count 错误，不允许改变 presence 排名或 anchor。

若 GT-count Atom set 过门而 deploy-count 不过门，失败只归于 count head。

### 2. Candidate recall

依次报告：

1. 所有 dense `(slot,y,x)` predicted anchor 对 GT 的 6px maximum matching；
2. presence 排名前 128 个 dense candidate 对 GT 的 6px matching；
3. GT count `K` 下最终 top-K selected set 的 6px matching。

all-dense 失败表示 grid、slot 或 offset 表示本身不能覆盖 GT；all-dense 通过但 top-128
失败表示 presence candidate ranking 丢失目标；top-128 通过但 top-K 失败表示 selector
排序不能在正确 count 内选全对象。

### 3. Selector identity

对每个 checkpoint 调用现有 `_build_atom_slot_targets`，获得该 checkpoint 的 supervision
positive dense identity `(slot,y,x)`。比较 GT-count top-K selected identity 与 positive
identity，报告 micro precision/recall、whole-set exact、selected-only 和 positive-only。

另外按稳定 `sample_key + GT atom index` 记录 step 400/800/1200/1600/2000/2400 的
target-to-dense-slot identity，并报告相邻 checkpoint 及 first-to-final churn。该审计只用于
定位 cell 内 Hungarian 或 selector 的移动身份，不改变训练 assignment。

### 4. Localization

- oracle 选择 supervision-positive slots，只使用其 predicted anchor，报告 6px micro、
  whole-set exact、median 和 p90 pixel error；
- 对 selected 且属于 positive identity 的 slots，报告 conditional 6px accuracy；
- 不得用 GT anchor 覆盖 predicted anchor 后再计作模型正确。

若 selected identity 已正确但 oracle-positive localization 未过门，失败归于 anchor offset
定位；若 localization 通过而 selected identity 未通过，失败归于 presence/selector。

## 与 P0-G 的组合诊断

P0-H 训练阶段不得调用、加载或训练 P0-G pair head，因此训练时不存在
predicted-Atom 到 GT Bond 的标签映射。这样才能保持 Atom-only 单变量。

P0-H final 后允许做一次不反传的 frozen P0-G composition shadow，但它不是 P0-H Gate：

1. P0-G checkpoint 和 pair head 参数保持冻结并记录 SHA；
2. 使用 P0-H 的 predicted Atom count、selected anchor 和 AtomPair universe；
3. predicted Atom 与 GT Atom 的 Hungarian/L2 `<=6px` 映射只用于离线评分；
4. 任何 unmatched predicted Atom 都使相关 GT Bond 不可表示，不得删除、修复或用 GT Atom
   替换；
5. 报告 target-bond representable coverage、整体 skeleton/typed graph，以及仅在 Atom-set
   exact 图上的 conditional pair 指标；
6. 原始 P0-G 在 GT Atom oracle 下必须单独 replay 为 `32/32`，证明权重和评估器未损坏。

Atom-only loss不会训练 P0-G 所消费的 384 维 Atom slot feature 接口。因此即使 Atom Gate
通过，frozen composition 仍可能因 feature distribution/interface 失败。此时结论必须写成
“P0-H Atom detector GO，frozen composition NO-GO”，并将适配或联合训练留给单独的 P0-I；
不得反过来篡改 P0-H Gate。

若未来 P0-I 需要训练 pair head，唯一允许的 target 映射是：forward 完成后对 selected
predicted Atom 与 GT Atom 做 detached 一对一匹配，仅接受 6px 内对应；只有两端都匹配的
predicted pair 才能参与 presence/type loss，含 unmatched Atom 的 pair 必须 ignore，不能
武断标成 negative。缺失 GT Bond 只记录为 unrepresentable coverage。该规则不属于 P0-H
训练协议。

## 禁止事项

- 禁止 resume P0、P0-G 或 P0-H 中间 checkpoint；
- 禁止在同一 run 追加 step、换 seed、搜索 loss 权重或挑选中间 checkpoint；
- 禁止 warmstart P0-G，或把 P0-G pair loss 当作 Atom feature 辅助监督；
- 禁止修改 `SpatialAtomSlotHead`、slot 数、loss、cell 内 Hungarian 或 top-K selector；
- 禁止加入 NMS、阈值校准、距离/近邻裁剪、valence、图 repair 或人工规则；
- 禁止向 Atom-only forward/loss 传入 GT Bond count、endpoint、Atom refs、type、display、
  direction 或 stereo；
- 禁止读取 MR-TADF、USPTO、fresh dev、sealed test 或 confirm 数据；
- 禁止把 smoke/capacity 结果创建为正式模型版本或 Git tag。

## 预期判读

- **GO**：说明现有 Atom head/loss/selector 在相同 2,400-step 预算内具有 overfit 容量，
  原 P0 的 joint Bond objective 是主要干扰之一；下一步处理 predicted Atom feature 与冻结
  P0-G relation 的接口。
- **NO-GO，candidate recall 失败**：先处理 dense proposal/offset 表示，不训练 Bond。
- **NO-GO，candidate 通过但 selector identity 失败**：下一步只研究 Atom 集合选择或稳定
  slot identity。
- **NO-GO，identity 通过但 localization 失败**：下一步只研究 anchor regression。
- **Atom GO、deploy count NO-GO**：保持 geometry/selector，单独处理 count head。

以上结论都只针对固定 32 张训练图的容量，不代表公开集、MR-TADF 或真实图片泛化。

## 实际运行结果

唯一一次固定运行已于 2026-08-02 完成。训练严格使用本文预注册的 seed、32 张图、
BF16、batch 4 和 2,400 steps；未 resume、未追加 step、未搜索 loss，也未读取外部集。
训练耗时 `214.55s`，共见到 9,600 个重复训练样本。

初始化审计通过：原 P0 reference Atom head 与 P0-H candidate 的 canonical tensor SHA256
均为：

```text
8bfd1e6d5d2763b99893085b1e3a3aefe76c73c5eb94ccd0fcfb241b45d82806
```

Atom-only head 共 `2,983,562` 个参数、39 个 parameter tensor；optimizer 中也只有这 39
项。冻结 vision 加 Atom head 共 `16,641,642` 个参数。

### Fixed-final Gate

| 指标 | 门槛 | step 2400 | 结论 |
|---|---:|---:|---|
| deploy Atom count exact | `>=30/32` | `31/32` | PASS |
| deploy Atom count MAE | 仅诊断 | `0.53125` | - |
| GT-count Atom 6px whole-set exact | `>=30/32` | `5/32` | FAIL |
| GT-count Atom 6px micro | 仅诊断 | `1369/1424 = 96.14%` | - |

因此 P0-H fixed-final 正式 `NO-GO`。step 2000 曾达到 `11/32` Atom set exact，但协议禁止
选择中间 checkpoint，不能用它替代 fixed final。

### Final 分层诊断

| 层级 | final |
|---|---:|
| all-dense 6px matching | `1424/1424`，`32/32` 图全覆盖 |
| presence top-128 6px matching | `1424/1424`，`32/32` 图全覆盖 |
| GT-count top-K 6px matching | `1369/1424`，`5/32` whole-set exact |
| top-K 与 supervision-positive identity 交集 | `1199/1424 = 84.20%` |
| identity whole-set exact | `0/32` |
| supervision-positive localization <=6px | `1424/1424` |
| localization median / p90 | `0.52px / 1.07px` |
| step 400 到 2400 supervision identity churn | `1000/1424 = 70.22%` |
| step 400 到 2400 GT-count top-K set churn | `70.01%` |

all-dense、top-128 和 positive localization 已全部通过，排除 grid/slot 容量、候选池覆盖和
anchor offset 精度是当前首要瓶颈。失败集中在最后从 top-128 压到真实 count 的 presence
排序；同时 cell 内按当前 predicted anchor 重做 Hungarian，使同一个 GT Atom 的监督 slot
身份持续移动。删除 Bond objective 后，count 从原 P0 的 `28/32` 提高到 `31/32`，但
whole-set exact 从 `17/32` 降到 `5/32`，所以“旧 Bond loss 是 Atom set 的主要干扰”这一
假设不成立；现有 dense focal objective 与动态 slot identity/整组选择协议才是下一战场。

本轮不运行 frozen P0-G composition shadow。P0-H Atom Gate 已失败，组合结果不会改变判定，
且会混入未训练的 384 维 Atom feature interface 问题。

### Artifact

目录：`outputs/smoke/chemvector-g3-p0h-atom-only-overfit32`

```text
last.pt SHA256:
0131bed9d1d1789adb577597f2f3f60f5715644d73465baf7428aaa6620cba95

metrics.json SHA256:
4dc9dac7983ae5450fac4c2518d27ad73a0f0b463a12ee5258d97a140a8f6a60

audit.json SHA256:
ca5912a66dcf0b14526a9dd8ca1b49b75aeeb24752f304bf98f72427daffee6e

atom_only_final_audit.json SHA256:
5c2e1942d68c7e433339dae78f1399d5b4b815ad59ee872885f962ef832d0815
```

该实验是容量 smoke，不创建正式模型 tag。
