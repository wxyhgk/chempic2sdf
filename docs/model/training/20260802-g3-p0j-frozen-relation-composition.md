# G3 P0-J Frozen Relation Composition

> 日期：2026-08-02
>
> 状态：已完成，fixed composition `NO-GO`
>
> 性质：冻结组件组合与 feature interface 因果审计；不训练，不创建 Git tag

## 唯一问题

P0-I 已在固定 overfit32 上把 deploy Atom count、GT/deploy Atom 6px whole-set 和
canonical slot identity 全部做到 `32/32`。P0-G 则已在 GT Atom oracle 下把全部无序
AtomPair 的 skeleton 和 typed graph 做到 `32/32`。P0-J 只回答：

> 两个已经单独通过容量 Gate 的冻结组件，能否在不向 forward 注入 GT Atom 的条件下直接
> 组合？若不能，断点是 P0-I 的 Atom feature interface，还是 P0-G relation 本身？

本轮不训练、不改 checkpoint、不做 repair，也不扩大数据。

## 冻结输入

- manifest：`Data/sdf/chemtraverse_v0_1_capacity/overfit_32.jsonl`；
- manifest SHA256：
  `8fbb55b26ed170a64d9f258f702432d5ec626049976f6d9194bcf6751033b8a0`；
- frozen vision：`outputs/checkpoints/chemvector-v4-clean-50k/last.pt`；
- vision SHA256：
  `65a443d26bbffd41359604ac8147fa62a66ee18ca877693ebb3dea0b0a637095`；
- P0-I：`outputs/smoke/chemvector-g3-p0i-canonical-cell-slot-overfit32/last.pt`；
- P0-I SHA256：
  `452a95c84cbb5bfb4a0e185f9955a963e5c126f77663fc8888052fdae619fe60`；
- P0-G：`outputs/smoke/chemvector-g3-p0g-gt-atom-pair-bond-capacity/last.pt`；
- P0-G SHA256：
  `f4f6f6030e1dc0802a143b960a83ed3aff3163b2853e648c9a058b9150b08ee0`。

所有模块必须 `eval()`、`requires_grad=False`，整个审计使用 `torch.no_grad()`。禁止读取
MR-TADF、USPTO、fresh dev、sealed test 或 confirm。

## Forward 协议

P0-I 先仅从图像特征预测：

- deploy Atom count；
- selected Atom anchor；
- selected dense identity `(slot,y,x)`；
- Atom presence score；
- P0-I native Atom feature。

然后对 P0-I selected Atom 枚举全部无序 `i<j` AtomPair。禁止距离、近邻、键长、空白或
价态裁剪。P0-G 的 `DirectAtomPairBond` 读取图像 ribbon、Atom feature、anchor 和 score，
预测 pair presence 与 chemical type。

GT Atom anchor、Atom ref、Bond count 和 Bond type 均不得进入 P0-I 或 composition forward。
GT 只允许在 forward 完成后用于离线评分。

## 两种 feature 模式

### A. P0-I native

直接使用 P0-I selected `atom_features`。P0-I Atom-only loss没有监督
`feature_projection/slot_embedding/slot_norm`，因此该模式专门测试原生 feature interface
是否碰巧兼容 P0-G。

### B. P0-G slot replay

保持 P0-I 的 count、anchor、identity 和 score不变；只运行冻结 P0-G `atom_head`，在 P0-I
预测的同一 `(slot,y,x)` 上重取 P0-G `slot_features`。不得根据 GT anchor重新选择 slot。
该模式移除 feature distribution mismatch，用于判断 P0-G relation 在预测 Atom 几何上是否
仍然成立。

两种模式唯一差异必须是 384 维 Atom feature。图像 ribbon、P0-I anchor、P0-I score、pair
universe、P0-G pair/count 权重完全相同。

## 评分映射

P0-I predicted Atom 与 GT Atom 只在评分侧做一对一 maximum-cardinality-first Hungarian，
距离阈值固定为 768px 图上的 `6px`。映射必须 detached，且不得回写 forward。

- unmatched predicted Atom 相关 pair 不得武断标为 negative；
- 两端都映射的 predicted pair 才能与 GT Bond 比较；
- 任一 GT Bond 有未匹配端点时记为 unrepresentable；禁止 repair、drop 或 GT 替换；
- chemical type 只在真实 positive pair 上评分；
- skeleton/typed graph exact 按整图集合比较。

必须先报告 P0-I Atom representable coverage，再报告 relation；不能把 Atom 失败算成 P0-G
relation 错误。

## 三个评测分支

1. `p0g_gt_atom_oracle_replay`：原 P0-G checkpoint 原样重放，确认 GT/deploy typed graph
   仍为 `32/32`；
2. `p0i_native_feature`：P0-I predicted Atom + P0-I native feature；
3. `p0g_slot_replay_feature`：P0-I predicted Atom + 同预测 slot 上的 P0-G feature。

每个 composition 分支都分别报告：

- GT Bond count top-K：隔离 pair presence/type；
- deploy Bond count top-K：包含冻结 P0-G count head；
- pair precision/recall/F1、positive type accuracy；
- skeleton exact、typed graph exact；
- count exact/MAE；
- Atom/Bond representable coverage。

## GO/NO-GO Gate

协议 Gate：

- P0-I deploy Atom count 与 6px set exact 均为 `32/32`；
- Atom mapping 为 `1424/1424`，GT Bond representable 为 `1508/1508`；
- P0-G GT Atom oracle replay 的 GT/deploy skeleton 与 typed graph 均为 `32/32`；
- 两个 feature 分支的 pair universe identity 完全相同；
- 所有 checkpoint、vision、manifest、实现脚本 SHA 已记录。

能力 Gate 只对 `p0g_slot_replay_feature` 生效：

| 模式 | 指标 | GO 门槛 |
|---|---|---:|
| GT Bond count | skeleton exact | `>=30/32` |
| GT Bond count | typed graph exact | `>=30/32` |
| deploy Bond count | count exact | `>=30/32` |
| deploy Bond count | skeleton exact | `>=30/32` |
| deploy Bond count | typed graph exact | `>=30/32` |

`p0i_native_feature` 只做接口诊断，不进入主 Gate。

## 判读

- **slot replay GO，native GO**：组件可直接组合，下一步再做真实 predicted-Atom 联合训练；
- **slot replay GO，native NO-GO**：relation 已成立，唯一缺口是 Atom feature interface；
  下一轮只训练 feature adapter，禁止重训定位和 relation；
- **slot replay NO-GO，P0-G oracle GO**：P0-G relation依赖 GT-oracle 时的 slot/score/几何分布，
  下一轮需要 predicted-Atom conditioned relation alignment；
- **P0-G oracle replay失败**：本轮无效，先修 checkpoint/evaluator；
- **P0-I Atom representable失败**：本轮无效，不能归因 relation。

以上结论仅针对固定 overfit32 的冻结组合，不代表泛化或完整 SDF 能力。

## 输出

固定目录：

```text
outputs/smoke/chemvector-g3-p0j-frozen-relation-composition/
```

固定主文件：

```text
composition_audit.json
```

## 实际结果

唯一一次冻结组合审计已完成。P0-I、P0-G 与 vision 全程 `eval()`、无梯度、无参数更新；
GT 只用于 forward 后的 6px 评分映射。

协议 Gate 全部通过：

- P0-I Atom mapping：`1424/1424`，Atom set `32/32`；
- GT Bond representable：`1508/1508`；
- pair universe：`37380`；
- 两个 feature 模式的 pair identity SHA 完全相同；
- P0-G GT Atom oracle replay：GT/deploy typed graph 均 `32/32`。

### P0-I native feature

| 指标 | 结果 |
|---|---:|
| positive type accuracy | `1299/1508 = 86.14%` |
| GT Bond count pair TP | `1498/1508 = 99.34%` |
| GT Bond count skeleton exact | `25/32` |
| GT Bond count typed graph exact | `4/32` |
| deploy Bond count exact | `8/32` |
| deploy skeleton / typed graph | `6/32 / 2/32` |

P0-I Atom-only objective没有训练 384 维 slot feature interface，因此 native 分支的 type 与
count head 分布明显不兼容 P0-G。

### P0-G slot replay feature

| 指标 | 门槛 | 结果 |
|---|---:|---:|
| positive type accuracy | 仅诊断 | `1507/1508 = 99.93%` |
| GT Bond count pair TP | 仅诊断 | `1498/1508 = 99.34%` |
| GT Bond count skeleton exact | `>=30/32` | `26/32` |
| GT Bond count typed graph exact | `>=30/32` | `26/32` |
| deploy Bond count exact | `>=30/32` | `32/32` |
| deploy skeleton exact | `>=30/32` | `26/32` |
| deploy typed graph exact | `>=30/32` | `26/32` |

slot replay 修复了 feature/type/count interface，但 presence top-K 仍用 10 个 false pair 替换
了 10 个 GT Bond，错误分散到 6 张图。因此 `p0j_gate.status=failed`，解释为
`predicted_atom_conditioned_relation_mismatch`。

P0-J 正式 `NO-GO`。结论不是“P0-G 不会识别键型”：type 已达到 `1507/1508`；剩余首要
问题是 P0-G relation 对 P0-I predicted anchor/score 分布不完全稳定。同时 native feature
仍是第二个独立接口缺口。

下一步先做冻结残差消融：保持 P0-G slot feature，分别替换 P0-I score 与 GT-anchor oracle，
定位 10 条 presence 错误来自 score 还是几何；在消融前不启动新训练。

### 冻结残差消融

2x2 anchor/score 消融随后完成，所有分支都使用 P0-G slot replay feature：

| Anchor | Score | GT-count typed graph |
|---|---|---:|
| P0-I predicted | P0-I | `26/32` |
| P0-I predicted | P0-G same-slot | `26/32` |
| GT oracle | P0-I | `32/32` |
| GT oracle | P0-G same-slot | `32/32` |

仅替换 score 完全不改变结果；仅把 pair ribbon 的端点换成 GT anchor，即恢复 pair TP
`1508/1508`、skeleton/typed graph `32/32`。因此 primary cause 明确为
`predicted_anchor_geometry_shift`。这里的 GT-anchor 分支只用于因果诊断，不是可部署结果。

下一轮 P0-K 只允许使用 P0-I 的真实 predicted anchor 对齐 P0-G pair head；冻结 P0-I
Atom、vision、P0-G Atom feature extractor 与 count head。不得回退 GT anchor，也不得把
score、OCR、数据量或 Atom detector 同时改动。

## Artifact

目录：`outputs/smoke/chemvector-g3-p0j-frozen-relation-composition`

```text
composition_audit.json SHA256:
c6b6b187bdda3034576582b086dd2a0264fde25616c8608a31bdb1ab6766a314

audit script SHA256:
33eec272c50d270d34e693b7ef2d00e57caf20cbe4f49c87d7de98ea3d0d842a

composition_residual_ablation.json SHA256:
58a9de618e042927eeacb3121e51b875a2615952d0e925922776fab691c7646a

residual audit script SHA256:
24d0d0046ff8ec6018e931daf0f24286f17660325941a877a17b7af1af0e7b8e
```

本轮不创建 checkpoint 或 Git tag。
