# G3 P0-K Predicted-Anchor Relation Alignment

> 日期：2026-08-02
>
> 状态：已完成，独立审计通过，正式 `GO`
>
> 性质：P0-I predicted Atom 条件下的 relation 容量对齐；不是正式模型，不创建 Git tag

## 因果依据

P0-J 冻结组合得到：

- P0-I Atom mapping `1424/1424`，GT Bond representable `1508/1508`；
- P0-G GT Atom oracle replay `32/32`；
- P0-G same-slot feature replay 的 Bond count `32/32`、type `1507/1508`；
- 但 pair presence top-K 错 10 条，typed graph 只有 `26/32`；
- 换 P0-G score仍为 `26/32`；
- 只换 GT anchor立即恢复 `32/32`。

因此 P0-K 只验证一个问题：

> 在保持 P0-I predicted anchor、P0-I score 和 P0-G same-slot feature 的真实部署输入下，
> 微调 P0-G pair head 能否吸收亚像素几何偏移并恢复整图 relation？

## 冻结与训练范围

冻结：

- vision checkpoint及全部参数；
- P0-I Atom head全部参数；
- P0-G Atom head全部参数；
- P0-G Bond count head全部参数。

唯一 trainable 模块：P0-G `DirectAtomPairBond`，共 `2,364,869` 个参数、32 个 parameter
tensor。它包括 ribbon encoder、Atom/score encoder、shared trunk、presence 和 chemical head。

初始化必须来自 P0-G winner：

```text
outputs/smoke/chemvector-g3-p0g-gt-atom-pair-bond-capacity/last.pt
SHA256 f4f6f6030e1dc0802a143b960a83ed3aff3163b2853e648c9a058b9150b08ee0
```

P0-I 固定为：

```text
outputs/smoke/chemvector-g3-p0i-canonical-cell-slot-overfit32/last.pt
SHA256 452a95c84cbb5bfb4a0e185f9955a963e5c126f77663fc8888052fdae619fe60
```

## Forward 与标签

训练 forward 严格使用：

- P0-I deploy Atom count、predicted anchor、predicted `(slot,y,x)` 与 score；
- P0-G 在同一 predicted slot 上重取的冻结 384 维 feature；
- 全部无序 AtomPair；
- predicted anchor 定义的视觉 ribbon。

禁止 GT Atom anchor、GT Bond count、GT Bond refs/type 进入 pair forward、ribbon、pair universe
或 top-K。

pair forward 完成后，P0-I predicted Atom 与 GT Atom 做 detached、maximum-cardinality-first、
6px 一对一映射。只有两端都映射的 pair 才能获得 presence/type 标签；unmatched pair必须
ignore，不能标 negative。本轮 P0-I 已要求每个 batch mapping 100%，否则立即停止训练。

loss 复用 P0-G 原有 `direct_atom_pair_loss`：全 pair weighted BCE presence，加真实 positive
pair 的 chemical cross-entropy。不得加入 valence、键长、邻近、repair 或其他辅助 loss。

## 固定训练协议

| 项目 | 固定值 |
|---|---:|
| data | overfit32，32 张 |
| seed | `20260713` |
| batch | `4` |
| updates | `400` |
| optimizer | AdamW |
| LR | `1e-4` |
| weight decay | `0.01` |
| max grad norm | `1.0` |
| precision | BF16 |
| evaluate/save | 每 `100` step |
| pair universe | 全部 `i<j`，不裁剪 |

manifest 顺序循环，不 shuffle。只按 fixed final step 400 判定；不选择中间 checkpoint，
不追加 step，不搜索 LR/loss/阈值。

实际启动命令：

```bash
.venv-train/bin/python -m \
  chemvector.training.graph.relations.train_g3_p0k_relation_alignment
```

固定输出：

```text
outputs/smoke/chemvector-g3-p0k-predicted-anchor-relation-alignment/
```

## Final Gate

协议 Gate：

- P0-I Atom mapping `1424/1424`、Atom set `32/32`；
- target Bond coverage `1508/1508`，pair universe `37380`；
- optimizer 恰好只包含 P0-G pair head 的32个参数；
- vision、P0-I、P0-G、manifest、代码 SHA 全部匹配；
- 训练期间没有 GT anchor/ribbon oracle或外部数据。

能力 Gate：

| 分支 | 指标 | 门槛 |
|---|---|---:|
| predicted Atom + GT Bond count | skeleton exact | `>=30/32` |
| predicted Atom + GT Bond count | typed graph exact | `>=30/32` |
| predicted Atom + deploy Bond count | count exact | `>=30/32` |
| predicted Atom + deploy Bond count | skeleton exact | `>=30/32` |
| predicted Atom + deploy Bond count | typed graph exact | `>=30/32` |
| GT Atom oracle replay | typed graph exact | `>=30/32` |

还必须报告 pair precision/recall/F1、positive type accuracy，以及相对 P0-J base 的 0 break
检查。若 predicted 分支提高但 GT-oracle 分支跌破30，仍为 `NO-GO`。

## 禁止事项

- 禁止训练/修改 P0-I Atom detector；
- 禁止训练 P0-G Atom feature或Bond count head；
- 禁止把 GT anchor 写入 ribbon；
- 禁止距离、近邻、键长或空白裁剪；
- 禁止读取 MR-TADF、USPTO、fresh dev、sealed test、confirm；
- 禁止用 native P0-I feature掩盖本轮 geometry 对齐目标；
- 禁止把容量 smoke checkpoint接前端或创建正式 tag。

## 判读

- **GO**：predicted-anchor relation alignment 容量成立；下一步再单独训练 P0-I native
  feature adapter，并扩大到非重复 scaffold 数据；
- **predicted GO / oracle break**：pair head过拟合 predicted 几何并遗忘 oracle，下一轮需
  双几何 replay；
- **两者都 NO-GO**：400-step/当前 pair head 无法同时容纳几何分布，需重新设计 ribbon
  的 anchor-perturbation invariance；
- **协议 coverage失败**：实验无效，不属于模型能力结论。

## 实际结果

本轮于2026-08-02按上述默认参数执行唯一一次400-step训练。固定 final step 400结果：

| 指标 | step 0 | step 100 | step 200 | step 400 |
|---|---:|---:|---:|---:|
| predicted + GT count skeleton exact | `26/32` | `31/32` | `32/32` | `32/32` |
| predicted + GT count typed graph exact | `26/32` | `31/32` | `32/32` | `32/32` |
| predicted + deploy count skeleton exact | `26/32` | `31/32` | `32/32` | `32/32` |
| predicted + deploy count typed graph exact | `26/32` | `31/32` | `32/32` | `32/32` |
| GT Atom oracle replay structural graph | `32/32` | `32/32` | `32/32` | `32/32` |

final 的 Atom mapping为 `1424/1424`，Bond coverage为 `1508/1508`，全图 pair
universe为 `37380`。presence precision/recall/F1、positive type accuracy均为 `1.0`；
相对 P0-J 的 `26/32` 提升6张，并且 final 全部32张正确，因此不存在逐样本回退。

训练耗时约64.5秒，处理1,600个重复采样图次。输出：

```text
outputs/smoke/chemvector-g3-p0k-predicted-anchor-relation-alignment/
```

关键 SHA256：

```text
step-000400.pt  3457d1b46511695ad5579e177317eb26a97dae48fbf88adf45f0d512fd7968db
last.pt         a91d5bb1052063c819b3ba71e26987c45d5e7e819281ff26fa7b3bd80f90eac5
final audit     ccba470b4157025afd1ba8dab73c5a063b69599191a26f2c9c75eab73d2989d8
trainer         71fc7de3a1785728423ae298caf378bb6261b24f845ff2b5f94890a18e83437d
```

`last.pt` 与 `step-000400.pt` 是分别原子写出的 PyTorch归档，文件字节 SHA不同；独立审计
对两者使用模型 state与协议字段核对，不以归档字节相等作为模型一致性条件。

## 独立审计

只读审计入口：

```bash
.venv-train/bin/python -m \
  chemvector.evaluation.audits.graph.audit_g3_p0k_relation_alignment
```

审计固定读取100/200/300/400四个 checkpoint，并验证：

- 全部训练参数逐项等于预注册值；
- optimizer只有 `DirectAtomPairBond` 的32个 parameter tensor、`2,364,869` 个参数；
- manifest、vision、P0-I、P0-G、trainer和初始 pair head SHA一致；
- pair forward只读取 predicted anchor，GT anchor输入计数为0；
- final predicted GT/deploy与oracle GT/deploy全部严格为 `32/32`，而不是原训练器中较松的
  `>=30/32`。

审计结果为 `protocol_status=passed`，见输出目录中的
`checkpoint_final_audit.json`。

## 结论与边界

正式结论为 `GO`：P0-J 的10条 presence错误确实来自 pair head与 P0-I predicted-anchor
几何接口未对齐；只训练 pair head即可完全吸收该偏移，同时不破坏 GT-anchor oracle能力。

这个结论仅证明 overfit32 的组合容量，不证明新 scaffold、字体或渲染风格上的泛化，也不代表
当前模型可以输出完整 SDF。P0-I native Atom feature仍是独立接口问题；下一轮应保持 P0-K
relation冻结，只验证一个小型 feature adapter能否把 native feature对齐到 P0-G same-slot
teacher feature。容量 smoke不创建 Git tag，也不接入前端。
