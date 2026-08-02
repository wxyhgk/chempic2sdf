# G3 P0-L Native Atom Feature Adapter overfit32

> 日期：2026-08-02
>
> 状态：已完成，协议有效，能力 `NO-GO`
>
> 性质：P0-I native Atom feature 到 P0-K relation 的接口容量实验；不是泛化实验

## 因果依据与唯一问题

P0-J 在相同 P0-I predicted Atom、anchor、score 和 pair identity 下只替换 Atom feature：

| Atom feature | GT-count typed graph |
|---|---:|
| P0-I native feature | `4/32` |
| P0-G same-slot replay feature | `26/32` |

P0-K 随后保持 P0-G same-slot replay feature，只用真实 P0-I predicted anchor/ribbon 对齐
pair head。fixed-final step 400 的结果为：

- predicted Atom + GT Bond count：skeleton / typed graph `32/32`；
- predicted Atom + deploy Bond count：count / skeleton / typed graph `32/32`；
- GT Atom oracle replay：GT/deploy structural graph `32/32`；
- pair precision、recall、F1 与 positive type accuracy 全部为 `100%`。

这证明 Atom set、predicted-anchor ribbon、Bond count 和 relation head 在固定 overfit32 上已经
可以共同工作；剩余未隔离的接口只有 P0-I native 384 维 Atom feature。P0-L 只回答：

> 冻结 P0-I、P0-G 和 P0-K 后，一个共享的逐 Atom 线性残差适配器，能否将 P0-I native
> feature 对齐到 P0-G same-slot feature，并让已冻结的 P0-K relation 保持完整图能力？

P0-L 不继续训练 relation，不修改 anchor、score、count 或 AtomPair 协议。

## 冻结输入与 checkpoint

固定 P0-I：

```text
outputs/smoke/chemvector-g3-p0i-canonical-cell-slot-overfit32/last.pt
SHA256 452a95c84cbb5bfb4a0e185f9955a963e5c126f77663fc8888052fdae619fe60
```

固定 P0-G：

```text
outputs/smoke/chemvector-g3-p0g-gt-atom-pair-bond-capacity/last.pt
SHA256 f4f6f6030e1dc0802a143b960a83ed3aff3163b2853e648c9a058b9150b08ee0
```

固定 P0-K relation：

```text
outputs/smoke/chemvector-g3-p0k-predicted-anchor-relation-alignment/last.pt
SHA256 a91d5bb1052063c819b3ba71e26987c45d5e7e819281ff26fa7b3bd80f90eac5
```

P0-K checkpoint 中的 pair head 必须覆盖 P0-G 原 pair head；P0-G Atom feature extractor 和
Bond count head仍从固定 P0-G checkpoint 加载。vision、P0-I Atom、P0-G Atom/count 和
P0-K pair head 全部 `eval()`、`requires_grad=False`，且训练前后 module SHA 必须完全一致。

数据继续固定为：

- manifest：`Data/sdf/chemtraverse_v0_1_capacity/overfit_32.jsonl`；
- manifest SHA256：
  `8fbb55b26ed170a64d9f258f702432d5ec626049976f6d9194bcf6751033b8a0`；
- 32 张图、1,424 个 predicted Atom、37,380 个无序 AtomPair、1,508 个 GT Bond；
- frozen vision checkpoint SHA256：
  `65a443d26bbffd41359604ac8147fa62a66ee18ca877693ebb3dea0b0a637095`。

不得读取 fresh dev、sealed test、confirm、USPTO 或 MR-TADF。

## 唯一 trainable 模块

P0-L 只增加一个共享逐 Atom residual adapter：

```text
normalized = LayerNorm(384, elementwise_affine=False)(native_feature)
adapted    = native_feature + Linear(384, 384)(normalized)
```

固定要求：

- `LayerNorm` 无 affine 参数；
- `Linear` 使用 bias；
- Linear weight 和 bias 全部 zero-init；
- trainable 参数只有 `384*384 + 384 = 147,840`；
- step 0 adapter 输出必须与 native feature bitwise 相同；
- adapter 对每个 Atom 独立、跨图片共享。

adapter 不得读取 anchor、score、slot id、cell 坐标、sample id、molecule id、GT Atom/Bond
字段或其他上下文。不得增加 MLP、attention、卷积、非线性 bottleneck 或第二个 residual
分支。

## Feature Teacher 与 Forward

对 P0-I deploy 选择的每个 predicted Atom，同时缓存：

1. `native_feature`：P0-I 自身在该 selected Atom 上输出的 384 维 feature；
2. `teacher_feature`：冻结 P0-G Atom head 在同一个 P0-I predicted `(slot,y,x)` 上 gather 的
   384 维 feature。

teacher identity 只来自 P0-I 的 predicted dense slot，不使用 GT Atom mapping。两组 feature
均由同一冻结 vision 输入生成；teacher 必须 `detach`，且不进入部署 forward。

训练与评测的 relation forward 固定使用：

- P0-I deploy Atom count、predicted anchor、predicted score 与 AtomPair identity；
- adapter 输出的 384 维 Atom feature；
- P0-I predicted anchor 定义的 ribbon；
- 冻结 P0-K pair head；
- 冻结 P0-G Bond count head，pooled Atom 输入改为 adapted feature。

全体有效 Atom 的无序对 `i<j` 全部保留，不做距离、近邻、键长、空白中点或 valence 裁剪。
GT Bond count 只允许在 forward 后的 GT-count 评测分支用于 top-K；不得进入 adapter、pair
forward、ribbon 或 deploy count。

## 唯一训练 Loss

P0-L 只计算 valid predicted Atom 上的 feature distillation：

```text
L = SmoothL1Loss(
      adapted_native_feature,
      stop_gradient(p0g_same_slot_teacher_feature),
      beta=1.0,
      reduction="mean"
    )
```

禁止加入以下任何目标：

- pair presence、chemical type 或 Bond count loss；
- GT Atom/Bond label loss；
- cosine、MSE、contrastive、triplet 或 feature variance loss；
- anchor、score、OCR、valence、graph repair 或 consistency loss。

cosine similarity、raw SmoothL1、NRMSE 可以在 no-grad evaluator 中报告，但不能参与训练、
选模或阈值搜索。

## 固定训练协议

| 项目 | 固定值 |
|---|---:|
| data | overfit32，32 张 |
| seed | `20260713` |
| batch | `4` |
| updates | `400` |
| optimizer | AdamW |
| LR | `1e-3` |
| weight decay | `0` |
| max grad norm | `1.0` |
| precision | BF16 |
| evaluate/save | 每 `100` step |
| trainable parameters | `147,840` |

manifest 顺序循环，不 shuffle、不增强。只按 fixed final step 400 判定；不生成或选择
`best.pt`，不追加 step，不换 seed，不搜索 LR、loss、adapter 结构或 feature threshold。

预注册独立输出目录：

```text
outputs/smoke/chemvector-g3-p0l-native-feature-adapter-overfit32/
```

当前文档只预注册协议；trainer、聚焦测试、审计与严格 CLI 校验完成前不得启动 GPU。

实际启动命令：

```bash
.venv-train/bin/python -m \
  chemvector.training.graph.relations.train_g3_p0l_native_feature_adapter
```

## Step 0 Baseline

由于 Linear 为零初始化，step 0 必须复现“P0-K pair head + P0-I native feature + predicted
anchor”的直接组合。必须在训练前保存：

- native-to-teacher raw SmoothL1、cosine 与 NRMSE；
- predicted GT/deploy count 下的 pair precision/recall/F1、positive type、skeleton 和 typed
  graph；
- P0-K same-slot replay control；
- P0-K GT Atom oracle replay control。

step 0 只定义相对 feature error baseline，不用于选择 checkpoint。

## Final GO/NO-GO Gate

### 协议 Gate

以下项目必须全部严格通过：

- P0-I predicted Atom mapping `1424/1424`、Atom set `32/32`；
- same-slot native/teacher feature pair `1424/1424`；
- GT Bond representable `1508/1508`、pair universe `37380`；
- adapter 恰好为无 affine LayerNorm 加 zero-init `Linear(384,384,bias=True)`；
- optimizer 恰好只包含 147,840 个 adapter 参数；
- vision、P0-I、P0-G Atom/count 和 P0-K pair head final SHA 与 initial 完全相同；
- adapter/distillation/pair forward 没有读取 GT feature、anchor、Bond refs/type/count；
- final artifact 严格来自 step 400，metrics 与 checkpoint 同 step。

任一协议门失败表示实验无效，不属于模型能力 `NO-GO`。

### Feature Interface Gate

在相同 1,424 个 valid Atom、相同 raw SmoothL1 口径下：

```text
final SmoothL1 <= 0.10 * step0 identity-adapter SmoothL1
```

即 feature distillation error 至少下降 90%。cosine 与 NRMSE 只报告，不另设可搜索门槛。

### Adapted Native Capability Gate

| 分支 | 指标 | 门槛 |
|---|---|---:|
| predicted Atom + GT Bond count | skeleton exact | `>=30/32` |
| predicted Atom + GT Bond count | typed graph exact | `>=30/32` |
| predicted Atom + deploy Bond count | count exact | `>=30/32` |
| predicted Atom + deploy Bond count | skeleton exact | `>=30/32` |
| predicted Atom + deploy Bond count | typed graph exact | `>=30/32` |

同时报告 pair precision、recall、F1 与 positive type accuracy，但 micro 指标不能替代整图
Gate。

### 严格 Control Gate

adapter bypass 后必须严格保持：

| Control | 要求 |
|---|---:|
| P0-K + P0-G same-slot replay，GT-count typed graph | `32/32` |
| P0-K + P0-G same-slot replay，deploy typed graph | `32/32` |
| P0-K GT Atom oracle replay，GT-count typed graph | `32/32` |
| P0-K GT Atom oracle replay，deploy typed graph | `32/32` |

四项均要求 0 break，不能放宽为 `>=30/32`。control 失败表示冻结权重、评估器或协议已改变，
整轮无效。

Feature Interface、Adapted Native Capability 任一门失败即为 P0-L `NO-GO`。只按 final
step 400 判定，不得选择中间 step。

## 禁止事项

- 禁止训练或修改 vision、P0-I、P0-G、P0-K；
- 禁止把 P0-G teacher feature 保留到部署 forward；否则应直接使用 same-slot replay，
  P0-L adapter 没有意义；
- 禁止用 GT Atom anchor、GT Atom identity、GT Bond label 构造 feature teacher；
- 禁止训练 pair head、Atom head或Bond count head；
- 禁止加入 relation/task loss、双几何 replay 或 native/teacher feature 混合推理；
- 禁止修改 predicted anchor、score、pair universe、ribbon 或 count 协议；
- 禁止读取外部数据、切换 checkpoint、追加 step、选择 best 或搜索超参数；
- 禁止把本实验 checkpoint 接入前端或创建正式 tag。

## 结论边界与主要风险

P0-I、P0-G、P0-K 和 P0-L adapter 都使用同一 overfit32。即使 final 达到 32/32，P0-L
也只能证明：

> 在这 32 张已见训练图上，一个共享逐 Atom 线性接口具有连接 P0-I native feature 与
> 已对齐 P0-K relation 的容量。

严禁将结果写成 feature 泛化、未知分子泛化、公开集提升、部署 GO 或端到端模型完成。
把 32 图内部切出几张也不能宣称完整模型泛化，因为冻结的 P0-I/P0-G/P0-K 已经训练过全部
32 图；最多只能称 adapter conditional split，本轮不做该实验。

其他主要风险：

- 1,424 个训练 feature 足以被 147,840 参数线性层记忆，feature error下降不代表新 scaffold
  可迁移；
- P0-I 当前 occupancy 全为 1，slot 1/2 和 collision feature 尚未验证；
- 全 AtomPair 仍为 `O(V²)`，P0-L 不解决 CPU 部署成本；
- SmoothL1 对齐全部维度，不保证每一维都对 relation 等价；若 feature loss 过门但 graph
  不过门，只能说明全维回归不是 relation-sufficient alignment；
- 若 graph 过门但 feature error 未过门，adapter 可能利用训练图特有方向获得结果，仍判
  `NO-GO`。

P0-L GO 后，下一阶段必须使用预先冻结、未被 P0-I/P0-K/adapter 训练消费的非重复 scaffold
数据，单独预注册接口泛化实验；本轮按协议不读取这些数据。

## 预期判读

- **feature 与 graph 均 GO，controls 32/32**：train-only interface capacity 成立；
- **feature GO、graph NO-GO**：通用 feature 拟合不等于 relation 所需接口，下一轮才允许
  task-aware adapter；
- **feature NO-GO**：共享单层 residual affine 容量不足，不能在本轮增加 MLP；

## 实际结果

本轮于2026-08-02按预注册参数执行唯一一次400-step训练。fixed-final结果：

| 指标 | step 0 | step 100 | step 200 | step 300 | step 400 |
|---|---:|---:|---:|---:|---:|
| SmoothL1 | `0.62843` | `0.23612` | `0.20913` | `0.19524` | `0.18618` |
| 相对 step 0比例 | `1.000` | `0.3757` | `0.3328` | `0.3107` | `0.2963` |
| cosine similarity | `0.1653` | `0.6930` | `0.7388` | `0.7605` | `0.7757` |
| positive type accuracy | `86.87%` | `88.13%` | `89.12%` | `89.99%` | `90.25%` |
| GT-count skeleton exact | `32/32` | `32/32` | `32/32` | `32/32` | `32/32` |
| GT-count typed graph exact | `6/32` | `5/32` | `7/32` | `8/32` | `7/32` |
| deploy typed graph exact | `4/32` | `4/32` | `7/32` | `8/32` | `7/32` |

step 200起 deploy Bond count、skeleton均为 `32/32`，说明 adapter足以恢复 count与presence
排序相关接口；但 final feature error只下降 `70.37%`，未达到预注册的90%，而键类型整图
仍只有 `7/32`。P0-K same-slot replay与 GT Atom oracle四个 control在全部 checkpoint均严格
保持 `32/32`。

输出目录：

```text
outputs/smoke/chemvector-g3-p0l-native-feature-adapter-overfit32/
```

关键 SHA256：

```text
step-000400.pt  70e97dc9915eb042342f3ca1fb799620cac3ac055cca5f514d64efccdacf0f0b
last.pt         6d85be58ba17074c78e685c6d75c472789d8c319bd695d103da60cd408b9ffc5
final audit     439579ed4a0b4762ea944618fe66783499730a6eb1d286b562d31723304758a3
trainer         3eb636f034f6ac9688fe97046533fa9f30385ed51d3b7fe93ad7e4cf43d00eb3
```

## 独立审计

只读审计验证100/200/300/400四个 checkpoint、全部固定参数和来源 SHA，并 strict-load真实
adapter。optimizer只有 Linear weight/bias两个 tensor、共147,840参数；teacher显式 detach，
训练 loss未读取 GT字段；P0-I、P0-G/P0-K冻结状态前后 SHA一致。每步均覆盖1,424个feature、
1,508个Bond和37,380个AtomPair。

审计结果：`protocol_status=passed`、`capability_status=failed`、`decision=NO-GO`，见
`checkpoint_final_audit.json`。

## 最终结论

P0-L 正式判为 `NO-GO`。不能在同一32图上追加 step、扩大 MLP或加入 relation loss；那只会
增加记忆容量，无法回答新样本迁移问题。当前证据表明 P0-I native feature没有以简单共享
仿射形式保留 P0-K所需的键类型语义。

后续结构暂时正式保留 P0-G same-slot semantic feature：它只读取图片和 P0-I predicted
`(slot,y,x)`，不是 GT oracle；P0-I仍独占 Atom identity/count/anchor/score，P0-K负责
predicted-anchor ribbon上的pair presence/type。下一步先在 graph head未见的固定 train-split
route上做零训练冻结审计，再决定是否扩量训练。容量 smoke不创建 Git tag，也不接前端。
- **graph GO、feature NO-GO**：可能是同集记忆，不接受为接口对齐；
- **任一 control break**：实验无效，先修冻结/评估协议。
