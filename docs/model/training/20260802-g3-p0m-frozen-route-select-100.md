# G3 P0-M Frozen Route Select 100

> 日期：2026-08-02
>
> 状态：已完成，协议有效，route-select `NO-GO`
>
> 性质：graph-head 未见 train-split 的零训练冻结路线选择；不是外部 benchmark

## 目标与当前结论

P0-K 已在固定 overfit32 上证明：使用 P0-I predicted Atom、predicted-anchor ribbon、P0-G
same-slot semantic feature 和已对齐 pair head时，GT/deploy skeleton 与 typed graph 可以
达到 `32/32`。P0-L 尝试用 147,840 参数线性 residual adapter 替换 P0-G semantic feature，
fixed-final 得到：

- SmoothL1 ratio `0.2963`，未达到 `<=0.10`；
- cosine `0.7757`；
- GT/deploy count 与 skeleton `32/32`；
- positive type accuracy `90.25%`；
- GT/deploy typed graph `7/32`；
- P0-K same-slot 与 GT Atom oracle controls 均保持 `32/32`。

因此停止在同一 overfit32 上扩大 adapter。当前结构接口固定为：

```text
P0-I Atom head
  -> predicted Atom identity / count / anchor / score / dense slot
  -> P0-G 在同一个 predicted (slot,y,x) 读取 384D semantic feature
  -> P0-K 在 predicted-anchor ribbon 上判断全 AtomPair existence / type
  -> P0-G frozen Bond count head
```

P0-G same-slot feature 只读取图片和 P0-I predicted slot，不使用 GT，不是 oracle。P0-G 自己
的 Atom presence、anchor 或 count 不得参与对象选择。

P0-M 只回答：

> 上述冻结路线在 graph heads 未训练过的固定 train-split 分子上，是否已有足够的 AtomPair
> presence/type 信号，值得进入独立的大规模训练协议？

P0-M 不训练任何模块，也不选择 checkpoint 或阈值。

## 唯一数据与权限边界

route-select manifest 固定为：

```text
Data/sdf/chemtraverse_v0_1_capacity/dev_route_select_100.jsonl
sample count: 100
molecule count: 25
SHA256: a31addd5d18dc153b093a9ec3879872401b6928fda40b100cb2e78c06e72a7a6
```

首次 forward 前必须验证 manifest 路径、字节 SHA、100 个 sample、25 个 molecule、sample
顺序和 schema。不得替换、追加、删除或根据图片/标签筛选记录。

route-select 还必须审计它与 overfit32 及 P0-I/P0-G/P0-K graph-head 训练样本的：

- sample key overlap；
- molecule id / InChIKey overlap；
- scaffold overlap。

sample key、molecule id和InChIKey overlap必须全部为0，才能称为“graph-head未见分子”。
当前manifest构建器没有承诺scaffold-disjoint，因此scaffold overlap必须如实报告，但不设
协议硬门；无论数值如何都不得称为“新scaffold”。冻结vision checkpoint可能已经消费同源
train分布和同一物理shard，因此P0-M不能称为完整模型未见图片、外部分布或泛化benchmark。

confirm manifest 固定保留为：

```text
Data/sdf/chemtraverse_v0_1_capacity/dev_route_confirm_400.jsonl
```

P0-M 执行前和执行期间禁止读取、解析、统计、哈希、预览或试跑 confirm400。只有
route-select 报告写出机器可读 `status=passed`，且本文件全部 Gate 同时通过后，才允许为
confirm400 新建独立预注册协议并首次读取。route-select `NO-GO` 时 confirm400 继续封存。

禁止读取 fresh dev、sealed test、USPTO、MR-TADF 或其他 confirm 数据。

## 冻结 checkpoint 与参数范围

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

固定 P0-K：

```text
outputs/smoke/chemvector-g3-p0k-predicted-anchor-relation-alignment/last.pt
SHA256 a91d5bb1052063c819b3ba71e26987c45d5e7e819281ff26fa7b3bd80f90eac5
```

vision、P0-I、P0-G、P0-K 全部 `eval()`、`requires_grad=False`。P0-K pair head覆盖 P0-G
原 pair head；P0-G Atom feature extractor 和 Bond count head保持原 checkpoint 权重。

P0-M 固定为：

```text
optimizer: none
updates: 0
backward: 0
trainable parameters: 0
```

评测前后必须逐模块记录 canonical tensor SHA。任一参数、buffer 或 mode 改变都使运行无效。
不得加载 P0-L adapter。

## 三条固定评测分支

### 1. GT Atom Oracle Relation

只用于隔离 P0-G/P0-K relation 在新 graph-head 样本上的信号：

- GT Atom anchor/slot 只进入明确标记的 oracle 分支；
- 使用 P0-G image-derived Atom feature；
- 枚举全部 GT Atom 的无序 pair；
- ribbon 由该 oracle 分支的 Atom anchor 定义；
- pair existence/type由冻结 P0-K 预测。

该分支不是部署结果，也不能与 same-slot 主路混写。

### 2. Same-Slot 主路

真实当前结构路线严格使用：

- P0-I deploy Atom count、predicted anchor、predicted score与selected `(slot,y,x)`；
- P0-G 在同一 predicted slot gather的384D semantic feature；
- P0-I predicted anchor 定义的 ribbon；
- 冻结 P0-K pair head和P0-G Bond count head；
- 全部 predicted Atom 的无序 pair `i<j`。

分别报告 GT Bond count top-K 和 deploy Bond count top-K。GT Bond count只在pair forward完成
后用于明确标记的GT-count评测，不得进入pair feature、ribbon或deploy count。

### 3. P0-I Native Feature 对照

除 Atom feature 改为 P0-I native 384D feature 外，Atom set、anchor、score、pair identity、
ribbon、pair head和count协议必须与same-slot主路完全相同。两条predicted Atom分支必须有
完全相同的pair identity SHA，才能做paired接口比较。

不得训练或加载 P0-L adapter，也不得在看到结果后混合 native/same-slot feature。

## GT 使用边界与评分映射

Same-slot与native两条预测分支的pair forward全部完成后，才允许将P0-I predicted Atom与
GT Atom做 detached、maximum-cardinality-first、L2 `<=6px` 一对一映射。该映射只用于：

- Atom recall/set exact；
- target Bond representable coverage；
- predicted pair 的presence/type评分；
- skeleton/typed graph exact。

映射不得改变 Atom set、pair universe、ribbon、presence logit、chemical logit或top-K。
unmatched predicted Atom相关pair不得伪造GT identity；unrepresentable Bond只计coverage
失败，不得repair。

所有分支禁止距离、近邻、键长、空白中点、valence或化学规则裁剪。每个有效Atom集合必须
枚举全部 `i<j`。

## Cell Occupancy 与 Slot 协议风险

P0-I 只在 overfit32 的 `occupancy=1` 上验证过 canonical cell slot。P0-M 必须在任何模型
评测之前，使用192×192 feature cell和GT normalized anchor统计：

| 统计 | 必须报告 |
|---|---|
| occupancy=1 | cell、Atom、image 数 |
| occupancy=2 | cell、Atom、image 数 |
| occupancy=3 | cell、Atom、image 数 |
| occupancy>3 | cell、Atom、image 数及sample key |
| identical normalized anchor collision | 数量、cell与sample key |

occupancy 1/2/3 都必须分别报告 P0-I Atom count、6px recall/set exact、same-slot
representable coverage和relation指标，不得只报总体平均。

`occupancy>3` 超过模型每cell 3 slots的硬容量；完全相同 normalized `(x,y)` 不能按P0-I
canonical规则无歧义排序。任一项非0都记为slot协议风险并使route-select Gate失败，不得删除
样本、扩大slot或用source atom id破tie后继续。

## 固定运行协议

| 项目 | 固定值 |
|---|---:|
| manifest | `dev_route_select_100.jsonl` |
| images / molecules | `100 / 25` |
| updates | `0` |
| trainable parameters | `0` |
| device | CUDA |
| precision | BF16 |
| batch | `4` |
| vision batch | `4` |
| threshold search | 禁止 |
| checkpoint selection | 禁止 |

样本严格按manifest顺序评测一次。允许按batch释放显存，但不得shuffle、增强、重复投票、测试
时训练或按分子挑选输出。

预注册输出目录：

```text
outputs/smoke/chemvector-g3-p0m-frozen-route-select-100/
```

只生成JSON审计、逐样本结构指标和状态文件，不生成训练checkpoint、optimizer、best模型、SDF
或前端结果。当前文档只预注册协议；评测代码和测试完成前不得运行。

实际运行命令：

```bash
.venv-train/bin/python -m \
  chemvector.evaluation.audits.graph.audit_g3_p0m_frozen_route_select
```

## GO/NO-GO Gate

### 协议 Gate

以下项目必须全部严格通过：

- route-select manifest SHA、100图、25分子和schema正确；
- 与graph-head训练数据的sample key、molecule id和InChIKey overlap均为0；scaffold
  overlap已报告且没有被写成生成器保证；
- vision、P0-I、P0-G、P0-K initial/final SHA完全相同；
- optimizer/backward/update均为0；
- predicted两分支pair identity SHA完全相同；
- 每个分支全 `i<j`，没有hard candidate filter；
- GT只在oracle分支或forward后的detached评分映射使用；
- occupancy>3为0，identical coordinate collision为0；
- confirm400和所有外部/封存数据均未读取。

协议门失败表示运行无效，不属于能力 `NO-GO`。

### Oracle Relation Gate

在GT Atom oracle分支全部target Bond上：

| 指标 | GO门槛 |
|---|---:|
| pair presence F1 | `>=95%` |
| positive chemical type accuracy | `>=95%` |

同时报告GT/deploy Bond count、skeleton exact和typed graph exact，但本轮不以whole-image exact
替代上述relation micro门。

### Same-Slot 主路 Gate

| 指标 | GO门槛 |
|---|---:|
| target Bond representable recall | `>=90%` |
| GT-count pair presence F1 | `>=95%` |
| representable positive chemical type accuracy | `>=95%` |

type accuracy只在“两端Atom都完成6px映射”的同一组representable positive Bonds上计算；
unrepresentable Bond不得从分母中静默删除，必须由representable recall单独承担。

同时必须报告：

- P0-I Atom count exact、6px micro和whole-set exact；
- deploy Bond count exact/MAE；
- GT/deploy skeleton与typed graph exact；
- pair precision、recall、F1和逐occupancy桶结果。

这些完整图指标当前只用于分解，不设生产门槛。

### Same-Slot / Native Paired Gate

在两条predicted Atom分支完全相同的representable positive Bond分母上：

```text
same-slot positive type accuracy - native positive type accuracy >= 5 percentage points
```

并且：

```text
same-slot GT-count skeleton exact images >= native GT-count skeleton exact images
```

不得用不同Atom mapping、不同pair universe或不同分母制造增益。

Oracle、Same-Slot与Paired三项能力Gate任一失败，route-select即为 `NO-GO`，confirm400保持
封存。全部通过时写出机器状态 `passed`，只表示当前路线有资格进入confirm协议，不表示模型
泛化或部署GO。

## 禁止事项

- 禁止训练、微调、蒸馏、校准或修改任何参数；
- 禁止加载P0-L adapter或重新训练native feature接口；
- 禁止让P0-G Atom detector参与主路对象选择；
- 禁止用GT anchor替换same-slot/native主路的predicted anchor；
- 禁止用GT Bond count驱动deploy分支；
- 禁止距离、近邻、键长、NMS、valence、OCR或repair；
- 禁止查看结果后改manifest、Gate、occupancy规则或checkpoint；
- 禁止读取confirm400、fresh、sealed、USPTO或MR-TADF；
- 禁止生成模型tag、接前端或宣称SDF能力。

## 结论边界与后续

P0-M 使用train split，并且vision可能见过同源训练分布。即使route-select全部通过，也只能
表述为：

> 冻结same-slot semantic feature路线在graph-head未见的train样本上显示出可扩量信号。

不得称未知分子泛化、外部泛化、SOTA、部署准确率或MR-TADF能力。100图route-select还是
路线选择开发集，不是最终验证集。

只有P0-M `GO`后，才能另建confirm400中文预注册，首次读取固定confirm manifest并运行一次
冻结确认。confirm仍通过后，才允许设计独立的大规模train协议；route-select/confirm样本都
不得回流训练。

P0-M `NO-GO`时按分解结果处理：

- oracle relation失败：P0-K/P0-G relation未跨graph-head样本迁移，先处理relation训练；
- oracle通过、representable不足：P0-I Atom/slot协议是主瓶颈；
- representable与presence通过、type失败：same-slot semantic type仍需扩训；
- same-slot不优于native：保留双feature provider缺少证据，重新评估接口成本；
- occupancy>3或collision非0：先扩展Atom slot协议，不能继续当前路线。

## 实际结果

本轮于2026-08-02按预注册协议只读运行一次。协议检查全部通过：100图/25分子、固定SHA、
sample/molecule/InChIKey overlap均为0，所有模块0 trainable、0 optimizer、0 backward，
评测前后state与mode完全一致。报告项scaffold overlap也为0，但生成器未承诺该属性，仍不
据此宣称新scaffold。

### Occupancy

| occupancy | cell | Atom | image |
|---|---:|---:|---:|
| 1 | 3,078 | 3,078 | 100 |
| 2 | 1 | 2 | 1 |
| 3 | 0 | 0 | 0 |
| >3 | 0 | 0 | 0 |

完全相同normalized anchor collision为0。route-select首次覆盖一个occupancy=2 cell，未超过
三slot硬容量；但单样本不足以证明collision slot泛化。

### 冻结能力

| 分支/指标 | 结果 | Gate |
|---|---:|---:|
| GT Atom oracle presence F1 | `98.42%` | `>=95%`，通过 |
| GT Atom oracle positive type | `81.51%` | `>=95%`，失败 |
| same-slot Atom recall | `87.92%` | `>=90%`，失败 |
| same-slot Atom count exact | `5/100` | 仅报告 |
| same-slot Atom count MAE | `12.36` | 仅报告 |
| same-slot Bond representable recall | `79.18%` | `>=90%`，失败 |
| same-slot GT-count presence F1 | `59.77%` | `>=95%`，失败 |
| same-slot representable positive type | `80.65%` | `>=95%`，失败 |
| native representable positive type | `75.71%` | paired baseline |
| same-slot相对native type增益 | `+4.94pp` | `>=5pp`，失败 |

same-slot与native两路AtomPair identity完全相同。两路whole-image Atom set、GT skeleton和
typed graph均为 `0/100`；25个分子中四种variant全部正确的分子数也均为0。GT Atom oracle
在GT count下 topology whole exact为 `75/100`，typed structural graph仅 `4/100`，进一步
确认presence迁移明显强于chemical type迁移。

按variant看，same-slot Atom recall为coordgen `84.94%`、original `91.69%`、RDKit
`89.74%`、stress `85.32%`；风格差异已经进入Atom selector，不是只发生在最终图指标。

输出与SHA256：

```text
outputs/smoke/chemvector-g3-p0m-frozen-route-select-100/route_select_audit.json
748471ced0c21f748ef0f189cf5a44d8899fea6e99d6fff2f1ad3c8051834db3

evaluator
3c1dd391b66acdc32d851c18b4095c3af0cb67949a001d524888a2f1a207263e
```

## 最终结论

`protocol_status=passed`，但能力Gate多数失败，P0-M正式为 `NO-GO`，机器字段
`confirm400_may_be_read=false`。confirm400继续封存。

这次失败不是“纯SVG路线完全不可行”：GT Atom oracle presence F1仍达98.42%，说明全
AtomPair视觉presence在graph-head未见分子上已有迁移信号。真正需要扩量的是两个可分离模块：

1. P0-I canonical Atom selector：count exact 5%、Atom recall 87.92%，导致只有79.18%的Bond
   可表示；
2. P0-G/P0-K semantic type：即使给GT Atom，positive type也只有81.51%。

下一阶段使用固定train_5k，先只训练canonical Atom selector并冻结评测；通过后再单独训练
semantic type/relation。禁止直接joint训练，也不读取confirm400。
