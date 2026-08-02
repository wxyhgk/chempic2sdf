# G3 P1-B Frozen Presence Ranker Train5k

> 日期：2026-08-02
>
> 状态：已完成；协议通过，能力 `NO-GO`
>
> 性质：train-distribution selector因果隔离；不是deploy count、外部泛化或完整结构模型

## 唯一问题

P1-A fixed-final在route-select100上的只读诊断为：

- all-dense top-128 coverage：`3072/3080 = 99.74%`；
- canonical fixed-positive定位：`3080/3080`，whole-set `100/100`；
- GT-count learned top-K：F1 `93.18%`，whole-set `14/100`；
- deploy count learned top-K：F1 `90.39%`，whole-set `2/100`；
- GT-count top-K canonical identity recall：`2581/3080 = 83.80%`。

候选容量与anchor定位已经基本解决，第一断点是presence ranking。P1-B只回答：

> 冻结P1-A整个Atom head后，一个零初始化轻量residual能否在不改变候选位置和count的情况下，
> 把全部canonical positive稳定排在第K边界negative之前？

P1-B不回答count、Bond、OCR、SDF或外部分布问题。

## 数据与禁读边界

训练和开发数据完全继承P1-A：

```text
train:
Data/sdf/chemtraverse_v0_1_capacity/train_5k.jsonl
5000 images / 1250 molecules
SHA256 0022245e1bbc98a638a94b03426b3cc12c09781ed355baafbd748ad918d66e21

route-select:
Data/sdf/chemtraverse_v0_1_capacity/dev_route_select_100.jsonl
100 images / 25 molecules
SHA256 a31addd5d18dc153b093a9ec3879872401b6928fda40b100cb2e78c06e72a7a6
```

固定train sample顺序SHA256：

```text
f336398a865d55a65b5efc2a3dd4f3b4e430434356a0fe59bd318ddfac9b6049
```

confirm400、fresh、sealed、USPTO、MR-TADF及其他外部数据继续禁止读取。route-select100只允许
step0与fixed-final两次只读评测，不得用于checkpoint、阈值或超参数选择。

## 初始化与冻结

固定P1-A final初始化：

```text
outputs/checkpoints/chemvector-g3-p1a-canonical-atom-selector-train5k/last.pt
SHA256 bf0d35f31dafc467fafe3fd1bcbb758d47685a4febc7d9ad6ff96d1032d12ec2
Atom head state SHA256 05386591037f0c9ae9624ef6befe25c7a82315faf720ab915a01532592e44b8a
```

vision仍固定为：

```text
outputs/checkpoints/chemvector-v4-clean-50k/last.pt
SHA256 65a443d26bbffd41359604ac8147fa62a66ee18ca877693ebb3dea0b0a637095
```

冻结范围：

- vision严格`eval()`、`requires_grad=False`；
- P1-A整个`SpatialAtomSlotHead`严格`eval()`、`requires_grad=False`；
- base presence、hidden、anchor、slot feature和count输出全部detach；
- optimizer只能包含新增presence residual的16,643个参数；
- 不加载P0-G、P0-K、Bond、AtomPair、OCR、ChemScene或SDF模块。

## 唯一新增模块

对冻结Atom head的`object_features [B,256,H,W]`应用：

```text
Conv1x1(256 -> 64) -> GELU -> Conv1x1(64 -> 3)
```

最后一层weight与bias严格全零，第一层使用框架默认初始化。最终分数为：

```text
adjusted_presence = base_presence.detach() + residual(hidden.detach())
```

因此step0必须与P1-A fixed-final bitwise等价。residual不接收GT count、GT anchor、candidate
mask或任何化学标签；禁止加入全局count token、图级attention、卷积空间上下文、NMS和几何
启发式。本轮只测试冻结局部语义是否足够完成排序。

## Canonical Ranking Loss

每张图的canonical positive集合`P`继续由GT anchor所属cell及同cell `(x,y)`升序slot定义，
不读取predicted anchor、presence或rank。令`K=|P|`，negative集合为其他全部dense slot。

1. 从当前adjusted logits中取`H=min(4K, |N|)`个最高分negative；top-H索引detach；
2. 对每个`p in P`和hard negative `n in H`计算`softplus(s_n - s_p)`；
3. 对图内全部pair取均值，再对batch取均值。

唯一loss：

```text
L_rank = mean_image mean_(p,n) softplus(s_n - s_p)
```

temperature固定1，无margin、focal、BCE、count CE、anchor loss、distillation、regularizer或
辅助loss。GT count只用于定义canonical positive数量、hard-negative预算及forward完成后的
known-count top-K评测，绝不进入residual forward。

## 固定训练协议

| 项目 | 固定值 |
|---|---:|
| train images / molecules | `5000 / 1250` |
| seed | `20260713` |
| batch | `4` |
| epochs | `5` |
| steps / epoch | `1250` |
| total updates / exposures | `6250 / 25000` |
| optimizer | AdamW |
| LR / weight decay | `3e-4 / 0.01` |
| max grad norm | `1.0` |
| precision | BF16 |
| workers / shuffle / drop_last | `0 / false / false` |
| hard negatives | `min(4K, |N|)` |
| checkpoints | `1250,2500,3750,5000,6250` |
| route evaluation | `step0,6250` only |

固定LR，无scheduler、warmup、gradient accumulation、增强、early stopping或best checkpoint。
每个epoch必须按相同manifest顺序完整streaming；不建立整集pixel、vision feature、hidden或logit
cache。正式运行中断后必须从固定P1-A final和空输出目录重跑。

预注册输出目录：

```text
outputs/checkpoints/chemvector-g3-p1b-frozen-presence-ranker-train5k
```

## 评测与Gate

route-select100在step0/final均固定使用GT count做top-K，768px下用maximum-cardinality-first、
L2 `<=6px`一对一匹配。另报告canonical slot identity recall、whole-set identity exact、
all-dense top-128 coverage及canonical fixed-positive定位。

### 协议Gate

- train/route manifest、样本数量、顺序SHA与身份不重叠审计通过；
- P1-A、Atom state及vision SHA完全一致；
- residual参数量恰为16,643，末层step0严格全零；
- optimizer只含residual，vision与Atom head initial/final state SHA不变；
- count logits与deploy count vector在step0/final bitwise相同；
- 5 epochs、6,250 updates、25,000 exposures及五个checkpoint精确成立；
- route只在step0/final运行，GT数据在residual forward后才参与target/loss或oracle评测；
- 无count CE、anchor/focal/Bond/OCR loss或forward；
- 禁读数据路径未出现且未运行。

协议Gate失败表示实验无效，不属于模型能力`NO-GO`。

### Fixed-Final Selector Capability Gate

| 指标 | GO门槛 |
|---|---:|
| known-count precision/recall/F1 @6px | 三者均`>=97%` |
| known-count whole-set exact | `>=50/100` |
| canonical identity recall | `>=95%` |
| all-dense top-128 positive coverage | `>=3072/3080` |
| canonical fixed-positive localization | `3080/3080` |

同时要求P1-A冻结count的deploy结果仍为count exact `13/100`、MAE `3.83`，用于证明本轮没有
暗中修count；它不属于P1-B能力Gate。任一selector能力门失败即正式`NO-GO`，禁止追加epoch、
更换loss、挑中间checkpoint或换seed重跑。

## 结论边界

- `GO`只表示当前train distribution上冻结dense feature足以通过独立ranker修复presence选择；
- `GO`后另立Count-only协议，不在P1-B内解冻或联合训练；
- `NO-GO`表示当前冻结hidden缺少可分信息或逐像素residual归纳偏置不足，不能靠追加epoch解释；
- 无论结果如何，都不得宣称未知分子、公开数据、MR-TADF、SDF或部署能力；
- 本实验属于G3容量/因果实验，不接前端、不生成SDF、不创建正式版本tag。

## 正式结果

唯一正式训练严格完成5 epochs、6,250 updates和25,000 exposures，训练与final评测总耗时
约565.75秒。每个epoch的sample顺序SHA均与预注册值一致，五个checkpoint完整生成；route只
在step0和fixed-final运行。

| 指标 | step 0 | step 6,250 | 变化 |
|---|---:|---:|---:|
| known-count P/R/F1 @6px | 93.18% | 93.60% | +0.42pp |
| known-count matched Atom | 2870/3080 | 2883/3080 | +13 |
| known-count whole-set exact | 14/100 | 16/100 | +2 |
| canonical identity recall | 83.80% | 84.22% | +0.42pp |
| canonical identity whole-set | 3/100 | 3/100 | 0 |
| deploy precision | 89.65% | 90.10% | +0.45pp |
| deploy recall | 91.14% | 91.59% | +0.45pp |
| deploy F1 | 90.39% | 90.84% | +0.45pp |
| deploy whole-set exact | 2/100 | 2/100 | 0 |
| top-128 geometry coverage | 3072/3080 | 3072/3080 | 0 |
| fixed-positive localization | 3080/3080 | 3080/3080 | 0 |

全部17项协议检查通过，包括：

- P1-A Atom head与vision initial/final SHA不变；
- optimizer只含16,643个residual参数；
- deploy count vector和完整count-logits stream SHA逐位不变；
- 同一dense scores在`K=1/GT/128`选择前后一致；
- 禁止组件未构造，GT count未进入dense forward；
- train/route顺序、身份不重叠、步数、曝光和checkpoint schedule全部成立。

能力Gate中只有top-128 coverage与fixed-positive localization两个守门项通过；F1、whole-set和
canonical identity三类主门全部失败，正式决策为`NO-GO`。final residual绝对值最大达到
`18.25`，但只找回13个known-count matched Atom，说明局部pointwise residual已经大幅改分，
却没有学到可迁移的整组排序规则。禁止追加epoch或扩大同类局部MLP。

关键产物：

```text
step-006250.pt
SHA256 6ead09e46a837e0316e8c8df940399df5b3422f5370112da5534512cc3c10fef

last.pt
SHA256 9d6df3a711d3a6f21e358daba0ded4860a0280706cfb5c35d024bf19c2b68e26

trainer source
SHA256 392b59e38d9f13a68b406e8770dca69f727076b58f60748c23e006a28bf5fa9e

ranker source
SHA256 423b7414f8eb6406f361a916630879fade2fb2c462dffd513f566d64b96ea7ef

checkpoint_final_audit.json
SHA256 fbd49bd18f3068960ee8a50101b9d6cc50737744c52937a97889e3042514e457

independent audit source
SHA256 147e9e5f76402bdd7788836230c4b9d8e4edd61ccc5556767820959dc5511233
```

独立final audit逐个重载五个checkpoint，重新核验optimizer residual-only、冻结lineage、trainer/
ranker源SHA、五次固定sequence SHA、6,250 updates和25,000 exposures，并分别重放step0与
fixed-final route100。保存指标逐字段一致，独立结论仍为
`protocol passed / capability failed / NO-GO`。P1-B相关13项测试全部通过。

## 决策与下一步

P1-B否定的是“冻结hidden上的逐像素局部residual足以修好排序”，不能扩大成“hidden完全没有
信息”。top-128仍以6px覆盖`99.74%`的GT Atom，因此下一轮应把问题缩成128个候选上的集合
选择：候选之间交换信息，利用相对位置、重复候选和全局基数上下文共同排序。该P1-C set
ranker必须另立协议；P1-B自身停止，不接Count-only、不解封confirm400。
