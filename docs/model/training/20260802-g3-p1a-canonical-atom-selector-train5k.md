# G3 P1-A Canonical Atom Selector Train5k

> 日期：2026-08-02
>
> 状态：已完成；协议通过，能力 `NO-GO`
>
> 性质：train-distribution Atom head transfer；不是外部泛化或完整结构模型

## 因果依据与唯一目标

P0-M 在固定 graph-head 未见 route-select100 上完成零训练冻结审计：

- GT Atom oracle pair presence F1 `98.42%`，说明 relation existence 已有迁移信号；
- GT Atom oracle positive type `81.51%`，semantic type 仍需后续单独扩训；
- P0-I deploy Atom count exact仅 `5/100`；
- P0-I Atom recall `87.92%`；
- target Bond representable recall `79.18%`；
- same-slot主路GT-count pair presence F1 `59.77%`、representable type `80.65%`。

首要断点是 P0-I canonical Atom selector没有从 overfit32 迁移到 train-distribution，导致大量
GT Bond在进入relation前已经不可表示。P1-A 因此只回答：

> warmstart P0-I canonical cell-slot Atom head，在固定 train_5k 上独立训练5 epochs后，能否
> 提高未参与本轮梯度的 route-select100 Atom count、6px集合精度与whole-set exact？

P1-A 不训练 P0-G semantic feature、Bond count、P0-K pair head或任何OCR/图修复模块。
P1-A fixed-final判定后必须依据实际断点另立协议，禁止joint训练。

## 固定训练数据

唯一训练 manifest：

```text
Data/sdf/chemtraverse_v0_1_capacity/train_5k.jsonl
sample count: 5000
molecule count: 1250
SHA256: 0022245e1bbc98a638a94b03426b3cc12c09781ed355baafbd748ad918d66e21
```

训练入口必须在读取图片前验证路径、字节SHA、schema、5,000个sample、1,250个molecule和
manifest顺序。禁止替换、删除、追加或按标签/图像筛选样本。

train_5k 与 route-select100 必须审计 sample key、molecule/InChIKey 与 scaffold overlap。
sample key、molecule和InChIKey overlap须为0；manifest构建器没有承诺scaffold-disjoint，
因此scaffold overlap只报告、不设硬门，也不得据结果宣称新scaffold。若数据交付中已有冻结
commitment，应核对commitment而不是重新挑选样本。

P1-A 只使用train split。vision可能已消费同源训练分布，因此结果只能称为
`train-distribution head transfer`，不能称完整模型未见数据、公开泛化或外部benchmark。

## 固定初始化与冻结范围

Atom head必须warmstart以下P0-I fixed-final：

```text
outputs/smoke/chemvector-g3-p0i-canonical-cell-slot-overfit32/last.pt
SHA256 452a95c84cbb5bfb4a0e185f9955a963e5c126f77663fc8888052fdae619fe60
```

vision继续使用P0-I绑定的冻结checkpoint：

```text
outputs/checkpoints/chemvector-v4-clean-50k/last.pt
SHA256 65a443d26bbffd41359604ac8147fa62a66ee18ca877693ebb3dea0b0a637095
```

固定范围：

- vision严格`eval()`、`requires_grad=False`；
- 仅P0-I `SpatialAtomSlotHead`进入optimizer；
- P0-G Atom/semantic feature、Bond count和P0-K relation均不加载到训练forward；
- 不构造Dense Bond、AtomPair、Bond target、ChemScene、SDF或OCR输出；
- optimizer从空状态新建，不恢复P0-I optimizer state。

训练前必须严格加载并记录P0-I Atom head canonical tensor SHA、参数名、参数量和optimizer参数
名。除Atom head外出现任何trainable参数都使协议无效。

## Canonical Cell-Slot Target 协议

P1-A 完整继承P0-I规则，不允许退回prediction-conditioned Hungarian：

1. 按192×192 frozen feature网格计算每个GT Atom所属cell；
2. 同一cell内GT Atom按normalized anchor `(x,y)` lexicographic升序；
3. 排序后第`n`个GT Atom固定映射slot `n`，即slot `0..occupancy-1`；
4. target assignment禁止读取predicted anchor、presence、rank、feature或count；
5. 完全相同normalized `(x,y)`显式拒绝，不使用source id、target id或SDF顺序破tie。

每个cell仍保留3个slot，禁止根据train_5k occupancy修改模型容量。训练和route-select都必须在
任何forward前统计：

- occupancy=1 的cell、Atom、image数；
- occupancy=2 的cell、Atom、image数；
- occupancy=3 的cell、Atom、image数；
- occupancy>3 的cell、Atom、image数及sample key；
- identical normalized coordinate collision数量及sample key。

`occupancy>3`或完全相同坐标任一非0，表示当前3-slot/canonical协议无法无损表达，正式运行
必须停止；不得跳过、替换样本、扩大slot或改tie规则后继续。

## 唯一模型与 Loss

模型保持现有 `SpatialAtomSlotHead`：

- 每cell 3个presence logits与anchor offsets；
- Atom count head；
- count-conditioned global presence top-K selector；
- maximum Atom 128；
- 模型结构、hidden channels、slot embedding与所有输出定义不变。

训练loss固定为P0-I canonical版本：

```text
L_atom = focal_heatmap_loss
       + 8.0 * smooth_l1_anchor_loss
       + 0.1 * atom_count_cross_entropy
```

禁止增加whole-set surrogate、duplicate suppression、NMS、count oracle、distillation、feature
loss、Bond loss、OCR loss或辅助化学约束。GT count只是count head监督标签，不得在训练forward中
teacher-force selected Atom集合。

## Streaming Batch 训练

P1-A 禁止把5,000张图片、pixel tensor或vision feature一次性放入GPU/整集cache。训练必须按
manifest顺序streaming：

1. 每次只读取当前batch的4张图片和Atom targets；
2. frozen vision在BF16 autocast下生成当前batch feature；
3. feature立即送入Atom head并计算loss；
4. 当前step结束后释放不再使用的pixel/feature graph；
5. 下一batch继续manifest顺序，epoch边界回到第一个sample。

允许操作系统page cache和小规模CPU预取，但不得建立包含全部5,000张vision feature的GPU或
CPU tensor cache。数据加载不得shuffle、随机采样或按分子重排。为保持完全确定性，正式协议
必须固定worker数、drop_last行为和每epoch sample-key SHA；5,000可被batch4整除，固定
`drop_last=False`时每epoch恰好1,250 steps且无残 batch。

实测多worker会按shard交错样本，因此正式固定 `num_workers=0`。单worker按manifest顺序消费
5,000个唯一sample的sequence SHA256固定为：

```text
f336398a865d55a65b5efc2a3dd4f3b4e430434356a0fe59bd318ddfac9b6049
```

每个epoch必须逐项核对sample key与该顺序，不接受“样本集合相同但顺序不同”。

## 固定训练协议

| 项目 | 固定值 |
|---|---:|
| train images / molecules | `5000 / 1250` |
| seed | `20260713` |
| manifest shuffle | `false` |
| batch | `4` |
| epochs | `5` |
| steps / epoch | `1250` |
| total updates | `6250` |
| optimizer | AdamW |
| LR | `1e-4` |
| weight decay | `0.01` |
| max grad norm | `1.0` |
| precision | BF16 |
| data workers | `0` |
| save interval | `1250` steps |
| route-select evaluation | step `0`与`6250` only |

LR固定，不使用scheduler、warmup、early stopping或gradient accumulation。每个step必须实际消费
4张图并执行一次optimizer update。五个epoch的sample顺序完全相同。

只按fixed-final step 6,250判定，不生成或选择`best.pt`。step 1,250/2,500/3,750/5,000只保存
审计checkpoint，不允许从中断点恢复本次正式运行；不得在route-select100上评测，也不得根据
训练loss选择其中任何一个。正式运行若中断，必须从固定P0-I warmstart重新开始并使用空输出
目录，避免未预注册的dataloader/RNG恢复语义改变样本顺序。

预注册输出目录：

```text
outputs/checkpoints/chemvector-g3-p1a-canonical-atom-selector-train5k/
```

正式运行已严格按本节协议完成。trainer、严格CLI、测试和独立审计均已落地；实际结果见
文末“正式结果与决策”。

训练前只读occupancy预检得到：occupancy=1为155,202个cell，occupancy=2仅7个cell，
occupancy=3为0。五个epoch内slot1 positive总曝光仅35次、slot2为0；因此P1-A的整体Gate主要
验证occupancy=1 transfer。occupancy=2只报告诊断，禁止据本轮结果宣称multi-slot已学会；
后续若需要可靠multi-slot能力，必须另备有足够collision样本的数据协议。

## Route-Select100 使用边界

固定开发manifest：

```text
Data/sdf/chemtraverse_v0_1_capacity/dev_route_select_100.jsonl
sample count: 100
molecule count: 25
SHA256: a31addd5d18dc153b093a9ec3879872401b6928fda40b100cb2e78c06e72a7a6
```

route-select100 已在P0-M被消费，P1-A只允许两次只读Atom评测：

1. step 0：warmstart P0-I加载完成、optimizer update之前；
2. fixed-final step 6,250：最后一次optimizer update后，对同一固定final state评测；随后将
   该state与final metrics一起原子写入final checkpoint。

禁止在中间checkpoint评测route-select，禁止根据step0/final以外的dev结果改LR、epoch、loss、
checkpoint、manifest或Gate。step0仅记录baseline；模型选择始终固定为final。

route-select评测也必须streaming，不得与train_5k混合或反向传播。Atom matching统一使用768px
图像上的maximum-cardinality-first、L2 `<=6px`一对一匹配。

## Confirm400 禁读边界

以下manifest继续禁止读取、解析、统计、哈希、预览或运行：

```text
Data/sdf/chemtraverse_v0_1_capacity/dev_route_confirm_400.jsonl
```

P1-A GO不自动解锁confirm400。P1-A只完成Atom selector阶段，semantic type尚未扩训；只有后续
阶段完成并另建中文confirm协议后，才能按新授权首次读取confirm400。P1-A `NO-GO`时confirm
继续封存。

同时禁止读取fresh dev、sealed test、USPTO、MR-TADF或其他confirm数据。

## Final Route-Select GO/NO-GO Gate

### 协议 Gate

以下项目必须全部严格通过：

- train_5k路径、SHA、5,000图、1,250分子和schema正确；
- train_5k与route-select的sample key、molecule和InChIKey overlap均为0；scaffold overlap
  已报告且没有被写成生成器保证；
- P0-I warmstart和vision SHA完全匹配；
- optimizer仅包含Atom head参数，vision initial/final SHA完全相同；
- streaming每epoch恰好5,000图/1,250 steps，总计25,000 exposure/6,250 updates；
- data workers严格为0，每epochsample顺序与固定sequence SHA逐项一致；
- step checkpoint恰好为1,250/2,500/3,750/5,000/6,250；
- route-select只在step0与6,250评测，final metrics与final checkpoint同step；
- train与route的occupancy1/2/3统计完整，overflow>3和coordinate collision均为0；
- confirm400及所有外部/封存数据均未读取；
- 没有Bond/AtomPair/OCR forward、loss或参数更新。

协议门失败表示实验无效，不属于模型能力 `NO-GO`。

### Fixed-Final Atom Capability Gate

在route-select100的P1-A deploy Atom输出上：

| 指标 | GO门槛 |
|---|---:|
| Atom count exact | `>=80/100` |
| Atom count MAE | `<=1.0` |
| Atom precision @6px | `>=97%` |
| Atom recall @6px | `>=97%` |
| Atom F1 @6px | `>=97%` |
| Atom whole-set exact @6px | `>=50/100` |

precision分母为deploy selected Atom数，recall分母为GT Atom数，true positive必须来自同一
maximum-cardinality 6px matching；whole-set exact要求count正确且所有predicted/GT Atom完成
一对一6px匹配。micro指标不能替代whole-set Gate。

还必须报告：

- step0到final的全部指标绝对变化；
- count error分布与MAE；
- occupancy1/2/3逐桶precision、recall、F1、whole-set exact；
- 按四种layout和Atom数量分桶的结果；
- overflow/collision sample清单，正常协议下应为空。

任一能力门失败即为P1-A `NO-GO`。不得延长epoch、选择中间checkpoint、重跑seed或修改Gate。

## 禁止事项

- 禁止resume P0-I optimizer或从P1-A中间checkpoint延长训练；
- 禁止在相同run增加epoch、换seed、搜索LR/WD/loss或选择best；
- 禁止整集GPU/CPU feature cache、shuffle、增强或数据重采样；
- 禁止修改canonical target、slot数、模型结构、top-K或count协议；
- 禁止训练vision、P0-G、P0-K、semantic feature、Bond count或OCR；
- 禁止读取Bond refs/type并以其影响Atom训练；
- 禁止根据route-select结果删除train样本或回填dev样本；
- 禁止读取confirm400、fresh、sealed、USPTO或MR-TADF；
- 禁止接前端、生成SDF、创建正式tag或宣称部署GO。

## 结论边界与后续

P1-A只使用train_5k训练，并在同源train-split route-select上判断head transfer。vision可能见过
同源分布，route-select也是已消费开发集。因此即使全部Gate通过，也只能表述为：

> canonical Atom selector在当前train distribution上完成了从overfit32到train5k的head
> transfer，并达到预注册开发门槛。

不得表述为未知分子泛化、公开集泛化、外部分布能力、SOTA或最终部署准确率。

- **GO**：冻结P1-A final Atom head，生成独立predicted Atom cache；下一阶段单独训练
  same-slot semantic type/relation，不做joint fine-tune；
- **count失败、geometry通过**：只处理count head，不重训relation；
- **recall/whole-set失败**：Atom candidate/selector仍是主瓶颈，confirm保持封存；
- **occupancy2/3显著失败**：扩展canonical multi-slot训练分布或目标，但必须另立协议；
- **协议失败**：修复实现后从固定P0-I warmstart重新开始，不能续无效run。

## 正式结果与决策

唯一正式训练已于2026-08-02完成：5个epoch各消费5,000张图、1,250次update，总计
25,000次exposure和6,250次update。每个epoch的sample顺序SHA256均为：

```text
f336398a865d55a65b5efc2a3dd4f3b4e430434356a0fe59bd318ddfac9b6049
```

训练实际耗时约682秒。固定step0到fixed-final的route-select100结果为：

| 指标 | step 0 | step 6,250 | 变化 |
|---|---:|---:|---:|
| precision @6px | 67.94% | 89.65% | +21.71pp |
| recall @6px | 87.92% | 91.14% | +3.22pp |
| F1 @6px | 76.65% | 90.39% | +13.74pp |
| count exact | 5/100 | 13/100 | +8 |
| count MAE | 12.36 | 3.83 | -8.53 |
| whole-set exact | 0/100 | 2/100 | +2 |

模型确实学到了train-distribution Atom选择能力，但六项预注册能力门全部失败，因此能力结论
是正式 `NO-GO`，禁止追加epoch、挑中间checkpoint或换seed补跑。

固定final checkpoint与关键产物：

```text
step-006250.pt
SHA256 4efb8f6709892ae66db1726fd80426f2cf4f3f781743b4e1266aa496f8f097db

last.pt
SHA256 bf0d35f31dafc467fafe3fd1bcbb758d47685a4febc7d9ad6ff96d1032d12ec2

trainer source
SHA256 9fe59edef699220269614768f9654a5bdeefa5ba2644edf99ba55a944686f15b

checkpoint_final_audit.json
SHA256 bccbca7e0806f0cbe66c80ada4143748ba0c74dbf70664805595335c9adf8225
```

独立final audit重新加载五个checkpoint并重放step0/final指标，确认：

- 五个checkpoint、6,250 updates、25,000 exposures及五次固定sequence SHA全部一致；
- optimizer仅含Atom head，vision冻结且状态未变；
- route-select只在step0和final运行；
- protocol=`passed`、capability=`failed`、decision=`NO-GO`；
- confirm400、fresh、sealed、USPTO和MR-TADF均未读取。

## Fixed-Final 只读断点诊断

在不改checkpoint、不改预注册Gate的前提下，独立audit额外运行了GT-count top-K诊断：

| 路径 | F1 @6px | whole-set exact |
|---|---:|---:|
| deploy count + learned ranking | 90.39% | 2/100 |
| GT count + learned ranking | 93.18% | 14/100 |
| GT count + canonical fixed-positive identity | 100% | 100/100 |

另有：

- all-dense top-128 coverage为`3072/3080 = 99.74%`；
- GT-count top-K canonical identity recall为`2581/3080 = 83.80%`；
- top-128 canonical positive identity recall为`99.286%`；
- canonical fixed-positive定位为`100%`，whole-set为`100/100`。

因此候选容量和anchor定位已经基本解决。当前第一瓶颈是presence/ranking：正确canonical slot
虽然几乎总在候选池中，却不能稳定排过第K边界negative。count是第二个独立瓶颈，但仅替换为
GT count仍只有`14/100`，不能解释全部失败。

共享trunk的只读梯度诊断进一步显示count CE与heatmap梯度cosine均值为`-0.0254`、中位数
`-0.0163`，25个batch中17个为负；即使count权重仅0.1，其梯度范数仍约为heatmap的
`1.197x`。这支持将count和selector拆开，而不是继续用同一Atom head联合训练。

下一实验固定为P1-B selector-only：冻结vision和P1-A整个Atom head，只在冻结dense hidden与
presence logits上增加zero-init轻量residual；使用GT count训练整组ranking，不使用count CE、
anchor loss或共享trunk梯度。P1-B若通过，再另立Count-only协议。P1-A自身不再继续训练。
