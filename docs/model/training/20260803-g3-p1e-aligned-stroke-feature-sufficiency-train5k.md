# G3 P1-E Aligned Stroke Feature Sufficiency Train5k

> 日期：2026-08-03
>
> 状态：已完成；独立审计协议通过，能力 `NO-GO`
>
> 性质：同参数三臂的ranker容量与候选对齐视觉因果实验

## 为什么不能继续换Loss或直接扩数据

P1-D在固定top-128和5,000张训练图上直接优化最弱positive对最强negative，fixed-final仍只有：

- hybrid top-M recall：`88.6761%`；
- whole-set exact：`261/5000 = 5.22%`；
- strict boundary exact：`260/5000 = 5.20%`。

五个checkpoint乘五个固定连续区块的完整重放没有发现达到阻断门槛的末段遗忘，因此当前失败
不能主要归因于无shuffle顺序。继续换ranking surrogate、追加epoch或增加同类图片都没有新的
因果信息。

## 当前Token的已确认缺口

P1-C/P1-D token只含：

```text
slot_features[384] + predicted anchor(x,y) + base presence logit
```

`slot_features`由共享cell hidden经过`1x1 feature_projection`后加slot embedding并做slot norm。
P1-A canonical loss只直接监督presence、anchor和count，不消费`slot_features`。只读tensor对比
确认以下五个参数从P0-I warmstart到P1-A fixed-final逐位完全相同：

| tensor | SHA256 |
|---|---|
| feature_projection.weight | `9a82b03fdd116a4265d51e408b66b5806bf0a178d8a5413d5e6c4eb2f64e4f32` |
| feature_projection.bias | `ddec9bd59a9aa528e3da9496801a34f81ed282a53a13688c1623899511f9ab04` |
| slot_embedding.weight | `c56a4718d724fae3a55c33675d16cc2f4da93515e6432c75c2f332b69d9ef07c` |
| slot_norm.weight | `702cc37cfd91cac6b67c8bd7cc3c9da19f7a52986a8435bc2d1269721a65ce78` |
| slot_norm.bias | `80422bc3d307b4a25bdafcc84ac7fb01cb55a09810e8b0f37bb12e0edb5c48ca` |

这不等于token绝对无信息：P1-A训练过的共享hidden仍经过该固定映射。P1-E必须同时隔离“原
60k ranker太小”和“缺候选对齐局部视觉”，不能只改其中一个后凭单臂结果下结论。

## 三个同参数实验臂

三个臂使用完全相同的初始化、参数量、候选、标签、loss、数据顺序和训练预算，只改变输入
patch与候选的对应关系：

1. `CAP`：每个候选输入全零patch，只测强ranker能否利用现有token；
2. `SHAM`：提取同图全部真实patch，但按`sample_key + epoch + seed`确定性无固定点置换；
   每个epoch更换置换，保留整图patch多重集合，破坏候选与局部笔画的对应；
3. `ALIGN`：每个候选使用自身predicted anchor对齐的真实patch。

patch只从冻结vision的`[B,96,192,192]` stroke feature map提取，以predicted anchor为中心，
`grid_sample`得到`9x9`区域。禁止读取GT anchor、target/count、candidate标签、元素、Bond、OCR
或SDF字段。SHAM置换必须对128个候选无固定点，且与ALIGN patch多重集合逐位相同。训练
epoch 1..5固定使用`seed + epoch = 20260714..20260718`，禁止五轮复用同一错配，避免强模型
把固定错配记成sample fingerprint。

patch坐标协议固定为`align_corners=False`、`padding_mode=border`。非方形coordinate-coded
真值测试必须证明anchor `((x+0.5)/W,(y+0.5)/H)`的patch中心精确落在`[y,x]`、水平和垂直
`+-1`落在对应相邻cell、x/y不颠倒，边界超出部分按border复制；该测试失败禁止训练。

## 固定模型

每臂唯一trainable模块为：

```text
current: LN(384, affine=False) -> Linear(384, 256)
geometry: MLP(3 -> 64 -> 256)  # predicted anchor x/y + frozen base logit
patch: Conv3x3(96 -> 64) -> GELU
       Conv3x3(64 -> 256) -> GELU -> spatial average

sum -> GELU -> LayerNorm(256)
4 x TransformerEncoderLayer(d=256, heads=8, FFN=512,
                            dropout=0, batch_first=True, norm_first=True)
Linear(256 -> 1), zero-init
ranked_score = frozen_base_logit + residual
```

每臂精确`2,427,713`参数。先构造一个master初始化，再用完整state load到另外两臂；三臂
initial canonical state SHA必须相同，但任何parameter/buffer的Python identity或storage
`data_ptr`不得共享。optimizer分别只含本臂ranker参数，且三个optimizer之间无共享参数。
CAP也必须执行同一个patch encoder但输入全零patch，禁止删模块导致参数量或优化动力学不同。
step0三臂score和selection必须逐位相同并复现P1-A stable-top128 baseline。

## 固定初始化、数据与禁读

初始化与冻结来源：

```text
P1-A fixed-final
outputs/checkpoints/chemvector-g3-p1a-canonical-atom-selector-train5k/last.pt
SHA256 bf0d35f31dafc467fafe3fd1bcbb758d47685a4febc7d9ad6ff96d1032d12ec2

vision
outputs/checkpoints/chemvector-v4-clean-50k/last.pt
SHA256 65a443d26bbffd41359604ac8147fa62a66ee18ca877693ebb3dea0b0a637095
```

- 训练：同一固定`train_5k.jsonl`，5,000 images / 1,250 molecules；
- 路线评估：同一固定`dev_route_select_100.jsonl`，只在共同step0与三臂fixed-final读取；
- 禁止加载P1-C/P1-D ranker或optimizer权重；
- confirm400、fresh dev、sealed test、USPTO、MR-TADF及其他外部数据继续禁读；
- vision与完整P1-A Atom head严格`eval()`、`requires_grad=False`，初末state SHA必须相同。

## 唯一Loss与训练预算

三臂都只使用P1-D固定boundary-band loss：每图最弱`min(8,M)`个positive乘最强
`min(8,N)`个negative，计算margin 0、temperature 1的pairwise softplus。禁止BCE、focal、
anchor、count、patch reconstruction、对比学习、蒸馏、辅助分类或任何Bond/OCR loss。

| 项目 | 每臂固定值 |
|---|---:|
| batch / workers / shuffle | `4 / 0 / false` |
| epochs / steps / exposures | `5 / 6250 / 25000` |
| optimizer | AdamW `lr=3e-4, wd=0.01` |
| clip / precision / seed | `1.0 / BF16 / 20260713` |
| scheduler / warmup / accumulation | none |
| checkpoint | 每1,250步，fixed-final only |

允许一次vision/Atom forward后依次计算三臂loss与backward，但每臂必须有独立optimizer、独立
gradient norm、独立checkpoint和完整epoch frozen-set/label SHA。禁止按中间loss或训练指标
停止、选best、追加epoch或只重跑其中一臂。

输出目录固定为：

```text
outputs/checkpoints/chemvector-g3-p1e-aligned-stroke-feature-sufficiency-train5k/
  cap/
  sham/
  align/
```

## Protocol Gate

能力指标生效前，以下硬门必须全部通过：

- 三臂各自参数量精确为`2,427,713`，initial state SHA完全相同，但parameter/buffer identity
  与storage均互不共享；
- 三个optimizer分别且只覆盖本臂全部参数，参数集合两两不相交；
- vision、P1-A Atom head的initial/final canonical state SHA不变；
- 每epoch train sample sequence SHA固定为
  `f336398a865d55a65b5efc2a3dd4f3b4e430434356a0fe59bd318ddfac9b6049`；
- top-128 candidate id/anchor/current feature、hybrid/canonical positive及原始ALIGN patch SHA
  跨三臂、跨epoch一致；
- 每个SHAM batch的patch multiset digest与ALIGN相同、128候选0 fixed point，epoch seed严格为
  `20260714..20260718`；
- 没有加载P1-C/P1-D ranker、checkpoint或optimizer；没有构造Bond/OCR/ChemScene/SDF模块；
- 每臂严格6,250 updates/25,000 exposures，只保存1,250/2,500/3,750/5,000/6,250五个
  checkpoint，final artifact与metrics同step；
- route100只stream共同step0一次和共同final一次，`route_evaluation_steps=[0,6250]`；final
  同一stream内计算CAP、SHAM五seed bank、ALIGN和ALIGN-as-SHAM五seed bank；
- step0/final count logits/vector不变，三臂ranked score不读取target/count，禁用数据无引用。

任一项失败均为protocol invalid，不能解释为模型能力`NO-GO`或`GO`。

## Fixed-Final评测

每臂fixed-final对完整train5k报告：

- hybrid top-M recall；
- hybrid top-M whole-set exact；
- strict boundary exact；
- candidate、anchor、feature、patch、positive identity SHA。

每臂fixed-final对route100报告原P1-D的known-count、deploy-count、canonical identity、top-128
coverage和fixed localization。SHAM final不使用训练epoch的任一置换，而固定使用以下五个
evaluation seed bank逐个重放train5k与route100：

```text
31001, 31013, 31019, 31033, 31039
```

必须报告每个seed、均值、最弱值与最强值；因果比较使用SHAM bank中表现最强的结果，避免用
一次偶然较差的错配夸大ALIGN收益。另将训练完成的ALIGN checkpoint在权重不变时输入同一
五seed SHAM bank，分别重放train5k与route100，作为patch-permutation反事实；反事实不得
参与选模。

## Gate与因果判读

每臂Train-Fit Gate：

```text
boundary exact >= 90%
hybrid top-M whole-set >= 90%
hybrid top-M recall >= 99.5%
```

route能力Gate继续固定为known P/R/F1均`>=97%`、matched`>=2988/3080`、whole
`>=50/100`、canonical identity recall`>=95%`、top-128 coverage至少`3072/3080`、fixed
positive localization严格`3080/3080`。

因果判读按以下互斥优先级执行，命中上项后不再套用下项：

1. CAP过Train-Fit：current token在强ranker下容量充分；原60k ranker表达不足，但不能声称
   aligned patch必要；
2. CAP不过、SHAM bank最强结果过Train-Fit：图像级patch集合信息充分，但不能声称候选局部
   对齐必要；
3. CAP与SHAM都不过，ALIGN过Train-Fit，且同权重ALIGN-as-SHAM bank最强结果相对ALIGN下降
   `>=15pp`，同时ALIGN相对独立训练SHAM bank最强结果也高`>=15pp`：candidate-aligned局部
   stroke信息是必要缺口；
4. 其他情况：`NO-GO / indeterminate`，只能按三臂中间指标设计新的独立假设，禁止把多个
   重叠规则包装成确定结论。

可额外报告各臂相对P1-D的提升，但`>=10pp`不再单独定义瓶颈归属。route GO只能作为迁移
证据，不能覆盖Train-Fit因果门或ALIGN反事实门。

无论结果如何，本实验不接前端、不生成SDF、不读取confirm/外部benchmark、不创建正式部署
版本tag。

## 训练前实现与复核

正式训练代码已完成，固定源码SHA为：

```text
trainer
3513a25b4c6cafed4d3b13f93194cb4ea31bfb16e99db5e61e8436fc78c8dd32

strong ranker / patch extractor
b4b60d1f6d2a1709df45e765dbb4b978ea77a6f3c715a8cf4a7de211cacf1374

boundary loss
7ea06a351b43d55987dda510974d27b85620d168c0d2df9ba46c0d623c9338c0

P1-C helper
845eabdd90bd386f77ca40744a27aa3b2f08f39978786a4d0f4f18efbeda478f

P1-D helper
f13cad9a1c4daeeeb8ae130a578ba2c54a6fb2a105d71408690f5d3b98461e87
```

trainer在读取数据前硬核对后四项依赖SHA；自身SHA写入audit和每个checkpoint，fixed-final后由
独立审计固定。实现与相邻回归共`33 passed`，覆盖：

- 三臂参数/state相同且parameter/buffer storage不共享；
- optimizer严格各臂独占；
- non-square、fractional、边界patch坐标真值；
- SHAM无固定点、每epoch换seed、patch multiset严格恢复；
- train/route五seed bank各只消费一次stream，route seed由train锁定；
- 互斥causal Gate、strict CLI和arm-local checkpoint payload。

RTX 5090 CUDA BF16单batch smoke通过：一次vision forward、三臂finite gradient、冻结Atom/
vision SHA不变、SHAM fixed point为0。smoke未读取manifest或checkpoint，不创建formal输出。

唯一正式训练命令：

```bash
.venv-train/bin/python -m \
  chemvector.training.atoms.train_g3_p1e_aligned_stroke_feature_sufficiency
```

## 正式结果

唯一正式训练按协议完成。每臂5 epochs、6,250 updates、25,000 exposures；纯训练约741.61
秒，包含完整train5k五seed bank、ALIGN反事实、route100和checkpoint写出的总耗时约904.03
秒。24项Protocol Gate全部通过，包括三臂隔离、五轮不同SHAM置换、跨臂冻结输入、五seed
final bank、train-locked route seed、冻结state和固定checkpoint schedule。

完整train5k fixed-final：

| Arm | top-M recall | whole-set exact | boundary exact |
|---|---:|---:|---:|
| P1-D baseline | 88.6761% | 5.22% | 5.20% |
| CAP | 84.8687% | 5.26% | 5.16% |
| SHAM best (`31033`) | 88.7600% | 5.64% | 5.58% |
| ALIGN | 88.3998% | 5.96% | 5.88% |
| ALIGN-as-SHAM best (`31001`) | 88.2074% | 5.58% | 5.50% |

SHAM五seed boundary exact只在`5.50%..5.58%`之间波动。ALIGN只比独立SHAM高`0.30pp`，
且将同一ALIGN权重输入SHAM patch后只下降`0.38pp`，远低于预注册的`15pp`因果门。CAP将
ranker从60,545参数扩大到2,427,713参数后，recall反而相对P1-D下降约3.81pp，未提高训练
可拟合性。

route-select100的known-count结果：

| Arm | matched / 3080 | F1 @6px | whole-set | canonical identity |
|---|---:|---:|---:|---:|
| P1-D baseline | 2874 | 93.31% | 17/100 | 84.81% |
| CAP | 2834 | 92.01% | 16/100 | 83.67% |
| SHAM train-locked | 2873 | 93.28% | 15/100 | 84.87% |
| ALIGN | 2870 | 93.18% | 15/100 | 84.77% |
| ALIGN-as-SHAM train-locked | 2870 | 93.18% | 16/100 | 84.84% |

top-128 coverage保持`3072/3080`，fixed-positive localization保持`3080/3080`。三臂route
Gate也全部失败，没有迁移增益。

正式结论：

```text
protocol passed / capability failed / NO-GO
causal_decision = no_go_indeterminate
```

这否定了两个具体而有限的假设：原60k ranker只是容量不足；`9x9` candidate-aligned
stride-4 patch是关键缺失信息。不能越界解释为视觉token完全无用、训练数据不足或整个G3路线
失败。

## 独立审计

独立审计重新加载三臂各五个checkpoint，核对arm-local payload/optimizer、epoch prefix、源码
与依赖SHA、last与final递归tensor相等，并在CUDA BF16下重新运行完整train5k五seed bank和
route100。重算JSON与保存JSON逐项相同，结论不变。

```text
CAP final checkpoint
SHA256 5c31296978672f35af4bd2a60a6d219f1ff5b1c1be230c73097c761c046dc393

SHAM final checkpoint
SHA256 573d45254f80eff116fce2ceeea8d003b519ecae85d65753e5e0a54d2cf75039

ALIGN final checkpoint
SHA256 445d75917ee44ebceeef752084d66c5b6a5234d3c48d965e877d05b403429aa4

checkpoint_final_audit.json
SHA256 71df145aed60de12bfaae6ce0e989ebe08fa10e59fad8c35a0ad126f88fb743c

independent audit source
SHA256 fd5e4665ae8a76c33e133bc6db7922862e1ec0d80bd0ee51c3afdce0a5d534ac
```

非阻断加固项：未来协议应把P1-A stream/evaluator、top128 target component和图模型helper也
纳入源码闭包硬锁；直接在Gate比较route step0/final六项冻结输入SHA；把顶层causal status与
route migration status分开命名。本次对应冻结输入独立比较均相等，不影响结论。

## 后续决策

停止扩大同类ranker、继续加入stride-4局部patch、换ranking surrogate或追加相同预算。下一步
先只读审计监督身份：比较固定hybrid-positive exact与允许同一GT 6px内候选等价置换后的几何
集合exact，并统计每个GT附近候选多重度。若大量“固定身份错、几何集合对”，则当前loss在
惩罚视觉等价候选，应转GT集合匹配/等价组监督；若两者同步失败，再研究更高分辨率对象证据。
