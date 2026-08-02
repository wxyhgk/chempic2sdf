# G3 P1-D Boundary-Band Set Ranker Train5k

> 日期：2026-08-03
>
> 状态：已完成；独立审计协议通过，能力 `NO-GO`
>
> 性质：P1-C的loss-only因果实验

## 唯一假设

P1-C标签覆盖train/route分别为99.80%/99.74%，固定attention也带来known matched `+18`和
canonical identity `+41`，说明标签与特征并非完全无信号。但当前每图约31个positive时，
`top-4M`几乎纳入全部约97个negative，数千个easy pair会稀释决定top-K exact的边界错误。

P1-D只验证：保持模型和所有数据完全不变，仅聚焦最弱positive与最强negative，是否能显著
提高整组选择。

## 固定不变量

- 从P1-A fixed-final重新初始化，不加载P1-C权重；
- 完全复用P1-C的60,545参数单层attention、stable top-128和hybrid标签；
- train5k、route100、batch4、5 epochs、6,250 steps、BF16、AdamW `3e-4/0.01`全部不变；
- vision与P1-A Atom head冻结，GT count/anchor不进入ranker forward；
- route只在step0/final读取；confirm400、fresh、sealed、USPTO和MR-TADF继续禁读；
- 不增加数据、epoch、层数、margin、辅助loss或P1-C蒸馏。
- 允许复用P1-C的冻结P1-A加载器、candidate SHA累加器和route evaluator；其源码SHA必须固定并
  进入Gate，但禁止加载任何P1-C checkpoint、optimizer或set-ranker权重。

## 唯一Loss

每图从representable positives取当前分数最低`min(8,M)`个，从negatives取当前分数最高
`min(8,N)`个，计算最多64个pair：

```text
L_boundary = mean softplus(score_strong_negative - score_weak_positive)
```

margin=0、temperature=1、band固定8。禁止BCE/focal/count/anchor/entropy/regularizer及其他loss。

## 训练与输出

固定协议与P1-C一致：无shuffle、scheduler、warmup、resume、best或early stop；每1,250步保存，
只用fixed-final。输出目录：

```text
outputs/checkpoints/chemvector-g3-p1d-boundary-band-set-ranker-train5k
```

除原有epoch候选/标签SHA外，fixed-final后允许一次train5k只读诊断，不参与checkpoint选择：

- `boundary_exact_images`：最弱positive严格高于最强negative；
- hybrid-positive top-M recall与whole-set exact。

重复forward的bitwise一致性只报告、不设硬门。GT隔离硬门使用forward签名与AST无target/count、
K=1/GT/128下ranked scores不变，以及step0/final候选与count SHA稳定，避免再次把浮点核差异
误判为标签泄漏。

## Gate与判别

route能力Gate完全继承P1-C：known P/R/F1均`>=97%`、matched`>=2988/3080`、whole
`>=50/100`、canonical identity`>=95%`，并守住top-128 coverage与fixed localization。

附加诊断用于确定下一步，不改变GO门：

- route过门：loss稀释是主因，P1-D `GO`；
- train boundary exact `>=95%`但route提升`<1pp`或whole`<30`：训练可拟合但冻结feature/分子
  迁移不足，下一步转feature/data隔离；
- train boundary exact `<90%`：当前token连训练边界都不可分，优先feature sufficiency；
- train与route都只小涨：停止继续改ranking surrogate。

无论结果如何，不接前端、不生成SDF、不创建正式版本tag。

## 正式结果

唯一正式训练按预注册完成5 epochs、6,250 updates和25,000 exposures。训练更新耗时约
560.76秒，包含完整train5k fixed-final诊断与route100 final评测的总耗时约673.62秒。
五轮的candidate、anchor、feature、hybrid positive和canonical positive SHA完全一致；
train5k仍有`154911/155216 = 99.80%`个GT Atom可由冻结top-128表示。

route-select100结果：

| 指标 | step 0 | P1-C final | P1-D final |
|---|---:|---:|---:|
| known-count F1 | 93.18% | 93.77% | 93.31% |
| known-count matched | 2870/3080 | 2888/3080 | 2874/3080 |
| known-count whole-set | 14/100 | 17/100 | 17/100 |
| canonical identity recall | 83.80% | 85.13% | 84.81% |
| canonical identity whole-set | 3/100 | 4/100 | 4/100 |
| deploy F1 | 90.39% | 90.71% | 90.39% |
| deploy whole-set | 2/100 | 2/100 | 1/100 |
| top-128 coverage | 3072/3080 | 3072/3080 | 3072/3080 |

P1-D相对step0只增加4个matched Atom，且低于P1-C的18个增量。聚焦边界没有改善整组选择，
因此“P1-C主要被easy negative稀释”的唯一假设被否定。

完整train5k fixed-final诊断为：

| 指标 | 结果 |
|---|---:|
| hybrid positive | 154,911 |
| top-M matched positive | 137,369 |
| hybrid top-M recall | 88.6761% |
| top-M whole-set exact | 261/5000 = 5.22% |
| strict boundary exact | 260/5000 = 5.20% |

`5.20%`远低于预注册的`90%`判别线。这证明固定60,545参数、单层64维attention在当前预算与
冻结token上连训练边界都没有拟合；它不能单独证明冻结token在任意模型下不可分，因此下一步
必须先做ranker容量对照，不能直接归因于数据不足或擅自解冻backbone。

## 独立审计

独立审计重新加载五个checkpoint，核对optimizer、lineage、源码、完整epoch顺序与冻结集合
SHA，并在CUDA BF16下重新计算step0/final route100及完整train5k诊断。保存指标与重算结果
完全一致，step0/final的target-free重复forward、selection stable和score immutable均为
`25/25`。最终结论：

```text
protocol passed / capability failed / NO-GO
```

关键产物：

```text
step-006250.pt
SHA256 a8fd94c36434ecb977c406b574e3d2c43d48f98d512aae491bd85a6eb3afd97c

last.pt
SHA256 743f194509d62ff9386b08563224aa327572275e1327b8296660c861554197f2

checkpoint_final_audit.json
SHA256 381de688b728bc09a1fe9edff53f4426ca5c51ea29ee7ed99c636646fd741103

independent audit source
SHA256 709a58e0ef7e5bdd93d0ae422745069455e07a85219f91549e80d6fa662073aa
```

## 后续决策

停止继续修改ranking surrogate、追加P1-D epoch或扩大同类训练数据。下一轮先做冻结输入不变的
ranker容量A/B：在同一top-128、同一hybrid标签上比较当前小ranker与更强集合ranker的训练
可拟合性。只有更强ranker仍不能拟合时，才进入局部raster或多尺度backbone feature增量实验；
只有更强ranker能拟合但route不迁移时，才讨论过拟合与特征泛化。confirm400与所有外部数据
继续封存。

## Checkpoint训练轨迹复核

为排除固定顺序、无shuffle造成的末段遗忘，独立只读脚本将五个epoch checkpoint分别重放
完整train5k，并按manifest顺序固定切成五个连续1,000-image区块。该复核不参与checkpoint
选择，也没有读取route以外的任何数据。

| step | 总体 boundary exact | block 1 | block 2 | block 3 | block 4 | block 5 |
|---:|---:|---:|---:|---:|---:|---:|
| 1,250 | 3.52% | 2.1% | 2.8% | 2.8% | 4.2% | 5.7% |
| 2,500 | 4.54% | 3.1% | 3.4% | 3.8% | 5.0% | 7.4% |
| 3,750 | 4.86% | 3.4% | 3.8% | 3.8% | 5.6% | 7.7% |
| 5,000 | 5.08% | 3.3% | 3.9% | 3.8% | 5.8% | 8.6% |
| 6,250 | 5.20% | 3.8% | 4.1% | 4.2% | 5.5% | 8.4% |

五个区块难度存在稳定偏差，但没有至少两个前四区块从早期最佳到final下降`>=5pp`，最后区块
也没有形成`>=5pp`持续改善或相对前四区块中位数`>=10pp`的遗忘证据。因此固定stream顺序
不是P1-D失败的主要解释，允许进入P1-E容量/特征因果实验。

```text
train_checkpoint_trajectory.json
SHA256 b0b7a7ded1a626033fc4db2ad6648aec7003a81c35b658c5444deba9ffdb349d

trajectory source
SHA256 70cfe77eb64e6b19e3655399a090f3281626559760154e0d66d7f2327502d2f5
```
