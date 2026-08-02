# G3 P1-C Top128 Global Set Ranker Train5k

> 日期：2026-08-02
>
> 状态：已完成；独立审计协议通过，能力 `NO-GO`
>
> 性质：固定候选集合上的全局选择因果实验；不是count、Bond或外部泛化实验

## 唯一问题

P1-B冻结dense hidden上的逐像素residual只使known-count F1 `93.18%→93.60%`、whole-set
`14→16/100`，即使residual绝对值达到18.25也只找回13个Atom。与此同时P1-A top-128
候选对GT Atom的6px geometry coverage仍为`3072/3080 = 99.74%`。

P1-C只回答：固定P1-A top-128成员后，让候选彼此交换一次信息，能否识别局部重复/竞争并
完成整组选择？它不接P1-B权重，不训练dense head，也不处理count。

## 数据、初始化与禁读

完全继承P1-B的train5k、route-select100、manifest SHA和固定顺序；P1-A fixed-final仍为唯一
初始化，vision和完整Atom head严格冻结。confirm400、fresh、sealed、USPTO、MR-TADF及其他
外部数据继续禁读。route-select100只允许step0/final两次只读评测。

## 固定Top-128身份

1. 从冻结P1-A base presence的110,592个dense slots确定性取128个成员；
2. BF16同分时先保留严格高于第128阈值的候选，再按flat id升序补齐等分候选；
3. `flat_id = slot*H*W + y*W + x`；取完后按flat id升序排列token；
4. residual不得改变成员、refill候选或读取base rank；
5. 每epoch审计每图128个flat id、完整候选序列SHA和冻结anchor/feature SHA。

## 唯一模型

每个token只含冻结的`slot_features[384]`、`anchor(x,y)`和base presence logit：

```text
LayerNorm(384, affine=False) -> Linear(384, 64)
MLP(3 -> 32 -> 64)  # anchor x/y + base logit
两支相加 -> GELU -> LayerNorm(64)
1 x pre-norm TransformerEncoderLayer(d=64, heads=4, FFN=128, dropout=0)
Linear(64 -> 1) zero-init
ranked_score = base_score.detach() + residual
```

无global/count token、sequence position embedding、candidate rank、第二层attention、元素、Bond
或OCR输入。anchor是token内容，模型必须对token排列permutation-equivariant。最终参数量已由
实现测试固定为`60,545`，训练前必须再次写入协议审计；若发生变化，禁止训练。

## 固定Hybrid Positive身份

监督在ranker forward完成后构建，且整个训练期间不变：

1. top-128内存在canonical fixed slot时，优先锁定为对应GT positive；
2. 对canonical slot缺失的剩余GT，在剩余候选上做冻结anchor的6px
   maximum-cardinality-first、minimum-distance匹配；
3. 同代价按flat id、canonical GT order破tie；
4. 不可表示GT不伪造positive、不refill；同一GT附近其余重复候选均为negative；
5. 每epoch记录positive flat-id SHA，必须逐轮一致。

该匹配只依赖冻结候选与GT，不读取ranker score，因此不同于moving prediction-conditioned
Hungarian。还必须分别报告canonical positive和geometry-rescue positive数量。

## 唯一Loss

每图representable positive数为`M`；从其余候选取当前最高
`min(4M, 128-M)`个hard negatives，对全部positive/negative pair计算：

```text
L = mean softplus(score_negative - score_positive)
```

margin=0、temperature=1。禁止BCE/focal、count CE、anchor、dense canonical、entropy、residual
正则、蒸馏及任何Bond/OCR loss。GT总count不参与loss，只在完整forward后用于known-count
top-K评测。`M=0`直接拒绝。

## 固定训练协议

| 项目 | 固定值 |
|---|---:|
| train | `5000 images / 1250 molecules` |
| batch / workers / shuffle | `4 / 0 / false` |
| epochs / steps / exposures | `5 / 6250 / 25000` |
| optimizer | AdamW `lr=3e-4, wd=0.01` |
| clip / precision / seed | `1.0 / BF16 / 20260713` |
| scheduler / warmup / accumulation | `none` |
| checkpoint | 每`1250`步，fixed-final only |
| route evaluation | step `0`与`6250` only |

不缓存整集pixel/vision/dense/candidate/label，不resume optimizer，不选择best，不追加epoch。输出：

```text
outputs/checkpoints/chemvector-g3-p1c-top128-global-set-ranker-train5k
```

## Gate

协议Gate要求：P1-A/vision SHA与state不变；optimizer仅含set ranker；candidate flat-id与hybrid
positive SHA跨epoch一致；step0 zero-init scores与P1-A逐位一致，stable tie-break选择作为P1-C
独立冻结baseline报告，不要求与旧CUDA `topk`同分顺序一致；K=1/GT/128不改变ranked
scores；targets置换不改变forward bytes；count logits/vector不变；五epoch、checkpoint与route
时点严格成立；禁用组件和数据未读取。

能力Gate：

| 指标 | GO门槛 |
|---|---:|
| known-count P/R/F1 @6px | 均`>=97%` |
| known-count matched | `>=2988/3080` |
| known-count whole-set | `>=50/100` |
| canonical selected identity recall | `>=95%` |
| fixed top-128 geometry coverage | `>=3072/3080` |
| canonical fixed-positive localization | `3080/3080` |

deploy count exact必须仍为13/100、MAE 3.83，只用于隔离证明。任一主门失败即`NO-GO`，禁止
追加attention层、epoch或改为DeepSets补跑。GO后才另立Count-only；confirm400继续封存。

## 结论边界

P1-C最多证明train distribution上固定top-128的全局集合选择可行，不能宣称外部泛化、完整
图、SDF或部署能力。本实验不接前端、不生成SDF、不创建正式版本tag。

## 训练前实现复核

正式训练前交叉审查发现并修正两项协议问题：

1. 初版组件漏掉“两支相加后的GELU”，现已恢复为
   `LayerNorm(GELU(feature_branch + geometry_branch))`，并新增forward hook结构测试；
2. P1-C stable flat-id tie-break与旧CUDA `topk`在完全同分时不保证选择相同，因此step0只要求
   base/ranked scores逐位相同；P1-C stable选择的step0指标独立冻结并报告，不再错误地强制等于
   旧topk的2870/14等数值。

hybrid matcher和epoch SHA审计已改为每batch一次性搬到CPU，避免逐原子GPU同步。真实train5k
首batch的5-step CUDA BF16 smoke为`0.1203秒/step`、峰值显存约`2.74 GB`，当前batch的
representable/target为`92/92`。该smoke不读取route、不保存checkpoint、不参与模型选择。

## 正式结果

唯一正式训练完成5 epochs、6,250 updates和25,000 exposures，总耗时约563.15秒。五轮的
candidate、anchor、feature、hybrid positive和canonical positive SHA完全一致；train5k的
representable positive为`154911/155216 = 99.80%`，其中canonical `154507`、geometry rescue
`404`。

| 指标 | step 0 | step 6,250 | 变化 |
|---|---:|---:|---:|
| known-count F1 | 93.18% | 93.77% | +0.58pp |
| known-count matched | 2870/3080 | 2888/3080 | +18 |
| known-count whole-set | 14/100 | 17/100 | +3 |
| canonical identity recall | 83.80% | 85.13% | +1.33pp |
| canonical identity whole-set | 3/100 | 4/100 | +1 |
| deploy F1 | 90.39% | 90.71% | +0.32pp |
| deploy whole-set | 2/100 | 2/100 | 0 |
| top-128 coverage | 3072/3080 | 3072/3080 | 0 |

attention确实比P1-B pointwise residual学到更多可迁移排序信号，但仍远低于预注册能力门。
final residual绝对值最大为`11.9375`，不能归因于zero-init未学动。

训练时保存的Gate曾因一次`target_free_forward` bitwise检查为false而标记protocol failed。独立
审计确认该检查比较了不同autocast上下文，不能作为GT泄漏证据；审计在相同精度下对step0和
final各重放25个batch，bitwise/allclose/selection stable/score immutable均为`25/25`、最大差
为0，同时AST与签名确认forward只接受`candidates`且无target/count标识。因此最终纠正结论为：

```text
protocol passed / capability failed / NO-GO
```

关键产物：

```text
step-006250.pt
SHA256 42b108e9f7d81c5440882a0aaa750ac8945dcbdf0840922c0939fa79e0b1c39b

checkpoint_final_audit.json
SHA256 07df120def248e825c3a34d8f7ad6d606a8a0c9a1e02a492d8124f9879c04e0d

independent audit source
SHA256 8dbe9df7e00a0d608a00e354ae7a5a2e429f43b9167dd6b007cf97d456a35cf7
```

## 后续决策

当前`min(4M,128-M)`在平均`M≈31`时几乎选入全部negative，再对数千个pair求均值，容易让
easy pair稀释真正决定top-K whole-set的边界错误。P1-D只替换为固定8×8 boundary-band loss；
模型、初始化、候选、数据和训练预算全部不变，不追加attention层或epoch。
