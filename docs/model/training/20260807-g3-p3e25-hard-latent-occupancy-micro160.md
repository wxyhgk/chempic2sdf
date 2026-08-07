# G3 P3-E25 Hard Latent Occupancy Micro160

> 日期：2026-08-07
>
> 状态：已完成，Micro Capacity NO-GO
>
> 模型：`chemvector-g3-p3e25-hard-latent-occupancy-micro160`

## 目标

P3-E24 证明旧query虽能读取192候选，却没有救回任何新增可达图片。本轮实现五路线中的第一条
`hard_latent_occupancy`：直接对192个candidate预测presence；每一步在GT兼容候选之间动态求
唯一一一匹配，不固定candidate ID；同一presence经Poisson-binomial得到Count。

本轮只做Micro Capacity。通过才允许扩到train3200/dev640；失败则记录NO-GO，不追加epoch、
不换loss、不运行SDF。

## 固定输入

- 候选：P3-E22 raw score前192项；
- normalized score：严格对`raw_scores[:192]`的logit按192项均值/标准差重新计算；
- scale rank：`0..191 / 191`；
- 冻结base/expert：P3-E7原checkpoint，128 query读取192 candidate；
- candidate evidence：41维；query/global evidence只提供冻结上下文；
- GT、mask、Count、split、variant、字体和scaffold不得进入model forward。

该协议命名为`true_global192`，不得复用P3-E23的top256-normalized shard。输入cache和源码在
训练前固定SHA。

## Micro160

只从P3-E21 train3200选择40个分子，每个保留四种variant，共160图：

- 五个Count桶1--20、21--40、41--60、61--80、81--100各8个分子；
- 四种variant在top192下都必须fully representable；
- 每桶先按固定SHA排序选择“至少一个variant在top128不可达、top192可达”的分子，再补普通组；
- 选择规则和sample-key digest在训练前写入audit，不根据step0或训练结果换样本。

这是容量集，训练和评估使用同一160图；不得把结果表述为泛化准确率。

## 模型与损失

复用`G3PoissonBinomialCount(h64)`的candidate occupancy结构，随机固定种子初始化。每图根据当前
presence logit，在6px GT-compatible矩阵中用Hungarian动态选择一组唯一positive candidate；
其余候选为negative。损失固定为：

```text
balanced_positive_negative_BCE
+ 0.1 * Poisson-binomial Count NLL
```

不训练base/expert、不加入OCR/Bond/SDF、不使用固定ID标签或坐标回归。训练100个完整epoch，
batch16，共1,000 optimizer steps、16,000 exposures；AdamW lr=3e-4、weight decay=0、
grad clip=5.0、BF16、fixed-final-only。presence末层weight固定零初始化、bias固定-1.0。禁止中间
checkpoint选模。Poisson-binomial内部虽产生
0--192分布，训练与解码只保留0--128并重新归一化，129--192不进入部署Count语义。

## 解码与门

Count取Poisson-binomial分布argmax；Atom取presence最高的Count个唯一candidate。另报真实Count
top-K以隔离Count与选择。固定门：

- predicted Count exact不低于152/160；
- predicted Count MAE不高于0.10；
- predicted Atom set exact不低于152/160；
- known-count Atom set exact不低于152/160；
- predicted Atom micro recall不低于99.9%；
- 每图动态positive最小概率减最大negative概率不低于0.8的图片不少于152/160，同时报告
  known-K真实top-K边界gap；
- 对本Micro集合中top128不可达、top192可达的图片，predicted Atom exact不低于50%；
- 五个Count桶和四种variant各自的Count exact与predicted Atom exact均不低于90%；
- Poisson分布有限且归一化，预测Count全部不超过128；
- 两次fixed-final评估逐项一致，训练步数/exposure/冻结状态全部一致。

全部通过为`hard_latent_occupancy_micro160_go`；否则NO-GO。无论GO/NO-GO，正式checkpoint、
训练结果和代码提交后创建与模型名完全一致的annotated tag，注释记录checkpoint SHA、160图、
1,000 steps、16,000 exposures和结论。

本轮不读取dev、旧confirm、confirm_v2、fresh_dev、USPTO、MR-TADF或sealed_test，也不运行
v0.10.10/v0.10.21 SDF；SDF只在Micro通过并完成下一阶段dev协议后运行。

## 实际结果

固定1,000步、16,000 exposures全部完成，重复评估和Count分布协议通过，但全部能力门失败：

| 指标 | 结果 | 门 |
|---|---:|---:|
| Count exact | 76/160 | ≥152 |
| Count MAE | 0.8188 | ≤0.10 |
| predicted Atom set exact | 36/160 | ≥152 |
| known-count Atom set exact | 47/160 | ≥152 |
| predicted Atom recall | 95.07% | ≥99.9% |
| gap≥0.8 | 29/160 | ≥152 |
| 新增可达 exact | 0/27 | ≥50% |

known-count micro precision/recall/F1均为95.67%，说明候选排序已学到大量局部信号，但整图集合仍
不成立。presence gap均值为-0.306、最小-0.994；hard assignment每步只把一个兼容替代峰设正，
其余兼容峰设负，等价候选之间持续发生相互矛盾监督。81--100桶predicted Atom exact为0/32，
也否定了在相同目标上继续追加epoch。

正式判定`hard_latent_occupancy_micro160_no_go`，不扩train3200、不运行dev/SDF。下一路线改为
soft coverage：所有GT兼容候选共享coverage目标，同时用兼容分量总质量约束为GT数，避免固定或
动态单一identity互相打架。

## 产物

- checkpoint：`outputs/checkpoints/chemvector-g3-p3e25-hard-latent-occupancy-micro160/step-001000.pt`；
- checkpoint SHA256：`43bd9e0128794ab3a48e4a699f30f4d04ec658cbff98dbeb6fadca8ca8f1435b`；
- audit SHA256：`2bea5768cd3ac9e83725032f1737bbe3b68c00098cc2b85a4efe02ff678642b4`；
- final metrics SHA256：`fff2096e55b23e856af204683318f45cb7f18ba98a41a0e3c9da24f0b57a04e9`；
- 训练耗时：47.63秒；
- dev、旧confirm、fresh和sealed均未读取。

结果提交后创建同名annotated tag：
`chemvector-g3-p3e25-hard-latent-occupancy-micro160`。
