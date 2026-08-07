# G3 P3-E26 Soft Coverage Occupancy Micro160

> 日期：2026-08-07
>
> 状态：已完成，Micro Capacity NO-GO
>
> 模型：`chemvector-g3-p3e26-soft-coverage-occupancy-micro160`

## 因果目标

P3-E25 的hard latent occupancy在训练Micro上known-count F1达到95.67%，但Atom exact仅47/160、
presence gap均值为负。原因是每步Hungarian只选一个兼容峰为正，其余兼容替代峰被当负，动态
identity产生相互矛盾监督。

本轮保持模型、true-global192 evidence、Micro160、初始化和训练预算不变，只把目标改为
identity-invariant soft coverage，验证等价候选监督是否是P3-E25的因果断点。

## 固定数据与模型

- 完全复用P3-E25的40分子/160图sample-key digest与五Count桶分布；
- 从P3-E22 raw前192重算true-global192 normalized score和rank/191；
- 冻结同一base/expert并重建41维candidate evidence；
- `G3PoissonBinomialCount(h64)`随机同种子初始化，presence末层weight=0、bias=-1；
- batch16、100 epoch、1,000 steps、16,000 exposures；
- AdamW lr=3e-4、weight decay=0、clip5、BF16、fixed-final-only。

不得读取dev、旧confirm、fresh或sealed，不运行SDF。

## Soft Coverage 损失

每图构造6px GT×candidate compatibility二部图，并按共享候选求连通分量。设candidate presence
为`p`：

1. 每个GT coverage：`1 - product(1 - p_j)`，对负对数取均值；
2. 每个连通分量的`sum(p_j)`应等于分量GT数，使用相对L1质量误差；
3. 与任何GT都不兼容的candidate使用negative BCE；
4. 同一presence的0--128截断Poisson Count使用NLL。

固定总损失：

```text
coverage_nll
+ 0.5 * component_relative_mass_l1
+ 0.5 * outside_negative_bce
+ 0.1 * poisson_count_nll
```

不生成hard identity标签，不缓存assignment，不把所有兼容candidate逐个标正。

## 固定门

解码仍为Poisson mode Count + presence稳定top-K。能力门与P3-E25保持可比：

- Count exact≥152/160且MAE≤0.10；
- predicted/known-count Atom set exact各≥152/160；
- predicted Atom recall≥99.9%；
- 新增可达图片predicted Atom exact≥50%；
- 五Count桶、四variant的Count/Atom exact各≥90%；
- 至少152/160图满足：最弱GT coverage≥0.95且最大分量相对质量误差≤0.10；
- Count分布有限归一化、Count≤128、两次评估一致、训练步数与冻结状态一致。

全部通过为`soft_coverage_occupancy_micro160_go`，否则NO-GO。失败不追加epoch，不改权重，不扩量；
无论结果如何都提交checkpoint与结果，并创建与模型名完全一致的annotated tag。

## Pre-step 实现事故

首次启动在任何optimizer step之前因CUDA不支持`int16 addmm`停止，只写出初始audit，无训练日志或
checkpoint。失败目录保留为
`outputs/checkpoints/chemvector-g3-p3e26-soft-coverage-occupancy-micro160-failed-prestep-addmm-short`。
修复仅将无梯度compatibility邻接乘法从int16改为FP32，并新增CUDA回归测试；数据、随机种子、
损失、超参、步数和门槛均未改变，因此按同一预注册协议从step0重启。

## 实际结果

修复后固定1,000步完整运行。相对P3-E25 hard路线，known-count Atom exact从47提高到49、F1
从95.67%提高到95.86%，但未形成整图容量：

| 指标 | P3-E25 hard | P3-E26 soft | 门 |
|---|---:|---:|---:|
| Count exact | 76/160 | 69/160 | ≥152 |
| Count MAE | 0.8188 | 0.8375 | ≤0.10 |
| predicted Atom exact | 36/160 | 38/160 | ≥152 |
| known-count Atom exact | 47/160 | 49/160 | ≥152 |
| predicted recall | 95.07% | 95.37% | ≥99.9% |
| 新增可达 exact | 0/27 | 0/27 | ≥50% |

coverage≥0.95且分量质量误差≤0.10只有28/160；最弱GT coverage均值0.328，最大分量相对质量
误差均值0.766。soft目标只带来2张known-count exact，仍没有正确tail hit。说明当前冻结
base/expert evidence上的独立candidate occupancy头不是足够的Micro表示，不能靠继续调整同一头的
identity loss解决。

正式判定`soft_coverage_occupancy_micro160_no_go`，不扩量、不运行dev/SDF。下一路线进入
`query_slot_occupancy`：解冻并训练128 query到192 candidate的pointer/存在性本体，使tail进入
可学习memory和唯一query分配，而不是继续在冻结evidence上叠presence头。

## 产物

- checkpoint SHA256：`8c2520a09fd432b97fa39051c73352e920eda8d2bb13c5a3a99f70c36da4cbb4`；
- audit SHA256：`4668453ca81a5c9e4f75ef0ad44d8bcb0d8e58ccccfe7194e43b92d017e5dad8`；
- final metrics SHA256：`b5fd8110ce4c0ac3d70d8dce5fc20c8bb9dd7a162c53800f123f19c863a04425`；
- 训练耗时：357.18秒；
- dev、旧confirm、fresh和sealed均未读取。

结果提交后创建同名annotated tag：
`chemvector-g3-p3e26-soft-coverage-occupancy-micro160`。
