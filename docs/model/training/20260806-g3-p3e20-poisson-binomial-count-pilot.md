# G3 P3-E20 Poisson-binomial Count Pilot

> 日期：2026-08-06
>
> 状态：完成，严格 NO-GO
>
> 输出：`outputs/smoke/chemvector-g3-p3e20-poisson-binomial-count-pilot2560-h64`

## 目标

P3-E19 证明区间重加权只能改善41–60，无法改善61–80。本轮不再把Count直接分类为129类，
改为对128个候选分别预测存在概率，再由Poisson-binomial动态规划得到0–128的总数分布。
该表示强制总数由局部对象贡献相加，验证是否能摆脱全局类别先验。

## 数据

- 只使用冻结的style90k evidence；fresh_dev、sealed_test、MR-TADF均不读取；
- 按molecule分组，四variant必须完整且只能进入一个split；
- 五个Count区间各选128个训练分子、64个开发分子；
- train：640分子、2,560图；dev：320分子、1,280图；
- 分子按固定seed hash排序，split在训练前固定。

## 模型与训练

- 每个candidate输出一个presence logit；
- candidate置换时presence等变，Count分布不变；
- 监督只有整图Count，不固定等价候选中的唯一正例；
- loss：Poisson-binomial Count NLL + 0.1相对期望Count误差；
- h64、AdamW、lr 1e-3、weight decay 0.01、clip 1.0；
- 30个完整epoch，1,200 step、76,800 exposure；
- fixed-final only，不选中间checkpoint。

## 门控

在同一1,280图dev上与P3-E16比较：

- pilot train exact至少50%；
- dev exact至少提高5pp，MAE至少下降0.25；
- 61–80与81–100各至少提高5pp；
- 1–20与21–40各自回退不超过5pp。

任一门失败即NO-GO，不进入style90k，也不读取fresh_dev。

## 结果

完成30个完整epoch、1,200 step、76,800 exposure；协议检查全部通过。模型可训练参数为
123,009。平均原始梯度范数18.06，固定clip 1.0全程处于强裁剪区间。

| 指标 | P3-E16 baseline | P3-E20 | 变化 |
|---|---:|---:|---:|
| dev exact | 28.36% | 29.22% | +0.86pp |
| dev MAE | 3.418 | 2.762 | -0.656 |
| dev 1–20 exact | 46.48% | 49.22% | +2.74pp |
| dev 21–40 exact | 49.22% | 48.83% | -0.39pp |
| dev 41–60 exact | 26.95% | 28.52% | +1.57pp |
| dev 61–80 exact | 8.59% | 11.33% | +2.74pp |
| dev 81–100 exact | 10.55% | 8.20% | -2.35pp |
| pilot train exact | - | 29.34% | 未过50%容量门 |

MAE明显改善，证明候选概率相加能学习数量尺度；但exact几乎没有提高，且训练集自身也未达到
容量门。仅用整图Count监督时，多个候选可以共同取中间概率，使期望总数接近标签，却不能把
每个候选极化为明确的存在/不存在对象。Poisson-binomial分布因此仍较宽，mode不够稳定。

## 决策

`poisson_binomial_count_pilot_no_go`。七个能力门中只有MAE改善与两个低Count保护门通过；不进入
style90k，不读取fresh_dev，也不追加相同Count-only epoch。

下一步保留可加和表示，但补充“等价候选集合下的动态一一匹配”局部监督：每次根据当前
occupancy分数在每个GT附近的等价候选中选择一组一一对应正例，而不是固定candidate ID。先在
小容量集验证presence是否能极化及Count exact是否随之提高，再决定是否扩量。

## 产物

- checkpoint：`outputs/smoke/chemvector-g3-p3e20-poisson-binomial-count-pilot2560-h64/step-001200.pt`；
- SHA256：`cd3cd23aea6ed765432f3dd222baa3fa858d3b3bf2d2cec14fdcdc94f83f91a9`；
- 训练时间：34.19秒；
- 本模型NO-GO，不替换P3-E16；
- 文档复核日期：2026-08-06。
