# G3 P3-E20 Poisson-binomial Count Pilot

> 日期：2026-08-06
>
> 状态：预注册，待运行
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

待运行后填写。

## 决策

待运行后填写。
