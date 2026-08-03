# G3 Adjacent Atom Capacity32 Manifest

> 日期：2026-08-03
>
> 状态：数据审计通过
>
> 性质：P1-T第二阶段容量门的固定训练子集

## 选择协议

不生成新图片，也不读取MR-TADF。只从固定train5k manifest读取GT Atom anchor：

1. 强制选择全部same-stride4-cell碰撞图；
2. 其余样本按整图最近Atom距离从小到大选择；
3. 相同距离按`sample_key`排序；
4. 固定取32图并记录源train5k、overfit32和结果manifest SHA。

## 结果

- 32张图，29个分子；
- train5k全部7张same-stride4-cell碰撞图均入选；
- 最近Atom距离范围`0.75px..6.7134428px`；
- 与原overfit32重叠2图：`train_000005_coordgen`、`train_000006_stress`；
- 样本序列SHA：`57d1133a4ca4766cc269cd26b1cbc38f98bc9489522e7cf9337ec2445c918ad0`；
- manifest SHA：`aa2ca0b1b671a3268c084169a4582052889806a3b4d12a5ad051cf519d4f8ef6`。

产物：

```text
outputs/audits/chemvector-g3-adjacent-atom-capacity32/manifest.jsonl
outputs/audits/chemvector-g3-adjacent-atom-capacity32/audit.json
```

下一训练必须从P1-T相同初始化重开，训练4,800 updates；不能从P1-T overfit checkpoint继续。
最终要求原overfit32与本专项集同时known/top128全Atom、全图100%，且7张碰撞图中的近邻Atom由
不同峰分别命中。任何一项失败都不得进入train5k。
