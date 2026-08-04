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
4. 选择完成后恢复train5k源顺序，满足WebDataset确定性流式加载；
5. 固定取32图并记录源train5k、overfit32和结果manifest SHA。

## 结果

- 32张图，29个分子；
- train5k全部7张same-stride4-cell碰撞图均入选；
- 最近Atom距离范围`0.75px..6.7134428px`；
- 与原overfit32重叠2图：`train_000005_coordgen`、`train_000006_stress`；
- 样本序列SHA：`6bb2e770590b486e4dec72282f6576520d2099477c37867a95bc04089a2cb803`；
- manifest SHA：`6e4eca6fcace2e9e20a29c0414972cc5b967d71da1df9f7e31034c083f07442f`。

最初按困难度直接写manifest的版本在P1-U启动前流式预检中被拒绝，未发生训练。该无效顺序审计
保留在`outputs/audits/chemvector-g3-adjacent-atom-capacity32-invalid-difficulty-order`，不属于正式
协议。

产物：

```text
outputs/audits/chemvector-g3-adjacent-atom-capacity32/manifest.jsonl
outputs/audits/chemvector-g3-adjacent-atom-capacity32/audit.json
```

下一训练必须从P1-T相同初始化重开，训练4,800 updates；不能从P1-T overfit checkpoint继续。
最终要求原overfit32与本专项集同时known/top128全Atom、全图100%，且7张碰撞图中的近邻Atom由
不同峰分别命中。任何一项失败都不得进入train5k。
