# G3 P1-U Adjacent Atom Capacity64

> 日期：2026-08-04
>
> 状态：协议通过 / 双域容量 NO-GO
>
> 性质：连续stride-2中心场的相邻Atom专项容量门

## 训练协议

P1-U从P1-T相同seed和相同P1-A/vision checkpoint重新初始化，不加载P1-T训练后的模型权重。
启动时逐项比较field、vision、最后一个stride-4 block和冻结base block的初始化SHA，四项完全一致。

每个epoch严格交替：

```text
overfit32 batch
adjacent32 batch
... 共16 batch、64次域曝光
```

两个集合重叠2张图，因此每个epoch是64次域曝光、62张唯一图片。固定训练4,800 updates、300个
epoch、19,200 exposures。模型仍是P1-T的92,035个可训练参数，不使用固定像素聚类或背景补点。

专项manifest最初按困难度直接写入，启动前被WebDataset源顺序预检拒绝，未发生训练。修正为
“按困难度选样、按train5k源顺序写入”后，完整epoch预检通过。正式manifest SHA：
`6e4eca6fcace2e9e20a29c0414972cc5b967d71da1df9f7e31034c083f07442f`。

## 结果

| 域 | known matched/exact | top128 matched/exact | 决策 |
|---|---:|---:|---|
| 原overfit32 | **1424/1424，32/32** | **1424/1424，32/32** | 保持100% |
| adjacent32 | 1461/1473，20/32 | 1468/1473，27/32 | 未过门 |

7张stride-4同cell碰撞图：

- known-count近邻对使用不同峰：`0/7`；
- top128近邻对使用不同峰：`4/7`；
- 峰数不足：`0`。

全部15项协议检查通过：初始化与P1-T一致、训练参数边界正确、冻结权重不变、4,800步、19,200
曝光、300个epoch、交替顺序和两次完整重放均一致。正式决策为`dual_capacity_no_go`，不进入5k。

## 表示根因

训练结束loss已降到约`0.012`，但整组指标仍失败。固定专项集的one-center target可表达性审计为：

| 网格 | 碰撞图片 | target丢失Atom |
|---|---:|---:|
| 192（stride 4） | 7 | 7 |
| 384（stride 2） | 3 | 3 |
| 768（stride 1） | **0** | **0** |

当前target builder在同一cell有多个Atom时只保留一个正center和词典序最小Atom的offset。因此384
网格中的3张图片在监督层已经丢失对象，增加epoch、调loss或扩大同一head都不能解决。正式审计：

```text
outputs/audits/chemvector-g3-p1u-target-resolution/audit.json
decision = stride1_required_for_lossless_one_center_targets
```

下一步P1-V只测试连续stride-1（768）中心场：保留P1-T双路径冻结base与最小视觉block解冻，
在连续384 feature之后再学习一次feature上采样，最后统一卷积成768 center/offset。仍从相同初始化
训练双域容量门；未达到两个域和7个近邻对全部100%，不得进入5k。

```text
decision = dual_capacity_no_go
checkpoint = outputs/smoke/chemvector-g3-p1u-adjacent-atom-capacity64/step-004800.pt
SHA256 = 0da73a5900c90035be8438e01b872c1fd818a797d03715228e557723966c2c5f
```
