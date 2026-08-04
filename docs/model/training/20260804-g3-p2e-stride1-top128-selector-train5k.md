# G3 P2-E Stride-1 Top128 Selector Train5k

> 日期：2026-08-04
>
> 状态：协议通过 / route因果NO-GO / 停止该loss

## 输入与上限

P2-D固定P2-A stride-1 top128候选：

| 域 | 候选可达matched | exact |
|---|---:|---:|
| train5k | 155081/155216 | 4937/5000 |
| route100 | 3073/3080 | 98/100 |

缓存同时保存完整GT mask与最大基数representable mask。训练只对可达GT做动态一对一匹配，最终
端到端指标仍使用完整GT和完整count，不删除困难标签。

## 训练协议

使用P1-X的105,089参数、3层集合selector，fresh初始化；train5k固定5 epochs、6250 updates、
25000 exposures。中心场与候选缓存完全冻结，route只读step0/final。预注册因果门为route recall
至少+0.5pp且exact至少+10；能力门为99% recall、85/100 exact。

## 结果

| 指标 | step0 | final | 变化 |
|---|---:|---:|---:|
| train full matched | 152423/155216 | 152789/155216 | +366 |
| train full exact | 3344/5000 | 3492/5000 | +148 |
| route full matched | 3022/3080 | 3018/3080 | -4 |
| route full exact | 63/100 | 61/100 | -2 |

模型在训练域有效，但route同步退化，属于分布过拟合。五个checkpoint的route曲线为：

```text
step0   3022 / 63
1250    3020 / 64
2500    3022 / 63
3750    3021 / 64
5000    3018 / 61
6250    3018 / 61
```

没有任何checkpoint同时提高matched和exact，因此不是简单早停问题。停止动态硬匹配加
boundary-band selector loss，不追加相同epoch。下一轮优先扩大/改进中心场数据和目标，因为P1-Z、
P2-A已经提供了强且一致的正向证据。

```text
model = chemvector-g3-p2e-stride1-top128-selector-train5k
checkpoint = outputs/checkpoints/chemvector-g3-p2e-stride1-top128-selector-train5k/step-006250.pt
SHA256 = 306b34bc5d710276015e5691a7aa42dffe0a0fabf86db49a13500613d94a53b2
final metrics SHA256 = 95934baec2262db84ee34a7da61e56608450612a2394a2c5c01bfe28dcb976d2
```
