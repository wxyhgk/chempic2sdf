# G3 P1-Q High Resolution Center Field Overfit32

> 日期：2026-08-03
>
> 状态：容量 NO-GO
>
> 性质：384x384中心场的冻结特征充分性实验

## 模型

将P1-A slot0 center双线性上采样到`384x384`作为冻结base。8,643参数adapter先在192网格做
`256->32`投影，再双线性上采样、depthwise `3x3`局部处理并输出center/offset残差。最终每个
field cell对应约2px，不使用固定像素聚类。

训练继续使用P1-P的one-hot center、offset和local peak margin，固定2,400 updates。P1-Q只回答：
双线性上采样冻结192特征，是否足够恢复相邻Atom？

## 结果

| 路径 | known matched/exact | top128 matched/exact | 峰/图 |
|---|---:|---:|---:|
| P1-P 192场 | 1414/1424，25/32 | 1421/1424，30/32 | 1118.91 |
| P1-Q 384场 | 1399/1424，19/32 | **1422/1424，30/32** | 2536.34 |

384场多恢复一个top128 Atom，但known排序显著回退，exact仍停在30/32。双线性放大没有创造
新的高分辨率视觉信息，adapter只能在插值特征上制造更多低分峰。方案容量NO-GO，不进入5k。

下一步不能继续扩大同一adapter。需要二选一：

1. 从stride-2视觉特征直接预测384中心场；
2. 使用可学习上采样并解冻P1-A/vision最后一个高分辨率block，重新联合训练连续heatmap。

两条路线都必须保留P1-N的regional-max对象协议、无背景补点和无固定像素聚类原则。

```text
decision = local_residual_capacity_no_go
checkpoint = outputs/smoke/chemvector-g3-p1q-high-resolution-center-field-overfit32/step-002400.pt
SHA256 = 55b67b15e2dfa3d1c986dc929774b9953490866410add2a66d75116d22bd7e28
```
