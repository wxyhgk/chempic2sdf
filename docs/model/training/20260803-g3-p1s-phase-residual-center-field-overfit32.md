# G3 P1-S Phase Residual Center Field Overfit32

> 日期：2026-08-03
>
> 状态：容量 NO-GO
>
> 性质：P1-A强base加四相位残差的隔离实验

## 模型

P1-S修正P1-R丢失强base的问题。冻结的P1-A slot0 center先双线性放大到`384x384`；另一路从
冻结的`96x192x192` stroke feature生成四个固定空间phase的center/offset residual，并以严格
零初始化叠加到base。模型共`7,628`个可训练参数，step 0与P1-Q base逐值相同。

训练协议与P1-R完全一致：32张图、2,400 updates、9,600 exposures、one-hot center、`8x`
offset、local peak margin、CUDA BF16。vision和P1-A全部冻结，不使用count或固定像素聚类。

## 结果

| 路径 | known matched/exact | top128 matched/exact | 峰/图 |
|---|---:|---:|---:|
| P1-Q 双线性384残差 | **1399/1424，19/32** | **1422/1424，30/32** | 2536.34 |
| P1-R 随机四相位场 | 1222/1424，3/32 | 1394/1424，21/32 | 31955.13 |
| P1-S base+四相位残差 | 1369/1424，15/32 | 1420/1424，29/32 | 31794.81 |

所有协议检查通过，step-0 base exact、重复评估一致、冻结权重SHA不变。但四项容量条件均失败，
不进入5k。

P1-S证明问题不只是P1-R缺少base。即使保留P1-A排序，直接把PixelShuffle的四个phase作为最终
center logit，仍会在相邻phase之间形成棋盘式局部极大值，使峰数从P1-Q约2,536个暴增到约
31,795个。regional maxima依赖连续场的局部拓扑；phase通道直接重排到logit破坏了这个拓扑。

因此停止“phase即logit”的冻结小头路线。下一实验必须先学习连续的高分辨率feature，再在
384网格上用共享局部卷积输出单一center field；同时只解冻最后一个stride-4视觉block，使旧
骨干能为上采样器重编码子格相位。仍先过overfit32，未过不得进入5k。

```text
decision = phase_residual_capacity_no_go
checkpoint = outputs/smoke/chemvector-g3-p1s-phase-residual-center-field-overfit32/step-002400.pt
SHA256 = 04614822ca2c23da4deda6107d1b6e9a063c4f354f6f32f6c4fd0a8485dd6d6b
```
