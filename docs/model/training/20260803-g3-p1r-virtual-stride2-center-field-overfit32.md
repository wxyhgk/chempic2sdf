# G3 P1-R Virtual Stride-2 Center Field Overfit32

> 日期：2026-08-03
>
> 状态：容量 NO-GO
>
> 性质：冻结 stroke 特征的子格相位可恢复性实验

## 模型

P1-R 不再对192场做双线性放大，而是从冻结的`96x192x192` stroke feature直接预测四个
有固定空间身份的stride-2子格：

```text
stroke feature [B,96,192,192]
 -> Conv1x1 96->64
 -> GELU
 -> Depthwise Conv3x3
 -> GELU
 -> Conv1x1 64->12
 -> PixelShuffle(2)
 -> center + offset-x + offset-y @ 384x384
```

四个phase不是可交换slot；每个phase对应固定的`2x2`空间象限。模型共`7,628`个可训练参数，
vision和P1-A全部冻结。训练目标为one-hot center、`8x` offset和local peak margin；不使用
count、背景补点、NMS或固定像素聚类。

固定训练协议：32张图、2,400 updates、batch 4、9,600 exposures、CUDA BF16。严格容量门要求
known-count与top128同时达到`1424/1424`、`32/32`，且峰数不足为0。

## 结果

| 路径 | known matched/exact | top128 matched/exact | 峰/图 |
|---|---:|---:|---:|
| P1-P 192局部残差 | 1414/1424，25/32 | 1421/1424，30/32 | 1118.91 |
| P1-Q 双线性384残差 | 1399/1424，19/32 | **1422/1424，30/32** | 2536.34 |
| P1-R PixelShuffle384 | 1222/1424，3/32 | 1394/1424，21/32 | 31955.13 |

所有协议检查通过，所有参数张量均发生更新，vision和P1-A SHA保持不变，重复评估完全一致。
但四个容量条件全部失败，因此不进入5k。

P1-R的失败不能解释为“stride-2表示无效”。它从随机初始化重新学习全部Atom排序，没有继承
P1-A已经很强的slot0场；结果是空间分辨率提高了，但高分对象排序和峰形退化。约3.2万个局部峰
说明独立phase场在没有强base约束时产生大量低幅棋盘峰，known-count因此明显差于P1-Q。

下一步只允许做保守的P1-S混合残差：保留P1-A slot0双线性384场作为冻结base，PixelShuffle
分支只学习四相位center/offset残差并从零初始化。这样能单独回答“子格相位残差能否修复P1-Q
剩余2个不可达Atom”，不会再次丢掉已有覆盖能力。P1-S仍须先过相同overfit32容量门。

```text
decision = virtual_stride2_capacity_no_go
checkpoint = outputs/smoke/chemvector-g3-p1r-virtual-stride2-center-field-overfit32/step-002400.pt
SHA256 = fb7eba15f697f619c373c0f917a011209f8dc782d6183046ec02b77622e3f615
```
