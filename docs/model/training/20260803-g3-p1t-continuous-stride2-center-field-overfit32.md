# G3 P1-T Continuous Stride-2 Center Field Overfit32

> 日期：2026-08-03
>
> 状态：第一阶段容量 GO
>
> 性质：连续高分辨率feature与最小视觉联合微调实验

## 模型

P1-T不再把四个PixelShuffle phase直接当作最终logit，而是先生成共享的连续高分辨率feature：

```text
训练路径 stroke [B,96,192,192]
 -> Conv1x1 96->128
 -> PixelShuffle(2)
 -> [B,32,384,384]
 -> GELU + Depthwise Conv3x3 + GELU
 -> Conv1x1 32->3
 -> center/offset residual
```

冻结副本的最后一个stride-4 ConvNeXt block专门产生P1-A base；可训练副本只为residual重编码
子格相位。因此训练不会让base间接漂移。最终center是冻结P1-A slot0双线性384 base与零初始化
连续residual之和。

可训练参数共`92,035`：decoder `12,835`，最后一个stride-4视觉block `79,200`。vision其余部分
和P1-A全部冻结。继续使用one-hot center、`8x` offset和local peak margin，不使用count、NMS、
背景补点或固定像素聚类。

## 协议修正

首次运行到step 450时发现：如果P1-A base与residual共享已解冻的视觉block，虽然P1-A参数冻结，
base feature仍会漂移。这次运行立即终止，产物保存在：

```text
outputs/smoke/chemvector-g3-p1t-invalid-shared-base-step450
```

它不计入训练结果。正式运行从相同seed重开，并用冻结block副本隔离base。最终协议确认冻结base
block、其余vision与P1-A均逐值不变，只有指定block和decoder更新。

## 结果

| 路径 | known matched/exact | top128 matched/exact | 决策 |
|---|---:|---:|---|
| P1-Q 双线性384残差 | 1399/1424，19/32 | 1422/1424，30/32 | NO-GO |
| P1-S phase直接作logit | 1369/1424，15/32 | 1420/1424，29/32 | NO-GO |
| **P1-T 连续384 feature** | **1424/1424，32/32** | **1424/1424，32/32** | **容量 GO** |

四种布局`coordgen/original/rdkit/stress`各自都是`356/356`、`8/8`。峰数不足为0，两次完整重放
的峰身份SHA和所有指标完全一致。固定2,400 updates、batch 4、9,600 exposures全部完成。

这个结果证明之前的瓶颈不是32图数据量，也不是Atom中心场路线先天不可行，而是表示错误：冻结
192 feature上的后处理不能产生可靠子格信息，phase直接作为logit又破坏连续场拓扑。先形成连续
384 feature并允许一个视觉block适配后，模型具备严格整组拟合能力。

注意：平均仍有约23,865个regional maxima；当前容量门依赖known-count/top128排序正确，这不等于
部署计数已解决。根据预注册协议，P1-T还不能进入5k。下一步建立`adjacent_atom_capacity_32`，
包含train5k全部7张same-stride4-cell碰撞图及最近邻困难样本；从相同初始化训练4,800 updates，
要求原overfit32与相邻集同时100%，且每对近邻由不同峰命中。

```text
decision = overfit_capacity_go
checkpoint = outputs/smoke/chemvector-g3-p1t-continuous-stride2-center-field-overfit32/step-002400.pt
SHA256 = 68774eab05873df289245b69da4fc9b6be95e0892f837397afd4bd245febeb52
```
