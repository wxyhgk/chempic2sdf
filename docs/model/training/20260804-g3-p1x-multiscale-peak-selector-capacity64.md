# G3 P1-X Multiscale Peak Selector Capacity64

> 日期：2026-08-04
>
> 状态：协议通过 / 双域selector容量GO
>
> 性质：固定P1-W候选上的known-count集合选择容量门

## 目标

P1-W已经证明下面这个固定、target-free的候选协议能覆盖两个32图域的全部Atom：

```text
P1-U stride-2 top96 + P1-V stride-1 top32 = 128 candidates
```

P1-X只回答一个问题：不做像素半径聚类、不固定某个candidate ID为唯一正例，一个小型集合模型
能否从这128个高召回峰中整组选出全部真实Atom？只有两个域都达到全Atom、`32/32`，才允许进入
5k泛化训练。

## 模型与输入

冻结P1-U、P1-V、P1-A和视觉骨干。每张图只在启动时重放一次两个中心场，然后缓存128个
候选。每个候选token包含：

- anchor `x/y`；
- stride-2或stride-1尺度标识；
- 尺度内rank；
- 原始中心分数和尺度内标准化分数；
- 对应中心cell的32维连续特征。

selector是3层、64维、4头的集合Transformer，共`105,089`个可训练参数。它对candidate排列
等变，输出每个候选的ranking residual。known count不进入模型，只在评估时对完整排序截取前N项。

## 动态一对一监督

同一个GT附近常有多个兼容峰，因此训练不使用固定candidate identity。每一步在6px兼容图上做：

1. 最大基数优先，必须覆盖每一个GT；
2. 一个候选最多分配给一个GT，一个GT只分配一个候选；
3. 完整匹配之间优先当前selector分数更高的候选；
4. 动态匹配的M个候选为positive，其余为negative；
5. 8x8 boundary-band loss直接拉开最弱positive与最强negative。

候选集合在训练前先独立做一对一可达性复核。任何图少一个GT都立即停止，不允许通过GT驱动的
候选合并、NMS或固定像素半径聚类修补。

## 固定训练协议

- overfit32与adjacent32交替，每域32张；
- batch 4，300 epochs，4,800 updates，19,200 exposures；
- AdamW，学习率`3e-4`，weight decay`0.01`，gradient clip`1.0`；
- CUDA BF16，固定seed，关闭TF32并启用确定性算法；
- 只训练105,089参数selector；
- 不读取Bond、OCR、ChemScene、SDF、USPTO或MR-TADF。

## 结果

| 域 | step 0 matched/exact | final matched/exact |
|---|---:|---:|
| overfit32 | 1199/1424，0/32 | **1424/1424，32/32** |
| adjacent32 | 1225/1473，0/32 | **1473/1473，32/32** |

两个域final precision、recall、F1均为100%。全部8项训练协议检查与4项能力检查通过：

```text
decision = dual_domain_selector_capacity_go
checkpoint = outputs/smoke/chemvector-g3-p1x-multiscale-peak-selector-capacity64/step-004800.pt
SHA256 = 630431b4d9aa0cc0620f9ae0e3e1a0b12587bafda8c00247b9b1b227b290052b
```

## 独立审计

独立进程重新加载checkpoint，重建两尺度候选缓存，并复算两域指标。首次重放没有继承训练时的
确定性CUDA/TF32开关：几何指标仍为100%，但等价峰identity和候选特征哈希不同，因此严格审计
按协议判失败。该记录保留在：

```text
outputs/audits/chemvector-g3-p1x-multiscale-peak-selector-capacity64-replay-without-determinism/
```

补齐确定性协议后，候选缓存、选择identity、保存指标与重复重放全部逐项一致：

```text
decision = audit_passed_dual_domain_selector_capacity_go
audit = outputs/audits/chemvector-g3-p1x-multiscale-peak-selector-capacity64-final/audit.json
```

## 结论与下一步

P1-X证明P1-W的多尺度候选表示和集合排序器具有完整选择容量。此前的低known-count exact不是
“128候选无法区分”，也不是必须依赖3px/3.5px/4px聚类；在连续特征、尺度信息和整组上下文下，
selector能同时处理原域与近邻Atom专项域。

这仍是64图容量实验，不是泛化结论。下一步P1-Y固定架构和候选协议进入train5k，并使用与训练
身份隔离的route-select作为泛化门；若route不能同步提高，应诊断风格/特征迁移，而不是追加容量轮次。
