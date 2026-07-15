# Oracle Semantic Layer 实验结论

实验已完成。模型对每张 combined 图片只推理一次，随后使用完全相同的顶点、
OCR 和原始边候选运行三种 repair：

1. `baseline`：repair 使用完整 combined 图片；
2. `hard bond-only`：repair 只使用真值 bond layer；
3. `bbox-aware bond`：使用真值 bond layer，并按真值文字 bbox 从文字框外缘开始
   采样线段。

使用 checkpoint：

```text
outputs/checkpoints/chemvector-v073-pixel-ocr-original-refine-1k/last.pt
```

完整输出：

```text
outputs/evaluations/chemvision-pilot-oracle-bbox-v073
```

## 样本

- validation + internal_test：1,000 张；
- 可评估：998 张；
- 排除：2 张；
- 排除原因：stress 布局在 ChemSVG v0.4 的 q1023 坐标量化后发生原子坐标
  碰撞。SDF 和数据层本身没有损坏。

## 结果

| 指标 | Baseline | Hard bond-only | Bbox-aware bond |
|---|---:|---:|---:|
| SDF 有效率 | 20.24% | 20.94% | 21.04% |
| 化学完全一致 | 4.51% | 3.61% | 4.61% |
| 完全拓扑 | 6.91% | 5.21% | 7.01% |
| Edge precision | 87.12% | 86.39% | 87.41% |
| Edge recall | 81.37% | 79.25% | 81.21% |
| 多余边 | 4,268 | 4,430 | 4,149 |

`hard bond-only` 明显变差，证明不能在推理时直接擦除文字。键线在文字框边缘
终止，中心到中心的固定比例裁剪会把真实键误判为中断；图不连通后，repair
又会从模型原始候选中补桥，可能重新加入错误边。

`bbox-aware bond` 的方向正确：

- 整体多余边减少 2.79%；
- 整体 edge recall 下降 0.155 个百分点；
- 完全拓扑提升 0.10 个百分点；
- large/extreme 285 张中，多余边减少 3.30%；
- large/extreme edge recall 提升 0.187 个百分点；
- large/extreme 完全拓扑提升 0.351 个百分点。

但它没有达到原定的 50% 伪边下降，也没有带来至少 5 个百分点的整体收益。

## 决策

当前结论为：

```text
NO-GO: 暂不生成 72 万张全量语义分层数据
```

这批数据证明自适应文字 bbox 是必要的，但也证明文字墨迹不是当前主要瓶颈。
当前模型在两个完全未见字体家族上的 baseline 化学完全一致率只有 4.51%，说明
主要问题是 OCR/顶点对字体和字号的分布外泛化。

下一项实验应使用现有 pilot train 的 9,000 张图片训练多尺度 OCR，并继续使用
500 张 validation 和 500 张 internal_test 作为未见字体测试。建议先比较：

```text
固定 r=20
三尺度 r=12/20/32
真值 bbox ROIAlign
```

只有多尺度 OCR 通过后，再决定是否训练 soft text/bond semantic head。
