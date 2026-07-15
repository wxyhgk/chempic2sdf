# 多尺度 OCR pilot 结论

ChemVision Layers pilot 已完成多尺度 OCR 实验。当前结论：10,000 张 pilot 足以
带来显著 MR-TADF 提升，暂时仍不需要生成 72 万张全量语义层。

## 最优方案

```text
共享 Pixel CNN
ROI 半径：12 / 20 / 32 px
B bbox jitter：100%
N bbox jitter：5%
训练步数：500
batch size：4
训练数据池：9,000 张
实际图片采样：约 2,000 次
```

checkpoint：

```text
outputs/checkpoints/chemvector-v075-pilot-multiscale-balanced-jitter-1500/checkpoint-0000500.pt
```

完整 MR-TADF 结果：

```text
outputs/evaluations/mr-tadf-v1-v075-multiscale-balanced-jitter-500/
```

| 指标 | v0.7.3 原图修正 | v0.7.5 |
|---|---:|---:|
| 有效 SDF | 81.36% | 89.73% |
| 化学完全一致 | 49.91% | 67.45% |
| 原子数一致 | 68.55% | 76.00% |
| 键数一致 | 68.64% | 76.18% |

## 已否决实验

- hard bond-only repair：真实键在文字框边缘被截断；
- 所有可见字符 bbox jitter：中心 OCR 降到约 84%；
- B/N 全量 bbox jitter：N 数量过多，压制 F/Cl/P 等字符；
- 只按总体 visible OCR 选择 checkpoint：MR-TADF 端到端退化；
- predicted-anchor 续训：实际峰 OCR 提升，但未改善完整 SDF。

## 决策

当前保持 `full/` 不生成。下一阶段应围绕剩余 113 个无效分子和 245 个有效错误
做错误分解，重点判断多文字峰是否需要显式 text-instance query/bbox head，而不是
继续无差别扩大普通训练图片。
