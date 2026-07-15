# 010：v0.7.5 ChemVision 多尺度 OCR

> 状态：当前最佳候选  
> checkpoint：`outputs/checkpoints/chemvector-v075-pilot-multiscale-balanced-jitter-1500/checkpoint-0000500.pt`  
> SHA256：`53b4855b367f5fda524f0a4442b80391ba056bc6410bcbabb3aa345c443dfd19`

## 目标

解决固定 20px ROI 在 small/extreme 字号上裁剪不完整或丢失细节的问题，同时验证
ChemVision Layers pilot 是否值得扩展到 72 万张全量数据。

## 数据

- `chemvision_layers_v0_1/pilot_10k`：2,500 个分子、10,000 张图片。
- train：9,000 张；validation 和 internal test 各 500 张。
- 两个验证 split 使用训练未见字体家族。
- MR-TADF 不进入训练，只用于外部门控。

## 模型与采样

- 同一锚点裁取 `12/20/32px` 三种 ROI。
- 三个尺度共享 Pixel CNN，由轻量 scale scorer 融合 64 个视觉 token。
- 新增参数约 1.86 万。
- 每个 B 增加一个 bbox 内锚点。
- N 只抽 5% 增加 bbox 锚点，避免 34,004 个 N 压制其他字符。
- 视觉、vertex、stroke 和 relation 全冻结，只训练 OCR head。

## 训练

```bash
.venv-train/bin/python -m chemvector.pretrain_ocr \
  --dataset chemvision-layers --pixel-ocr --pixel-multiscale \
  --visible-jitter-copies 1 --n-jitter-probability 0.05 \
  --init-checkpoint outputs/checkpoints/chemvector-v073-pixel-ocr-original-refine-1k/last.pt \
  --train-shards 'Data/sdf/chemvision_layers_v0_1/pilot_10k/shards/train-*.tar' \
  --validation-shards \
    Data/sdf/chemvision_layers_v0_1/pilot_10k/shards/validation-000000.tar \
    Data/sdf/chemvision_layers_v0_1/pilot_10k/shards/internal_test-000000.tar \
  --steps 1500 --batch-size 4 --workers 4 \
  --validate-every 500 --validation-samples 1000 \
  --ocr-learning-rate 5e-5 --vision-learning-rate 0 \
  --output-dir outputs/checkpoints/chemvector-v075-pilot-multiscale-balanced-jitter-1500
```

最优点是 500 step、batch 4，实际约 2,000 次图片采样。不能只按总体 OCR 数字
选择后期 checkpoint，因为 B/F/Cl 的权衡会让端到端 SDF 退化。

## MR-TADF 完整结果

输出：

```text
outputs/evaluations/mr-tadf-v1-v075-multiscale-balanced-jitter-500/
```

| 指标 | v0.7.3 原图修正 | v0.7.5 |
|---|---:|---:|
| 有效 SDF | 81.36% | 89.73% |
| 化学完全一致 | 49.91% | 67.45% |
| 原子数一致 | 68.55% | 76.00% |
| 键数一致 | 68.64% | 76.18% |

完全正确 742、有效但错误 245、无效 113。标准键长、长键、极端长键的完全一致
率分别为 72.37%、62.71%、56.14%。

## 已否决实验

- hard bond-only repair：截断文字框边缘的真实键。
- 所有可见字符 bbox jitter：中心 OCR 降到约 84%。
- B/N 全量 jitter：N 数量过多，压制 F、Cl、P。
- predicted-anchor 续训：实际峰 OCR 提升，但完整 SDF 未提升。
- 困难替换锚点续训：合成验证提高，MR-TADF 200 张门控降到 32.5%。
- checkpoint 权重插值：没有同时改善准确率和有效率。

## 错误分解与决策

358 个错误中，255 个主要是顶点检测或定位，95 个是纯 OCR；并发标签中 142 个
含 OCR、122 个含多余键、64 个是明确交叉错连、78 个多余键穿过第三个原子。

结论为 `GO`：保留多尺度 OCR checkpoint；`NO-GO`：暂不生成 72 万张全量语义层。
下一阶段优先改进顶点召回与定位稳定性，OCR 作为并行分支持续维护。

完整 benchmark 和诊断见
[MR-TADF v1 文档](../../benchmarks/MR_TADF_V1.md)，数据实验原始报告见
[`MULTISCALE_OCR.md`](../../../Data/sdf/chemvision_layers_v0_1/reports/MULTISCALE_OCR.md)
和
[`ORACLE_SEMANTIC_LAYER.md`](../../../Data/sdf/chemvision_layers_v0_1/reports/ORACLE_SEMANTIC_LAYER.md)。
