# 005：v0.5 顶点、连线与墨线修复

> 状态：完成  
> 主要 checkpoint：`outputs/checkpoints/chemvector-v5-vertex-10k/last.pt`

## 视觉优先生成

`--visual-first --ocr-fusion --stroke-scoring` 在第一个 stroke 之前用 OCR 改写非碳
标准元素、重建 decoder cache，并从全部顶点对中选择有线条支持的候选：

```bash
.venv-train/bin/python -m chemvector.evaluate \
  --checkpoint outputs/checkpoints/chemvector-v5-joint-replay-10k-stage1/last.pt \
  --shards Data/sdf/webdataset_full_v0/shards/validation-000000.tar \
  --samples 20 --generate-samples 5 --beam-width 4 \
  --visual-first --ocr-fusion --stroke-scoring \
  --stroke-threshold 0.5 \
  --output-dir outputs/evaluations/chemvector-v5-visual-first-eval
```

Stroke scorer 使用 stride-4 特征沿两点之间采样带状区域，不设置键长上限。10k
训练后，阈值 0.5 的全点对验证达到 99.72% edge recall 和 84.82% precision。

## 生成后墨线修复

`repair_v4` 直接检查顶点之间是否存在连续墨线。若直线上存在中间顶点，只保留
相邻线段，避免把两条共线键合成一条跨点长键。修复器还会合并同一文字内部产生
的重复峰，并根据小环中的芳香预测传播芳香键型。

检测式推理使用 20px OCR 取样半径。在 500 张验证图片、15,220 个锚点上，元素
融合准确率为 97.12%，高于 24px 半径的 96.98%，并消除 GD906 邻近 N 泄漏到
上方碳顶点的问题。

```bash
.venv-train/bin/python -m chemvector.infer_detected_v5 \
  --checkpoint outputs/checkpoints/chemvector-v5-vertex-10k/last.pt \
  --image Data/images/GD906.png \
  --output-dir outputs/predictions/chemvector-v5-auto-gd906-raw
```

该流程依次执行透明背景白底合成、等比例画布准备、顶点检测、OCR、stroke 生成、
墨线修复、重复峰合并、SDF 写入和预览渲染。GD906 输出为 84 个原子、104 条键、
1 个连通分量，分子式 `C80H44BFN2`，无需手工覆盖或按键长补边。

## 500 张 scaffold 独立盲测

```bash
.venv-train/bin/python -m chemvector.evaluate_detected_v5 \
  --checkpoint outputs/checkpoints/chemvector-v5-vertex-10k/last.pt \
  --shards Data/sdf/webdataset_full_v0/shards/validation-000000.tar \
  --samples 500 \
  --reject-keys-file Data/sdf/chemmolseq_v0_3_depiction_reject_keys.txt \
  --output-dir outputs/evaluations/chemvector-v5-e2e-validation-500
```

500 张图片对应 125 个未见分子，每个分子四种布局：

- 生成完成率：100%。
- 合法 SDF：48.6%。
- 完整拓扑：17.2%。
- 化学完全一致：12.4%。
- 至少一种布局合法的分子：85.6%。
- 至少一种布局完全一致的分子：31.2%。
- 修复成功样本顶点 precision/recall：98.55%/98.67%。
- 修复成功样本边 precision/recall：96.95%/97.24%。
- 正确边上的键型准确率：96.27%。

四种布局的完全一致率为 RDKit 18.4%、CoordGen 15.2%、Original 13.6%、
Stress 2.4%。257 个修复失败中，断图 120、价态冲突 112、Kekule 失败 25。

## 决策

全点对和墨线修复路线保留，因为它不依赖键长剪枝。瓶颈从点线定位逐步转向布局
鲁棒性、价态约束、芳香环和立体化学。
