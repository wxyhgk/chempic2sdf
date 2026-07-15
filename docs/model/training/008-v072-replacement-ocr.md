# 008：v0.7.2 平衡字符替换 OCR

> 状态：完成  
> 最佳输出：`outputs/checkpoints/chemvector-v072-ocr-balanced-3k`

## 目标

不再使用 PaddleOCR 伪标签。在完整分子图中替换少量原子显示字符串，SDF、顶点、
键和布局保持不变，只计算 OCR loss。标签作为 UTF-8 普通字符串，不执行化学解析。

覆盖全部 118 个元素、带氢/电荷/同位素写法、`A/X/R1/R(n)/*` 占位符，以及
`Ph/Me/Et/tBu/Boc/TMS` 等简写。

## 失败的高比例课程

首次 10,000-step 实验每张图替换 2–8 个字符，并让周期表长尾、占位符和简写占比
过高。合成验证可见字符串准确率达到 83.49%，但 MR-TADF 完全一致从 33.82%
降到 15.73%，B 识别率从 59.62% 降到 33.69%。该方案被否决。

## 平衡配置

- 65% 图片不替换，保留原始分布。
- 35% 图片只替换 1–3 个锚点。
- 118 个元素保留在低比例长尾池。
- `B/N/O/F/Cl/Br/S/Se` 提高权重，B 权重约为 P 的 10 倍。
- 学习率从 `1e-4` 降到 `3e-5`。
- 只训练 3,000 step，随后使用外部 benchmark 门控。

```bash
.venv-train/bin/python -m chemvector.pretrain_ocr \
  --dataset molecule-replacement \
  --init-checkpoint outputs/checkpoints/chemvector-v5-vertex-10k/last.pt \
  --train-shards Data/sdf/webdataset_full_v0/shards/train-*.tar \
  --validation-shards Data/sdf/webdataset_full_v0/shards/validation-000000.tar \
  --steps 3000 --batch-size 4 --workers 4 \
  --ocr-learning-rate 3e-5 --vision-learning-rate 0 \
  --output-dir outputs/checkpoints/chemvector-v072-ocr-balanced-3k
```

## MR-TADF 外部门控

| checkpoint | 有效 SDF | 化学完全一致 | 正确/错误/无效 |
|---|---:|---:|---:|
| v0.7.1 基线 | 71.91% | 33.82% | 372 / 419 / 309 |
| 高比例全字符 10k | 37.09% | 15.73% | 173 / 235 / 692 |
| 平衡字符替换 3k | 83.64% | 44.82% | 493 / 427 / 180 |

B `59.62% -> 68.50%`，O `85.82% -> 98.18%`，F `92.48% -> 96.41%`，
S `65.79% -> 92.11%`。训练数据未加入 MR-TADF。

## 决策

完整分子字符替换有效，但必须保留原始字符分布。下一阶段针对 stride-8 特征无法
保留细笔画的问题，引入 Pixel OCR。
