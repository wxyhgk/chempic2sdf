# 可复现的多样案例库 v1

只生成隔离案例，不训练、不改推理、不查询评估身份。所有样本保持
`split=quarantine`、`training_authorized=false`、`external_overlap_status=unchecked`。

## 这批增加什么

固定选择 768 个不同化学结构，各画 16 种样式，计划 12,288 个案例。
候选池来自既有 v0.2 代码模板、12 个标签场景种子和明确固定的 32 个旧 TRAIN 分子身份。
不读取 manifest 中的开发条目、任何评估样本文件或私有身份索引。
多数化学身份是已有模板／训练来源，不能把 12,288 张图说成 12,288 个新分子。

用 GF(4) 的 OA(16,5,4,2) 覆盖五个四水平因子的全部两两组合：

| 因子 | 水平 |
| --- | --- |
| 字体 | DejaVu Sans、DejaVu Serif、Liberation Sans、Nimbus Roman |
| 字号 | 12、18、24、32 px |
| 结构旋转 | 0、90、180、270 度 |
| 键长上限 | 28、40、52、64 px |
| 线宽 | 0.8、1.6、2.4、3.2 px |

16 种设计不是 4⁵ 种全组合，也不是 16 个独立渲染器。成图拒绝、冲突隔离和像素去重后，
个别分子的组合覆盖可能不完整；看实测报告，不能只看设计表。

源结构优先保留标签种子和固定 TRAIN32，再按元素、键级、端点类型、官能团、
局部邻接环境、尺寸及立体等特征的递减支持权重选样，确定性 ID 打破平局。
不以模型成败选择分子。结构独特性按 canonical isomeric SMILES 计算，
另报连接身份及非空 Murcko 骨架，避免把许多侧链变体当作大量不同骨架。

## 同源与质量边界

- 新渲染接口复用冻结的矢量归属、立体保护排版和严格同 SVG 像素／掩膜生成。
- 不改历史 renderer；原 DejaVu Sans 下的 PNG、SDF、全部像素层有逐字节回归测试。
- 新增字体禁止 fallback，字体文件绑定 SHA256。原图均为干净 768×768，不做后期放缩或噪声。
- 拥挤、裁剪、退化连接等已知成图拒绝整图记录；未知错误严格终止，不吞错补数量。
- 保留拒绝、重复、同像素异身份三类记录；同像素异身份涉及的整个身份隔离。
- 不删除原生成物、不自动发布、不把生成成功当识别率或训练效果。
- 审计中将“正确可见字符串能否被现解析器接受”单列；这不是 OCR 识别准确率。

## 运行

从仓库根目录运行，数据目录及旁边的 `-runtime` 目录必须不存在。

```bash
CUDA_VISIBLE_DEVICES='' .venv-train/bin/python -m Data.sdf.case_diversity_v1.build \
  --output outputs/datasets/NEW_SMOKE_DIRECTORY --smoke --workers 8 --wall-seconds 240

CUDA_VISIBLE_DEVICES='' .venv-train/bin/python -m Data.sdf.case_diversity_v1.build \
  --output outputs/datasets/NEW_FULL_DIRECTORY --workers 16 --wall-seconds 1200

CUDA_VISIBLE_DEVICES='' .venv-train/bin/python -m tools.audit_case_diversity_v1 \
  --dataset outputs/datasets/NEW_FULL_DIRECTORY --output outputs/audits/NEW_AUDIT_DIRECTORY
```

单分子全部 16 种样式为提交单位，单任务 180 秒硬上限；外层运行预算包含选样、
生成、哈希复核及图库。超时／失败不自动重试，已提交案例保留，未返回任务为 pending。
只有完整对账、检查和图库完成后才写 `summary.json`，无摘要不能声称全量完成。

## 文件

- `profiles.py`：固定两两组合覆盖和显式字体绑定。
- `catalog.py`：来源约束、去重、结构覆盖选样和骨架计数。
- `rendering.py`：独立多字体同源渲染，不修改历史代码。
- `coverage.py`：实际可见布局／文字／键线场景及内部质量分区。
- `build.py`：有限 CPU 并行生成、完整分母、哈希、图库和完成标记。
- `tools/audit_case_diversity_v1.py`：独立逐文件、SDF/索引、标签、像素去重及解析接口审计。

产物中的 `index.html` 包含原始成图（也包含后续建议排除的重复／冲突项），
不可绕过 `quality-partition.json` 直接当训练集。`style-matrix.png` 仅为前景缩略预览，
真正模型输入在 `samples/*.png`。未来训练前仍需身份准入、跨批去重、目标适配、
真实模型候选可达性，以及另行固定的训练配方与预算。
