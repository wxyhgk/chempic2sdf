# MR-TADF v1 Benchmark

MR-TADF v1 用于检验大型 B/N 富集、多环、拥挤布局和超长键分子的图片到 SDF
能力。该基准与现有 18 万训练分子没有结构或 Murcko scaffold 重叠。

## 数据构建

数据入口只能是审计后的
`Data/MR-TADF_SDF/audit/approved.csv`，禁止直接遍历原始 SDF 目录。

```bash
.venv-train/bin/python -m Data.sdf.build_mr_tadf_benchmark \
  --approved-manifest Data/MR-TADF_SDF/audit/approved.csv \
  --source-dir Data/MR-TADF_SDF \
  --output-dir Data/benchmarks/mr_tadf_v1
```

固定渲染协议：

- 768×768 PNG；
- 黑白配色；
- 保留源 SDF 二维坐标和相对键长；
- 旋转角度 0°、线宽 2.0、字体 16、padding 0.08；
- 每个分子只生成一张图，不使用随机增强；
- PNG 通过尺寸和非空像素验证后才写盘。

输出结构：

```text
Data/benchmarks/mr_tadf_v1/
├── images/            1,100 张输入 PNG
├── annotations/       原子、键和像素坐标
├── ground_truth/      1,100 份真值 SDF
├── manifest.csv       benchmark 加载入口
├── runtime_rejected.csv
└── summary.json
```

当前构建结果为 1,100/1,100 成功，运行时拒绝为 0，总占用约 102MB。按最大键长
与中位键长之比分层：

- 标准键长 `<1.75×`：695；
- 长键 `1.75×～2.5×`：177；
- 极端长键 `>=2.5×`：228。

v1 一旦用于模型对比，不再修改图片或真值。渲染协议变化必须建立 v2，不能覆盖
v1，否则不同模型的结果不可比较。

## 模型评测

```bash
.venv-train/bin/python -m chemvector.benchmark_mr_tadf \
  --checkpoint outputs/checkpoints/chemvector-v5-vertex-10k/last.pt \
  --benchmark-manifest Data/benchmarks/mr_tadf_v1/manifest.csv \
  --output-dir outputs/evaluations/mr-tadf-v1-v071 \
  --device cuda
```

评测器只加载一次模型，逐样本追加 `records.jsonl`，中断后使用完全相同的命令即可
续跑。`benchmark_config.json` 固化 checkpoint 与 manifest 的 SHA256，避免在同一
结果目录中混入不同模型或数据版本。

主要指标：合法 SDF、分子式一致、去立体结构一致、InChIKey 完全一致、原子数和
键数一致，以及三档键长难度的分层结果。

## v0.7.1 基线

稳定 checkpoint 在 RTX 5090 上的完整 1,100 张结果：

- 合法 SDF：71.91%；
- 分子式一致：34.09%；
- 去立体结构一致：33.82%；
- InChIKey 完全一致：33.82%；
- 原子数一致：61.27%；
- 键数一致：60.91%；
- 平均推理时间：0.099 秒/张；
- 失败分类：断图 23、价态 187、Kekulé 99。

InChIKey 完全一致率按难度分层：

- 标准键长：36.69%；
- 长键：34.46%；
- 极端长键：24.56%。

结果位于 `outputs/evaluations/mr-tadf-v1-v071/summary.json`。后续每个模型版本使用
独立输出目录，不覆盖该基线。
