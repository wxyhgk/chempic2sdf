# 自研混合训练集已发布

**后续运行已完成**：第一轮 GPU 续训在 1,800 步早停，完整分子严格匹配 170/800 → 66/800，保留初始模型，未部署；用量上界 0.11517 GPU-hours。见 `20260915-mixed-continuation-result.md`。本文下方保留发布时状态，不应据此重复发布或再自动启动训练。

真实小规模混合数据已获得本地研究准入：**训练 2,325 图，验证 944 图，全部 3,269 图通过发布后 loader 检查**。没有运行真实 optimizer、GPU 训练或产生新的准确率；本轮只在 CPU 读取现有模型以固定初始权重和推理参数。

## 实际发布

| 用途 | 域 | 分子 | 图片 |
| --- | --- | ---: | ---: |
| train | 原子/键/官能团课程 | 73 | 657 |
| train | 旧 train 身份完整分子重绘 | 193 | 1,668 |
| validation | 课程结构 | 16 | 144 |
| validation | committed fresh-dev 原图/SDF | 200 | 800 |

目录：`outputs/datasets/chemvector-mixed-release-20260915/`。

- `manifest.json` SHA256：`47e5ed3958aa20d89b871513afb7a90c6b8c0b82dfd365b5801967788afe3b9a`。
- `admission.json` SHA256：`28d4a9343a87a4d33cb5ae02819d05c9b20031348cb2ee02849d844c17b8afc4`，绑定精确 manifest，`training_authorized=true`，仅限本地研究。
- `sources/curriculum/`、`sources/full_molecule/`：2,469 个独立副本，原隔离数据不改。
- `source-evidence/`：原始清单与完整分子 source-provenance 快照，保留原分子 ID 和来源。
- `promotion-plan.json`、`promotion-review.json`、`source-use-review.json`、`publication-report.json`：计划、明确审查、使用依据和实测检查。

PNG、SVG、SDF、文字记录及各 mask 共 **19,752 个文件逐字节复制**；graph/layers 只改变用途标记，meta 更新这些标记、相关哈希和来源链接，不化学修复或重画。发布后独立重算 **27,159 个原始样本文件**及源清单哈希，全部未变，原 summary 仍为 `training_authorized=false`。

首次复制发现旧 graph 没有 `split`，隔离状态在 metadata/layers，发布器当时严格拒绝。按实际旧 schema 修正后通过。失败现场在 `chemvector-mixed-release-20260915-incomplete-graph-schema/`，没有样本副本或 admission，不是可训练发布。

## 修正了两个真实的划分冲突

旧课程提案按 core/scaffold 划分，以下结构却共享连接层键：

| 结构对 | 旧提案 | 最终处理 |
| --- | --- | --- |
| `CN` / `C[NH3+]` | train / validation | 全部 validation |
| `C[S-]` / `CS` | train / validation | 全部 validation |

现在对分子 ID、连接层、新 core/scaffold 求传递并集，保留已有 validation 一侧。18 图从课程 candidate_train 移出，课程训练 **675 → 657**，课程验证 **126 → 144**。没有旧验证进入训练，完整分子重绘不改作未见验证。如果一个连通组同时包含 old-train replay 与已有 holdout，发布器拒绝，不自动牺牲任一侧。

最终所有来源的 train/validation 分子 ID、连接层及新 core/scaffold 均无交集；固定的 fresh-dev 200 分子 / 800 图不变。

## 审查依据与边界

精确输入见 `releases/20260915-mixed-spec.json`；本地工程审查见 `releases/20260915-promotion-review.json`，不是第三方或签名 custodian 的新证明。

- 沿用 287 候选的独立私有 `checked_conservative` 凭据；当前 282 是精确子集，没有新查询或隐藏样本读取。
- 沿用公开异常的保守排除结果，完整排除 5 个风险分子，不声称公开库的 13 条全局身份异常已修复。其他已知公开缓存阳性继续排除。
- 两个零失败目标适配报告绑定原始清单；发布时重新验证每张复制图的哈希、SDF 身份、同源标注和目标张量，最后检查整个 mixed loader。
- 手写课程来自项目 catalog；ChEMBL 官方 FAQ 标注 CC BY-SA 3.0；COCONUT 官方数据文档标注 CC BY 4.0，其在线服务条款保留原所有者限制；DejaVu 字体使用依据来自安装包许可。保留署名和来源，不称 COCONUT 为 CC0，不授权分发或部署。官方 URL 与适用限制见 `releases/20260915-source-use.json`。
- fresh-dev/sealed 已知的新口径 1 个连接层、3 个分组交集仍是评估限制。本轮仅授权开发验证，不宣称两者严格独立；sealed 仍为 `000`，没有读取样本。
- 这仍是小规模、单布局/两字体试点，不证明大规模真实文档覆盖或课程有效性。旧完整分子计划的 88 个失败渲染仍在历史记录中。

## 唯一固定配方

计划 SHA256：`0f14a90c82cc57b2745ccf384f9f9d6bc3358d886a6d1a4b52a0195f5899fe42`。

- 现有自研 ChemVector，不引入 MoE、MolGrapher 或新架构。
- 混合周期 `full_molecule, full_molecule, curriculum`，2:1；seed=20260914。
- batch=1，最多 6,000 optimizer steps；每 600 步验证；连续 3 次不改善早停。
- 单 GPU，最多 43,200 秒（12 小时），包括检查、初始化、基线、训练、验证和保存；无自动重试/断点恢复。
- 九个 checkpoint SHA 与推理阈值固定；实际 provenance 不符，在模型转移到 GPU 前拒绝。现有容量 128 atoms / 192 bonds；atom=-0.25、bond=-0.5、pair=0.85、role=0.85。
- 保留七组单套学习率，OCR 转写冻结，不做超参数搜索。
- **只用 800 张 full_molecule 开发验证的 exact 数、然后 valid 数选 checkpoint**。144 张课程验证另报，不允许小图进步掩盖整图退步；无提升保留 initial。

`protocol.py` 拒绝 CLI 覆盖绑定的步数、batch、验证间隔、patience 或时限；历史 release 未声明新协议则保留原行为。验证 JSON 同时有总体 `metrics` 与选模 `selection_metrics`。

## 验证与下一步

**285 passed，32.70 秒，CPU**；模块编译和 `git diff --check` 通过。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-train/bin/python -m pytest -q tests/chemvector/test_curriculum_*.py \
  tests/chemvector/test_ctab_inventory.py tests/chemvector/test_full_molecule_data.py
```

覆盖独立复制、旧 graph 缺省字段、失败不留下可启动 admission、原文件不变、带电连接层对/传递分组、协议不可覆盖，以及微型 CPU 训练中“小图提升、整图退步仍保留初始模型”。合成 optimizer 不涉及真实模型/数据。

**数据发布阻塞已解除。下一轮应运行，而不是重复筛库或发布。** 先只读检查 GPU 可用性/占用，用本 release 做固定初始评估和一次定额续训，保留原模型作逐图配对对照。若 CUDA 数值/内存或其他运行问题失败，保留失败记录，不自动换配方或清零预算。没有新成绩时不得将准入成功表述为准确率提升。
