# 2026-09-14：冻结后的完整预训练 OCSR 对照准备

## 当前结论

自研研究参照已冻结，部署默认未改。下一项只回答：**一个官方完整预训练系统，在
相同输入、相同完整化学图评分下，与冻结参照有什么差距？** 不先训练新架构，也不
把更换 OCR 当作完整系统对照。选择 MolScribe 是为了建立参照，不预设它更好。

快照：`outputs/baselines/chemvector-v01016-reference-20260914/`。
独立复核通过：5,215 个文件、462,232,915 字节。
manifest SHA256：`16b34c058652edf8fe0a72db1eaf7f11df389d90909e5b09168c485e0bcb21b4`。
研究参照 v0.10.16 非立体完整图正确 59/200；部署默认 v0.10.10 为 55/200。
两者都是重复使用的开发集成绩，不是独立泛化结论。

冻结范围及复现限制见 [冻结决定](../model/baselines/20260914-frozen-reference-and-exploration.md)。
本笔记在快照生成后新增，不属于该快照，不改动其文件。

## 官方来源核验

本轮通过 GitHub API 读取官方仓库，不依赖旧调研文档中的安装状态。

- 仓库：[thomas0809/MolScribe](https://github.com/thomas0809/MolScribe)。
- 当前 main commit：`7296a30413eb55436702011efdff78131f66d162`，提交时间
  `2025-01-09T18:36:50Z`；后续运行应固定此源码版本或显式记录改选版本。
- [LICENSE](https://github.com/thomas0809/MolScribe/blob/7296a30413eb55436702011efdff78131f66d162/LICENSE)
  是 MIT。此处核验的是代码许可证，不把它自动等同于独立权重的使用许可。
- [官方 README](https://github.com/thomas0809/MolScribe/blob/7296a30413eb55436702011efdff78131f66d162/README.md)
  明确链接权重仓库 `yujieq/MolScribe`。快速示例使用 `swin_base_char_aux_1m.pth`，
  实验复现使用 `swin_base_char_aux_1m680k.pth`；不是同一个文件，不应混用成绩。
  初始对照优先核验后者；最终需记录权重仓库 revision 和下载文件 SHA256。
- [interface.py](https://github.com/thomas0809/MolScribe/blob/7296a30413eb55436702011efdff78131f66d162/molscribe/interface.py)
  的 `predict_image_file(..., return_atoms_bonds=True, return_confidence=True)` 返回
  `smiles`、`molfile`、原子/键和置信度。完整系统原生提供 molfile，主评分不必先把
  SMILES 重新绘制成 SDF。原始图字段可能含缩写节点，不能直接假设与展开后的化学图
  原子逐一对应。

MolScribe 本身仍包含原子、坐标和键预测。因此即使它胜出，也只能说明具体系统有
差距，不能据此证明“所有原子/键建模都不成立”，更不能单独归因于 MoE 或循环深度。

## 运行准备与当前缺口

[setup.py](https://github.com/thomas0809/MolScribe/blob/7296a30413eb55436702011efdff78131f66d162/setup.py)
标记包版本 1.1.1，依赖 `numpy>=1.19.5,<2.0`、`timm==0.4.12`、
`OpenNMT-py==2.2.0`、`albumentations==1.1.0` 等。
[实验 requirements](https://github.com/thomas0809/MolScribe/blob/7296a30413eb55436702011efdff78131f66d162/requirements.txt)
对 timm 和 albumentations 改用特定 Git revision；两套依赖不是完全相同的协议。

当前 `.venv-train` 实测为 Python 3.13.5、torch 2.10.0、NumPy 2.4.3、RDKit 2026.3.3；
未安装 torchvision、timm、OpenNMT-py、albumentations。NumPy 已有明确约束冲突，
不能在原环境直接安装或降级。独立环境的 Python/PyTorch 组合尚未验证，不声称
安装完就能运行。上游还使用未显式指定 `weights_only` 的 `torch.load`，以及
`strict=False` 的模型加载；需核验可信权重的加载兼容性和缺失/多余参数，不能
静默接受不完整加载后报告准确率。

本轮 GitHub API 可达；raw.githubusercontent.com 请求超时，Hugging Face 元数据请求
出现 DNS 解析失败。只能说明这些请求失败，不能推出整机网络或 CUDA 不可用。
官方权重位置已由 README 确认，但权重 revision、哈希、模型卡许可和实际下载可用性
尚未核验。本轮没有安装依赖、下载 checkpoint 或运行 MolScribe 推理。

## 下一轮的最小实验

1. 在独立环境固定官方源码和一个官方权重，先用已有输入做少量推理冒烟检查。
   记录依赖、权重 SHA、加载情况及 CPU/GPU 实际状态；兼容性改动单独记录。
   不加入 ChemVector 的检测器、OCR 或修补规则，不训练、不按真值挑结果。
2. 跑既有固定 200 张开发 PNG，保持输入身份和分母；保存每张原始返回值、molfile、
   错误及耗时。预测端不读取真值，不下载或解封新的评估数据。
3. 使用已建立的共同 SDF 评分语义，单独输出完整图正确率、立体正确率、有效但错误、
   无效/失败，并列出相对冻结参照救回和丢失的样本及四种版式分组。推理依赖留在隔离
   环境，评分沿用当前共同 RDKit 环境。外部对照另存，不改旧实验的固定 arms 或快照。

已有 `profile_comparison.scoring.score_sdf` 提供共同单分子评分入口，但其完整运行器
绑定原三组实验，不能把 MolScribe 冒充其中一组。对照应保留独立工件并复用评分语义。
主指标直接评分原生 molfile；若以后增加 SMILES 兜底或重新生成坐标，必须作为独立
结果披露，不能覆盖原生失败，更不能声称还原了图中位置。

读完配对结果后再决定是否以外部系统作为探索基座；200 图不足以支持上线、独立泛化
或预训练数据无重叠的结论。若第三方没有改善，就以共同失败样本定位问题，不自动
进入另一轮架构替换。
