# 自研续训：数据准入交接与明确阻塞项

本轮完成候选身份核验请求的整理，并排除了“换一份 prepared SDF 就能补全公开身份”的可能。**没有启动训练，也没有新的识别准确率结果。** 不改模型架构、原数据、权重和前端默认配置。

**后续状态索引**：固定候选的公开异常保守筛查已完成，剩余 282 分子 / 2,469 图，见 `20260914-public-candidate-screen.md`；完整分子原图/SDF 验证入口已接入，真实 200 分子 / 800 图通过 CPU 核验，见 `20260914-fresh-dev-original-input.md`。下文保留交接时的阻塞和计数，不应据此重复已完成的筛查或接口工作。当前仍无真实训练 admission，也未开启 GPU 训练。

## 后续进展：已完成获准的隔离排重

用户随后批准独立数据侧流程。固定的 287 个候选与全部 3,000 个 fresh/sealed 分子，在完整 InChIKey、连接层及保守 core/scaffold 并集上均为零命中，未解析身份为零。最终凭据为 `outputs/audits/chemvector-private-identity-review-20260914-conservative/receipt.json`，状态 `checked_conservative`，不是 training admission。

发现 2 条旧骨架记录与重算分组不一致，检查时保留两者参与排除，没有修改真值；另有 33 条旧骨架差异经请求既定的去立体/同位素规则一致。评估两部分在新口径下仍共有 1 个连接层身份、3 个骨架组，完整键交集为零。新规则下不能声称 fresh/sealed 完全结构组独立。

模型侧仅收到脱敏凭据，未收到私有 ID、SMILES、InChIKey 或样本内容；图片、SDF、tar 未进入数据侧 worker。原 sealed 目录维持 `000`，没有 GPU 或权重更新。实现、访问授权和结果边界见 `Data/sdf/chemvector_joint_v1/IDENTITY_REVIEW.md`。

**因此下面关于“尚无独立凭据”的描述是交接时的历史状态。** 当前仍需处理公开评测的 13 条身份异常，以及真正完整分子的验证接入/新分组隔离问题；没有自动批准任何候选用于训练。

## 当前候选，不是已批准训练集

| 来源 | 有图且未命中旧公开缓存的分子 | 图片 |
| --- | ---: | ---: |
| 原子/基本键/官能团课程 | 89 | 801 |
| 旧 train 身份重新渲染的完整分子 | 198 | 1,712 |
| 合计 | 287 | 2,513 |

另外 12 个课程分子、108 张图被列入已知公开重合排除项；两个完整分子没有任何成功渲染，不进入本次核验请求。原文件均保留。801 张课程图包含此前 train/validation 两种提案，不能全部称为训练图；本请求不分配 split。

两类来源之间共有三个 core/scaffold 组：环己烷、苯环、吡啶环。未来混合发布必须跨来源统一分组，不能各自随机划分。复核此前课程提案，这些共有组的课程分子均为 candidate_train，**不是已经发生训练/验证污染的证据**。完整分子来自旧 train，不能被重新命名成未见验证集。

本地交接产物：`outputs/audits/chemvector-identity-review-request-20260914/`。

- `request.json`：输入清单及实现 SHA256、评估 commitment、候选计数、所需检查和返回约束。
- `candidate-identities.json`：287 个候选的 canonical SMILES、完整 InChIKey、连接层、忽略立体/同位素的分组、对应图片键。
- `known-public-exclusions.json`：12 个已知重合分子及全部变体。
- `no-render.json`：两个没有合格图的分子。

候选文件 SHA256：`ec0b92bc31fdc7a72b1262f516beb355d0ae009c85b61422a4f2fd79084bcfea`。

请求只绑定当前 manifests 和 artifact hash 清单，不声称重新核验了所有图片 payload；前轮像素/目标适配证据仍单独保留。这不是 launcher 可接受的 release 或 admission，不具备批准功能，没有发送给任何外部服务。

## 公开身份异常：备用容器无法解决

只读核查前轮 13 个异常 ID，对三个公开 manifest、13 个源结构和 9 个 prepared SDF 验证 SHA256。

9 个 prepared SDF 去掉容器结束分隔符后的 MolBlock 与官方源文件**逐字节相同**，重新运行解析和 InChIKey 转换均复现原失败。其余 4 个是 pending_superatom_protocol 记录，没有备用 SDF。因此补全身份数为 **0**，未解决数仍为 **13**。

证据：`outputs/audits/chemvector-public-identity-alternatives-20260914/report.json`。这是只读诊断转录，不是新的完整公开排除索引。没有改价态、删原子、展开缩写、用原子数/分子式代替身份证明，也没有读取公开图片。

## 为什么旧 fresh/sealed 汇总还不够

旧生成器 `candidate_from_mol` 和 `validate_dataset` 检查旧 molecule ID、完整 InChIKey 及保留立体信息的旧 scaffold。现在课程分组忽略立体/同位素，并额外排除连接层命中。旧 `old_dataset_scaffold_overlap=0` 对原协议有意义，但不能自动推出新规则同样通过。

本轮只读取允许的 aggregate commitment/validation，没有打开 fresh/sealed 逐样本清单、图像或结构，也没有绕到 `work/fresh_molecules.plan.csv` 等副本获取隐藏身份。

所需核验必须由**已获授权的独立数据侧**执行：按固定评估 commitment，对整个候选清单核对 fresh_dev 与 sealed_test；返回请求 SHA256、覆盖/未解析计数以及候选侧的排除 ID，不返回隐藏 ID、结构、图像或匹配对象。完整键、连接层、core/scaffold 检查分别记录。缺失或未知不能填成“不重合”。

当前仓库未提供可直接消费的这类凭据；本请求不授权开启隐藏目录，也不伪装成受信服务/签名回执。需要项目负责人指定执行方或明确批准这一独立数据侧流程后才能继续此项。

## 接下来只推进三个交付物

1. 取得公开异常记录的获准排除处理，以及不泄露隐藏样本的 fresh/sealed 核验结果；否则保持 quarantine。
2. 接入真正的完整分子 development validation，不能拿旧 train 重绘替代。当前 release 消费 same-SVG 目标，legacy fresh-dev 仍需要经过约定的数据适配；这份身份请求不解决该接口。
3. 绑定真实 mixed-data release、单套配方和停止条件，再做一次不超过 12 GPU 小时的自研续训，不做架构/学习率搜索。

不再通过增加 renderer 变体或重复 CPU smoke 来掩盖上述数据依赖；当前这一步需要明确的数据侧协作。

## 复现与验证

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-train/bin/python -m chemvector.training.curriculum.review_request \
  --curriculum-root outputs/datasets/curriculum-v0.2-ink-aware-20260914 \
  --full-molecule-root outputs/datasets/full-molecule-v0.3-stereo-aware-20260914 \
  --exclusion-index Data/sdf/chemvector_joint_v1/work/old_overlap_index.json \
  --evaluation-commitment Data/sdf/chemvector_joint_v1/fresh_eval_12k/commitment.json \
  --output /absolute/path/to/new-review-request
```

新目录必须不存在。工具只读取明确指定的隔离候选 manifests、旧公开缓存、aggregate commitment；不会跟随 commitment 里的 CSV/tar 路径读取评估样本。

`test_curriculum_review_request.py`、`test_curriculum_admission.py`、`test_curriculum_release.py` 合计 **41 passed，8.86 秒，CPU**。覆盖连接层排除全部变体、跨来源分组、旧 train 用途、不跟随隐藏 payload 路径、输入哈希绑定、无效来源/commitment 拒绝、不可覆盖输出及请求不能授权训练。没有 GPU 任务、optimizer 更新或重跑前轮成图。
