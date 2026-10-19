# 自研续训任务：接口与发布合同

**最新执行结果**：用户确认的 `freeze-role-v1` 试训已正常结束，800 步按两次不改善早停，完整分子正确数 initial=170、400 步=106、800 步=117（均 /800）。保留 initial、未部署、不自动重跑；单卡耗时 323.450 秒。角色头状态确实冻结，但没有避免整图退步。见 `docs/model/training/20260915-expanded-freeze-role-result.md`。

**最新发布**：`outputs/datasets/chemvector-expanded-release-20260915/`，训练 13,151 条 / 12,985 张不同图片，原 944 条验证及旧训练行不变。本批私有身份凭据只继承，公开复核、来源用途审查与独立发布已完成；新增 10,826 张，排除 298 张与旧训练完全相同的图。私有核验的一次授权已经使用，禁止再次查询或补候选。用户已另行明确回复“开始训练”，确认单卡 30 分钟 / 1,200 步这一次任务，输出为 `outputs/experiments/chemvector-expanded-freeze-role-20260915/`；没有自动重试或部署权限。详情见 `docs/model/training/20260915-expanded-training-readiness.md`。

2026-09-15 已完成真实 mixed release：`outputs/datasets/chemvector-mixed-release-20260915/`，训练 2,325 图、验证 944 图。原 quarantine 来源不动；批准仅绑定独立副本，不授权扩量、对外分发或部署。**第一轮 GPU 续训已完成并在 1,800 步早停，完整分子成绩退步，保留 initial，未部署**。耗时 414.60 秒、用量上界 0.11517 GPU-hours；不要自动重跑或恢复。结果见 `docs/model/training/20260915-mixed-continuation-result.md`，下一步是无更新接口诊断。

上述接口诊断已完成：固定 24 张分层图上，仅回退 role 将正确数从 4 恢复到 15；CPU 58.15 秒、GPU 和优化器更新均为 0。该诊断不授予组合模型部署或新训练权限。完整结果和下一项固定开发集验证见 `docs/model/training/20260915-text-interface-diagnostic.md`。

`publish.py` 采用固定计划 + 明确审查的两阶段发布，全部检查通过才写最终 admission，不是忽略隔离标记的后门。既有诊断报告仍不能单独授权训练。

## 先检查，后启动

```bash
CUDA_VISIBLE_DEVICES='' .venv-train/bin/python -m chemvector.training.curriculum.launcher \
  --manifest /absolute/path/to/reviewed-manifest.json \
  --admission /absolute/path/to/reviewed-admission.json --check-only
```

检查会验证清单绑定、来源/分组规则和每个样本的完整 payload；不加载模型、不创建输出目录。当前 `outputs/audits/chemvector-curriculum-admission-20260914/report.json` 会立即被拒绝，包括仅运行 `--check-only` 的情况。

获准后，入口还必须显式给定 `--output`、`--max-steps`、`--wall-seconds`。默认 CPU；只有显式 `--gpu-device 0` 才指定单卡。历史默认 `--batch-size 1`、`--validate-every 100`、`--patience 3`。旧 mixed release 绑定 `--max-steps 6000 --batch-size 1 --validate-every 600 --patience 3 --wall-seconds 43200`，保持原样，不自动重跑。

新 expanded release 绑定协议 v0.2 / `freeze-role-v1`：`--max-steps 1200 --batch-size 1 --validate-every 400 --patience 2 --wall-seconds 1800`。参数必须显式匹配，不接受任意覆盖。角色头从优化器和 role loss 中排除并保持 eval，OCR 仍冻结；bbox 及其余五组照原学习率更新。保存/加载带有配方标记，验证及结束时检查 role 摘要不变。旧 v0.1 的七组更新行为不变。原 quarantine 数据不能直接用于训练；以上短回归试训已经执行，不是一万多张图的一轮完整训练，也不是重复启动许可。

所有正式 CLI worker 都由外部进程监控时间，时间包含初始化、数据核验、训练、验证与保存；超时记录失败并终止本次进程组，不自动重试。最多 1 张 GPU、配置时限不超过 12 小时；实际 wall/GPU-hour 上界估计记录在 runtime.json，进程清理及调度可能带来小幅超时开销。学习率是既定单套参数分组，不暴露网格搜索。

## 清单必须包含什么

Manifest schema：`chemvector-continuation-release-0.1`。原 `curriculum-same-svg` 保持兼容；原图/SDF 验证使用显式 `curriculum-with-original-validation`，训练部分仍为 same-SVG。

| 字段 | 要求 |
| --- | --- |
| `sources` | 来源名 → 明确的绝对目录；训练目录采用已发布 same-render bundle 格式 |
| `samples` | 每行有 source、sample_key、molecule_id、canonical_smiles、group_key、split、domain、meta_sha256 |
| `split` | 只能为 train 或 validation；禁止 candidate_* / quarantine |
| `domain` | curriculum 或 full_molecule；训练必须包含两者，验证必须有 full_molecule |
| `train_cycle` | 显式域序列，例如 full_molecule、full_molecule、curriculum 表示 2:1；此例不是已冻结的真实配方 |
| `seed` | 整数；固定各域打乱顺序 |

分组由 canonical 结构重新计算，不信任任意填写的 group_key；分子 ID、InChIKey 连接层和忽略立体/同位素的 Murcko/core 均检查跨 split 重叠。所有图片变体必须一起分组。域名称是受审数据用途，不声称能仅凭名称自动证明“大分子覆盖”。

训练样本必须已经在 metadata、graph、layers 中有一致的已准入标记，metadata 指定 `split=train`、`external_overlap_status=checked`。验证样本必须明确属于 validation 且排重已检查，仅以 diagnostic loader 读取，不能进入 optimizer。metadata 的 SHA256 写入清单，并继续绑定图像、图、文字和 SDF 哈希。输入在每次访问时再次验证。

### 原始完整分子验证来源

混合格式额外声明 `validation_sources`，键必须对应 `sources` 中的来源名。每个 binding 包含：

```json
{
  "format": "fresh-dev-original-image-sdf",
  "access_plan": "/absolute/path/to/fixed-access-plan.json",
  "access_plan_sha256": "<exact plan bytes SHA256>"
}
```

该来源所有行只能是 `split=validation, domain=full_molecule`，不能混入训练；必须完整保留访问计划所选的全部分子和四种变体。行在通用字段外还包含 `inchikey, variant, output_shard, png_sha256, sdf_sha256`。这些字段连同 `meta_sha256` 与 committed `samples.csv` 逐项对齐；随后用 SDF 重新检查 canonical 身份和新分组。

`fresh_dev.py plan` 只读取原 aggregate commitment/validation，固定它们的原始字节、全部 commitment 字段及基于分子 ID 的选择策略。先把计划独占写盘并固定 SHA256，再由 `fresh_dev.py check --plan ... --plan-sha256 ... --output ...` 读取 fresh-dev manifest 和所选原图/SDF。`check` 产生的 `validation-candidates.json` 是验证清单片段，**不是已批准训练 release**。

此验证来源无需修改旧 metadata 的用途标记，也不需要同源 SVG 字符记录；它的用途约束来自独立验证合同。整 tar 首次核验 SHA256 并建立只读偏移索引，后续每次访问检查文件签名、图像和 SDF 哈希。拒绝软/硬链接成员、路径越界、sealed 路径、重复成员、多条 SDF、身份不一致和非 768px RGB 单帧 PNG。不解包到磁盘、不重画、不转换图片、不导入旧 builder。缓存局限于单次任务的 tar 索引，不缓存解码图像或评分真值。

批准报告沿用 `chemvector-curriculum-admission-0.1`，除 training_authorized=true、external_overlap_status=checked、coverage_gaps=[] 外，必须有：

- `approved_manifest_sha256`：精确 manifest 原始字节 SHA256；混合顺序、路径或 seed 改变也需重新审查绑定。
- `checked_images`：与清单一致。
- `release_checks`：source_rights、public_benchmark_identity、fresh_eval_identity、group_split、model_targets 均为 checked。

这些字段是受信审查结果的消费合同，不是密码学证明，更不是把字段填全就真的完成外部排重。固定候选的私有身份核验与公开异常保守排除已有单独凭据，但不等于获得训练准入；全局公开身份异常仍未被修复。代码不读取 sealed 测试内容，也不会自行编造覆盖证据。

## 输出与模型选择

输出必须是全新目录。任务记录 manifest/admission 绑定、来源权重哈希、固定阈值、混合配方和参数组配置；逐步保存样本身份、域计数、loss、梯度和可达监督计数。

先在完整发布验证集上评估初始权重，然后按固定间隔验证。模型选择优先完整 SDF 身份 exact 数，其次 valid 数；相同成绩保留较早权重。本次固定协议只使用 full_molecule 的 800 图作为选模指标，144 张课程图单独保留，不能用小图提升掩盖整图退步。未改善可以选择 initial.pt，不把最后一步冒充最优。评价是 development validation，不是独立 sealed-test 结果。

- `initial.pt`：完整初始结构权重包。
- `step-XXXXXX.pt`：验证确有改善时独立发布的完整包。
- `last.pt`：最后一步完整包，不自动部署。
- `validation-XXXXXX.json`：全量逐图预测 MolBlock、严格身份结果、无效/fallback、救回/改错。
- `run.json`、`steps.jsonl`、`report.json`：配方、过程和最终选择。
- `runtime.json`、`worker.log`：父进程最终状态和日志；超时/worker failure 不能因为留下 checkpoint 就当作任务成功。

完整包保存 vision/atom/bond/set_head/pair/role/bbox 的 state 和架构配置。通过 `bundle.load_bundle(path, expected_sha256=...)` 独立恢复结构模型，不读取旧上游权重。冻结 OCR 仍是单独固定哈希依赖；`evaluation.FrozenOCRPredictor` 可用于加载后的完整分子推理。未注册前端新 profile，旧推理默认配置不变。

**不支持 optimizer 断点续跑或自动恢复任务**。包中的 progress 用于审计，不是续跑许可，避免重新启动时悄悄清零步数/时间预算。保存/加载目前指完整模型权重的恢复。

## 当前验证边界

评估计算已与训练 collator 解耦：`evaluation.evaluate()` 只从原始 768px RGB 图像生成输入张量，真值可为单条 `target_molblock` 或旧 `graph.canonical_smiles`，两者不可同时出现。无需 SVG 字符/框等训练标签，像素归一化与原训练路径逐位一致。`release.RecordLoader` 现支持上述显式混合格式，launcher 在加载模型前完成两种来源的 payload 检查。身份评分辅助函数位于不依赖模型的 `validation_truth.py`，旧 evaluation 导入接口保持可用。

真实固定子集已核验 **200 分子 / 800 原图**，四种 variant 各 200，数据来自 committed fresh_dev，不是旧 train 重绘。与剩余 282 个训练候选的分子 ID、完整键、连接层、新 core/scaffold 交集均为零。结果在 `outputs/audits/chemvector-fresh-dev-input-20260914/`；不含模型预测或准确率，未开启 GPU。fresh_dev/sealed 在新分组下的旧交集问题仍存在，不能声称它们完全独立，也未重划或删除评估样本。详见 `docs/model/training/20260914-fresh-dev-original-input.md`。

固定候选的公开筛查保留 282 个分子 / 2,469 张图，并继承前轮隐藏排重凭据。2026-09-15 已在上述独立发布副本中分配 train/validation 并完成准入，源目录仍 quarantine。公开 13 条全局身份异常仍保留，5 个不能排除风险的候选整体隔离；历史筛查详见 `docs/model/training/20260914-public-candidate-screen.md`。

CPU 合成测试覆盖实际 optimizer 更新、混合域采样、验证模式隔离、无效输出分母、早停、最佳模型选择、整套权重一致性、写盘失败不发布、旧权重路径不回退、数据与身份变更拒绝。完整身份评分单测覆盖同位素、电荷、立体中心和盐。

另有真实旧权重的无更新 CPU 保存/加载检查，乙醇样本前后预测身份相同；不声称它验证了泛化效果。旧配方已经完成过真实 GPU 续训并发生退步；新冻结配方也已完成一次 800 步真实 GPU 试训，role 摘要不变、梯度有限，但开发整图仍退步。两次均未部署新模型。
