# 训练/推理状态诊断：Dropout 改变候选，但未见明显监督覆盖下降

本次承接 selected 匹配修复后的排查，只运行固定 CPU 对照。**没有修改训练配方、Dropout 配置、学习率、参数分组、损失权重或推理阈值，没有更新权重或启动 GPU 训练。**

结论：视觉 Dropout 确实解释了本次观察到的 train/eval 预测张量差异；但在固定样本上，正键监督可达率差别很小，不能据此认定它是大幅退步的主因。暂不把“关闭 Dropout”作为已确定的修复。

## 代码上确认的训练条件变化

旧入口与当前续训的状态不同：

- `train_joint_bond_set_multidomain.py:405`：vision 和 atom 使用 `eval().requires_grad_(False)`；set_head 可训练。
- `train_direct_atom_pair_bond.py:430`、`train_direct_atom_role.py:188`、`train_direct_atom_text_bbox.py:173`：上游结构模块使用推理模式并冻结。
- 当前 `continuation.py:46` 的 `train()` 会启用前五个视觉 stage 的训练模式，仅保持未用 stage 和冻结角色头为 eval。
- `core/model.py:152` 的 ConvNeXtBlock 使用 `Dropout2d`；实际 checkpoint 中参与前向的 10 层为 `p=0.1`。

这说明续训改变了原下游模块熟悉的视觉特征分布，但“有分布变化”本身还不是造成验证集回归的因果结论。

## 固定对照协议

复用此前已固定的 12 张训练图（8 张完整分子、4 张课程图），initial/last 各一份保存权重。读取这些已准入训练样本仅用于 diagnostic；不访问封存样本，不做新的私有身份核验。

随机种子在加载模型/像素前固定为 `20260915、20260916、20260917、20260918`。不搜索种子，不因结果更换样本。每次前向独立恢复 RNG，四种模式为：

| 模式 | 模型主状态 | 参与结构前向的视觉 Dropout |
| --- | --- | --- |
| eval | 推理 | 关闭 |
| train | 训练 | 开启 |
| train_vision_dropout_off | 训练 | 仅临时关闭这些 Dropout |
| eval_vision_dropout_on | 推理 | 仅临时开启同一批 Dropout |

两个确定性模式每图各运行一次；两个随机模式每图各运行四个种子。共 `12 × 2 × (1+1+4+4) = 240` 次前向。模式切换不修改 Dropout 的 p、任何参数值或 requires_grad；离开对照上下文后恢复每个模块原本的混合 train/eval 状态。

使用修复后的 selected 匹配语义和原 6 px 半径。真值只在预测之后计算可达监督，不进入结构前向，也不补入候选。

## 对照结果

“正键可达率”指真值键两端均有已选且成功匹配的候选，可以生成正键监督；**不是键预测准确率，更不是完整分子的识别准确率**。

| 权重 | 模式 | 已选原子匹配 / 真值原子 | 可达正键 / 真值键 | 正键可达率 |
| --- | --- | ---: | ---: | ---: |
| initial | eval | 159/160 | 165/166 | 99.40% |
| initial | train，四种子合计 | 637/640 | 661/664 | 99.55% |
| last | eval | 160/160 | 166/166 | 100.00% |
| last | train，四种子合计 | 639/640 | 663/664 | 99.85% |

四个种子是同一批样本的重复观察，不能当作四倍独立图片；该组是训练样本，不是新验证集。表内微小升降不支持统计显著性或泛化收益结论。

精确张量对照包括视觉特征、raw 原子/键候选、set 输出和 selected 候选：

- **24/24** 个“模型×图片”对照：train 中仅关闭视觉 Dropout，与 eval 的张量摘要完全相同。
- **96/96** 个“模型×图片×种子”对照：eval 中仅开启同一批视觉 Dropout，与正常 train 完全相同。
- 相对同图 eval，initial 的 48 次随机前向中，2 次选择数量变化；合计丢失 1 个原来可达的目标键，同时恢复 2 个原来不可达的目标键。
- last 的 48 次随机前向中，3 次选择数量变化；合计丢失 1 个原来可达的目标键，无新增可达目标键。

所以，在本次固定范围内，Dropout 的直接状态影响可被隔离，但没有观察到大量原子/正键监督被切断。

## C=N 单例与梯度检查

上次核验中，C=N 在一次 train 前向里只留下 1 个原子。本次预先固定的四个种子下，initial 的 C=N **4/4 都留下 2 个原子，保留 1 条正键监督**。没有为复现那次失败而继续搜索随机种子。

上次前向使用此前计算推进后的 RNG 状态，本次每个前向独立设种子，不能把二者当作同一随机状态的重放。上次单例是风险提示，不应据此推断常态发生率。

另外在 initial 的 C=N 上执行一次 `train_vision_dropout_off` 模式的实际 forward/backward：vision、atom、bond、set_head、pair、bbox 六组均有有限非零梯度，角色头保持冻结，2 个匹配原子、1 条正键、1 条文字框监督。未创建 optimizer。

这证明“只关闭 Dropout”不等于冻结梯度，但**未证明这样训练更好**。本次没有将该诊断模式接入续训入口或训练协议。

## 不能从本次结果推出什么

- 没有运行最终连键分类和 OCR/SDF 准确率对照，也没有训练。因此不能排除 Dropout 对下游预测或长期优化的影响。
- last 在这 12 张图中可达全部真值原子/键，但 eval 多选了 3 个未匹配候选，initial 为 0。监督覆盖高不代表候选精度、键分类或完整分子正确率高。
- 上次验证集上的大幅回归和上游/bbox 组件对照证据仍然成立。本次检查没有恢复准确率，也没有证明分子图路线不可行。

## 下一优先级

1. 保持已修复的 selected 标签接线，暂不叠加 Dropout 配方变更。
2. 优先核对联合损失与更新兼容性：旧配方冻结的 vision/atom/bond 现在收到哪些任务梯度；原子、键、set 损失是否在共享视觉参数上压过或冲突于最终 pair 任务；哪些参数组的名义学习率改变。先用固定样本做 CPU 梯度归因，不能只凭原始 loss 数值或参数变化百分比断言主因。
3. 根据归因结果锁定一次有界重训的唯一改动和停止条件，不同时试架构、Dropout、学习率和数据配比。此前 GPU 授权已执行完，不自动复用。

## 文件、核验与资源

- 诊断实现：`chemvector/training/curriculum/mode_diagnostic.py`；其入口固定样本来源、模式、种子和 300 秒上限，不接受运行时搜索参数。
- 测试：`tests/chemvector/test_curriculum_mode_diagnostic.py`，新增 7 项，检查混合状态异常恢复、冻结角色头、随机状态不变、交叉模式精确重放、权重不变和空候选行为。
- 相关完整测试命令：`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-train/bin/python -m pytest -q tests/chemvector/test_curriculum*.py tests/chemvector/test_direct_atom_pair_bond.py tests/chemvector/test_direct_atom_role.py tests/chemvector/test_direct_atom_text_bbox.py`。
- 测试结果：**325 passed in 49.54s**，不是全仓库测试声明。
- 真实诊断目录：`outputs/audits/chemvector-train-eval-modes-20260915/`，含 `worker.log`、`report.json`、`runtime.json`。
- completed，**162.101 秒**，上限 300 秒；一次执行，无自动重试，GPU=0，optimizer steps=0。
- initial/last 模型参数和缓冲区、模块模式、requires_grad 与 Dropout p 均在检查后保持不变。原 checkpoint、来源权重、输入与历史记录摘要核验通过；数据发布、前端和既有训练协议未改。
- worker.log SHA256：`bf29c08d400bc9dd64f6901314819d6b35b448cc4f0f30dbc1e86b128fbe7510`。
- report.json SHA256：`bfbaf0970176fdf7c16069d371a13180ef3b15700d9f265dc9f57b9a3d09ce35`。
- 诊断代码 SHA256：`ff3ed9b4593bc5cca30dd14dd1059507b5920befc692f5ae0eb2871c8e178794`。
