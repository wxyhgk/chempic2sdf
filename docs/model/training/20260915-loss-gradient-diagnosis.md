# 联合损失梯度归因：共享层有冲突，但不能直接解释全部回退

本次只做固定 CPU 求导与公式对照，**未创建真实模型的 optimizer、未更新权重、未修改学习率/损失权重/Dropout/参数分组/阈值，未启动 GPU 训练**。selected 标签修复保持不变。

结论：原子、键、set 辅助损失在共享视觉层上的合成梯度较大，并与 pair 梯度存在局部冲突；但这种冲突不等于实际 AdamW 更新会让连键损失上升。8 个固定检查点按空状态 AdamW 首步方向计算的本地 pair 方向导数均为负，不能把“梯度冲突”直接定为验证集退步的主因。

## 固定检查范围

在加载模型和像素之前，固定此前 12 张训练图清单中的索引 `0、1、8、11`：

- `full_molecule_expansion:curr_d1248897c226b214__serif`。
- `full_molecule_expansion:curr_b7a85b1f6804b7df__font_small`。
- `curriculum:curr_a5f70e40f982d786__font_small`。
- `curriculum:curr_8419bd7a8bf844e9__bond_short`（C=N）。

即前两张完整分子、第一张课程图和既有 C=N 见证；不是按梯度结果挑图。initial/last 各检查一次，均为正常 train 模式，每次前向独立固定种子 `20260915`。4 张图 × 2 份权重，共 8 个状态。

每个状态只前向一次，在**同一个计算图、同一组预测候选与采样对**上分别求 atom、bond、set_head、pair、bbox 五项梯度，再求总损失梯度。冻结角色头的 loss 为零，不伪造第六项有效监督。`autograd.grad` 不写入参数 `.grad`，空分支照实报告。

没有读取新测试图、封存样本或私有身份清单；已准入训练样本以 diagnostic 用途读取。

## 各项损失实际影响哪些参数

8 个状态中观察到的非零梯度来源与当前接线一致：

| 参数组 | 有非零梯度的损失 |
| --- | --- |
| vision | atom、bond、set_head、pair |
| atom | atom、set_head、pair |
| bond | bond、set_head |
| set_head | set_head |
| pair | pair |
| bbox | bbox |

因此 bbox 回归不能简单归因于“别的损失直接向 bbox 参数传入冲突梯度”。其输入锚点与可达样本会受上游影响，但当前 bbox 损失不会直接更新视觉编码器或原子锚点。角色头继续冻结。

## 共享视觉层确实存在局部冲突

定义 `g_aux = g_atom + g_bond + g_set`，并与 `g_pair` 在同一参数空间比较。梯度范数比不是参数更新倍数，负余弦也不代表整个系统的更新方向一定有害。

initial 的四个检查点：

| 样本 | 辅助/连键梯度范数比，vision | aux 与 pair 余弦 | 总梯度与 pair 余弦，vision |
| --- | ---: | ---: | ---: |
| 完整分子 serif | 14.89 | −0.0789 | −0.0118 |
| 完整分子 font_small | 137.65 | −0.0386 | −0.0313 |
| 课程 font_small | 103.16 | −0.0524 | −0.0428 |
| C=N | 9.82 | −0.0577 | +0.0441 |

initial 的 aux 与 pair 在 vision 上 4/4 方向夹角大于 90°；合成总梯度与 pair 为 3/4 负余弦。last 的对应计数均为 2/4，aux/pair 范数比约 35.25–397.08。

这些结果说明原本主要服务于下游的视觉特征，现在受到更强的多项辅助监督；但 pair 损失本身已较小时，梯度比值也会变大，不能直接据此设定新的 loss 权重或声称模型受损同样的倍数。

## 为什么不能只看负余弦就判定训练方向错误

实际 launcher 使用 AdamW、weight_decay=0，并对全部可训练参数做 norm=1 的统一裁剪，不是沿未缩放总梯度直接走一步 SGD。

本次只按公式构造一个未应用的参考方向：假设 optimizer 的一、二阶矩全空，使用当前各组学习率、eps=1e-8、全局裁剪和首步偏差修正。各参数方向为：

```text
d = -lr * clipped_gradient / (abs(clipped_gradient) + 1e-8)
```

逐组计算每项损失的方向导数 `g_task · d`，再汇总全部参数组。**8/8 状态的 pair 汇总方向导数为负，其他四项损失也均为负。** 例如 initial 的完整分子 serif，虽然 vision 总梯度与 pair 余弦为负，按 AdamW 参考方向计算的 vision 部分导数仍为负；参数自适应缩放不能被原始余弦替代。

这个参考有严格限制：

- 没有执行参数更新；方向导数不是一次有限步长后实际 loss 的变化量。
- 对 last 也假设空状态，仅作统一的局部参考，不是重放它训练中的真实历史 momentum。
- 不覆盖跨样本梯度、长期优化、候选 top-k/阈值跳变或验证集准确率。
- 因此，它既不能证明续训无害，也不能证明需要梯度手术、关闭某项损失或更换架构。

## 训练图上的损失下降，不等于验证集恢复

在相同固定种子、修复后的标签语义下，四张图重算的总损失从 initial 到 last 均下降：约 `5.40→2.75、4.93→2.54、12.70→1.75、20.85→1.99`。两份模型的离散候选与可达标签可能不同，这不是相同固定预测上的纯损失对照。

历史完整分子开发集却从 `170/800` 退到 `117/800`，且此前上游结构/bbox 组件对照已有回归证据。这说明只盯训练 loss、梯度有限性或训练图监督覆盖不足以保证完整分子效果；本次并未定位全部泛化损害的来源。

## 数值核验与失败记录

首次检查的梯度和重建门槛设为 `2e-5`，第一张真实图的误差 `4.5008e-5` 超过门槛，9.579 秒后退出；原失败目录 `outputs/audits/chemvector-loss-gradients-20260915/` 保留。

FP32 在单次联合反传与多次分项反传中可能以不同顺序累加共享路径。第二次明确记录各组误差并使用 `1e-3`（0.1%）的数值门槛，模型与损失代码未改；另加测试确认漏掉真正的损失项仍会被拒绝。8 个状态最大相对误差为 **0.000189546，即 0.01895%**。这不是调训练阈值，也不是模型准确率容差。

## 下一步建议：只验证已经确认的标签修复

目前已确认并修复的是 selected 匹配标签缺陷；Dropout 与梯度对照都没有充分证据支持叠加新的配方修改。建议下一轮先做单一改动的短训：

- 从原 initial 重开，不接着退步的 last 训练。
- 使用同一已准入数据、原采样顺序和种子；模型、冻结角色头、学习率、loss 权重与 Dropout 均保持原配方，仅使用已修复的 selected 标签接线。
- 拟定最多 **400 步**，batch=1，每 **100 步**使用原固定开发集验证，完整分子 800 张为主指标；课程 144 张仍独立报告。
- 连续两次完整分子 exact 低于初始基线即停；预算耗尽也停，无自动重试或续跑。
- 单卡墙钟上限 **600 秒**（最多 0.1667 GPU-hour 的保守上限，不是预计耗时）；若基线复现不一致，先停查，不开始优化。
- 未超过初始 exact 时保留原 initial，不将退步权重接入前端；即使改善，也单独核验后再决定部署。

**以上是待确认的执行建议，本次未训练。** 新的步数/验证频率需要单独的运行协议快照与一次新的 GPU 授权，不覆盖旧 manifest/admission 或历史实验协议，不重复私有身份排重。短训无改善只说明该受限预算内未验证收益，不能证明模型路线不可行或标签修复无必要。

## 产物与验证

- 实现：`chemvector/training/curriculum/gradient_diagnostic.py`，固定样本与模式、300 秒 CPU 上限，不接受运行时数据/种子/优化器搜索参数。
- 新测试：`tests/chemvector/test_curriculum_gradient_diagnostic.py`，8 项，覆盖已知冲突、空梯度无方向、梯度和、真实 tiny 模型接线、RNG/状态不变、漏项检测、AdamW 公式。只有合成参数公式测试构造 optimizer，真实模型诊断不构造。
- 相关测试：`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-train/bin/python -m pytest -q tests/chemvector/test_curriculum*.py tests/chemvector/test_direct_atom_pair_bond.py tests/chemvector/test_direct_atom_role.py tests/chemvector/test_direct_atom_text_bbox.py`，**333 passed in 49.81s**，不是全仓库测试声明。
- 最终目录：`outputs/audits/chemvector-loss-gradients-20260915-v2/`，包含 plan、8 个完整梯度表、state/输入摘要检查和 runtime receipt。
- completed，**59.299 秒**，CPU 上限 300 秒；加首次失败共约 68.879 秒，不含测试。没有自动重试，GPU=0，optimizer steps=0。
- 参数、缓冲区、模式、requires_grad 与空 `.grad` 在真实核验前后不变；来源权重、原 checkpoint 和绑定输入摘要检查通过。前端、数据发布和训练协议未改。
- worker.log SHA256：`18d670c3f9a030c95e34153de6f6c4ca776b1df580547016099a862219f552af`。
- report.json SHA256：`00440c76be6a87ce5f72e1543586c7dc7ff96bd8ac01c67dee9b01e78df07866`。
- 诊断代码 SHA256：`96e6ca2ead7efcc73e5290c90f4a6bc9b7cb0c9770b944c38beefa50c9956529`。
