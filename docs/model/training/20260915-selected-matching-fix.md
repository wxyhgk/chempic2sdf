# 续训 selected 匹配修复与 CPU 核验

本次落实此前排查的第一项修复：**raw/set 监督保留原匹配，最终 pair、role、bbox 监督对实际选中的候选重新匹配**。未改架构、学习率、损失权重、采样配方或阈值，未启动 GPU 训练。原排查报告和历史实验记录保持不变。

## 代码改动

`chemvector/training/curriculum/continuation.py:118` 不再直接遮掉 raw 映射中未选候选的条目，而是在 `torch.no_grad()` 下复用既有 `match_atom_targets(selected, batch, config)`，供连键和文字监督共同使用。

- raw/set 仍接收 `associate_predicted_proposals()` 对所有 raw 候选建立的标签。
- selected 匹配不改变候选坐标、数量、选择结果；未注入真值候选。
- 保留既有 Hungarian 匹配和 6 px 半径；空集合和半径外候选不会被强行匹配。
- 未修改通用 `build_pair_targets()` 或负例采样算法。正例数恢复后，既有采样配方可能选择更多训练对，这是标签修复的结果，不是改了采样比例。
- `freeze-role-v1` 仍不更新角色头；`role_rows` 是可达标签计数，不表示执行了角色头更新。

## 回归测试

新增 `tests/chemvector/test_curriculum_selected_matching.py`，用受控候选接入实际 continuation loss、两层匹配、采样和下游模型。不是只测试一个与训练入口脱离的匹配 helper。

10 项新用例覆盖：未选近邻不能抢走已选候选的连键/文字标签、原始匹配仍供 set 使用、真实非键保持负例、正负 presence 梯度方向、单候选/空集合/部分集合、6 px 边界、batch 隔离，以及冻结/非冻结角色头两种配方。

先在旧接线上运行：9 failed、1 passed；修复后新用例全部通过。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-train/bin/python -m pytest -q tests/chemvector/test_curriculum*.py
# 310 passed in 47.30s

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-train/bin/python -m pytest -q \
  tests/chemvector/test_direct_atom_pair_bond.py \
  tests/chemvector/test_direct_atom_role.py \
  tests/chemvector/test_direct_atom_text_bbox.py
# 8 passed in 1.01s
```

共 **318 项通过**。这些是相关测试，不宣称全仓库测试通过；合成单步测试不使用真实 checkpoint 或真实训练数据。

## 固定真实样本核验

复用此前 probe 在加载模型/像素前固定的 12 个训练样本 key（前 8 张完整分子、前 4 张课程图），不重新挑样、不访问封存样本。加载保存的 initial 与 last，各执行 12 次修复接线核验，共 **24 次**。

核验调用实际 `continuation_losses()`，检查传入连键目标的映射；对历史接线的重放只替换监督映射，使用相同预测张量。24 次均复现此前记录中的匹配计数；raw/set 损失逐项完全相同，预测张量未被修改。

initial 中三张受影响图片：

| 样本 | 已选原子匹配，修复前→后 | 正键对，修复前→后 | 文字框监督行，修复前→后 |
| --- | ---: | ---: | ---: |
| `curr_eacef754193a270a__font_large` | 13→16 | 11→15 | 3→6 |
| `curr_2d374a1e00db024e__font_large` | 13→16 | 12→18 | 2→4 |
| `curr_8419bd7a8bf844e9__bond_short`（C=N） | 0→2 | 0→1 | 0→1 |

在这组固定检查中，先前 **11 个被送入负例的正键对全部恢复**，恢复 **8 个已选原子匹配、6 条文字框监督**；没有出现原正键被改成负键的反向翻转。last 在同一组中没有匹配变化。这不是全数据集错误率或模型准确率估计。

C=N 的预测概率仍为 `0.9909895062`，没有因核验发生变化。旧负标签的单对 BCE 导数是 `+0.9909895062`；修复后实际反向传播得到 `−0.0090104938`，化学键标签为 double。该梯度检查采用 **eval 模式、启用 autograd** 来保持原见证的预测候选；六个可训练参数组都有有限非零梯度，角色头保持冻结。

## 不能忽略的训练模式覆盖差异

另执行一次真实 train 模式的 C=N 前向/反向，不创建 optimizer。该次前向只选出 1 个原子，所以 pair rows=0，pair 没有梯度；其他五组梯度有限且非零，角色头仍冻结。模型训练模式有 10 个活动的视觉 `Dropout2d(p=0.1)`，eval 模式则保留 2 个候选。

这是**一次随机训练前向的覆盖现象**，不是 GPU 步骤重放、发生率估计或 Dropout 解释全部退步的证明。本次没有关闭 Dropout、降低阈值或强行提供第二个原子。匹配修复处理的是“候选已经选出但标签丢失”，不能解决“原子根本没被选出”。

还有一张 `curr_fe8ee15bb86d27a7__bond_short`：原图有 2 个原子，initial 在 eval 下只选出 1 个；修复前后都没有原子对。不能把它误称为单原子图片。

## 产物与执行记录

- 核验脚本：`outputs/audits/verify_selected_matching_20260915.py`。
- 最终完成目录：`outputs/audits/chemvector-selected-matching-fix-20260915-v3/`，含固定计划、逐样本对照、两个模式的反向记录、输入/权重摘要和 runtime receipt。
- 最终检查耗时 **43.468 秒**，CPU 上限 300 秒；GPU=0，optimizer created=false，optimizer steps=0。
- 首次目录 `chemvector-selected-matching-fix-20260915/` 保留：15.936 秒后因脚本错误地断言每张图都会调用 pair loss 而退出。
- 第二次目录 `chemvector-selected-matching-fix-20260915-v2/` 保留：38.812 秒完成全部 24 次标签对照后，因脚本假定 train 模式 C=N 一定有原子对而退出。
- 两次均为核验脚本假设错误，未改生产代码来迎合断言。第三次显式记录空分支，并分开 train 模式覆盖检查与 eval 模式见证梯度检查；无自动重试。三次合计约 98.217 CPU 墙钟秒，未含单元测试。
- 最终 worker.log SHA256：`5843e73dbc448287c33c70e909883151ef36db4ddde63a27ffe0deb58616b36d`。
- 最终脚本 SHA256：`7677d7797d2a1520358641371f87a8f1c83c16d288414213f07ee0d52f85edf6`。
- 修复后 continuation.py SHA256：`8df3a021b022b67c201964b6157fc02cc4d1a90c3509ca94227a0c921ca7eb94`。

initial/last 的参数和缓冲区摘要在检查前后相同，原 checkpoint 文件、来源权重以及绑定的数据/历史记录摘要核验通过。未重跑私有身份排重、未读取封存样本内容、未修改数据发布或 admission，未加载或训练 OCR，未改变前端部署。

## 结论与下一步

标签接线缺陷已修复并通过真实样本检查，**准确率恢复尚未验证**；保留原 initial，不部署失败的 last，也不启动新训练。

下一步优先用固定样本、固定随机种子核对 train/eval 的候选数与正键可达监督，避免把训练时空分支误判为匹配标签问题；再核对先前发现的参数分组、学习率和图级约束差异，形成一次有界重训方案。此前一次 GPU 训练授权已经执行完，本报告不复用该授权。
