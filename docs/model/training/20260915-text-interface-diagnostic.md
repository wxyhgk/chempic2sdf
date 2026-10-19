# 文字接口对照：角色头是已定位的回归来源，尚未获得可替换基线的模型

上一轮整图开发验证从 170/800 退步到 66/800 后，本轮没有继续训练。固定 24 个不同分子的 24 张原图，只在 CPU 上复用已保存的 initial/last 权重做五组推理对照。

**仅将 last 的文字角色头换回 initial，正确数从 4/24 恢复到 15/24；仅换回文字框头没有救回整图。** 这说明文字角色接口的权重变化确实造成了这些样本中的一部分退步，不再只是根据元素计数提出的猜测。它不证明整个训练回归都来自 role，也不证明冻结 role 后重新训练一定有效。

## 固定方案与边界

- 原训练与开发验证集合、四个推理阈值不变：atom=-0.25、bond=-0.5、pair=0.85、role=0.85。
- 先根据已有 initial/last 结果分层，每种样式抽 3 张组成退步、1 张其他退步、1 张保持正确、1 张始终错误；每层按 `sha256(20260915:key)` 排序，全局分子 ID 不重复。共 24 张，四种样式各 6 张。旧实验救回组不在本次定额抽样内。
- **这是按旧错误有意富集的诊断集合，不是新的验证集或准确率估计。** 其中 initial 正确 20 张是抽样设计的结果，不可与整体验证的 21.25% 直接比较。
- 在读取图片、加载模型前保存 `plan.json`，固定样本、全部五个对照和来源哈希；没有根据新结果换样本或追加对照。
- 每张图仅计算 initial/last 两份结构预测。三组替换对照复用完全相同的 last 候选、坐标、原子对、键输出和编译张量，每次调用后核对完整张量摘要。不是把初始模型的结构偷偷换回来。
- role/bbox 接收原像素和预测锚点；OCR 始终使用同一份冻结权重。真值仅用于旧结果分层和推理后的身份评分，不进入模型或裁框。
- 使用严格 SDF 身份评分，保留立体、电荷、同位素、碎片及无效结果；未改变 sanitize/fallback 行为。
- CPU 共 **58.15 秒**，600 秒上限；`status=completed, exit_code=0`。GPU 使用、优化器更新、自动重试均为 **0**。
- initial/last 的 48 次 CPU 基线回放在预测身份、真值身份、有效性、严格匹配和 fallback 上全部重现原 GPU 保存结果。结构包与 OCR 的参数/缓冲区摘要运行前后完全一致。
- 封存目录仍为 `000`，未读取 sealed 样本、重新查询私有身份或重建数据发布。前端和已有 checkpoint 未修改，没有选择或部署组合权重。

## 结果：正确图片数量，而非总体准确率

| 对照 | 组成退步 /12 | 其他退步 /4 | 保持正确 /4 | 始终错误 /4 | 合计 /24 |
| --- | ---: | ---: | ---: | ---: | ---: |
| initial 全部组件 | 12 | 4 | 4 | 0 | 20 |
| last 全部组件 | 0 | 0 | 4 | 0 | 4 |
| last 结构、initial role、last bbox | 9 | 1 | 4 | 1 | 15 |
| last 结构、last role、initial bbox | 0 | 0 | 4 | 0 | 4 |
| last 结构、initial role、initial bbox | 8 | 1 | 4 | 1 | 14 |

仅回退 role，救回 16 个新增错误中的 **10 个**，另救回 1 个 initial/last 都错的案例。四个保持正确的控制样本均未改错，但控制组很小，不能推断完整开发集零退步。还有 6 个新增错误未修复，不能把角色头视为全部原因。

## 接口级证据

在 last 的相同候选位置上，仅回退 role 共使 **42 个候选重新进入 OCR**，分布于 15 张图；没有候选被反向关闭。原本两组都进入 OCR 的候选，解码文本变化为 0。新开启位置的 OCR 字符串计数为 O=13、N=13、S=7、Cl=3、OH=3、NH=2、F=1。这是模型输出计数，不是这些局部标签全部正确的独立认证。

例如 `fresh_dev_001883_original`，结构、bbox 与 OCR 权重保持 last 不变：

| last 候选索引 | last role 概率 | initial role 在同一位置的概率 | 重新开启后 OCR 文本 |
| --- | ---: | ---: | --- |
| 53 | 0.7732 | 0.9973 | F |
| 55 | 0.2421 | 0.9194 | Cl |
| 63 | 0.6272 | 0.9894 | OH |

固定文字阈值为 0.85。last 将这三个位置挡在 OCR 入口外；换回 role 后读取文字，最终 SDF 从错误恢复为严格匹配。这里可以具体定位到预测候选，不再只是根据整个分子的 N/C 等计数猜测。

目前确认的是“**同一预测位置，续训后的 role 权重使文字判断掉到现有阈值以下**”。仍需区分训练标签/匹配质量、域与样式分布、遗忘、置信度校准等上游原因。不能直接把阈值调低并宣称修复，也不能据此判定原子检测或结构预测全部正确。

## 为什么不一起回退 bbox

两者一起回退，比只回退 role 少正确 1 张，但不是所有图片都同向变化：

- `fresh_dev_001209_original`：在 initial role 下，initial bbox 让某候选从 `NH` 读为 `N`，该图恢复正确。
- `fresh_dev_001205_coordgen`：initial bbox 将 `Cl` 读为 `Cl2`，原本 role-only 已恢复的图再次出错。
- `fresh_dev_000965_coordgen`：initial bbox 将几个 `O` 读为 `OH`，最终出现价态错误，无效结果继续计入分母。

因此 bbox 变化也有局部影响，但证据不支持“所有文字组件一律退回最初权重”。role 与 bbox 要分开检查，不能将换回旧 role 的收益归给 OCR 重训或图结构改进。

## 工程产物与复现

`outputs/audits/chemvector-text-interface-20260915/`：

- `plan.json`：预先固定方案，SHA256 `8765f94946642024eaca89013339cf4a2ef305171bf8002429502065e478329b`。
- `report.json`：全部分层计数、初始/末尾回放一致性、模型状态摘要、输入与实现关联。
- 24 个 `fresh_dev_*.json`：候选坐标、结构摘要、每个候选的角色概率、框、原始 OCR、解析/附着记录、fallback 前后原子属性及每组预测 SDF。
- `runtime.json` 与 `worker.log`：实际 CPU 时限、耗时、状态及逐图进度。

代码职责分离：`interface_plan.py` 只负责固定分层；`interface_diagnostic.py` 负责预算内对照与证据保存；`evaluation.py` 将现有推理拆成 `predict_structure` / `from_structure`，默认 `__call__` 仍顺序调用同一流程，未修改决策阈值或化学规则。可选 trace 只记录，不参与决策。

最终相关 CPU 回归 **175 passed（23.03 秒）**，覆盖固定抽样/身份拒绝、五组模块接线、默认与缓存推理一致、trace 不影响输出、无效结果分母、禁止训练样本进入诊断，以及原有续训、bundle、release、fresh-dev、runtime 和 protocol 合同。额外复核全部 9 个旧 checkpoint 与前端 registry 哈希不变，诊断计划与报告记录的实现哈希一致。`git diff --check` 通过；上述合成测试不计入真实模型效果。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-train/bin/python -m chemvector.training.curriculum.interface_diagnostic \
  --run outputs/experiments/chemvector-mixed-continuation-20260915 \
  --manifest outputs/datasets/chemvector-mixed-release-20260915/manifest.json \
  --admission outputs/datasets/chemvector-mixed-release-20260915/admission.json \
  --output outputs/audits/chemvector-text-interface-new \
  --wall-seconds 600
```

输出目录必须不存在。以上为复现命令，不是自动启动下一轮的指令。

## 诊断结束时的下一步建议（历史记录）

1. 保留 initial 作为现有基线，停止七组全部更新的原续训配方，不重跑、不扩大训练预算。
2. 下一项无训练验证只固定一个候选：**last 结构 + initial role + last bbox + 原冻结 OCR**。在完整的既定 800 张开发图上检查救回、改错和各样式结果，尤其检查这次 4 张正确控制图没有覆盖的风险。不要搜索多个阈值或把本次 24 图的最高计数当成可部署指标。
3. 同时优先审计 role 的训练正负标签、预测锚点匹配及样式覆盖。只有明确其漂移来源、检查完整开发集后，再决定是否进行一次“冻结 role、训练其他组件”的定额实验；**推理时换回 role 不等于已验证训练时冻结 role**。

本轮已经完成的是退步定位，不是获得新模型、更高的整体验证准确率或新训练协议的成功验证。

## 扩量后的提案更新

随后完成增量数据发布及 opt-in `freeze-role-v1` 接线，真实权重的 4 图 CPU 无更新检查通过。用户已确认并执行从 initial 出发、冻结 role/OCR 的单卡 30 分钟 / 1,200 步回归试训：实际在 800 步早停，117/800 低于 initial 的 170/800；role 状态确实保持不变，未部署。详见 [本次冻结训练结果](20260915-expanded-freeze-role-result.md)。

上述完整 800 图混合权重推理诊断尚未执行，不作为已完成证据。本次直接冻结训练的提案替代“先做该推理诊断”的原先顺序；它也不证明 role 监督漂移根因已查清或冻结有效。详情见 [扩量后开训准备](20260915-expanded-training-readiness.md)。
