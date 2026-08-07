# G3 P3-E23 Top192 周期恢复审计

> 日期：2026-08-07
>
> 状态：预注册，待运行
>
> 审计：`chemvector-g3-p3e23-top192-cycle-recovery`
>
> 输出：`outputs/audits/chemvector-g3-p3e23-top192-cycle-recovery`

## 目标

P3-E22 已证明 top192 是 train/dev/confirm 同时通过 99% Candidate Oracle 门的最小固定
预算。本轮不训练新模型，只验证现有 128-query base/expert checkpoint 能否读取 192 个候选，
并重建 P3-E21 五路线共用的 top192 geometry、pointer 和 P3-E16 Count evidence。全部门通过后，
才恢复五路线 Micro Capacity；任一门失败即停止，不运行完整 SDF。

## 固定输入

- 候选来源：P3-E22 top256 cache 的前 192 项，不重新运行视觉模型；
- 候选 cache SHA256：
  `c796da5a33cfc954e3f4f25d9da9d70e70296a9b76d32e0736fd6d5cd2655720`；
- query 数固定 128，candidate 数固定 192，Count 类别固定 129；
- base/expert/P3-E16 checkpoint 与 P3-E21 完全相同，权重 `strict=True` 加载；
- train/dev/confirm manifest 与 P3-E21 完全相同；
- 不读取 `fresh_dev`、USPTO、MR-TADF 或 `sealed_test`。

只允许复制 checkpoint 的 `G3AtomQuerySetConfig` 并把 `candidates` 从128改为192。不得修改
query数、网络层数、权重、阈值、候选排序或按图片动态改变预算。

## 兼容审计

必须同时满足：

1. base 和 expert 均能 `strict=True` 加载原 state dict；
2. 单 batch 输出 shape 固定为：existence `[B,128]`、pointer `[B,128,192]`、
   count `[B,129]`；
3. P3-E16 DeepSets 可直接接收 128 个 query 与 192 个 candidate evidence，输出 `[B,129]`；
4. 相同输入重复运行的 argmax 与保存 tensor digest 一致；
5. 所有 tensor 有限，candidate ID 在每图内唯一。

## Oracle 与 Count 门

重建后的 train/dev/confirm Candidate Atom recall 必须各自不低于99%，并逐项复现 P3-E22
top192 的 matched/target/exact 计数。

P3-E16 的 128-candidate 原基线为：

| split | Count exact | MAE |
|---|---:|---:|
| train | 30.6250% | 3.4244 |
| dev | 28.5938% | 3.5078 |
| confirm | 32.5000% | 3.3578 |

192-candidate evidence 必须满足：

- dev 与 confirm 各自 Count exact 回退不超过 2pp；
- dev 与 confirm 各自 MAE 增加不超过 0.10；
- dev+confirm 合并 Count exact 回退不超过 1pp；
- 输出不得出现 NaN/Inf 或超出 0--128 的 Count。

这些门只判断旧 P3-E16 是否可作为固定基线，不允许据此微调模型或挑选阈值。若 Candidate
门通过但 Count 门失败，下一周期必须预注册 top192 Count 适配；不得在同一次运行中改回
top128 posterior。

这里的原 `confirm640` 已参与 P3-E22 的候选预算选择，因此在本轮只记作
`budget_consumed_diagnostic`。它不得用于五路线 Top-2、winner、阈值或 checkpoint 选择。
五路线恢复前必须另建与 train/dev/原 confirm 在 molecule、InChIKey、scaffold 和图片哈希上
零重叠的封存 `confirm_v2`；只有 Top-2 身份、checkpoint SHA、阈值、排序器和评测代码全部
提交后，才允许一次解封并同时评测已锁定 Top-2。

## 决策与后续

- 全部门通过：`top192_cycle_recovery_go`，冻结共享 cache，恢复 P3-E21 的五路线 Micro
  Capacity；Dev 后改用预先封存的 `confirm_v2` 承担确认门，五路协议与双下游完整 SDF 门不变；
- 任一门失败：`top192_cycle_recovery_no_go`，停止五路训练并记录具体失败层。

本轮是零训练兼容/分布审计，不产生 checkpoint，不创建模型 tag。只有后续实际训练出的每条
route 才按模型名 commit 并创建完全同名 annotated tag，包括 NO-GO route。
