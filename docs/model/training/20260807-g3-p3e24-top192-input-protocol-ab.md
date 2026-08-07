# G3 P3-E24 Top192 输入协议 A/B

> 日期：2026-08-07
>
> 状态：预注册，待运行
>
> 审计：`chemvector-g3-p3e24-top192-input-protocol-ab`

## 问题

P3-E23 证明网络权重形状支持 128 query × 192 candidate，但 P3-E16 Count 在当前 top192
evidence 上崩溃。P3-E22 提取 256 候选时，`normalized_scores` 使用整组候选做逐图均值/标准差，
P3-E23 又把 `scale_ranks` 改为 rank/191；因此原前128项也不再复现模型训练时的输入。

本轮零训练比较两种预先固定的 192 输入协议，先用真实 Count 的 known-count 解码隔离 Count
错误，判断 base/expert pointer 与冻结 router 是否仍可用。

## 固定两臂

1. `global192`：P3-E23 已生成协议，192项共同标准化 logit，rank 为 `0..191 / 191`；
2. `prefix192`：candidate ID、视觉特征、坐标和 raw score 仍取 P3-E22 前192项；
   normalized score 使用前128项 logit 的均值/标准差变换全部192项，rank 为 `0..191 / 127`。

`prefix192` 的前128项必须与 P3-E21 的 candidate ID、视觉特征、坐标、raw score、
normalized score 和 scale rank 逐位一致；新增64项允许 normalized score 与 rank 超出旧训练范围，
但不得裁剪、重新排序或按图片改变公式。

不加入第三臂，不重训权重，不调 router policy。冻结使用 P3-E7 的既有 policy，真实 Count 只进入
评估解码，不进入模型 forward。train/dev/旧 confirm manifest不变；旧 confirm 仅作
`budget_consumed_diagnostic`，不参与协议选择。

## 指标

每臂在 train/dev/旧 confirm 报告：

- base、expert 和冻结 router 的 known-count Atom set exact；
- base/expert matched atoms 与 recall；
- 相对 top128 的候选集合 overlap、前128 pointer 分布漂移和新增 tail 候选选择数；
- P3-E16 Count exact/MAE仅作分层诊断，不与 known-count 结构指标混合。

## Query Extension 门与选择

只用 dev640 选择输入协议。某臂必须同时满足：

- 冻结 router known-count exact 相对 top128 回退不超过2pp；
- router matched-atom recall 相对 top128 回退不超过0.2pp；
- base 与 expert 各自 known-count exact 回退不超过3pp；
- 所有输出有限、重复运行一致、前192 candidate recall不低于99%；
- 没有 Count、split、variant、字体、scaffold 或 GT mask 进入模型 forward。

两臂都通过时选择 `prefix192`，因为它保留 checkpoint 的前128输入契约；只有 prefix 失败且
global 通过时才选择 global。两臂均失败则 `top192_query_extension_no_go`，下一步必须在192输入上
适配 query/pointer，不训练 Count 来掩盖结构回退。

若 query extension 通过但所选臂 P3-E16 的 dev exact回退超过2pp或MAE增加超过0.10，则判定
`top192_query_extension_go_count_adaptation_required`，授权下一周期只用train evidence重新适配
Count；不得直接恢复五路线。若 Count 同时兼容，则判定 `top192_input_protocol_go`。

本轮不读取 fresh_dev、USPTO、MR-TADF、封存 confirm_v2 或 sealed_test，不产生 checkpoint，
因此不创建模型 tag。

