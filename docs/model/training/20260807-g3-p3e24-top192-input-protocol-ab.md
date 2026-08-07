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
评估解码，不进入模型 forward。train只报告结构分布并作为未来Count训练来源；dev是唯一协议
选择集。旧 confirm 不在本次选择脚本中读取。

## 指标

每臂在 train/dev 报告：

- base、expert 和冻结 router 的 known-count Atom set exact；
- base/expert matched atoms 与 recall；
- 相对 top128 的候选集合 overlap、前128 pointer 分布漂移和新增 tail 候选选择数；
- P3-E16 Count exact/MAE仅作分层诊断，不与 known-count 结构指标混合。

## Query Extension 门与选择

只用 dev640 选择输入协议。为避免浮点边界歧义，兼容门固定为整数：

- 冻结 router known-count exact 相对 top128 最多净少12张；
- router matched atoms 相对 top128 最多净少64个；
- base 与 expert 各自 known-count exact 相对 top128最多净少19张；
- 所有输出有限、重复运行一致、前192 candidate recall不低于99%；
- 没有 Count、split、variant、字体、scaffold 或 GT mask 进入模型 forward。

兼容还不足以证明新增候选有用。P3-E22 在dev上将Candidate Oracle从32053/32400提高到
32191/32400，新增138个可达原子、30张整图。效用门固定为：

- router matched atoms相对top128净增至少69个，即吃到oracle增量的一半；
- paired exact rescues不少于breaks；
- top128已可达保护集的paired breaks不超过6/640；
- 至少正确选择一个索引128--191的tail candidate。
- 在top128整图不可达而top192整图可达的30张dev图片中，至少救回1张Atom set exact。

两臂都通过时选择 `prefix192`，因为它保留 checkpoint 的前128输入契约；只有 prefix 失败且
global 通过时才选择 global。两臂均失败则 `top192_query_extension_no_go`，下一步必须在192输入上
适配 query/pointer，不训练 Count 来掩盖结构回退。

结构选择阶段完全不计算P3-E16。选定臂后立即写
`query_protocol.commitment.json`，绑定臂、公式、candidate/cache/checkpoint/router policy、源码和
dev指标；此后不允许因Count结果换臂。随后只对已选臂计算train/dev P3-E16：

- dev exact回退不超过2pp且MAE增加不超过0.10，同时train同方向无异常：
  `top192_input_protocol_go_count_compatible`；
- 结构门全部通过，dev Count超过任一门，且train也超过至少一个同口径门：
  `top192_query_extension_go_count_adaptation_required`；
- 结构门未过：`top192_query_extension_no_go`；
- 兼容门过但效用门未过：`top192_shape_compatible_but_not_useful`。

Count仍固定0--128共129类。未来与192 candidate统一接口连接时，只允许在右侧补64个`-inf`，
不得训练没有标签支持的129--192类。

本轮结果及commitment提交后，才允许另起脚本读取旧confirm，且只能标记为
`post_selection_consumed_diagnostic`；无论结果好坏都不得改变协议或Count授权。

本轮不读取 fresh_dev、USPTO、MR-TADF、封存 confirm_v2 或 sealed_test，不产生 checkpoint，
因此不创建模型 tag。
