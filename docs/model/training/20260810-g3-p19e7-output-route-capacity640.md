# G3 P19-E7 Output-route Capacity640

> 日期：2026-08-10
>
> 状态：已完成，容量 GO
>
> 模型：`chemvector-g3-p19e7-output-route-capacity640`

## 目标

P19-E6 三路线 Oracle 为 553/640，融合为 538，只有 37 张路线分歧。本轮先在已经消费的
calibration640 做严格容量实验，验证部署可见的输出统计是否有能力记住正确路线；不宣称泛化。

## 输入与动作

动作固定为 `KEEP_FUSION / USE_NEW / USE_OLD`，默认 `KEEP_FUSION`。输入只包含：

- 三路线选边数量及两两 symmetric difference；
- 新/旧/融合归一化 margin 的均值、方差、分位数、极值及阈值附近 pair 数；
- 三张预测图的 degree、isolated、components、cycle rank；
- 分歧 pair 上的 margin 摘要。

禁止输入 GT bond count、FP/FN、stress operation、目标 action 或任何像素 GT。GT 只用于生成
capacity 动作标签和评估。

## 固定训练协议

- 冻结旧模型、新模型、融合 alpha/threshold 与 vision/Atom；
- calibration640 全量构建 feature cache，eval640 不读取；
- 128 hidden 两层 MLP，最后一层零初始化，step0 必须全部 KEEP；
- full-batch、5,000 step、AdamW `3e-3`、weight decay 0、class-balanced CE；
- checkpoint 优先零 harmful，再按 exact、rescue、target action accuracy 选择。

## GO/NO-GO

- calibration 必须 640 张，Oracle 相对融合 headroom 至少 10；
- 可观测路线分歧至少 20 张，USE_NEW/USE_OLD 标签都存在；
- best 必须 0 harmful；
- best exact 距 Oracle 不超过 2 张，rescue 距 headroom 不超过 2 张；
- 冻结 source state 不变。

GO 只表示输出特征具备容量。下一步必须从未使用 full80k 余量建立 scaffold 隔离的
train/calibration/holdout，重新训练同一 router；当前 calibration 与 eval 都不能再用于选模。
无论 GO/NO-GO 都保存 checkpoint 并创建同名 annotated Git tag。

## 实际结果

capacity640 的可观测路线分歧为 123 张；以融合为默认，三路线 Oracle 为 559/640，相对融合
543/640 有 16 张 headroom。动作标签为：

- `KEEP_FUSION`：624；
- `USE_NEW`：9；
- `USE_OLD`：7。

78 维纯输出特征输入 27,267 参数 MLP。step0 最后一层全零，严格输出 640 个 KEEP，复现融合
543/640、0 rescue、0 harmful。step 250 达到并保持：

- selected exact 559/640，追平 Oracle；
- 16 rescue、0 harmful；
- 640/640 target action correct；
- 动作输出 KEEP/NEW/OLD 为 624/9/7；
- stress 从 91/160 提高到 101/160；
- crossed、crowded、long、near 分别从 7/23/47/14 提高到 9/27/49/16。

## 工程复核

首次运行完成训练后因 `torch.bool` 检查值无法 JSON 序列化而停止；同时复核发现
`torch.logit(float32 threshold)` 使融合基线成为 542，而 P19-E5 的 Python double 常数协议应为
543。修复为完全复用 P19-E5 的 `math.log` 常数语义，并新增逐 pair 决策一致性测试；失败产物
保留在同名 `-failed-json-protocol` 目录。修复后确定性重跑得到上述正式结果。

## 决策

正式决策为 `output_route_capacity640_go`，全部门通过。这只证明特征与 MLP 有拟合容量；640 张
训练样本已用于 P19-E4/E5 calibration，不能作为泛化结果，也不允许接前端或部署。

下一步 P19-E8 必须从 full80k 未使用余量建立 scaffold 隔离的 train/calibration/holdout，固定
78 维 schema、27,267 参数架构和默认 KEEP。选模仍以零 harmful 优先；当前 P19-E4 eval640
已消费，只能保留为最终一次固定 shadow，不能参与 P19-E8 特征、阈值或 checkpoint 选择。

## 产物

- best step：250；
- best：`outputs/checkpoints/chemvector-g3-p19e7-output-route-capacity640/best.pt`；
- best SHA256：`9cd4f454ab3fe0f511cbfd2ed4e4e1e264960ffb173cb1afed933071d8871e86`；
- cache SHA256：`0b73c961ed711c00aa399ba909cfd6707e5b9abea81d259d9662e2188c078e89`；
- result SHA256：`14a40e0092750690aaed7f9709c9fd4336ec919cc227b3b11cd19e4b136f11cb`；
- 总耗时：38.21 秒。
