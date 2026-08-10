# G3 P19-E7 Output-route Capacity640

> 日期：2026-08-10
>
> 状态：已预注册，等待运行
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
