# G3 P4-E5 Peak Budget Dev1024

> 日期：2026-08-07
>
> 状态：预注册，待运行
>
> 审计：`chemvector-g3-p4e5-peak-budget-dev1024`

## 目标

P4-E4证明P4-E3单源top256 dev recall为99.7377%，旧train5k并集不能补齐。P4-E5只在已经暴露
的dev640上把同一新场提取到top1024，区分两种原因：

- 真峰存在，但仍排在top256之后；
- 真峰没有成为可用regional maximum，继续扩大候选预算无效。

## 固定协议

- checkpoint固定为P4-E3 final，SHA256
  `1c358d0c936c687fa0e40ea5e1d9364dd98a663bb762d294db76b43d5f009593`；
- 只读旧dev640，不重跑pilot，不读confirm_v2或外部/封存集；
- 一次提取top1024完整plateau regional maxima；
- 固定报告128、192、256、384、512、768、1024预算；
- 每档报告总recall/exact、四variant和Count桶；
- top128必须严格重放32,195 matched、550 exact；
- 不训练、不保存候选cache、不使用像素聚类或后处理规则。

## 决策门

top1024 recall不低于99.9%时为`tail_ranking_capacity_go`，并报告首个过门预算；说明后续应改变
排序/候选表示，而非center生成。若top1024仍低于99.9%，为
`regional_maxima_representation_no_go`，停止继续扩大同一候选预算，转向减少碎峰并恢复缺失真峰。

本轮零训练，不创建tag。
