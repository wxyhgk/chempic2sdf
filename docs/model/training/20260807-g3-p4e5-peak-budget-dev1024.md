# G3 P4-E5 Peak Budget Dev1024

> 日期：2026-08-07
>
> 状态：已完成，实用regional-maxima表示NO-GO
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

## 实际结果

P4-E3 top128严格重放通过。随着预算扩大，收益在768后完全饱和：

| 预算 | matched / 32,400 | recall | exact / 640 |
|---:|---:|---:|---:|
| 128 | 32,195 | 99.3673% | 550 |
| 192 | 32,286 | 99.6481% | 578 |
| 256 | 32,315 | 99.7377% | 584 |
| 384 | 32,335 | 99.7994% | 593 |
| 512 | 32,339 | 99.8117% | 594 |
| 768 | **32,342** | **99.8210%** | **597** |
| 1024 | **32,342** | **99.8210%** | **597** |

top1024仍缺58个Atom，低于99.9%门；768到1024一个Atom和一张exact都没有增加。剩余错误不是
把256简单放大到更高实用预算可以解决的。

分桶显示缺失不是单一大分子问题：top1024的1–20、21–40、41–60、61–80、81–100 recall分别
为99.8243%、99.8406%、99.9001%、99.8532%、99.7477%。其中21–40从top128起始终缺6个Atom，
说明部分GT附近根本没有随预算出现的可匹配峰。variant中stress最低（99.6790%），original为
99.7778%，coordgen/rdkit为99.9259%/99.9012%。

单次top1024提取及七档匹配耗时32.90秒；regional maxima均值56,763、p90 85,310、最大102,894。

最终决策`regional_maxima_representation_no_go`。这里的“表示NO-GO”限定于当前P4-E3场和实用
top1024协议，并不声称图片中不存在视觉证据。下一步审计58个未匹配GT在中心场中的局部状态：
相邻GT互相支配、GT cell不是regional maximum、文字/交叉线假峰，或真峰排名低于1024。停止
继续增大预算、停止当前source union和selector训练。

## 产物

- audit SHA256：
  `a2fcf57fd393a570f15218b39b3a7988226f6c56d1cdd49aeddd3f0f672929b5`；
- 运行耗时32.90秒；
- pilot、confirm_v2、fresh、USPTO、MR-TADF和sealed_test均未读取。
