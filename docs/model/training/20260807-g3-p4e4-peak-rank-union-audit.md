# G3 P4-E4 Peak Rank Union Audit

> 日期：2026-08-07
>
> 状态：已完成，Candidate Oracle NO-GO
>
> 审计：`chemvector-g3-p4e4-peak-rank-union-audit`

## 目标

P4-E3使单一新center field的dev top128 recall达到99.3673%、exact达到550/640，但仍低于
Candidate Oracle 99.9%门；同时regional maxima均值从3.03万升至5.68万。本轮不训练selector，
只审计新场与旧train5k场的精确source-id并集能否过门，以及新场的实际提取成本。

## 固定协议

- 输入为旧train5k stride-1 checkpoint和P4-E3 final checkpoint；
- 读取P4-E0 pilot_train 20k和旧dev640，只读取PNG与Atom GT；
- 每个source各提取top256完整plateau regional maxima；
- 比较new-first 128、160、192三个固定顺序，随后由旧源补齐，再由新源剩余项补齐；
- 只按共同stride-1 source-id去重，不使用像素距离、NMS或固定聚类；
- 对每个顺序报告top128/192/256 candidate recall与exact；
- 胜者按“最小同时过pilot/dev 99.9%的预算、dev exact、pilot exact、预注册prefix顺序”选择；
- 只保存胜者的256×72 source-aware cache，checkpoint与特征协议沿用P4-E1；
- 报告两个source在pilot/dev的regional maxima均值、p90、最大值和完整提取耗时。

P4-E3在dev top128必须严格重放为32,195 matched、550 exact，否则协议失败。confirm_v2、fresh、
USPTO、MR-TADF和sealed_test均不得读取。

## GO/NO-GO

至少一个预注册并集顺序在某个预算下同时满足pilot/dev recall不低于99.9%，且P4-E3 dev重放
严格一致，才为`peak_rank_union_audit_go`。GO只允许进入source-aware selector pilot；不代表
可以读取confirm_v2或运行SDF。NO-GO则停止当前并集，不追加同损失训练。

## 预期产物

- `outputs/audits/chemvector-g3-p4e4-peak-rank-union-audit/audit.json`
- `outputs/audits/chemvector-g3-p4e4-peak-rank-union-audit/pilot_train.pt`
- `outputs/audits/chemvector-g3-p4e4-peak-rank-union-audit/dev.pt`

本轮零训练，不创建模型tag。

## 实际结果

P4-E3的dev top128严格重放为32,195 matched、550 exact，checkpoint和提取协议一致。两个source
完整收集后，三个预注册并集顺序在三个预算下均未同时通过pilot/dev 99.9%门：

| 输入 | pilot top256 recall / exact | dev top256 recall / exact |
|---|---:|---:|
| train5k单源 | 99.8778% / 19,394 | 99.4383% / 546 |
| P4-E3单源 | 99.9326% / 19,650 | **99.7377% / 584** |
| new-first128并集 | 99.9376% / 19,689 | 99.5370% / 570 |
| new-first160并集 | 99.9467% / 19,729 | 99.6543% / 584 |
| new-first192并集 | **99.9525% / 19,759** | 99.7222% / **592** |

pilot在P4-E3单源top256已过99.9%，dev仍差85个Atom；最佳并集能增加8张dev exact，但recall
反而比P4-E3单源低0.0154个百分点。旧场与新场的漏点高度相关，在256容量下没有形成有效补集。
因此不保存无效的source-aware cache，也不启动selector。

完整plateau提取成本：

| source | 20,640图耗时 | pilot峰均值 / p90 | dev峰均值 / p90 |
|---|---:|---:|---:|
| train5k | 538.07秒 | 33,100 / 34,281 | 32,477 / 34,326 |
| P4-E3 | 673.17秒 | 59,263 / 88,593 | 56,763 / 85,310 |

总审计耗时1,298.68秒。P4-E3比train5k提取慢25.1%，并且p90峰数约为2.6倍；当前SciPy
plateau实现只能用于离线cache，不能作为CPU在线推理方案。

最终决策`peak_rank_union_audit_no_go`。下一步只在已暴露dev上将P4-E3单源预算扩大到1024，
区分256之后的排序尾部与regional-maxima表示缺失；不重跑pilot、不读confirm_v2。

## 产物

- audit SHA256：
  `9746bc2c49a0822a7c735b2cf4351b4439ad85528f9b50d45e0b0eab4005df51`；
- 无winner，未生成pilot/dev cache；
- confirm_v2、fresh、USPTO、MR-TADF和sealed_test均未读取。
