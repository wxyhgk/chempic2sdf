# G3 P4-E4 Peak Rank Union Audit

> 日期：2026-08-07
>
> 状态：预注册，待运行
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
