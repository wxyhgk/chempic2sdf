# G3 P3-E33 Candidate Source Union

> 日期：2026-08-07
>
> 状态：预注册，待执行cache-only审计
>
> 审计：`chemvector-g3-p3e33-candidate-source-union`

## 假设

P3-E32中train25k把13/27旧tail图拉回top128，却在自身top192漏5张；train5k的top192为160/160。
两个center field共享同一stride-1 192×192网格，因此可以按精确`source_id`合并，不需要固定像素
半径、聚类或NMS。若错误集合真正互补，192容量并集应同时保留train25k恢复点和train5k兜底点。

## 固定策略

只读取两个冻结top256 cache，生成四种确定性顺序：

- train25k前128，再按train5k排名补充，最后补train25k剩余；
- train25k前160，再按train5k排名补充；
- 两个source按归一化rank取best-rank并集；
- 两个source逐rank交替。

同一`source_id`只保留第一次出现的candidate；不做像素距离去重。每路均固定256条，报告top128、
top192、top256和known-Count raw结果。winner先按top192 exact，再按top128 exact、tail恢复、回退、
raw exact排序，不允许结果后修改优先级。

## GO/NO-GO门

- winner top192 candidate exact为160/160；
- winner top128 exact不低于train25k的145/160；
- 至少保留train25k恢复的13张旧tail图；
- 旧133张可达图最多回退1张；
- 四种策略、160张样本和256个唯一source ID完整。

全部通过才为`candidate_source_union_go`。本轮不训练、不读PNG，不创建tag。
