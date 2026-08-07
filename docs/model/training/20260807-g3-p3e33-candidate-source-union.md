# G3 P3-E33 Candidate Source Union

> 日期：2026-08-07
>
> 状态：已完成，192容量NO-GO / 256容量可达性GO
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

## 实际结果

winner为`train25k160_then_train5k`：先保留train25k前160，再按train5k rank补满共同source-id
并集。结果如下：

| 容量 | Atom recall | candidate exact |
|---|---:|---:|
| top128 | 99.757% | 145/160 |
| top192 | 99.927% | 154/160 |
| top256 | 100% | 160/160 |

winner完整保留P3-E32的13张旧tail恢复，只回退1张旧可达图；known-Count raw仍为56/160。四种
固定策略都不能在192容量同时容纳两个中心场的互补峰，因此未通过160/160门，正式决策为
`candidate_source_union_no_go`。但256容量达到160/160，并保留更强的train25k前缀，证明并集方向
成立、最小充分容量是256而不是192。

下一训练周期不得继续在Micro160上更换loss。应固定本轮256 winner输入，从style_train约100k中
排除P3-E21 train/dev/confirm的molecule/scaffold后建立训练源，再比较两条可泛化路线：source-aware
256-candidate set selector，以及继续训练center field使互补峰回到较小预算。Micro160只作门，不作
训练数据；MR-TADF、USPTO和sealed_test仍不参与选模。

## 产物

- audit SHA256：`22fdfc5215ab072a605f0aba84002300a7ec3a1cab76f6095f51105720d3fa3d`；
- union top256 cache SHA256：
  `b8cd08b05e5caf95ee44163e1e1ea9d02ea8ab9696aba2bac2b73e8859e2f4cd`；
- 不读PNG、不训练、不读取dev/confirm/fresh/外部或sealed_test；
- 本轮零训练，不创建tag。
