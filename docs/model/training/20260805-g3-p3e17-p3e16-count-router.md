# G3 P3-E17 P3-E16 Count 接回冻结 Router

> 日期：2026-08-05
>
> 状态：零训练集成审计 GO
>
> Count checkpoint：P3-E16 style90932 epoch6 h64
>
> 输出：`outputs/audits/chemvector-g3-p3e17-p3e16-count-router-audit`

## 目标

P3-E8 使用 P2-W 自带 Count 时，P3-E7 base/expert router 从 known-count 的111/256、36/100
跌到 predicted-count 的4/256、0/100。P3-E16 已首次通过独立 Count 四门，本轮只替换 Count，
不训练模型、不重选 P3-E7 阈值，验证 predicted-count Atom set 是否恢复。

## 固定协议

- base：P2-W，冻结；
- expert：P3-E1，冻结；
- Count：P3-E16，冻结；
- router：P3-E7 固定规则：

```text
base_joint_score_mean <= -1.8862037658691406 -> expert
```

- 正式推理 batch4，与 P3-E7/P3-E8 相同；
- 额外重放 batch16，只检查 Count 预测一致性；
- route_select 不读取；
- route_confirm 与 historical route100 不参与阈值或 checkpoint 选择；
- MR-TADF 未参与。

需要特别区分两组“confirm”：P3-E16 Count Gate 的 confirm256 与 P3-E7 route_confirm 是不同
样本。本轮在原 P3-E7 route_confirm 上重新计算 Count，不能直接复用 P3-E16 的128/256。

## 协议校验

- known-count route_confirm 精确复现 P3-E7：111/256；
- known-count route100 精确复现 P3-E7：36/100；
- 356 张样本的 batch4/batch16 Count 预测差异：0；
- predicted router exact 严格等于“known router exact 与 Count correct”的交集；
- 固定阈值未重新选择。

## 结果

| 集合 | P3-E8 旧 Count | P3-E16 Count | known-count 上限 |
|---|---:|---:|---:|
| route_confirm | 4/256 | **79/256** | 111/256 |
| historical route100 | 0/100 | **21/100** | 36/100 |

P3-E16 在 route_confirm 的 Count 为124/256、MAE1.055；route100为41/100、MAE1.04。
集成相对P3-E8分别净增75和21张，预注册改善门通过。

## 误差分解

### Route Confirm

```text
Count correct                         124 / 256
known router exact                    111 / 256
known exact 且 Count correct           79
known exact 因 Count 错误丢失           32
Count correct 但 known router 错误      45

Count正确条件：
base exact                             11 / 124
expert exact                          107 / 124
router exact                           79 / 124
```

### Historical Route100

```text
Count correct                          41 / 100
known router exact                     36 / 100
known exact 且 Count correct            21
known exact 因 Count 错误丢失            15
Count correct 但 known router 错误       20

Count正确条件：
base exact                              4 / 41
expert exact                           34 / 41
router exact                           21 / 41
```

Count 已不再是唯一断点。route_confirm 剩余 known 上限损失中的32张来自Count错误；而在Count
正确的124张里，expert可做对107张，保守router只采用到79张。P3-E7的零break阈值保护base，
同时漏掉了相当多expert收益。route decision 在错误Count下有19/256、5/100张变化，但这些
样本因基数错误本来就不可能整图exact。

## 决策

- P3-E16 Count 接入固定 router：`GO`；
- 该结果是 predicted-count Atom candidate set，不是完整 Atom chemistry、Skeleton 或 SDF；
- 不用已消费的 route_confirm 重选阈值，也不直接切 always-expert，因为 P3-E1 expert 曾破坏
  capacity hard 域；
- 下一步保持所有组件固定，在不读取 sealed_test 的前提下，对 `fresh_eval_12k/fresh_dev`
  held-out 字体域做无选择 Count + Atom set 审计，确认提升是否跨风格成立。

## 产物

- audit：`outputs/audits/chemvector-g3-p3e17-p3e16-count-router-audit/metrics.json`；
- Count SHA256：`50b7f07a723468f1ef50b21011cc65320af9bad7c6ecfd88be3f5549444ef236`；
- 本轮无新训练 checkpoint，因此不创建模型 tag；
- 文档复核日期：2026-08-05。
