# G3 P6-E2 Candidate Peak Reachability

> 日期：2026-08-08
>
> 状态：已完成，GO

## 问题

P6-E1在generic/real fit达到99.59%/90.06%，但calibration/holdout只有58.06%/56.72%。
本审计不再训练，先判断失败是“正确目标没有生成”还是“多个可行目标中选错”。

## 固定协议

- 冻结P6-E1最终checkpoint和上游center field；
- 只读取同一calibration 31动作与holdout 67动作；
- 从每个32x32输出热图提取3x3区域极大值，按logit与固定像素顺序排序；
- 候选提取完全不读取GT，事后才计算top-1/2/4/8中距GT最近的Oracle距离；
- 两域top-8的6px恢复率均至少80%、Oracle MAE均不超过3px才GO；
- 不读dev、`confirm_v2`、MR-TADF、USPTO或任何封存集。

若GO，下一模型保留96px编码器，显式训练source、多个目标候选和邻接结构的联合选择；若
NO-GO，则单目标heatmap没有产生足够正确假设，停止训练reranker，先改变输出表示。

## 结果

| 候选预算 | calibration恢复率 / Oracle MAE | holdout恢复率 / Oracle MAE |
| ---: | ---: | ---: |
| top-1 | 58.06% / 6.48px | 56.72% / 5.78px |
| top-2 | 77.42% / 4.23px | 74.63% / 3.93px |
| top-4 | 100.00% / 1.84px | 89.55% / 2.47px |
| top-8 | 100.00% / 1.78px | 92.54% / 2.24px |

calibration和holdout平均分别只有7.32和6.31个区域极大值，因此top-8接近读取完整候选集，
但仍是严格target-free生成。四个预注册门全部通过，决策为
`structured_peak_selection_capacity_go`。

## 结论

P6-E1的主要问题不是没有生成正确位置，而是对多个合理峰做了错误的单点argmax。top-4已在
两个域都超过能力门，top-8只额外恢复holdout 2个动作；下一模型固定最多8个候选，以列表级
目标训练共享结构选择器，不再回归新的坐标，也不改变96px编码器和候选生成协议。

产物：

- `outputs/audits/chemvector-g3-p6e2-candidate-peak-reachability/audit.json`，SHA256
  `41a5df076f83f95ee415a8be85339c9dbbb2e8a7f2090df1c7456a606ed8103b`；
- `records.jsonl`，SHA256
  `8725b1845c05157ee062e0a1f6fd47a59e023b7f8ff75cbe21ca67d024e9e1c9`；
- `dev_confirm_external_sealed_read=false`。
