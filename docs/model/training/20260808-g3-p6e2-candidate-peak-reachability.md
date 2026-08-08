# G3 P6-E2 Candidate Peak Reachability

> 日期：2026-08-08
>
> 状态：已预注册，等待审计

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
