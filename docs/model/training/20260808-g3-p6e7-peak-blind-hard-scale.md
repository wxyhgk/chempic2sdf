# G3 P6-E7 Peak-Blind Hard Scale

> 日期：2026-08-08
>
> 状态：已完成，NO-GO

使用P6-E6的1,339个新hard-choice，构建500与1,339两臂。500臂是完整臂的确定性子集并覆盖
500个scaffold；easy从既有generic top-1正确池中取等量嵌套子集。两个115,581参数peak-blind
模型使用相同随机初始化、32 hard+32 easy、D4和AdamW `3e-4`。

每条hard固定约192次曝光，因此两臂分别训练3,000与8,034步。大臂须hard fit至少90%，在
calibration/holdout均比小臂和P6-E1第一峰至少提高5pp，且两域MAE均不比小臂差，才判定
`hard_scale_go`。不读取dev、`confirm_v2`、MR-TADF、USPTO或封存集。两臂均保存checkpoint并
创建各自同名annotated tag。

## 结果

| 指标 | hard500 | hard1339 |
| --- | ---: | ---: |
| hard fit | 85.80% / 3.57px | 90.29% / 2.85px |
| easy fit | 99.60% / 0.72px | 99.63% / 0.65px |
| calibration | 19.35% / 11.66px | 29.03% / 10.81px |
| holdout | 25.37% / 10.96px | 26.87% / 10.86px |

大臂通过hard fit、calibration规模增益和两个MAE不回退门，但holdout规模增益只有1.50pp，且
两个冻结域都远低于P6-E1第一峰58.06%/56.72%，正式`peak_blind_hard_scale_no_go`。

数据扩量确实提高拟合与calibration，因此不是完全无效；但永久删除峰强度丢失了强泛化先验。
下一模型应固定原峰logit为base score，只学习零初始化的视觉residual，并通过easy保持损失约束
不得破坏原第一峰，而不是在“完全相信峰强度”和“完全删除峰强度”之间二选一。

checkpoint：

- hard500：`outputs/checkpoints/chemvector-g3-p6e7-peak-blind-hard500/step-003000.pt`，
  SHA256 `6de07b4fa806d76d84f815831b888ae5c5d7d3987296062a4396d8d595fe99f6`；
- hard1339：`outputs/checkpoints/chemvector-g3-p6e7-peak-blind-hard1339/step-008034.pt`，
  SHA256 `c71cb5cb3b1dddd41e043b86febf74f4b01c9f23780defb7888f2e6f95c26965`。

周期audit SHA256：`81db33e2084db2a8c38661f8117c5469e18c187a40fa6214e9538cdb21ceeb82`；
hard cache SHA256：`8e52e72e0e92fa63211f7769189796f5c3409bb59a832eb9864c52f2b18f115e`。
`confirm_v2_read=false`，`dev_and_external_read=false`。
