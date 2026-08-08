# G3 P5-E11 Expert Fusion

> 日期：2026-08-08
>
> 状态：已完成，NO-GO

## 目标

P5-E5真实专家在calibration较强，P5-E9通用专家在holdout较强；P5-E10证明继续真实微调会损害
holdout。本轮零训练验证两个专家是否在预测层互补。

## 固定协议

- 分别在4--13px有效环带内softmax归一化P5-E5和P5-E9 heatmap；
- P5-E5真实专家权重固定扫描`0/0.25/0.5/0.75/1`，其余权重给P5-E9；
- calibration若有权重过80%/3px门，按恢复率最高、MAE最低、真实权重最小选择；否则仍按同序；
- 选择后固定重放holdout；另报告两个端点专家“任一成功”的Oracle union上限；
- 不训练、不读dev、`confirm_v2`、外部或封存集，不创建tag。

最终融合仍须两域80%/3px才GO。失败则停止当前局部方向头的权重/融合调节。

## 实际结果

calibration按冻结规则选择P5-E5真实专家权重0.5：26/31、83.87%，但MAE仍为4.17px；固定到
holdout后只有37/67、55.22%，MAE 6.24px，四项联合门失败，最终
`expert_probability_fusion_no_go`。holdout最佳固定权重实际是纯P5-E9（44/67、65.67%）。

两个端点专家逐样本Oracle union为：

| 域 | 6px恢复率 | MAE |
|---|---:|---:|
| calibration | 83.87% | 3.61px |
| holdout | 82.09% | 3.41px |

专家确实互补，理想路由的恢复率可超过80%；但即使使用不可实现的Oracle逐样本选择，两域MAE仍
超过3px。因此继续训练router、扫描融合权重或做局部MoE都无法通过当前冻结门。P5局部恢复主线
正式停止；下一路线必须扩大可见结构范围，显式利用键/邻接几何，而不是继续调32px局部方向头。

audit SHA256：`eace64ebfff5e7f49899d7607824545ba6ce17a9a098604932b9002dc921c0c2`。本轮零训练、
不创建tag；dev、`confirm_v2`、外部和封存集均未读取。
