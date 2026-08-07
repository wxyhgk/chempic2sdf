# G3 P4-E7 Offset Radius A/B

> 日期：2026-08-08
>
> 状态：预注册，待运行
>
> 审计：`chemvector-g3-p4e7-offset-radius-ab`

## 目标

P4-E6中15个6px无峰漏点距最近完整峰只有6.12–10.57px。P4-E7固定P4-E3的峰身份和排序，
比较预测offset坐标与同一source-id的stride-1 cell center，并报告6/8/10/12px曲线，判断offset头
是否把本来正确的峰推过了6px门。

## 固定协议

- 只读已暴露dev640；候选身份、score和顺序逐项固定；
- 预算固定为top128和top1024；
- anchor模式为预测offset和cell center `(x+0.5,y+0.5)`；
- 半径固定为6、8、10、12px，全部使用最大基数优先Hungarian；
- P4-E3预测offset/top1024/6px必须重放32,342 matched、597 exact；
- cell center在6px至少增加15个matched、aggregate exact不退且paired exact零break，才判定
  `offset_head_harmful_go`；否则为`offset_head_not_primary`；
- 10px曲线只做诊断，不修改正式6px指标；
- 不训练、不保存cache、不读pilot/confirm_v2或外部/封存集。

本轮零训练，不创建tag。
