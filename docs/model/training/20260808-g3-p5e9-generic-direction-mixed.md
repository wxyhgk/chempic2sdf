# G3 P5-E9 Generic Direction Mixed

> 日期：2026-08-08
>
> 状态：已完成，NO-GO

## 目标

P5-E8提供50,000个一般原子source→anchor方向对。本轮保持P5-E5的45,353参数完整crop头，验证
通用方向预训练能否解决degree=2等局部几何的跨scaffold方向泛化。

## 固定训练

模型名：`chemvector-g3-p5e9-generic-direction-mixed50k-fit1378`。

1. 冻结P4-E3对fit图在线生成50k image/center-field crop cache；generic candidate score/rank置0；
2. 随机初始化方向头，50k通用对训练10,000步，batch64、AdamW `3e-4`；
3. 不保存阶段一checkpoint，只在线记录原calibration/holdout指标；
4. 继续5,000步混合微调，每批32通用对+32真实残差，学习率降为`1e-4`；
5. D4增强、weight decay `1e-4`、clip 1与既有heatmap loss保持不变。

最终calibration/holdout 6px恢复率都至少80%、MAE都不超过3px才GO。不读dev、`confirm_v2`、
外部或封存集。只保存最终checkpoint；无论GO/NO-GO，提交结果并创建与模型名相同的annotated tag。

## 实际结果

50k cache用时242.73秒，SHA256
`1a16432eb8c10a0217692c3c97a12dab3c1158f276a3b3966924f11a97b46fdb`。阶段一通用预训练的
真实残差迁移结果很低：calibration 16.13%/MAE 10.95px，holdout 25.37%/MAE 10.34px，证明
synthetic source与真实候选存在明显域差异，不能直接替代真实动作。

1:1混合微调后：

| 域 | 6px恢复率 | MAE |
|---|---:|---:|
| generic fit 50k | 99.55% | 0.69px |
| real fit 1,378 | 66.40% | 5.36px |
| calibration | 64.52% | 5.73px |
| holdout | 65.67% | 5.36px |

四项正式门全部失败，最终`generic_direction_mixed_no_go`。相对P5-E5，calibration从77.42%下降到
64.52%，但holdout从53.73%提高到65.67%；通用监督显著减少训练域记忆和跨scaffold落差，当前
1:1混合则对真实candidate分布适配不足。这个结果不支持继续增加generic比例，也不支持退回纯
真实残差；下一步只允许从最终checkpoint做短程、低学习率真实残差校准。

checkpoint：
`outputs/checkpoints/chemvector-g3-p5e9-generic-direction-mixed50k-fit1378/step-015000.pt`，SHA256
`b852bdc369639c8e1ceeed9e8c8b36ae4e785dae25fa24cb3666d009edd1e42e`；audit SHA256
`92b4bb8b6083bb72ef6f78b8b946e61f731118428d600e1d5401927c4ebc8939`。没有读取dev、
`confirm_v2`、外部或封存集。
