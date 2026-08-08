# G3 P5-E10 Real Calibration

> 日期：2026-08-08
>
> 状态：已完成，NO-GO

## 目标

P5-E9的holdout从53.73%升到65.67%，但真实fit只有66.40%、calibration 64.52%，说明generic
正则有效但1:1混合对真实candidate适配不足。本轮只做一次有界低学习率真实残差校准。

## 固定协议

- 初始化固定为P5-E9最终checkpoint；模型名`chemvector-g3-p5e10-real-calibrated-fit1378`；
- 只用1,378真实动作，延续AdamW状态，学习率`5e-5`，D4和其他配置不变；
- 固定运行到2,000步，只在追加`0/250/500/1000/2000`步记录calibration；
- 若有候选同时达到80%恢复率和3px MAE，选择最早候选；否则依次按恢复率最高、MAE最低、步数
  最少选择；
- 只保存所选状态，随后重放calibration并评估holdout；中间状态不落checkpoint。

最终两域都达到80%/3px才GO。不读dev、`confirm_v2`、外部或封存集。实际模型无论GO/NO-GO
都提交结果并创建同名annotated tag。

## 实际结果

calibration候选曲线：

| 追加步数 | 6px恢复率 | MAE |
|---:|---:|---:|
| 0 | 64.52% | 5.73px |
| 250 | 64.52% | 5.79px |
| 500 | 70.97% | 5.55px |
| 1,000 | 70.97% | 5.44px |
| 2,000 | 67.74% | 5.64px |

无候选过80%/3px门，按冻结规则选择1,000步。所选模型holdout为62.69%、MAE 5.58px；相对P5-E9
的65.67%/5.36px反而退化。四门全失败，最终`real_calibration_no_go`。真实动作适配与generic
泛化并非简单训练步数折中，不再继续扫描学习率、比例或更多step。

checkpoint：
`outputs/checkpoints/chemvector-g3-p5e10-real-calibrated-fit1378/step-016000.pt`，SHA256
`d20b5ee6065d0ce4113a66b7df5f821e381d5739696b9fa6b1843d078ba37f52`；audit SHA256
`04b730bfad04c3ad393414d359bad473184384fa560ebff5dbc36ce65cc5042b`。没有读取dev、
`confirm_v2`、外部或封存集。
