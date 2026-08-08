# G3 P5-E4 Contextual Refine Scale Curve

> 日期：2026-08-08
>
> 状态：已预注册，等待训练

## 目标

P5-E2在264个动作上fit 100%，但跨scaffold失败；P5-E3已经提供1,114个无评测泄漏的新动作。
本轮只改变动作数量，直接检验数据不足假设。

## 固定协议

两个模型都使用P5-E2的45,353参数完整crop上下文头、相同heatmap loss、D4增强、AdamW
`3e-4`、weight decay `1e-4`、batch64和clip 1：

| 模型 | 动作 | 步数 | 约曝光/动作 |
|---|---:|---:|---:|
| `chemvector-g3-p5e4-contextual-secondary-peak-fit528` | 528 | 6,000 | 727.3 |
| `chemvector-g3-p5e5-contextual-secondary-peak-fit1378` | 1,378 | 15,660 | 727.4 |

528由原264动作加264个确定性选择、互不相同scaffold的新动作组成；1,378使用原264加全部1,114。
calibration/holdout和80%恢复率、3px MAE门完全不变。不读dev、confirm_v2、外部或封存集。

两个实际模型无论GO/NO-GO都分别提交结果，并创建与模型名完全一致的annotated tag。若指标不随
规模同步改善，停止局部refine路线；若单调改善但未过正式门，只能说明数据假设成立，不能部署。
