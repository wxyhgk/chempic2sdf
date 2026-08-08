# G3 P5-E4 Contextual Refine Scale Curve

> 日期：2026-08-08
>
> 状态：已完成，两个模型均 NO-GO

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

## 实际结果

| 动作数 | fit 6px / MAE | calibration 6px / MAE | holdout 6px / MAE |
|---:|---:|---:|---:|
| 264（P5-E2） | 100.00% / 0.54px | 45.16% / 7.61px | 32.84% / 9.10px |
| 528（P5-E4） | 100.00% / 0.66px | 51.61% / 7.33px | 38.81% / 8.60px |
| 1,378（P5-E5） | 99.27% / 1.09px | 77.42% / 4.87px | 53.73% / 6.39px |

P5-E4和P5-E5的四项正式门都失败，均为`contextual_scale_no_go`。但两个评测域的恢复率随规模
同步上升、MAE同步下降，证明“264个稀有动作不足”是一个真实瓶颈，不是主观猜测。另一方面，
动作扩大5.22倍后holdout仍只有53.73%，说明数据量不是唯一断点；不能按当前斜率盲目外推并继续
堆同分布动作。

P5-E4 checkpoint：
`outputs/checkpoints/chemvector-g3-p5e4-contextual-secondary-peak-fit528/step-006000.pt`，SHA256
`386ae9a4b172179ac820c8a377a8621350149c729964f639bb549b19c6992577`，audit SHA256
`ce94af7c5c22cddeb9a43d07ea5509ea3787b9e3f727ae86b7590a6b86dd2da7`。

P5-E5 checkpoint：
`outputs/checkpoints/chemvector-g3-p5e5-contextual-secondary-peak-fit1378/step-015660.pt`，SHA256
`7c259f50d9a6846a7d748d7de3b02c7b4cb55b1d3fcd7a2e422f0f63db624bcd`，audit SHA256
`8548bd5c939ec28e919ee9b2d16a8b024e8d4128f65698479f45106606d93156`。

两个模型均未读取dev、`confirm_v2`、外部集或封存集。

## 结论与边界

规模假设获得`partial_go`：更多独立动作显著改善泛化，但当前局部refine仍不可部署。下一步不再
直接训练，先把P5-E5的holdout错误按variant、source距离、局部中心场可达性、文字邻近与多键
局部结构拆桶，判断剩余错误是否混合了不同任务。只有出现一个可预注册、跨域一致的错误子类，
才为该子类改变输入或目标；不直接读取confirm或用MR-TADF调参。
