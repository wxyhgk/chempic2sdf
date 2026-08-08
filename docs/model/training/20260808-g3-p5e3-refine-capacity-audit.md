# G3 P5-E3 Refine Capacity Audit

> 日期：2026-08-08
>
> 状态：已完成，GO

## 目标

P5-E2用45,353参数的完整crop上下文模型把264个fit动作做到100%、MAE 0.54px，但在
calibration/holdout只有45.16%/32.84%。本轮不训练，只判断未用训练域能否把稀有refine动作
从264扩到足以检验规模效应的数量。

## 数据边界

来源固定为P4-E0 `full_train` 80,000图。严格删除：

1. 原pilot的5,000个分子；
2. P4-E9 calibration和holdout的全部1,425个scaffold。

剩余必须恰为58,376图、14,594分子、13,168 scaffold。只运行冻结P4-E3 center field和既有
P4-E9 Oracle定义，收集`ADD_REFINE`；不读取dev、confirm_v2、外部集或封存集。

## 固定门

- refine动作至少800个；
- 动作分子至少500个；
- 覆盖四种variant；
- 与pilot calibration/holdout scaffold重叠为0；
- 所有输入checkpoint和manifest SHA256一致。

全部通过才允许预注册一次固定模型、固定步数的规模曲线。失败则停止局部refine路线，不用
MR-TADF补训练，不放宽半径或动作定义。本轮不训练，因此不创建模型tag。

## 实际结果

冻结骨干完整处理58,376张合格图片，用时1,854.90秒。得到：

| 指标 | 结果 |
|---|---:|
| refine动作 | 1,114 |
| 动作图片 | 992 |
| 动作分子 | 883 |
| 动作scaffold | 875 |
| 单图最多动作 | 7 |
| coordgen/original/rdkit/stress | 96/111/121/786 |

800动作、500分子、四variant、零评测scaffold重叠等全部门通过，最终决策为
`scaled_refine_capacity_go`。与原fit 264动作合并后可得到1,378个训练动作，是原规模的5.22倍；
其中stress占新增动作70.56%，与原calibration/holdout残差主要来自stress的现象一致，但规模曲线
必须逐variant报告，不能只看总指标。

audit SHA256：`1895a93e8e90a36d33189bcaef3ba56bd31aba86ef6f648d28d7b5c2c0436c74`；
动作文件SHA256：`ac65fd980d2e659ad0ddab32f4d4f62ff79e4881e71267d42f793fe3719c0dcd`。
本轮没有训练，不创建tag；`confirm_v2`、dev、外部和封存集均未读取。

## 下一步

允许预注册P5-E4固定规模曲线：保留P5-E2的264点，增加确定性选择的264个新动作形成528点，
再使用全部1,378点；模型、D4、loss、optimizer和每动作累计曝光量保持一致。只有跨scaffold
calibration与holdout随规模同步改善，才支持“数据不足”解释。
