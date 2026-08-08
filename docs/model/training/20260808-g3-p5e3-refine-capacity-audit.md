# G3 P5-E3 Refine Capacity Audit

> 日期：2026-08-08
>
> 状态：已预注册，等待审计

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
