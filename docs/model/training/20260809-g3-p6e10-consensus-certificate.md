# G3 P6-E10 Consensus Certificate

> 日期：2026-08-09
>
> 状态：已预注册，等待calibration审计

冻结P6-E9三臂，global为primary。只有三臂选择同一个非prior候选才具备执行资格；证书分数固定为
global最终selected分数减去prior winner的global最终分数。calibration只扫描该单标量阈值，在
harmful=0条件下依次最大化rescue、最小化执行数、提高阈值。

至少救回100/643 hard且零harmful才GO并允许单次解锁holdout；否则停止。holdout当前保持000，
dev、confirm、MR-TADF、USPTO和封存集均不读取。本轮零训练，不创建模型tag。
