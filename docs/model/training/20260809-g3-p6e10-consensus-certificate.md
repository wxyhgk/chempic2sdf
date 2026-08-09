# G3 P6-E10 Consensus Certificate

> 日期：2026-08-09
>
> 状态：已完成，NO-GO

冻结P6-E9三臂，global为primary。只有三臂选择同一个非prior候选才具备执行资格；证书分数固定为
global最终selected分数减去prior winner的global最终分数。calibration只扫描该单标量阈值，在
harmful=0条件下依次最大化rescue、最小化执行数、提高阈值。

至少救回100/643 hard且零harmful才GO并允许单次解锁holdout；否则停止。holdout当前保持000，
dev、confirm、MR-TADF、USPTO和封存集均不读取。本轮零训练，不创建模型tag。

## 结果

- 三臂一致且改选的候选：255；
- 零harmful winner阈值：`2.920013904571533`；
- 执行48次，rescue 46、harmful 0、useless 2；
- 未达到至少100 rescues门，决策为`consensus_certificate_calibration_no_go`。

三臂共识能形成高精度证书，但覆盖率只有643个hard中的7.15%，不足以解锁holdout。下一方向若
继续，必须在train split用冻结三臂的连续margin/entropy/agreement特征训练独立OOF safety模型，
不能在同一calibration上继续手工搜索第二套规则。

audit SHA256：`dc6f80c13ff9a7f26a4971af5935e7fe42b57c55f59930e7bf556925dc37c3fe`；
records SHA256：`9b87f87c99eae53a284b0466401bce083cfe6a1230679fa6a6d046ab4f6b84bf`；
holdout保持mode `000`且未读。
