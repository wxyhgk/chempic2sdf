# G3 P6 Residual Long Cycle

> 日期：2026-08-09
>
> 状态：已闭环，在P6-E9安全门停止

本周期按预注册自动分支协议执行。P6-E8扫描58,376张隔离训练图，得到6,552 hard和27,504 easy，
三个scaffold split零重叠并选择cap2.0，数据门GO。随后P6-E9完整训练global、local17、multiscale
三个零初始化residual臂，各31,026步。

三个模型在calibration hard均恢复约一半目标，但在easy产生38/61/58个harmful，因此没有满足
零harmful门的winner。两个holdout cache始终为000，P6-E10 trigger、dev、confirm_v2、整图关系、
SDF与外部benchmark均未运行。该停止证明当前缺口是安全改写prior的证书，不是候选可达性、模型
参数量或候选中心分辨率。

可恢复入口：

```bash
.venv-train/bin/python -m chemvector.training.atoms.run_g3_p6_residual_long_cycle --audit-existing
```

状态写入`outputs/checkpoints/chemvector-g3-p6-residual-long-cycle/cycle_state.json`，绑定各阶段audit
SHA256。只有未来出现`residual_calibration_winner_selected`，状态机才允许进入P6-E10。
