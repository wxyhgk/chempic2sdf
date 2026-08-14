# v1 P1-A-C 二值墨迹内部确认

> 日期：2026-08-14
>
> 状态：已预注册，等待一次性运行
>
> 审计：`chemvector-v1p1ac-binary-ink-internal-confirm`

## 固定输入

- P1-A checkpoint SHA256：
  `490b5daae3a2059c3d335785a69e0c2a41eed32f0237ef77fc3556c36d134a74`；
- P1-A validation result SHA256：
  `ce17cfe8fb0e41b877d3a0ece8d0f6f1de3f082404135b49f67394c21a647612`；
- safe manifest SHA256：
  `eb9da30c85aaede01b635294f53dca4454825f13db55002cb2df843edb49f11b`；
- bond/text 预测阈值均固定为 0.5。

本轮零训练、不重新读取 validation、不选择 checkpoint、step、阈值或门槛。只对此前未读的
496 张 internal test 执行一次前向。

## GO/NO-GO

internal test 必须同时满足：bond IoU 90%、text IoU 75%、联合前景 IoU 90%、stress bond
IoU 88%、bond-on-text-only 不超过 10%，且样本数严格为 496。全部通过才允许把 P1-A
语义前端交给下一阶段 BondObject 训练；任一失败都记为 confirm NO-GO，不回到 validation
调参。
