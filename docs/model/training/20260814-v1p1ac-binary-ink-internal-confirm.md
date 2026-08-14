# v1 P1-A-C 二值墨迹内部确认

> 日期：2026-08-14
>
> 状态：GO；一次性 internal test 全部门通过
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

## 结果

预注册提交为 `92a1362`。冻结 checkpoint 在 496 张 internal test 上一次前向耗时 6.28 秒：

| 指标 | Validation | Internal test | 门 |
|---|---:|---:|---:|
| Bond IoU | 97.72% | 98.09% | 90%，通过 |
| Text IoU | 91.85% | 91.13% | 75%，通过 |
| 联合前景 IoU | 99.21% | 99.21% | 90%，通过 |
| Stress bond IoU | 97.56% | 98.09% | 88%，通过 |
| Bond-on-text-only | 9.36% | 9.70% | 10%，通过 |

九项检查全部通过，P1-A-C 正式 GO。结果 SHA256：
`8ef07a24545236e494aa18bd741ca1adb6342f425eaf5349443141cf5e2a14e1`。

## 决策

P1-A checkpoint 可以作为 v1 BondObject 阶段的冻结语义前端。internal test 的文字区域污染率
9.70% 距 10% 门只有 0.30pp，因此下游必须单独报告 text-adjacent BondObject 的 precision，
不能只报总体端点或键召回。本轮 internal test 已消费，后续不得再用它选择结构、损失、step
或阈值。
