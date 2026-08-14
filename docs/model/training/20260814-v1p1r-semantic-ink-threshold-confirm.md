# v1 P1-R 语义墨迹阈值确认

> 日期：2026-08-14
>
> 状态：NO-GO；联合前景门差 0.54pp，internal test 未读
>
> 审计：`chemvector-v1p1r-semantic-ink-threshold-confirm`

## 原因

P1 固定 0.5 阈值正式 NO-GO，但 bond/text precision 为 99.60%/99.78%，recall 仅
71.94%/65.18%。训练目标是 0--255 soft alpha，评测 GT 前景定义为 `alpha > 8/255`；固定
0.5 可能只保留了笔画核心，系统性丢弃抗锯齿边缘。

本轮不训练、不改 checkpoint、不改变 P1 结论。先在 validation 做固定网格 PR 审计，只有
validation 通过原能力门后，才一次性读取 internal test 确认。

## 固定输入与阈值

- P1 checkpoint SHA256：
  `e4f726b80f73e46a71619f8f4a42ff3b0042a3156cc86c0a4f584154b574263c`；
- safe manifest SHA256：
  `eb9da30c85aaede01b635294f53dca4454825f13db55002cb2df843edb49f11b`；
- bond/text 独立扫描同一固定 26 点：`0.01..0.20`，步长 `0.01`，再加
  `0.25,0.30,0.35,0.40,0.45,0.50`；
- 每个通道选择 validation IoU 最大值；并列先选 precision 更高，再选阈值更高者；
- internal test 不得参与阈值、模型或门槛选择。

## 两级门

Validation 必须先同时满足 P1 原门：bond IoU 90%、text IoU 75%、联合前景 IoU 90%、
stress bond IoU 88%、bond-on-text-only 不超过 10%。若失败，脚本不得读取 internal test。

Validation 通过后，锁定两个阈值并一次性读取 496 张 internal test。internal test 必须通过
完全相同的五个门，才判 P1-R GO。通过只证明语义墨迹分离与 alpha 校准成立；下一周期才训练
BondObject。失败则按位置区分：validation 失败表示模型/损失能力不足，internal test 失败表示
validation 校准过拟合。

## 结果

预注册提交为 `f9dd3c9`。26 点扫描在两个通道都选择了阈值 `0.01`：

| Validation 指标 | 固定 0.5 | 校准 0.01 | 门 |
|---|---:|---:|---:|
| Bond IoU | 71.73% | 90.09% | 90%，通过 |
| Bond precision / recall | 99.60% / 71.94% | 99.43% / 90.56% | - |
| Text IoU | 65.09% | 80.90% | 75%，通过 |
| Text precision / recall | 99.78% / 65.18% | 99.29% / 81.36% | - |
| Stress bond IoU | 72.11% | 90.30% | 88%，通过 |
| Bond-on-text-only | 1.68% | 3.01% | 10%，通过 |
| 联合前景 IoU | 71.23% | 89.46% | 90%，失败 |

校准后四项门通过，但联合前景仍比门槛低 0.5355pp。因此 validation calibration 未通过，
脚本没有构造 internal-test loader，`internal_test_read=false`。P1-R 严格 NO-GO，结果 SHA256：
`70bb5cee663a9f0d2e46d827bddd15dba9518bf8f0d203d9018551144b2b57fd`。

## 决策

不能在看到结果后把阈值网格扩到 0.01 以下，也不能把 90% 门降到 89.46%。曲线证明模型已
学会 bond/text 分离，剩余问题是 soft-alpha 边缘置信度与二值任务目标不一致。下一版从 P1
checkpoint 继续固定一轮，将 `alpha > 8/255` 直接作为二值训练 target，并恢复固定 0.5
评测。若仍失败，再扩大模型或改变损失；不继续做阈值搜索。
