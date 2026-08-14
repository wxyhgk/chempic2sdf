# v1 P1-A 二值墨迹微调

> 日期：2026-08-14
>
> 状态：Validation GO；等待一次性 internal-test 确认
>
> 模型：`chemvector-v1p1a-binary-ink-finetune`

## 假设

P1-R 在阈值 0.01 已将 bond/text IoU 提到 90.09%/80.90%，说明模型学会了语义分离；但
联合前景 89.46% 未过 90% 门。继续降低阈值会违反预注册。根因假设是：P1 用 soft alpha
监督概率，固定 0.5 只覆盖笔画核心，而任务真正需要的是“该像素是否属于可见墨迹”。

## 固定协议

- 初始化 P1 best SHA256：
  `e4f726b80f73e46a71619f8f4a42ff3b0042a3156cc86c0a4f584154b574263c`；
- safe manifest SHA256：
  `eb9da30c85aaede01b635294f53dca4454825f13db55002cb2df843edb49f11b`；
- 所有 target 统一变为 `alpha >= 8/255` 的二值 bond/text mask；
- 模型结构和损失权重不变，全部 1.34M 参数继续训练一轮；
- 只读 8,856 train，AdamW、学习率 `1e-4`、batch 4、bfloat16；
- 496 validation 固定以 bond/text 阈值 0.5 评测；
- internal test、MR-TADF 和 USPTO 不读取。

本轮没有 checkpoint 搜索：一轮终点就是唯一候选，不根据中间 validation 早停或选 step。

## GO/NO-GO

继续使用 P1 原五项 validation 门：bond IoU 90%、text IoU 75%、联合前景 IoU 90%、stress
bond IoU 88%、bond-on-text-only 不超过 10%。同时要求固定阈值确为 0.5、internal test
读取数为 0。

GO 后冻结 P1-A，再以同一个固定 checkpoint 和 0.5 阈值一次性读取 internal test。NO-GO
则说明仅修改 soft/binary target 不足，应改变模型容量或目标结构，不能继续搜索阈值。

## 结果

预注册提交为 `e45aab9`。固定一轮完整读取 8,856 张，耗时 169.03 秒，终点就是唯一
checkpoint：

`outputs/checkpoints/chemvector-v1p1a-binary-ink-finetune/best.pt`

SHA256：`490b5daae3a2059c3d335785a69e0c2a41eed32f0237ef77fc3556c36d134a74`。

固定 0.5 的 496 张 validation 结果：

| 指标 | P1 soft target | P1-A binary target | 门 |
|---|---:|---:|---:|
| Bond IoU | 71.73% | 97.72% | 90%，通过 |
| Bond precision / recall | 99.60% / 71.94% | 98.07% / 99.64% | - |
| Text IoU | 65.09% | 91.85% | 75%，通过 |
| Text precision / recall | 99.78% / 65.18% | 99.15% / 92.59% | - |
| 联合前景 IoU | 71.23% | 99.21% | 90%，通过 |
| Stress bond IoU | 72.11% | 97.56% | 88%，通过 |
| Bond-on-text-only | 1.68% | 9.36% | 10%，通过 |

8 项门全部通过，validation 决策 GO。`result.json` SHA256：
`ce17cfe8fb0e41b877d3a0ece8d0f6f1de3f082404135b49f67394c21a647612`。

## 判断

相同结构、相同数据只对齐监督定义后，bond/text IoU 分别提高 26.0pp/26.8pp，说明 P1 的
主要失败不是数据不足或模型太小，而是 soft-alpha 概率目标与二值结构任务不一致。当前
bond-on-text-only 9.36% 只比 10% 门低 0.64pp，是唯一紧门；因此 checkpoint 暂不接
BondObject，必须保持模型和 0.5 阈值不变，一次性读取 internal test 确认。
