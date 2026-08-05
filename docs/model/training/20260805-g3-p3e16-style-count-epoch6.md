# G3 P3-E16 Style90k Count 六次完整遍历

> 日期：2026-08-05
>
> 状态：Count Gate GO；尚未接回 Atom selector/router
>
> 基础 checkpoint：P3-E15 style90932 epoch3 h64
>
> 输出：`outputs/smoke/chemvector-g3-p3e16-deepsets-count-style90932-epoch6-h64`

## 目标

P3-E15 在累计三次完整遍历后已通过 dev exact 与 confirm MAE，但 confirm exact 只有
120/256，historical route100 为 39/100。训练域 exact 仍在提高，continuation loss 未饱和。

本轮恢复 P3-E15 的模型与 AdamW optimizer，再做三个完整 style90k epoch，使每张唯一图片
累计至少出现六次。模型、普通 CE、学习率、evidence、评估集和门控全部不变；只读取最终
checkpoint，不选择中间 step。

## 训练总账

```text
source global step = 4,402
source samples seen = 281,344

continuation epochs = 3
continuation exposure = 272,796
continuation steps = 4,263

final global step = 8,665
final samples seen = 554,140
```

最终包含六次完整的 90,932 图遍历，外加 P3-E14 matched-exposure 的 8,548 次额外 exposure。
训练与保护集 molecule/scaffold 重叠仍为0；MR-TADF未用于训练或选择。

训练命令：

```bash
.venv-train/bin/python -m \
  chemvector.training.atoms.train_g3_p3e16_style_count_epoch6
```

## 预注册门

- dev Count exact ≥ 50%；
- confirm Count exact ≥ 50%；
- confirm MAE ≤ 1.0；
- historical route100 Count exact ≥ 40%；
- 分桶仅解释 fixed-final，不参与 checkpoint 选择。

## 总体结果

| 集合 | P3-E15 exact / MAE | P3-E16 exact / MAE | 变化 |
|---|---:|---:|---:|
| style90932 | 42,581/90,932 / 1.174 | **44,275/90,932 / 1.053** | +1,694 / -0.121 |
| dev256 | 134/256 / **0.867** | **148/256** / 0.902 | +14 / +0.035 |
| confirm256 | 120/256 / **0.910** | **128/256 / 0.926** | +8 / +0.016 |
| historical route100 | 39/100 / 1.120 | **41/100 / 1.040** | +2 / -0.080 |

四项 Count 门全部通过。step、exposure、最终 global step、samples seen 和重复评估也全部通过。
但 dev/confirm MAE 相对 P3-E15 轻微回退，不能把这轮描述为所有指标单调改善。

## Count 分桶

P3-E16 fixed-final：

| 真值 Count | style90k exact | dev exact | confirm exact | route100 exact |
|---|---:|---:|---:|---:|
| 1–20 | 52.22% | 58.33% | 50.00% | 68.75% |
| 21–40 | 53.17% | 64.06% | 56.67% | 37.50% |
| 41–60 | 26.87% | 16.67% | 22.73% | 25.00% |
| 61–80 | 12.10% | 无样本 | 无样本 | 无样本 |
| 81–100 | 10.97% | 0/4 | 无样本 | 无样本 |

confirm 的主要残差已明确集中在 `41–60`；该桶 MAE 2.20，且相对 P3-E15 的 11/44 退到
10/44。style90k 的 61–100 虽比 P3-E15 改善，但绝对准确率仍低。继续用总 epoch 平均施压
会主要优化占 72.6% 的 21–40 样本，不能自动解决大 Count。

## Variant 分桶

| Variant | style90k exact | dev exact | confirm exact | route100 exact |
|---|---:|---:|---:|---:|
| coordgen | 55.14% | 62.50% | 53.13% | 44.00% |
| original | 52.68% | 62.50% | 50.00% | 52.00% |
| rdkit | 52.19% | 70.31% | 56.25% | 48.00% |
| stress | 34.76% | 35.94% | 40.63% | 20.00% |

stress 在所有集合均最低。confirm stress 从 P3-E15 的21/64提高到26/64，但 route stress 从
6/25退到5/25；风格残差仍不能只靠继续增加普通 epoch 解决。

## 决策

- 独立 Count Gate：`GO`，这是 G3 Count 主线首次四门全部通过；
- 端到端能力：尚未验证，不能写成 Atom set、Skeleton 或 SDF GO；
- P3-E16 成为当前 Count checkpoint；
- 停止继续训练相同 Count 单头。下一步冻结 P3-E16 Count，替换 P3-E8 中失败的 P2-W Count，
  在固定 P3-E7 base/expert router 上重新运行 predicted-count Atom set；route100仍只做历史诊断。

## 产物

- checkpoint：`step-008665.pt`；
- SHA256：`50b7f07a723468f1ef50b21011cc65320af9bad7c6ecfd88be3f5549444ef236`；
- final samples seen：554,140；
- P3-E16 协议与分桶单元测试通过；
- 文档复核日期：2026-08-05。
