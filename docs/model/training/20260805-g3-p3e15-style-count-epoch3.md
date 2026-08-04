# G3 P3-E15 Style90k Count 三次完整遍历

> 日期：2026-08-05
>
> 状态：能力门 NO-GO；exposure 因果 GO
>
> 基础 checkpoint：P3-E14 style90932 h64
>
> 输出：`outputs/smoke/chemvector-g3-p3e15-deepsets-count-style90932-epoch3-h64`

## 目标

P3-E14 在与 P3-E10 相同的 exposure/step 下，把训练 molecule 从 829 扩到 22,733，confirm
从 108/256 提高到 117/256、MAE 1.27 降到 1.08。但 style90k 训练域自身仅 42.14% exact，
说明一次完整遍历仍欠拟合。

本轮只验证 exposure：恢复 P3-E14 的模型和 AdamW optimizer 原状态，在完全相同的
90,932 张 evidence 上再做两次完整随机遍历。模型结构、loss、学习率、冻结 evidence、评估
集合与门控均不变。

## 数据与隔离

- 训练：P3-E14 固定的 90,932 图、22,733 molecule、19,806 scaffold；
- evidence index SHA256：`14444a819fedf62130452de0aa85a6d62b061f644deb0dda4f46978a1c68ebb9`；
- dev256、confirm256 与 historical route100 逐样本保持不变；
- 训练与保护集 molecule/scaffold 重叠为 0；
- MR-TADF 未参与训练、阈值、epoch 或 checkpoint 选择。

## 续训总账

P3-E14 已完成一次全量遍历，并为 matched-exposure 额外重复 8,548 张。本轮新增两个完整
unique-data epoch：

```text
P3-E14 global step = 1,560
P3-E14 samples seen = 99,480

P3-E15 continuation epochs = 2
continuation exposure = 181,864
continuation steps = 2,842

final global step = 4,402
final samples seen = 281,344
```

因此 checkpoint 包含三次完整 unique-data 遍历，外加 P3-E14 为计算匹配产生的 8,548 次
额外 exposure。后者在 audit 中单列，不能把 `281,344 / 90,932` 简写成“正好 3 epoch”。

每个 epoch 都是独立固定随机排列，1,420 个 batch64 加一个 batch52；不保存或选择中间
checkpoint。

训练命令：

```bash
.venv-train/bin/python -m \
  chemvector.training.atoms.train_g3_p3e15_style_count_epoch3
```

## 预注册门

- dev Count exact ≥ 50%；
- confirm Count exact ≥ 50%；
- confirm MAE ≤ 1.0；
- historical route100 Count exact ≥ 40%；
- route100 只做 fixed-final 历史诊断。

## 结果

| 集合 | P3-E14 exact / MAE | P3-E15 exact / MAE | 变化 |
|---|---:|---:|---:|
| style90932 | 38,319/90,932 / 1.538 | **42,581/90,932 / 1.174** | +4,262 / -0.364 |
| dev256 | 120/256 / 1.234 | **134/256 / 0.867** | +14 / -0.367 |
| confirm256 | 117/256 / 1.078 | **120/256 / 0.910** | +3 / -0.168 |
| historical route100 | 37/100 / 1.280 | **39/100 / 1.120** | +2 / -0.160 |

正式门通过两项：dev exact 与 confirm MAE。confirm exact 为 46.875%，route100 exact 为
39%，仍低于 50%/40% 门槛。续训 step、exposure、global step、samples seen 与确定性重复
评估全部通过。

## 解释

所有集合均改善且没有平均指标回退，因此额外 exposure 的因果方向成立。训练域 exact 仍只有
46.83%，训练 loss 在 continuation 末尾仍下降，说明当前并未出现明显饱和。

但是 confirm exact 只增加 3 张，明显慢于 dev 的 14 张和 MAE 的改善。Count 预测正在靠近
真实整数，但未同步转化为足够的整图 exact；不能因为 MAE 已过门就宣布 Count 已解决。

## 决策

- 部署能力：`NO-GO`，不接 router、前端或端到端 SDF；
- exposure 因果：`GO`；
- P3-E15 全面优于 P3-E14，成为新的 Count 研究 checkpoint；
- 下一轮 P3-E16 保持同一模型/optimizer，再增加三个完整 unique-data epoch，使总完整遍历数
  达到 6；仍只评估 fixed-final，同时按 Count 区间和四种 variant 报告错误，判断 confirm
  exact 的慢增长来自数据分桶还是表示上限。

## 产物

- checkpoint：`step-004402.pt`；
- SHA256：`f67ba0ceccabc5ca54217c75c009f685de81ee6a0eaed6fb1516901bdb5fb823`；
- continuation 训练时间：约 10.36 秒（含最终双重评估）；
- 训练协议单元测试通过；
- 文档复核日期：2026-08-05。
