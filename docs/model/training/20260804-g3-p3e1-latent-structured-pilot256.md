# G3 P3-E1 Latent Structured Pilot256

> 日期：2026-08-04
>
> 状态：NO-GO；latent 方向有效，固定整图 margin 否决
>
> 基础 checkpoint：P2-W `step-004800.pt`
>
> 输出：`outputs/smoke/chemvector-g3-p3e1-latent-structured-pilot256`

## 目标

验证不固定 query/candidate/GT 身份的 latent valid-vs-best-invalid 结构目标，能否在保护
capacity64 的同时学习 192 张 general 图片。Count 使用完全冻结的独立 P2-W 模型，本轮结构
训练不含 Count CE、coordinate loss、旧 factorized loss 或固定 query identity。

## 数据与协议

- capacity：easy32 + hard32；general：固定前 192 张 eligible 图片。
- 每批 `3 general + 1 capacity`；30 epoch、1,920 step、7,680 exposures。
- 每张图恰好曝光 30 次；训练图片共 256 张。
- AdamW，学习率 `3e-4`，weight decay `0.01`，BF16，梯度裁剪 `1.0`。
- 可训练结构参数 292,693；Count 模型和结构模型的 Count head 完全冻结。
- 未读取 route100；MR-TADF 未参与训练、选门或 checkpoint 选择。
- 只保存 fixed-final；不追加 epoch，不选中间 checkpoint。

复核发现 general192 只有 57 个 molecule，其中 99 张来自 capacity 中 33 个 molecule 的其他
绘图风格。因此本实验只能证明样本/风格层面的学习信号，不能表述为分子隔离泛化。后续实验必须
另设 molecule-disjoint 集合。

## 预注册门控

- easy/hard known-count 与 frozen-count 均为 `32/32`。
- general known exact 至少 `190/192`，matched recall 至少 99.9%。
- capacity margin `64/64`；general margin至少 `183/192`。
- 固定 step/exposure、Count不变、两次 final replay 完全一致。

## 结果

| 指标 | Step0 | Final |
|---|---:|---:|
| easy matched / exact | 967/967，32/32 | 966/967，31/32 |
| hard matched / exact | 807/807，32/32 | 704/807，2/32 |
| general matched / exact | 5663/6202，9/192 | 6082/6202，155/192 |
| capacity margin satisfied | 1/64 | 28/64 |
| general margin satisfied | 0/192 | 126/192 |
| general structured loss | 1.756 | 0.285 |

训练协议检查全部通过，但能力门全部未过。general 的提升很大，证明 latent 等价集合目标有明确
能力信号；hard capacity 的灾难性回退则否决当前损失与混合协议。

## Margin E1b 审计

P3-E1 使用：

```text
relu(1 + mean(best_invalid_edges) - mean(best_valid_edges))
```

在 P2-W 训练起点，capacity64 本来已全部预测正确，且 64 张的 best-invalid 都只漏 1 个 GT、
只替换 1 个 candidate。按覆盖缺失量计算，`63/64` 已满足单位局部 margin；旧固定整图公式却
只有 `1/64` 满足。旧公式等价于对只错 1 个 Atom 的无效解要求约 `K` 个总 log-score gap，
导致已经正确的图仍承受强重塑梯度。

训练后 hard32 中 30 张的无约束 MAP 已变为无效，best-invalid 缺失量扩展到 1 至 8；这不是
“训练还不够”，而是 margin 定义与真实覆盖错误不成比例。域平均梯度 cosine 在 P3-E1 final
均为正（easy-hard 0.36、easy-general 0.53、hard-general 0.44），不支持先上 PCGrad。

审计结论：`fixed_whole_image_margin_misscaled`。

## 决策

`NO-GO`。不扩到 4,937 图，不进入推理或前端。保留 latent valid/best-invalid solver，下一轮
从同一个 P2-W 初始化做 coverage-calibrated margin A/B，并将 molecule-overlap 因果样本与
molecule-disjoint 能力样本分开。

## 产物校验

- checkpoint：`step-001920.pt`
- SHA256：`7880bc31c906c4afbbfd7265188c68921a115329aa795903ccbbac07c3529a3b`
- 模型名：`chemvector-g3-p3e1-latent-structured-pilot256`
- tests：14 项 latent/best-invalid/P3-E1 测试通过。
- E0b：`outputs/audits/chemvector-g3-p3e0-best-invalid-solver/metrics.json`
- E1b：`outputs/audits/chemvector-g3-p3e1-margin-geometry-audit/metrics.json`
