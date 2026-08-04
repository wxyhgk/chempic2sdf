# G3 P3-E6 Transactional Capacity Guard Pilot256

> 日期：2026-08-04
>
> 状态：capacity 保护 GO / general 与 margin Gate NO-GO

## 协议

沿用 P3-E5 的 coverage margin 和 A-GEM 方向，但不再相信一阶近似。每个 epoch 对同一个候选
AdamW update 依次尝试固定倍率：

```text
1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128
```

每次实际更新后立即检查 known-count capacity easy/hard；任一不是 `32/32`，就同时回滚模型和
optimizer state。8 个倍率均失败则该 epoch 跳过更新。general 指标不参与步长或 checkpoint
选择。30 epoch、7,680 exposures、final-only Gate。

## 结果

| 指标 | Step0 | E5 A-GEM | E6 Transactional |
|---|---:|---:|---:|
| easy matched / exact | 967/967，32/32 | 964/967，29/32 | **967/967，32/32** |
| hard matched / exact | 807/807，32/32 | 767/807，5/32 | **807/807，32/32** |
| general matched / exact | 5663/6202，9/192 | 6079/6202，110/192 | 5843/6202，37/192 |
| capacity coverage margin | 63/64 | 30/64 | 62/64 |

前 6 epoch 能接受 `1/4` 到 `1/64` 的更新；第 7 至 27 epoch 连 `1/128` 也无法保持 capacity，
协议正确回滚并跳过。第 28 至 30 epoch 又接受 3 次更新。总计仅 9/30 次实际更新被接受。

这证明共享模型具有 general 学习信号，但 P2-W 的严格 capacity 解位于极窄参数边界；继续细分
步长只会让有效学习趋近于零，不是可扩展训练路线。

## 决策与产物

严格 `NO-GO`：margin 与 general Gate 未过，不进入4,937图或推理。capacity事务保护机制本身
GO，可作为后续安全实验工具。下一步冻结 P2-W 为 base，将 general learner 作为独立 expert，
先做 base/expert oracle 并集与保守路由零训练审计；不再强迫一个共享参数点同时承担两域。

- checkpoint：`step-000030.pt`
- SHA256：`2570583e1b8b3d7d63a0a9f12cdf83effdb0928c76bfc8f657194f9976f2a0d7`
- 输出：`outputs/smoke/chemvector-g3-p3e6-mixed-coverage-transactional-pilot256`
