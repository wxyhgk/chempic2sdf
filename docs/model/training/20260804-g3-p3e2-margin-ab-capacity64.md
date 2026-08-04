# G3 P3-E2 Margin A/B Capacity64

> 日期：2026-08-04
>
> 状态：两个臂均 NO-GO；旧 margin 因果否决，coverage margin 仍受 microbatch 级联影响

## 目标与协议

从同一个 P2-W checkpoint 重开，只训练已达到 easy/hard `32/32` 的 capacity64。两个臂唯一
差异是结构化 margin：

- `capacity-fixed`：`mean` 口径固定 margin 1；
- `capacity-coverage`：margin 为 `missing_gt / atom_count`。

两臂均为 batch4、30 epoch、480 update、1,920 exposures，每图恰好 30 次；AdamW `3e-4`，
Count/coordinate/factorized loss 禁用，292,693 个结构参数可训练。只看 fixed-final，不读取
route100 或 MR-TADF。

## 结果

| 指标 | P2-W Step0 | 固定 margin | coverage margin |
|---|---:|---:|---:|
| easy matched / exact | 967/967，32/32 | 966/967，31/32 | 960/967，29/32 |
| hard matched / exact | 807/807，32/32 | 769/807，12/32 | 708/807，2/32 |
| capacity selected-margin | 固定1/64；coverage63/64 | 22/64 | 28/64 |

没有 general 数据参与时，固定 margin 仍使 hard 从 `32/32` 降到 `12/32`，直接证明 P3-E1
回退不是 general 域冲突的必要结果，旧 margin 本身就会破坏正确解。

coverage margin 虽把起点 active 图从 63 张降到 1 张，但 batch4 中该单图梯度仍触发强更新；
其它 60 张约束在该 step 不可见，错误随后逐 batch 级联。两臂训练梯度长期强裁剪。

## 决策与产物

两个臂均 `NO-GO`。下一步保持 coverage margin，将 capacity64 的 16 个 microbatch 梯度全部
累积后每 epoch 只更新一次。

- fixed checkpoint SHA：`c9adbb86072854984963a45a435f864418ca7c107f9364cb439ebde190375e65`
- coverage checkpoint SHA：`c643f12f5d87018f09a53821bcb94d23516369ffa8c105f192256f6c56e324c8`
- 输出：`outputs/smoke/chemvector-g3-p3e2-capacity-*-margin-pilot64`
