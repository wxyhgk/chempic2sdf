# pair-scale 芳香 minimum threshold 离线审计

本次是对已保存 15 张 trace 的 CPU 重算，只使用 report 中的 pair presence/chemical logits：

- model forwards = 0；
- training = false；
- sealed test = false；
- production threshold 未修改；
- 不重新编译 SDF，因此不是准确率实验。

工具与结果：

- [audit tool](../../../tools/audit_pair_scale_aromatic_thresholds.py)
- [audit report](../../../outputs/audits/chemvector-pair-scale-aromatic-thresholds-20260925/report.json)

## 结果

pair arm `pair-015443` 的 broken 组 accepted aromatic cycles 合计：

| minimum | broken cycles | rescued cycles |
| ---: | ---: | ---: |
| 0.00 | 24 | 26 |
| 0.02 | 21 | 22 |
| 0.03 | 18 | 21 |
| 0.05（当前） | 16 | 21 |
| 0.08 | 16 | 21 |

降低 minimum 确实能让部分 broken 图重新接受环：例如 `000263` 在 0.05 时为 0 个环，0.02 时为 2 个；`000120` 在 0.05 时 1 个，0.00 时 2 个。

但这不是安全修复：

1. 它只重做 promotion，不能恢复已被 presence 删除的边，也不能修复 chemical argmax 或超价边；
2. 低到 0.02 时，rescued 组也有多个图的 accepted cycles 下降，说明该行为对低置信度边并不单调安全；
3. 例如 `000314` 即使最低门槛降到 0.02，也只有 1 个环，仍低于 initial 的 3 个；`000676` 仍低于 initial；
4. 因此不能直接把 production minimum 从 0.05 改成 0.02 或 0.00。

## 决策

- threshold sweep：只读诊断完成，**不改变 production**；
- 降低 minimum threshold：**NO-GO**，不能作为当前退化修复；
- 主要修复方向仍是 chemical/aromatic 稳定性保护，presence 单独处理；
- 下一次若做训练，应冻结 presence，只对 chemical head 做受限保护实验，并以 15 张 witness 的 fallback/invalid 零回退作为硬门槛。
