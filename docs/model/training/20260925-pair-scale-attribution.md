# 万图 pair-only 训练：完整分子退步归因

## 范围

本记录只读取已保存的 `validation-000000.json` 与 `validation-015443.json`，
没有重新加载模型、读取开发图片、启动前向或修改训练产物。
因此结论是**预测输出层面的归因**，不是对隐藏 pair logits 的直接因果证明。

## 总体变化

完整分子开发集 800 张：

- exact：170 -> 167；6 张救回，9 张破坏；
- valid：552 -> 550；
- fallback：51 -> 62，增加 11 张。

课程开发集 144 张：exact 63 -> 96，但这不能抵消完整分子退步。

## 9 张被破坏的完整分子

审计工具逐个比较了 800 张完整分子的保存 MolBlock、valid、exact、fallback、error
和图计数。结果是：9 张 exact 被破坏，6 张 exact 被救回；9 张被破坏中有 6 张
同时出现芳香原子和芳香键下降。

### 芳香结构退化：6 张

这些样本的最终输出仍可能是 valid，但芳香原子和芳香键显著减少，部分结构被转成脂环/非芳香键。
其中 4 张触发 fallback，说明局部键型变化最终影响了整图化学编译。

| 样本 | 最终现象 |
| --- | --- |
| `fresh_dev_000314_rdkit` | 芳香原子 15 -> 5，芳香键 16 -> 5，环数 2 -> 0；valid 但 exact 破坏 |
| `fresh_dev_000746_rdkit` | 芳香原子 20 -> 0，芳香键 21 -> 0；fallback |
| `fresh_dev_000953_stress` | 芳香原子 15 -> 0，芳香键 16 -> 0；fallback |
| `fresh_dev_000263_rdkit` | 芳香原子 9 -> 0，芳香键 10 -> 0；fallback |
| `fresh_dev_000120_coordgen` | 芳香原子 12 -> 0，芳香键 12 -> 0；fallback |
| `fresh_dev_000676_stress` | 芳香原子 16 -> 10，分子从 1 个组件变成 2 个；valid 但 exact 破坏 |

### 化学连接/价态破坏：3 张

| 样本 | 最终现象 |
| --- | --- |
| `fresh_dev_000221_rdkit` | P=O 中的 O 脱离，输出变成两个组件；valid 但 exact 破坏 |
| `fresh_dev_001900_coordgen` | 大分子出现 `Explicit valence for atom # 48 C, 5`，输出 invalid |
| `fresh_dev_001091_rdkit` | 大分子出现 `Explicit valence for atom # 18 C, 5`，输出 invalid |

因此 9 张破坏并不是统一的“键存在性下降”：主要是芳香/键型退化，其次是断开或价态非法。

课程图作为对照：144 张中 exact 为 63 -> 96，34 张救回、1 张破坏；没有观察到芳香计数下降。

## 6 张救回的完整分子

救回主要来自原先的芳香结构恢复：

- `fresh_dev_000883_rdkit`：芳香原子 0 -> 12，fallback 消失，输出变成 exact；
- `fresh_dev_001800_coordgen`：芳香原子 0 -> 27，fallback 消失，输出 exact；
- `fresh_dev_001539_coordgen`：芳香原子 0 -> 48，fallback 消失，输出 exact；
- `fresh_dev_000437_coordgen`：保持芳香环并修复一个非芳香键，输出 exact；
- `fresh_dev_000054_rdkit`：修复一个键型，输出 exact；
- `fresh_dev_001091_coordgen`：修复一个额外连接，输出 exact。

这表明同一更新方向同时能够恢复芳香结构，也能够把原本正确的芳香结构推向退化。
全局平均 loss 或课程图 exact 无法表达这种风险。

## 当前能下的结论

1. 训练已覆盖全部万图数据，因此“训练量不足导致本轮没有真正训练”不成立。
2. pair-only 更新对课程域有效，但对完整分子中的芳香环和复杂连接不稳定。
3. 训练后 fallback 增加 11 张，且破坏样本集中在芳香结构和大分子，说明整图化学约束没有被训练目标充分保护。
4. 由于保存结果没有 pair logits、候选行和编译 trace，不能进一步断言每张图是 presence logit、chemical logit、芳香提升还是 role/OCR 导致。
   这需要另一个明确授权的诊断运行，不能从当前 JSON 过度推断。

## 下一步建议

- 暂停继续训练 pair-only 或追加第二遍；不使用 `pair-015443.pt`。
- 先设计一个只读、固定少量 broken/rescued 图的 prediction trace 诊断，保存 pair presence、chemical logits、芳香提升前后和编译 trace。
- 若确认主要是 chemical/aromatic 类型漂移，下一轮应加入**芳香与非芳香边界保护/图级一致性损失**，而不是单纯增加样本或降低阈值。
- 若确认是 presence 改变导致 P=O/大分子断边，则应把键存在性与键类型分开约束，并对完整分子回放加权。
- 新诊断或新训练都必须重新绑定数据用途、预算和授权；本次万图训练授权已消耗。

## 证据

- [只读输出层归因审计](../../../outputs/audits/chemvector-pair-scale-attribution-20260925/report.json)
- [审计工具](../../../tools/audit_pair_scale_attribution.py)
- [完整训练结果](20260925-pair-scale-training-result.md)
- [运行报告](../../../outputs/experiments/chemvector-pair-scale-20260924/report.json)
- [初始开发输出](../../../outputs/experiments/chemvector-pair-scale-20260924/validation-000000.json)
- [最终开发输出](../../../outputs/experiments/chemvector-pair-scale-20260924/validation-015443.json)
- [最终决策](../../../outputs/experiments/chemvector-pair-scale-20260924/decision-015443.json)
