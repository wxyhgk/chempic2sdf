# 连键端点扰动 A/B：首个检查点停损

## 结论

本次单次授权已执行并消耗。GPU、fresh cache、执行设备接口等价性和初始两臂一致性均通过；
在 attempt 29，两臂各更新 25 次后触发 `stop_regression`，没有继续到 attempt 58。
增强没有通过本次安全改善门槛，不能部署，也不能将中途结果称为完整单遍对照。
`report.complete=true` 表示正常保存了停损结果，不表示实验成功。

这次并非所有指标都下降：总正确键、整图 exact 数量增加，但同时破坏了原正确键和结构有效性。
不能用净增抵消这些退步。无增强对照同样失败，故不能把全部问题归因于端点扰动。

## 执行范围与核验

- 用户原授权表述：`可以开给你`，随后确认 `开干`。
- 新授权用途 `local_pair_anchor_ab58_once`，固定 58 张原训练／回放图，无新增样本。
- 使用自研模型原 initial，只更新 pair head；上游、文字角色及 OCR 冻结。
- GPU 0：NVIDIA GeForce RTX 5090，PyTorch 2.10.0+cu128，CUDA 12.8，FP32。
- 外层运行收据耗时 **15.484 秒 / 600 秒**；worker 报告耗时 13.996 秒。
- 29 次图像尝试／臂，其中 4 次无原监督跳过，实际 **25 次更新／臂，合计 50 / 108**。
- 58 次冻结上游前向；108 次采样等价性前向；70 次训练 pair 前向；232 次评估图像前向
  （两臂各初始和 attempt 29，共四轮全 58 图评估）。
- fresh cache 原目标键 1,150，可达 1,103；预计完整一遍 54 次更新／臂，未超限。
- 每臂实际使用 659 条原监督行；增强臂另使用 370 条偏移监督行，5 次非空更新没有合格增强行。
- 无自动重试、续训、CPU 回退、新身份核验、开发／封存样本读取或默认模型替换。
- 事后只读核验：241 个报告绑定产物哈希一致；初始两臂指标完全一致；由保存的全量记录重新计算
  `paired_decision` 与原 decision/report 完全一致；运行进程已退出。

## 固定 58 图结果

下表均在正常原位置、全部候选对上评估，不在偏移子集上评估；这是训练内检查，不是独立测试准确率。

| 指标 | initial | 无增强，25 次更新 | 有增强，25 次更新 |
| --- | ---: | ---: | ---: |
| raw 存在与化学键型正确 / 1,150 | 1,024 | 1,032 | 1,031 |
| 芳香提升后存在与化学键型正确 / 1,150 | 1,030 | 1,039 | 1,037 |
| 原正确 raw 键被破坏 | — | 6 | 9 |
| 新修好的 raw 键 | — | 14 | 16 |
| 原正确 promoted 键被破坏 | — | 6 | 9 |
| 新修好的 promoted 键 | — | 15 | 16 |
| SDF exact / 58 | 19 | 21 | 21 |
| SDF valid / 58 | 53 | 52 | 52 |
| fallback 图数 | 3 | 4 | 6 |
| 确认假连接总数 | 1 | 1 | 1 |
| 相对 initial 新确认假 pair ID 数 | — | 0 | 0 |
| unknown 中预测存在的 pair 数 | 18 | 22 | 23 |

raw/promoted 是局部化学键型指标，不是最终 SDF 的逐键或立体化学准确率。
unknown 不能直接当假边或训练负例。两臂均保住原 19 张 exact 图，新增的 2 张 exact 都来自 witness 组；
32 张旧完整分子的 exact 保持 8 张，没有增加。

## 退步定位

对照有 8 张图触发保护，增强有 9 张；所有这些图在本轮停止前都尚未进入更新。
固定 manifest 前 26 张是 witness，后 32 张是 old_full_molecule。
截至 attempt 29，已访问 26 张 witness + 3 张旧完整分子，实际更新为 22 + 3。
因此，本次结果存在明显的训练前缀不均衡，尚不能证明混合完整一遍仍会同样退步，
也不允许为了观察后半遍而绕过已约定的停损。

增强臂破坏的 9 条原正确 raw 目标键如下（ID 为本次目标键编号，不是候选 pair 行号）：

| sample_key | 键 ID |
| --- | --- |
| `curr_537bc20c087abbbd__font_small` | 29 |
| `curr_e60caa32d96d15cb__font_small` | 4, 6 |
| `curr_3b3fc1cbb2cc2063__bond_short` | 33 |
| `curr_4ef506af64e8f0b8__bond_short` | 45 |
| `curr_08d02946e879406f__font_large` | 6 |
| `curr_5c99d711703c1c0c__font_small` | 6, 9 |
| `curr_ef00e509a2b31f0a__serif` | 16 |

保存的编译 trace 进一步确认：

- `curr_cb5b7da426023d14__font_large`：两臂均新增芳香 Kekulization 失败并触发 fallback。
- `curr_3b3fc1cbb2cc2063__bond_short` 和 `curr_ef00e509a2b31f0a__serif`：仅增强臂新增该类失败。
- `curr_20d6303f5788a4d7__serif`：两臂均从有效变无效，报告 O 的显式价态为 3，编译失败。
  该图局部正确键集合未下降，说明仅保护正确键总数不足以保护最终结构。

这些是保存输出中的直接观察，不等于已完成训练梯度或单条边到编译失败的因果归因。

## 后续优先级（尚未执行或授权新试训）

1. 先复用本轮 logits、pair 行及编译 trace，拆分退步来自连键存在性、键型变化还是芳香／编译联动；
   尤其核对两臂共同破坏与增强独有破坏，不需要再跑模型。
2. 优先设计从训练开始就交错基础样本与完整分子的对照，并明确旧正确键保留约束。
   当前“先基础前缀、后完整分子”的顺序是待验证因素，不能直接宣称根因已证实。
3. 原增强方案暂不扩大数据量、方向数、轮次或预算；先解决无增强对照也无法保住旧能力的问题。
   新顺序／新目标若需试训，应另定单一变量与预算，不能复用本次已消耗授权。

## 证据

- [授权记录](releases/20260924-pair-anchor-ab-authorization.json)
- [固定协议](20260924-pair-anchor-ab-protocol.md)
- [运行报告](../../../outputs/experiments/chemvector-pair-anchor-ab-20260924/report.json)
- [停损决定](../../../outputs/experiments/chemvector-pair-anchor-ab-20260924/decision-029.json)
- [运行收据](../../../outputs/experiments/chemvector-pair-anchor-ab-20260924/runtime.json)

绑定 SHA256：

| 产物 | SHA256 |
| --- | --- |
| authorization | `78a2bb73535c11acfda3ebd160707157fa0536e165477ebd24a9b01fb050fe83` |
| report | `9aed03254e095bcd8ccceb7bdd49bd1fed94e0e7e1325a400a0070d8f9831e53` |
| decision-029 | `c9c5b75c0f3ee4720749348e5f2613e3785e5dd25f6ab0e2b50ae6553077d40c` |
| runtime | `413e863554d4dcded4d0b2a383f0c26a150b3804ac7b7fb6bb80fe54161075dd` |
| cache-manifest | `a3720bc1cf934e27ae2f22c6c8b5cbfdfc6a72fb1cff865ec0f1f97ed43069b9` |
