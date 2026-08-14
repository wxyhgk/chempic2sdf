# v1 P0-R Edge-centric 安全清单

> 日期：2026-08-14
>
> 状态：GO；安全训练输入清单已冻结
>
> 审计：`chemvector-v1p0r-edge-centric-safe-manifest`

## 原因

v1 P0 的 10,000 张中有 9,947 张完整通过，但覆盖率 99.47% 低于预注册 99.50%，因此原实验
保持 NO-GO。53 张异常来自 38 个分子，错误仅为坐标重合或视觉几何与源立体标签冲突；让模型
学习这些图会制造同图异义标签。

本轮不修图、不补图，也不降低 P0 门槛。只建立新的训练输入契约：一个分子的任一布局发生上述
协议错误，该分子的 original、rdkit、coordgen、stress 四张全部退出。

## 固定输入

- P0 `result.json` SHA256：
  `2061c375971eac10cee49018d1fffedf18d66e57a41e361833b09139ceeaf143`；
- P0 `skip_diagnostics.json` SHA256：
  `86268c54faf905d5e359befab45da0ab25da51a73b00441335d6c1eefb241ceb`；
- ChemVision pilot10k 的五个 tar 继续使用 P0 已冻结 SHA256；
- 不读取 USPTO、MR-TADF、sealed test 或模型预测。

## 输出

生成 `outputs/audits/chemvector-v1p0r-edge-centric-safe-manifest/manifest.jsonl`。每行只记录：

- sample key、molecule id、split、variant 和 stress operation；
- 原 tar 相对路径；
- PNG、四层图和三个 JSON 的原成员名。

清单不复制、修改或生成图片。预计排除 38 个完整分子、152 张图片，保留 2,462 个分子、
9,848 张图片。

## GO/NO-GO

- 原始输入严格为 2,500 个分子、10,000 张；
- 排除严格为 38 个完整四布局分子，不允许只删失败布局；
- 安全清单严格为 2,462 个分子、9,848 张，四布局各 2,462 张；
- 对安全清单逐张重新执行 ChemScene 对象不变量和 SDF/InChIKey 往返，必须 100% 通过；
- incident ports 必须始终等于 BondObject 数量的两倍；
- 零新 PNG，P0 结论仍为 NO-GO。

全部通过后，P0-R 只说明“训练输入清单可用”，不说明视觉模型有效。下一周期才能在 train split
上做 Text/Bond/Endpoint 小型容量实验；validation/internal test 只能按原 split 评估。

## 实际结果

预注册提交为 `203f6f1`。全量执行耗时 87.11 秒，12 项门全部通过：

| 指标 | 结果 |
|---|---:|
| 原始分子 / 图片 | 2,500 / 10,000 |
| 排除分子 / 图片 | 38 / 152 |
| 安全分子 / 图片 | 2,462 / 9,848 |
| train / validation / internal test | 8,856 / 496 / 496 |
| 每种布局 | 2,462 |
| 安全 Oracle 往返 | 9,848 / 9,848 |
| AtomObject / BondObject | 324,808 / 352,264 |
| incident ports | 704,528 |

四类 stress 仍完整覆盖：crossed-bonds 616、crowded-subgraph 608、long-bond 617、
near-nonbonded 621。清单构建未读取任何模型输出，没有补图或重渲染，P0 原始 10k 仍保持
NO-GO。

冻结产物：

- `manifest.jsonl`：9,848 行，SHA256
  `eb9da30c85aaede01b635294f53dca4454825f13db55002cb2df843edb49f11b`；
- `result.json`：SHA256
  `b497e3c9d0abaca99621d79c02398845f79a183409550c958fb7f1d55cf647e0`。

决策：安全清单 GO。下一周期只能使用 8,856 张 train 图片做容量训练；496 张 validation
用于固定评估，496 张 internal test 继续保留，不能参与 checkpoint、阈值或超参数选择。
