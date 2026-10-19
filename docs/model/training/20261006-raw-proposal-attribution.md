# 1,200 图 raw proposal / selector 端点归因（2026-10-06）

## 范围与安全边界

本次是对固定 chemical-head preparation 绑定的 1,200 张训练图做只读重放。使用
`initial.pt`、生产 `structure_forward`、raw atom decode、当前 selector mask 以及
`match_atom_targets`（6 px）。没有 optimizer step，没有写 checkpoint，没有读取
sealed test，也没有把 GT 坐标注入模型输入。

## 总体结果

| 项目 | 数量 | 说明 |
|---|---:|---|
| target atoms | 36,304 | 目标原子总数 |
| raw atom proposals | 149,127 | decode 后的 dense proposals（每图最多 128） |
| raw-matched targets | 36,072 | raw 对目标的 6 px Hungarian 匹配，99.36% |
| selected proposals | 36,184 | set/selector mask 保留的 proposals |
| selected-matched targets | 35,799 | 最终 pair 阶段可用，98.61% |
| raw correspondence gap | 232 | raw 阶段没有唯一匹配（其中 3 个有 6 px 内竞争近邻） |
| selector drop | 273 | raw 已匹配、但被 selector mask 丢弃 |

因此，当前 505 个未匹配目标中，约 46%（232）是 dense/raw 端点缺口，约 54%
（273）是 selector 保留问题。不能把全部缺口归咎于 selector，也不能用 pair head
补齐 raw 缺失。

## 元素与文字角色分层

### raw 缺口（232）

| 类别 | 数量 |
|---|---:|
| visible text | 193 |
| implicit C | 39（其中 3 个是有近邻但 Hungarian 未取到） |
| O text | 112 |
| N text | 64 |
| 其他文字（Cl/Se/Br/I/P/S/Si/C） | 17 |

O/N 文字占 raw 缺口 176/232（75.9%）；这是 dense atom field/字体样式问题，
不是 pair 化学类型监督问题。

### selector drop（273）

| 类别 | 数量 |
|---|---:|
| implicit C | 181 |
| visible text | 92 |
| O text | 37 |
| N text | 33 |
| 其他文字（Br/Se/I/Cl/Si/B/F/C） | 22 |

selector drop 主要是隐式 C 的密集局部竞争，而不是 O/N 文字识别本身。

## 样式分层

表中 gap 是目标原子数减去 selected-matched；括号内为
`raw gap / selector drop`。

| style | target | raw matched | selected matched | gap |
|---|---:|---:|---:|---:|
| font_large | 5,606 | 5,501 | 5,430 | 176 (105 / 71) |
| serif | 6,098 | 6,019 | 5,971 | 127 (79 / 48) |
| line_thick | 6,383 | 6,372 | 6,325 | 58 (11 / 47) |
| font_small | 6,540 | 6,532 | 6,503 | 37 (8 / 29) |
| reference | 5,931 | 5,924 | 5,885 | 46 (7 / 39) |
| bond_short | 4,060 | 4,046 | 4,017 | 43 (14 / 29) |
| bond_long | 572 | 566 | 562 | 10 (6 / 4) |
| line_thin | 638 | 637 | 632 | 6 (1 / 5) |
| rotated | 476 | 475 | 474 | 2 (1 / 1) |

font_large 与 serif 合计贡献 303/505（60.0%）gap；二者的 raw 缺口主要就是
O/N 字符。line_thick 的主要问题已经转成 selector 对隐式 C 的保留（47/58）。

## 决策建议

1. 不继续训练 pair presence/chemical head；它无法让 232 个 raw 缺失端点变得可达，
   且上一轮 joint 训练已经出现边级回归。
2. 不直接训练一个混合 selector loss。先拆成两个低成本、互不混淆的实验：
   - **A：dense/raw endpoint field**：只覆盖 `font_large/serif` 的 O/N，验证 raw
     recall；冻结 selector、pair、OCR，最多做 overfit 小切片，再做 bounded trial。
   - **B：selector retention**：只覆盖 raw 已匹配但被丢弃的隐式 C（尤其
     `line_thick`、`reference`、`font_large`），加入 initial matched retention
     约束；不把 raw-missing 样本伪造成 selector 标签。
3. A、B 均需同时满足：目标类 recall 上升、initial 已匹配端点 retention 100%、
   fixed dev valid/exact/fallback 不下降。否则立即 NO-GO，不增加步数、不解冻 shared
   trunk。
4. 下一轮训练优先级应是 A（真实 raw 缺口集中在 O/N 与 font_large/serif）；B 只能在
   A 的 shadow 证明 selector 输入已有覆盖后再做。两者都不应修改生产 initial。

## 复核绑定

重放使用 `outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt`
及 `outputs/audits/chemvector-edge-label-reachability-20261005/report.json` 绑定的
1,200 图 training-only plan；sealed test 查询数为 0，optimizer steps 为 0，部署变更为 0。
