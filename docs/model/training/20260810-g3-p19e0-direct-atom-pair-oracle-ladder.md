# G3 P19-E0 Direct AtomPair Oracle Ladder

> 日期：2026-08-10
>
> 状态：已完成，旧权重复用 NO-GO
>
> 类型：零训练复用审计，不创建模型 tag

## 目标

P18 已证明 Dense Bond endpoint residual 在 capacity64 成功但 fresh640 回退。P19 将 Bond identity
固定为无序 AtomPair，不再生成独立 Bond endpoint 或 incidence pointer。本轮先判断已有 v0.10.3
Direct AtomPair 权重能否直接复用。

## 固定协议

- 数据与顺序固定为 P18 fresh640；
- 路线 A：GT Atom anchor + 旧冻结 Atom dense feature + v0.10.3 Direct AtomPair；
- 路线 B：P4 top128 经 6px Oracle 一一选择后的 anchor + 同一旧 Atom feature + Direct AtomPair；
- Atom feature/score 在 route anchor 对应的旧 dense cell 采样，保持 v0.10.3 的 128D 输入协议；
- 枚举全部有效无序 AtomPair，不使用距离、近邻、键长或空白中点裁剪；
- 固定 v0.10.3 threshold `0.85`，不训练、不校准、不搜索 checkpoint；
- GT/P4 Oracle 只用于候选上限审计，不得称为部署结果；
- 不运行 Dense Bond、set/incidence，不读取 sealed test、confirm、USPTO 或 MR-TADF。

## GO/NO-GO

- GT Atom route skeleton exact ≥80%；
- GT Atom stress skeleton exact ≥60%；
- GT Atom edge recall ≥97%；
- P4 Oracle Atom exact 必须复现 P18 的 608/640；
- 所有冻结 state 前后完全一致。

全部通过表示旧 v0.10.3 可直接作为 P19 relation 起点；失败只表示旧权重/特征风格不可直接复用，
下一步仍可在同一稳定 AtomPair identity 上做 train-only capacity64，不能退回 Dense Bond endpoint。

## 实测结果

| Atom 路线 | Skeleton exact | Edge F1 | Stress exact |
|---|---:|---:|---:|
| GT Atom | 469/640（73.28%） | 98.60% | 64/160（40.00%） |
| P4 Oracle Atom | 464/640（72.50%） | 98.47% | 62/160（38.75%） |

GT Atom 路线的普通三种布局已经达到较高水平：

- coordgen：133/160（83.13%）；
- original：134/160（83.75%）；
- rdkit：138/160（86.25%）。

但 stress 分桶差异很大：

- crossed_bonds：0/41；
- crowded_subgraph：29/44；
- long_bond：29/41；
- near_nonbonded：6/34。

GT Atom 的 Edge precision/recall 分别为 98.78%/98.41%。P4 Oracle Atom exact 为
608/640，精确复现 P18；所有冻结 state 前后未变化。运行未读取 sealed test、confirm、
USPTO 或 MR-TADF。

## 决策

正式决策为 `existing_direct_atom_pair_reuse_no_go`。GT Atom overall 未达到 80%，stress
未达到 60%；Edge recall、P4 Atom coverage 与冻结状态门通过。

这个结果否定的是旧 v0.10.3 权重的直接复用，不是否定 AtomPair identity。相反，给定正确
Atom 后整体 Edge F1 仍为 98.60%，且普通布局 skeleton exact 为 83%--86%，明显优于 P18
Dense Bond endpoint 路线。当前残差已经收缩为 AtomPair presence 在 crossed_bonds 与
near_nonbonded 上的假边/漏边。

下一步固定 AtomPair 为全部无序原子对，从新初始化 relation 头做 train-only capacity64：
32 个 easy 与 32 个 hard，hard 必须覆盖 crossed_bonds 和 near_nonbonded；不使用 Dense Bond、
endpoint regression、incidence、距离或近邻裁剪。容量通过后才允许用固定 checkpoint 重放
同一 fresh640。

## 产物

- 审计目录：`outputs/audits/chemvector-g3-p19e0-direct-atom-pair-oracle-ladder-fresh640/`；
- `audit.json` SHA256：`186a2c2406fd7c4c47017234a976277dee0f8bfbf282826a17daee887be66c12`；
- `rows.json` SHA256：`6bba91215c28ef9c630d451eeab77f6f94a85bd6230a61b310ca95f926091e52`；
- prediction SHA256：`4eeb5146d6b34e6646c0573db2822c53e8ccc35fedce903e771e4c567954325f`；
- 耗时：18.81 秒；
- 本轮无训练，不创建模型 tag。
