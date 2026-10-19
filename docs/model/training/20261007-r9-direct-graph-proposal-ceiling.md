# r9 direct Atom-pair graph proposal ceiling

本轮验证了不经过 line-first/LSD endpoint 的自研路线。输入图片先产生
AtomObject 候选；对保留下来的 Atom 候选枚举全部无序 pair；现有 frozen
DirectAtomPair head 直接判断 pair 是否为键。候选构造不读取 graph/layers
身份；这些文件只在 proposal 完成后用于事后指标。没有 OCR、LSD、训练、GPU
或 production checkpoint 写入。

脚本：`tools/probe_direct_graph_proposal_ceiling_r9.py`

报告：
`outputs/audits/chemvector-identity-aligned-direct-graph-proposal-ceiling-r9/report.json`

## 固定协议

- 数据：r6 targeted diversity 的 `fresh_holdout`，95 张图，四个 profile；
- 权重：production initial，仅验证 SHA，不改写；
- Atom：frozen AtomObject peak + frozen set presence（阈值 `-0.25`）；
- Edge：所有无序 Atom pair，DirectAtomPair presence `p >= 0.85`；
- 事后 Atom 映射半径：16 px；
- CPU deterministic，单线程，`torch.use_deterministic_algorithms(True)`；
- sealed test 未读取。

## 结果

| 指标 | 结果 |
|---|---:|
| Atom precision / recall | 0.997743 / 0.981132 |
| Edge precision / recall | 0.994845 / 0.934625 |
| dangling rate | 0.003851 |
| bond type accuracy | 0.963731 |

四个 profile 的 edge recall 分别为 `0.9952 / 0.9330 / 0.9347 / 0.8756`，
最低仍高于固定门槛 `0.850031`。edge precision、dangling 和 type accuracy
也满足当前 proposal gate。该结果显著高于 r7 endpoint/segment-port 路线，说明
主要问题不是 Atom pair 图推理，而是 LSD endpoint 表示造成的不可达端点。

## 决策

direct Atom-pair 图路线通过 deterministic ceiling，可进入下一阶段 tiny CPU
scorer（仅训练 pair/graph sidecar，仍不改 production）。必须先固定训练数据、
正负 pair 采样和 early-stop 规则；CPU scorer 通过后才能申请一次受限 GPU
shadow。不能把本报告直接当作 production 验收，也不能读取 sealed test。
