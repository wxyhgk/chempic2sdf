# r5 node + frozen edge sidecar failure attribution

本审计只读重放现有 node MLP 与 frozen edge sidecar，未训练、未读取 sealed test、未修改 production checkpoint。GT 仅在 proposal 完成后用于归因。

脚本：`tools/audit_line_first_node_edge_failure_attribution_r5.py`

报告：`outputs/audits/chemvector-identity-aligned-line-first-node-edge-failure-attribution-r5/report.json`

## 474-image holdout

- GT bonds：3234；recovered：2581，recall `.7981`；
- unreachable：653（`.2019`）；
- node-drop：`0`；edge-score miss：`0`；
- selected edge precision：`.9598`；FP：108；dangling：153（`.0496`）；self-loop：`0`；
- keep-all 与 drop-isolated A/B 结果完全相同。

六个 profile 的 recall 分别为 `.7161/.7848/.7180/.8571/.8609/.8516`。因此当前主要瓶颈不是 node keep、edge threshold、isolated-node 过滤或 bond type，而是 raw line endpoint/candidate reachability，尤其 cov13 的 endpoint 检出。

## 决策

不启动 GPU node/type/router shadow；不继续调 ownership 或 isolated-node 过滤。下一步只做 target-free raw endpoint reachability / line-detector 数据审计，按 profile、线宽、旋转、芳香环和 junction 分层；只有 reachability ceiling 改善后才重新评估训练。
