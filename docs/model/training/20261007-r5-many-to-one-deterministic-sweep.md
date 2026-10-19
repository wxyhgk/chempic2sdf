# R5 many-to-one deterministic sweep

为确认 many-to-one 路线的轻微 recall 增益能否用纯确定性规则修复 precision，
在 frozen edge/type sidecar 上做了 ownership 参数 replay。生产 checkpoint、selector、
OCR、edge/type state 和 sealed split 均未触碰。

探索半径 `{20,24,28,32}`、score lambda `{0,4,8,12}`、candidate score floor
`{0,.2,.4,.6,.8}`、duplicate merge cap `{8,12,16,20,24}`（共 400 组合）；edge
threshold 固定为 r5 train-only 阈值。完整组合的报告保存在
`outputs/audits/chemvector-identity-aligned-line-first-many-to-one-deterministic-sweep-r5/report.json`。

Hungarian+merge baseline full holdout 是 P/R=`.965917/.850031`。扫描中最高 recall
为 `.859926`，但 precision 仅 `.957975`（r=28, λ=12, score floor=0, merge=24）；
较平衡点 P/R=`.960554/.858380`。没有任何组合同时达到 baseline precision 和 recall
floor，因此不能通过 deterministic 参数修复达到严格 gate。

结论：many-to-one 不启动 GPU edge/type shadow；下一步若继续只能改变 proposal/identity
契约，而非重复 edge scorer 或继续扩大参数 sweep。
