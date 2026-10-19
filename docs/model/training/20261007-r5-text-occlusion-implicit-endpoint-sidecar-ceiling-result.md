# r5 text-occlusion / implicit-endpoint sidecar：CPU ceiling 结果

本次审计是只读的 CPU ceiling；没有读取 sealed test，没有训练，没有修改 production checkpoint、selector、OCR 或 edge/type head。

## 结果

脚本：`tools/audit_line_first_lsd_text_seam_additive_ceiling_r5.py`

报告：`outputs/audits/chemvector-identity-aligned-line-first-lsd-text-seam-additive-ceiling-r5/report.json`

holdout（474 图）：

- suffix proposals：5243；suffix precision `0.7344`，suffix recall `0.5053`；
- prefix 保留并加入 suffix 后：edge precision `0.8305`，edge recall `0.9103`；
- oracle sanitize `0.3186`，dangling rate `0.7814`。

严格门槛要求 union precision 至少保持当前基线、sanitize 不退化，并满足 profile floor。本次全部未满足，因此该 text-seam / implicit-endpoint 规则 **NO-GO**，不进行 CPU fit 或 GPU shadow。

## 失败归因

当前规则用 glyph bbox 与 LSD 端点的距离/方向作为 seam 证据，产生了大量“线段靠近文字框但并非同一化学连接”的候选；同时 prefix 中的未归属端点会把悬空传播到 suffix。仅靠扩大训练量不能消除这一结构性误报。

## 后续约束

冻结该路线及其参数 sweep，不再进行 GPU 尝试。下一步应做失败分层（文字遮挡、真实隐式碳、悬空线段、字体/缩放 profile），只有提出能在 deterministic replay 中先达到严格 precision/sanitize gate 的新证据，才重新考虑 sidecar。
