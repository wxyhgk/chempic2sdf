# r5 targeted 6k quarantine panel

建立了一个 identity-disjoint 的 6k targeted manifest，覆盖短键、30–60° 旋转、DOUBLE/TRIPLE、芳香平行线和 heteroatom/junction 场景。

- manifest：`outputs/audits/chemvector-identity-aligned-line-first-targeted-6k-manifest-r5/manifest.json`
- 1000 个新连接身份，计划 6000 图（800 train / 100 dev / 100 fresh holdout，6 styles）
- 并行渲染 accepted `5975`，rejected `25`；拒绝原因均为 typography clearance 擦除 bond segment
- contract audit：5975/5975 文件、hash、graph identity、SDF identity 全通过；fixed source overlap 为 0
- quarantine 数据：`outputs/datasets/chemvector-identity-aligned-line-first-targeted-6k-r5-parallel/`

fresh holdout 的 target-free direct-LSD proposal ceiling：

- edge precision `.9916`
- edge recall `.7893`
- unresolved/dangling `.2635`
- node precision/recall `.7697/.8558`

六个 profile 的 edge recall 为 `.740-.810`，因此 deterministic proposal gate 未通过。数据 contract 是通过的，但这批数据还不能授权 CPU sidecar 或 GPU shadow；下一步仍需修复 candidate ownership/text-seam endpoint 连接，不能把“5975 张图已生成”当成训练准入。

串行中断目录 `chemvector-identity-aligned-line-first-targeted-6k-r5` 已标记 `INCOMPLETE`，不可使用。
