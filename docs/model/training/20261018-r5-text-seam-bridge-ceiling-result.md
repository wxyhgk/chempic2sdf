# r5 text-seam → line-endpoint bridge ceiling

脚本：`tools/audit_line_first_text_seam_bridge_ceiling_r5.py`

报告：`outputs/audits/chemvector-identity-aligned-line-first-text-seam-bridge-ceiling-r5/report.json`

该审计固定 node keep、edge sidecar 和原 edge set，仅增加 target-free glyph seam bridge（单个 glyph/owner、距离和方向约束）；未训练、未读 sealed test、未改 production。

train-selected `r24_c75` 在 474-image holdout：

- baseline edge P/R：`.95984/.79808`；
- union edge P/R：`.95983/.82004`；
- bridge 仅 77 条，bridge precision `.95946`、recall `.02195`；
- oracle sanitize `1.0`；
- profile recall：cov13 base48 `.7458`、cap52 `.7978`、stroke16 `.7458`；cov33 三组 `.8776/.8776/.8757`。

其它距离/方向配置的 union recall 仅 `.814-.820`，没有达到 precision `.965917`、recall `.850031` 与 profile floor。因此该 bridge 没有足够可训练入口，GPU NO-GO。下一步应针对 double-bond/文字遮挡做专门 deterministic 数据审计，而不是继续扩大 bridge 参数 sweep。
