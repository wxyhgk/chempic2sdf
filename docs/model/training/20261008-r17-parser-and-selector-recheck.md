# r17 admission/parser and selector recheck

日期：2026-10-08

## 完成项

- 新增 `tools/r17_manifest_rows.py`，统一展开当前的 `rows[]` 和历史的
  `split -> molecules[] -> rows[]` manifest；同时支持只有分子身份字段的
  `molecules.json` catalog。
- `audit_r17_new_source_prior_use_ledger.py`、两个 r17 source builder 都复用
  该展开逻辑。
- 对 v6、v7、targeted t1 重新生成独立 admission receipts（parser-r2）。

三批的 `molecule_id`、`canonical_smiles`、`connection_identity`、
`legacy_scaffold` 历史重合均为 0；ledger 覆盖、必需 provenance、样本/图像
唯一性、train/holdout 分子隔离均通过。报告仍保持 `training_admissible=false`，
因为 admission 只是身份闸门，不是训练授权。

## selector risk-curve 修复

`audit_visible_on_selector_risk_curve.py` 现在同时记录每个 policy 的 aggregate
和按 split 的 counters/rates，报告版本为 0.2。新报告：

`outputs/audits/chemvector-visible-on-selector-risk-curve-20261017-r2/report.json`

1000 张 public train-only 图、CUDA replay、冻结权重未改变。aggregate 结果仍显示
固定阈值只能交换 precision/coverage：baseline offset 0 的 strict coverage 为
0.9046、reachable-16 为 0.9894；top-k=96 虽 reachable-16 为 0.9974，
candidate precision 仅 0.5424。该报告不授权 selector 训练或 GPU。

## 数据决定

v7 的 holdout O-aromatic 只有 1 个分子、train N-aromatic 只有 20 个 incidence
但仅 2 个分子；targeted t1 的 holdout 配额通过，但 train N-aromatic 仅 1 个分子。
因此 t1 不能作为下一轮 calibration source。下一轮应重新生成带 train-side
分子配额的 pair source，并再次执行 admission → strata → CPU pair/calibration →
graph integration → target perturb；selector sidecar 和 GPU 保持冻结。
