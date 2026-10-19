# r9 direct AtomPair tiny CPU scorer

在 direct graph proposal ceiling 通过后，运行了一个受限的 CPU sidecar 试训。
sidecar 只接收 frozen DirectAtomPair 的 256 维 hidden、基础 presence logit
和 6 个图像几何标量；Atom 候选、pair 枚举、vision、AtomObject、DirectAtomPair、
OCR 和 production graph 全部冻结。该 sidecar 不是 production checkpoint。

脚本：`tools/train_direct_graph_pair_scorer_cpu_r9.py`

报告：
`outputs/audits/chemvector-identity-aligned-direct-graph-pair-scorer-cpu-r9/report.json`

## 固定训练协议

- train-only：96 张图（四个 r6 profile 各 24 张）；
- fresh holdout：95 张图，四 profile 全量 replay；
- train/holdout molecule identity disjoint；
- `DirectPairResidual(262 -> 32 -> 1)`，8,973 参数；
- AdamW，500 CPU steps，单线程 deterministic；
- presence threshold 在 train-only 上选定为 `0.95`，没有查看 holdout；
- 未使用 GPU，未读 sealed test，未写 production checkpoint。

## Holdout 对比

| 指标 | frozen baseline | CPU scorer |
|---|---:|---:|
| edge precision | 0.994845 | 0.992424 |
| edge recall | 0.934625 | **0.951574** |
| dangling rate | 0.003851 | 0.005025 |
| type accuracy | 0.963731 | 0.960560 |

四 profile scorer edge recall：`0.9904 / 0.9378 / 0.9598 / 0.9187`；最低
profile precision `0.9745`、dangling `0.0151`、type accuracy `0.9215`，均过
当前 proposal gate。相较 frozen baseline，recall 提升约 1.70 个百分点，precision
下降约 0.24 个百分点，dangling 仍远低于 0.05。

随后在未参与 sidecar 训练的 dev 96 图上复用该 train-only threshold（没有重调）：
scorer edge P/R=`1.0000/0.9651`，dangling=`0.0154`，type accuracy=`0.9735`；
baseline 为 `0.9988/0.9570`、`0.0072`、`0.9769`。这支持 sidecar 的 recall
提升并非只在训练图上成立，但 dangling 有小幅上升。

## 决策

该 tiny CPU scorer 在完整 fresh holdout 上通过 graph gate，且相对 baseline 有
严格 recall 改善，因此 direct AtomPair graph 路线具备申请一次受限 GPU shadow 的
证据。但 sidecar 仍不能直接部署；下一步若申请 GPU，只能在独立 shadow 输出、
800 steps/30 分钟上限、每 200 steps fixed holdout 检查、无改善立即早停，且不改
production checkpoint。
