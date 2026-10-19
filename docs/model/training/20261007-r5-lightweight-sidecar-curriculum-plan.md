# R5 低预算 sidecar / curriculum 计划

目标是找出仍有真实 proposal headroom 的方向，再决定是否消耗一次 GPU
shadow；edge/type scorer、pair scorer、port endpoint scorer 均不重复。

## 1. 已否决：graph-hypothesis ceiling → graph reranker

> 预检结论（已执行）：72 组 graph-hypothesis sweep 的最佳组合为
> `merge=24, tau=.2, cycle=on, threshold=.004`，full holdout
> P/R/sanitize=`.9753/.7319/.9557`；六个 profile 的 recall 仍有
> `.685/.718/.731/.774/.766/.718`。由于 oracle ceiling 本身低于现有
> Hungarian+merge baseline 的 recall，graph reranker 路线 NO-GO，不启动训练。

对每个冻结 proposal 只生成有限的 target-free 图假设：

* merge 半径 `{12,16,20,24}` px；
* edge confidence 只使用 train-only 预先固定的少数阈值；
* cycle demotion 与 topology guard 的固定组合。

CPU gate 先计算 `oracle-best-K`（K≤12）是否相对当前
Hungarian+merge+guard baseline 提供至少 1 个百分点 edge recall 或 sanitize
余量；同时报告每个 profile 的 ceiling、dangling、self-loop 和 graph validity。
若 oracle-best-K 没有余量，graph critic 没有可学习空间，立即 NO-GO（本轮已满足）。

只有 ceiling 通过，才拟合约 1–3k 参数的 graph-level reranker。输入只能是
图内统计（边置信度、节点度、几何残差、环/芳香边计数、价态压力、孤立节点和
dangling 计数），不读 atom id、元素、GT bond 或渲染目标。训练标签仅来自
train identity 的后验图质量，holdout 只做固定候选集 replay。

CPU gate：full holdout 的 precision、recall、oracle sanitize 必须同时严格高于
baseline，且六个 profile 的 recall≥0.75、sanitize≥0.90、dangling≤0.05；任何
一项失败都不启动 GPU。

若通过，才允许一次 isolated ≤800-step GPU shadow；200-step full-holdout 检查
没有改善即早停。sidecar、production checkpoint、selector、OCR、bond/type head
保持隔离。

## 2. 次选：局部尺度不变的 ownership scorer

一个简单的无训练 junction 交点修正（incident segment 的最小二乘无限直线交点）
已做过快速 CPU replay，结果由 P/R/S=`.9747/.7746/.9979` 变为
`.9723/.7588/.9979`，dangling 从 `.384` 升至 `.613`，因此该 anchor 方案已否决。
后续若再尝试，只能使用 candidate-conditioned line-support cost，并先通过
target-free reachability gate。

将 endpoint-candidate 特征改为局部规范化坐标：以 provisional node 为原点，按
局部 median bond length 缩放，并加入候选 score/rank、局部角度/degree/density。
它只改变 Hungarian cost regularizer，禁止 threshold/drop candidate 或改变 frozen
prefix。先做无训练 normalized-distance sweep；只有 full holdout graph recall 不降、
sanitize 与所有 profile 严格改善才可做 ≤500-step CPU fit，之后才评估 GPU。

此前绝对坐标 ownership MLP 虽然 pair AUPRC 很高，但 graph replay recall 明显
下降，因此不能直接上 GPU。

## 3. 数据 curriculum

proposal route 通过 ceiling 后才扩数据。新样本必须与固定 holdout identity/scaffold
隔离，优先 clean raster panel：40% junction/多重键，40% hetero/valence，20%
aromatic bridge；每个 identity 至少 4 个字体/线宽 profile，并保留旧 train replay
（建议新旧约 1:2）。先做 loader、线段可达性和 oracle ceiling gate，再训练 sidecar。
扩数据本身不能作为 GPU 训练理由。

所有 gate 失败均记录 NO-GO，不读取 sealed test，不修改 production initial SHA。
