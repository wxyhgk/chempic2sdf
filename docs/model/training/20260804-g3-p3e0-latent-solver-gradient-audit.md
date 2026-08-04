# G3 P3-E0 Latent Solver 与梯度审计

> 日期：2026-08-04
>
> 状态：latent-positive solver GO / pilot协议需先补best-invalid margin

## 精确 latent positive

P3-E0实现了不固定query、candidate或GT排列的精确正集合solver。对每个GT的兼容candidate组，
只枚举同时兼容多个GT的candidate归属；归属固定后用query-GT Hungarian求全局最高分并回溯candidate。
小规模穷举、共享candidate、固定基数唯一性以及query/candidate/GT排列不变性测试全部通过。

全量4,937图兼容统计：

| 指标 | 结果 |
|---|---:|
| candidates | 631,936 |
| 同时兼容多个GT | 73 |
| 受影响图片 | 45/4,937 |
| 单图最多歧义candidate | 5 |
| 最大归属组合 | 32 |
| 每GT兼容candidate p50/p95/max | 1 / 3 / 8 |
| solver p95 | 约1.6ms/图 |

两次solver摘要完全一致，工程预检GO。

## 等价身份稳定性

固定capacity64加前192张eligible general，比较P2-W/P3-B/P3-D的BF16、FP32、batch4/32和
`1e-4`确定扰动。P2-W legacy score的代表结果：

| 对照 | latent query exact | latent candidate-set exact |
|---|---:|---:|
| BF16 vs FP32 | 190/256 | **249/256** |
| `1e-4`扰动 | 214/256 | **254/256** |
| batch4 vs batch32 | 256/256 | 256/256 |

query身份显著变化时，最终candidate集合几乎不变，直接证明固定query边Hamming会惩罚结构等价解。
P3-D的BF16/FP32 latent query exact更只有36/256，而candidate仍有231/256。

fixed-K理论分数`existence_logit + logsoftmax(pointer)`与现legacy `logsigmoid + logsoftmax`差异很大：
P2-W两公式latent query/candidate exact为110/256、235/256；P3-D为0/256、151/256。因此分数公式
不能在P3-E中静默切换，必须单独受控消融，首个pilot继续保持train/decode一致的legacy公式。

## 梯度与裁剪

对同一256张图、三个checkpoint分别审计object、pointer、5x coordinate、boundary、Count、
structured梯度；全部使用`autograd.grad`，权重摘要前后相同，训练更新为0。

关键证据：

- P2-W capacity已经拟合，capacity object/structured平均梯度norm仅约0.14/0.20；general则约
  29.5/716。正确capacity replay几乎没有反向力，普通遗忘不能被“样本仍在训练批次”排除。
- P2-W capacity与general Count平均梯度cosine为`-0.39`；Count必须与结构阶段完全拆开。
- P2-W口径下P3-B目标pre-clip median约109，89% batch触发clip；P3-D structured+Count median
  约1,917，78%触发。P3-B/P3-D final上历史目标审计均为100% batch触发clip。
- boundary梯度median约403，远高于object 47、pointer 7.8；不能直接把boundary weight设为1。
- 单batch Count与各结构loss cosine均接近0而非稳定强负；主要问题是尺度、跨域梯度和正确样本
  零梯度，不支持把全部回退简单归因Count冲突。

## 决策

- `latent_solver_preflight_go`。
- 原P3-E1暂缓：简单valid gap在capacity已正确时为0，无法提供防遗忘margin。
- 下一步先实现等价类不变的best-invalid oracle：loss按GT覆盖有效/无效区分，不能按query边Hamming。
  capacity样本也必须满足`best valid score >= best invalid score + margin`，才有保护梯度。
- P3-E结构阶段删除Count CE、coordinate L1和旧factorized BCE/CE；Count使用完全冻结P2-W路径报告，
  另立独立Count实验。
- best-invalid solver先做小规模穷举、Hall witness规模、重复摘要与延迟审计；未通过不得进入192图pilot。

```text
solver audit = outputs/audits/chemvector-g3-p3e0-latent-solver-audit/metrics.json
gradient audit = outputs/audits/chemvector-g3-p3e0-gradient-conflicts/metrics.json
training updates = 0
decision = solver GO / training protocol revision required
```
