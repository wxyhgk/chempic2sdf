# 007：v0.7.2 因子化 Edge/Bond head

> 状态：NO-GO，未发布  
> 输出：`outputs/checkpoints/chemvector-v072-visible-relation-5k`

## 假设

旧 stroke scorer 用一个五分类同时预测 `NONE/SINGLE/DOUBLE/TRIPLE/AROMATIC`。
候选方案拆成 Edge head 和 Bond head：前者判断是否存在可见线条，后者只在真实边
上区分 `SINGLE/DOUBLE/TRIPLE`。

Render JSON 中的 `AROMATIC` 是化学属性，不一定能从局部线条辨识。RDKit 会把
芳香环绘制为交替单线和双线，所以芳香边只参与 Edge loss，芳香性由顶点、环结构
和修复器全局恢复。

## 训练

```bash
.venv-train/bin/python -m chemvector.pretrain_relation \
  --init-checkpoint outputs/checkpoints/chemvector-v5-vertex-10k/last.pt \
  --output-dir outputs/checkpoints/chemvector-v072-visible-relation-5k \
  --train-samples 5000 --batch-size 4 --learning-rate 1e-4 \
  --negative-ratio 4 --anchor-jitter-pixels 12 \
  --stroke-band 9 --stroke-band-radius 12 \
  --sample-shuffle-buffer 4096 --train-pair-encoder
```

训练冻结视觉 stem，解冻 pair encoder。

## 局部验证

- Edge precision/recall/F1：85.90% / 99.77% / 92.32%。
- 可观察键型准确率：96.23%。
- SINGLE/DOUBLE/TRIPLE recall：97.25% / 92.75% / 90.63%。
- 3,144 条芳香边只参与 Edge loss。

## 端到端门控

同一批 500 张 CPU 盲测中，完整拓扑仍为 20.6%，化学完全一致仍为 13.2%，
合法 SDF 从 43.0% 降到 41.2%；价态失败 236，Kekule 失败 46。

```text
outputs/evaluations/chemvector-v072-visible-relation-cpu-validation-500/summary.json
```

## 决策

局部分类指标提升没有转化为端到端收益。该 head 只保留为全图求解器的概率输入，
不更新 `VERSION`，不打 tag。下一阶段需要在全图范围联合选择边和键级，并显式满足
价态、连通性和芳香环约束。
