# G3 P3-E8 Predicted Count Router 审计

> 日期：2026-08-04
>
> 状态：Count 主瓶颈确认
>
> 训练更新：0

## 目标

将 P3-E7 固定策略实现为可复用 shadow 组件，并将同一个路由从 known-count 切换到部署可用的
P2-W predicted Count。base/expert 强制使用同一个 base Count，禁止通过不同 Count 偷换路由口径。

冻结策略仍为：

```text
base_joint_score_mean <= -1.8862037658691406 -> expert
```

route-select 不再读取，阈值不重新选择。

## 结果

| 集合 | Count exact / MAE | Known-count Router | Predicted-count Router |
|---|---:|---:|---:|
| route-confirm 256 | 9/256 / 8.61 | 111/256 | **4/256** |
| historical route100 | 3/100 / 7.66 | 36/100 | **0/100** |

confirm 的 base/expert predicted-count oracle 也只有 `8/256`；route100 只有 `1/100`。这说明不是
路由阈值在 predicted 模式下失效，而是错误 Count 已在解码前决定输出错误的 Atom 数量。

## 决策

`predicted_count_is_primary_router_bottleneck`。Atom selector 的 known-count 能力与安全路由保留，
后续停止调整 matching/margin。下一实验重做独立 Count，输入必须包含 candidate score 分布、
query existence 与 pointer utility 的集合统计；禁止继续使用 `memory.mean -> linear 129-class`。

组件：`chemvector/components/atoms/objects/g3_base_expert_router.py`

审计：`outputs/audits/chemvector-g3-p3e8-predicted-count-router-audit/metrics.json`
