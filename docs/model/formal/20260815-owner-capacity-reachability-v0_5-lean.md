# Owner Capacity Reachability v0.5 Lean 算术内核

日期：2026-08-15

## 目的

容量澄清协议要求每个 unreachable axis 提供可机检证书。本阶段先形式化与 Python 实现无关的有限算术核心，禁止用
“没构造出 fixture”冒充不可达证明。

## 已证明

Lean 模块 `OwnerCapacityReachabilityV05` 证明：

- fixed-R0 的 owner option 数为 2，且 `constraints = 1 + 3 * owners = 7`；
- 两个 owner 只有 `[L]`、`[R]`、`[L,R]` 三种 canonical 非空 domain class；
- 若无向 8-neighbor 图满足握手上界 `2E <= 8V` 且 `V <= 4096`，则
  `E <= 16384 < 32769`；
- 若二元 domain 满足 `raw = 2^A`，则 `A <= 20` 推出 `raw <= 2^20`，而
  `raw > 2^20` 必然推出 `A > 20`；
- 若 straight-carrier 的 fragment count 满足 `fragments <= carriers^2` 且 `carriers <= 8`，则
  `fragments <= 64`。

这组定理解释了为什么 `domain_classes`、`adjacency_edges`、`constraints` 和 `raw_assignments` 的高边界在固定 R0
下不能独立命中，也给出 atomic-fragment 不可达证明的算术末端。

## 严格边界

Lean 模块没有定义或证明：

- Python constructor 的 gate 顺序和 measurement point；
- Python 8-neighbor edge 构造确实满足握手假设；
- straight segment compiler 确实满足 `fragments <= carriers^2`；
- canonical JSON 的字段级最大字节数；
- 任一 reachable axis 的真实 closed `L/H` source。

这些绑定必须由冻结的 Python proof checker、source witness 和独立 verifier 完成。特别是
`max_ambiguous_variables` 的 `H=21` 同时使理论 raw count 超过 `2^20`，不满足单轴 isolated fixture，因此严格分类仍为
逐轴 NO-GO，不能因为 ambiguous gate 在 raw gate 之前就记作 isolated GO。

## 验证

```text
Lean 4.33.0 / Lake 5.0.0
目标模块：2/2 jobs passed
全工程：  33/33 jobs passed
公开定理：7
sorry / custom axiom / unsafe / native_decide：0
```

模块 SHA256：

```text
4001dcedb98d51eb5a04be9714d6adf9f3178e35701dbe111027c99263fec092
```

本阶段不改变 `training_authorized=false`。
