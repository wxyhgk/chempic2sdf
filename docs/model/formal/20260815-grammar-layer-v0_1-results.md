# Grammar Layer v0.1 全量结果

日期：2026-08-15

最终规则与审计提交：`882bea848abf81c4fa34cadf024118b9eec036ff`

结论：审计执行 `PASS`，可辨识性数学门 `NO-GO`，`training_authorized=false`。

## 执行完整性

- A/B 各枚举 66,712 个 core scene 和 4 个固定反例，共 66,716 个 Scene。
- 两份 manifest 均为 66,716 行；scene ID 重复 0，未分类 0。
- Final A 用时 690.42 秒，Final B 用时 691.97 秒；runtime 不进入 commitment。
- production preprocessing 新增异图碰撞 0。
- 10 项 Python 测试和 Lean `lake build` 12/12 通过。

互斥层计数：

| 层 | Scene 数 |
|---|---:|
| `simple_line_angle_valid` | 19,824 |
| `crossing_allowed` | 16,833 |
| `adversarial_invalid` | 30,059 |

累计域结果：

| 域 | Scene | Observation | 唯一 Observation | 冲突类 | 冲突 Scene |
|---|---:|---:|---:|---:|---:|
| `simple_domain` | 19,824 | 19,823 | 19,822 | 1 | 2 |
| `crossing_domain` | 36,657 | 36,649 | 36,641 | 8 | 16 |
| `all_domain` | 66,716 | 53,206 | 49,688 | 3,518 | 17,028 |

全域每个 observation 最多对应 9 个不同化学图签名。`crossing_allowed` 独占层内部碰撞为
0，但累计 `crossing_domain` 仍有 8 类；这证明不能用 exclusive layer 的数字冒充加入 crossing
后的可辨识性。

## A/B Commitment

以下四个承诺产物逐字节一致：

```text
summary.json
0e361850b5a9746c9b92d35c55e6dd934a538103296cf61dd69e78a9c9dab990

validation.json
f007ea92c2845c3c72372f2a8d21344f318886878953f9927a1b88d837115e0d

collision_certificates.json
f97262008a50edf26b263745b9058533820be8bb72fdffcc3f9c38e6f9839485

scene_manifest.jsonl
f7519924b13b685f235a90dbd837772ced626cdffc9bbbacd4ff423d22c406f7
```

独立 NetworkX verifier 的 A/B 输出也逐字节一致：

```text
independent_review.json
10db28ce9262a7af4a0c1601f58b40fca362c067cc06ade4f679d6cb1ec90b71
```

verifier 检查 3,518 份证书中的 13,670 对不同签名 witness，按 Node `color` 和 Edge
`order` 独立判断图同构；错误判异 0。渲染字段 `display_type` 不参与化学图身份。

首轮 `run_a/run_b` 因 manifest 把域外 Scene 继承成域内 collision class，被实现审批拒绝；
共发现 22,533 个错误域字段。提交 `882bea8` 改为按 Scene 自身层判断域成员，并新增覆盖全部
manifest 行的 `manifest_domain_collision_classes_match_scene_scope` 硬门。上述 SHA 只来自修正后
的 `final_a/final_b`，首轮 SHA 不属于正式结果。

## NO-GO 原因

`simple_domain` 唯一残留冲突的 production observation 为：

```text
fb2ace9b1d2655c08512fc20b859f5e191b4f6b4489a746b7d4b453d5c61d24a
```

它同时对应：

- 一根 order=2 的双键；
- 两个互不连接的单键分量。

两者 raw raster 和 production tensor 完全相同。不存在任何只读该 tensor 的确定性模型可以对
二者都正确；增加图片数量、模型参数或训练轮数不能解除这个碰撞。

`crossing_domain` 的 8 个冲突类同理说明：声明“允许 crossing”并没有自动解决 crossing 与
junction 的替代解释。未激活全局 prior 时必须保留 ambiguity 或拒识。

## 下一数学门

不开启训练。建立 Grammar v0.2 的两个全局、互斥先验候选并先预注册：

1. `terminal_aligned_parallel_strokes_are_one_multiple_bond_v1`：相同 terminal envelope 的
   2/3 条平行 stroke 只保留为一根 double/triple Bond，排除同像平行独立单键解释。
2. `unmarked_transversal_crossing_is_disconnected_v1`：无 marker 的内部 transversal crossing
   只保留不连接解释，并对整个 observation class 排除同像 junction，而不是逐 Scene 打标签。

v0.2 必须证明 prior 非空、coverage-preserving、同一 observation 最多保留一个化学图签名，
并在未知枚举域上重新运行 A/B。它是绘图协议，不是纯视觉定理；文档和模型输出必须保留这个
边界。通过后才进入 Oracle field 左逆，不直接进入训练。
