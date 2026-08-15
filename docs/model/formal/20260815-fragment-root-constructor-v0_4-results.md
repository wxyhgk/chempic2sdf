# Fragment Root Constructor v0.4 微场景结果

日期：2026-08-15

## 裁决

```text
exact geometry research probe:       GO
root-relative fragment cell exact:   GO
PNG -> base stroke completeness:      NO-GO
fragment support/payload completeness:NO-GO
v0.4 production schema freeze:       NO-GO
public proposer / UniqueGraph / SDF:  NO-GO
training:                            NO-GO
```

本周期证明的不是“模型会识别分子”，而是一个更窄的确定性层：给定 canonical base stroke 集合，所有 exact
几何交点可以统一原子化；给定每条 carrier 的 canonical ordered cut，全部相邻 fragment cell 可以双向精确重建。

## 实现结果

Python 微探针只使用整数 cross product 和 `Fraction`：

- X：1 intersection、2 split site、4 fragment；
- T：1 intersection、1 split site、3 fragment，不生成零长 child；
- Double-X：2 intersection、4 split site、7 fragment；
- Three-way：同一 exact point 合并为一个 region，每个 parent 只切一次，共 6 fragment；
- 非整数 Q4 交点保存为约分有理数，不先舍入；
- 共线正长度 overlap closed reject；共线单点 endpoint touch 保留；
- 删除 fragment 并刷新自签 count 仍报 `FRAGMENT_UNIVERSE_INCOMPLETE`；
- 正确集合但顺序/ID 错误报 `FRAGMENT_UNIVERSE_NONCANONICAL`；
- duplicate body 和 duplicate ID 分别独立拒绝；
- base stroke、intersection 和 fragment 超容量时不输出部分 root。

## 修复的互审问题

互审实际发现并修复三类问题：

1. T-touch 位于 parent 端点时，初版 fragment endpoint 没有保留 intersection ID；
2. 共线单点 endpoint touch 被 parallel 分支跳过；
3. Python `False == 0`、`4.0 == 4` 导致 declared root 的 bool/float 伪装被误判为 exact。

最终比较器先递归检查 closed JSON shape 和原生 scalar type，再比较完整 canonical JSON bytes。8 类 bool/float
变异全部返回 `INVALID`。

## 验证

| 检查 | 结果 |
|---|---:|
| fragment root 专项测试 | 22/22 |
| `formal/chemgraph/audit` 全量测试 | 205/205 |
| Lean 构建 | 29/29 jobs |
| Oracle 随机 8-line 检查 | 998 supported 全部满足计数与非零区间；2 overlap closed reject |

Lean 证明：

- `n` 个 ordered cut 恰好产生 `n+1` 个 cell；
- 每个输出 cell 都来自 delimiter 列表中的相邻 pair；
- 每个合法相邻 pair 都出现在输出，因而是 root-relative exact；
- 每个 carrier 的 fragment fiber 非空；
- 零 cut 时唯一 cell 是 `start -> finish`。

Lean 文件 SHA256：

```text
2c871b6e1be2bbfeb03f49b5e04172a82c9f0381399205e3a6f85395778baf39
```

## 新发现的 v0.3 合同冲突

v0.3 schema/protocol 声称每个 interpretation 最多 512 个 fragment，但 `ownership_component_count` 上限为 128，
Validator 又按 fragment ID 集合重算该 count，因此实际有效 fragment 上限只有 128。v0.4 不能继承这两套互相冲突
的容量；root object、symbolic variable 和输出 byte capacity 必须分别 census 后冻结。

## 下一门槛

当前 atomic fragment 只有 exact geometry，没有像素 support。下一周期必须从 matching 派生 base stroke，并实现
parent support 到 child fragment 的确定性半开区间分配，证明：

```text
child supports are pairwise disjoint outside named intersection support
union(child supports, named intersection supports) = parent support
```

若两个不同 exact cut 量化到同一 Q4 点、任一 child 无 support 或 support 无法唯一分配，只能 closed reject。
这一层通过前仍不进入 OCR、键级或模型训练。
