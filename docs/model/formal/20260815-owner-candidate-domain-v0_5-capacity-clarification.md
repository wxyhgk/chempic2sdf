# Owner Candidate Domain v0.5 容量澄清协议

日期：2026-08-15

状态：预注册，尚未执行确认实验。

## 原问题

R0 修订协议 `5a8bdd9` 同时规定：

```text
R0 intersection count = 2
R0 owner option count = 2
MAX_INTERSECTIONS = 8
MAX_OWNER_OPTIONS = 16
每个容量轴 N 必须由完整 constructor 成功
```

这些要求不可同时满足。Lean 定理
`fixed_two_intersection_r0_cannot_reach_original_declared_ceilings` 已形式化证明固定 `2/2` profile 不可能命中
`8/16` 的 exact-success 边界。因此原 full-capacity 条款保持 NO-GO，不能由 R0 core 的 `873/873` GO 覆盖。

## 两类合同

从本协议起，容量被拆成两个互不替代的合同。

### A. Frozen-R0 语义合同

只验证冻结 observation 的真实派生数量与语义：

```text
carriers = 3
intersections = 2
atomic fragments = 5
pixel variables = 58
owner options = 2
domain classes = 3
adjacency edges = 188
constraints = 7
ambiguous variables = 10
raw assignments = 1024
```

这些数量必须从 source 派生并经过相应 safety gate，但不得为了命中 generic ceiling 添加 dummy、padding、重复
owner、fragment owner 或 residual owner。

### B. Generic reject-only safety 合同

对每个容量轴 `limit=N`，只授权下列含义：

```text
derived_count <= N  => capacity gate 本身不拒绝
derived_count >  N  => 整份 closed reject CAPACITY_EXCEEDED
```

第一行不表示 constructor 必须语义成功；输入仍可因 source closure、R0 profile、incidence、seed、connectivity 或其他
语义条件拒绝。第二行不得通过 truncate、top-k、先到先得、整数溢出或 partial artifact 绕过。

因此直接调用 `capacity_gate(N)` 和 `capacity_gate(N+1)` 只能证明 helper 的局部边界，不能证明完整 constructor
的 end-to-end 容量。

## 下一确认实验

实验必须固定当前 R0 core commit `bb72eff`，不修改 candidate semantics。每个 axis 生成一份来源可审计的 mutation
manifest，并按以下三类报告，不得把类别混在一个通过率里：

在运行任何确认样本前，必须先提交冻结的 `axis_plan`。其 axis key 集合必须与冻结 `LIMITS` **严格相等**，无缺失、
无额外 key、无重复。每个 axis 必须记录：expected category、source-builder SHA256、独立 count 公式、measurement
point（raw、canonicalized、deduplicated 或 compiled）和 limit。边界字段使用 closed tagged union：

```text
reachable:   {L <= N, H > N, L_witness, H_witness}
unreachable: {L_optional, H: null, unreachability_certificate}
```

禁止为 unreachable axis 直接伪造内部 count 填补 `H`。运行后禁止更换类别、公式或 measurement point。

不能假设离散派生轴恰好可达 `N` 或 `N+1`：raw assignment 是 domain size 的乘积，constraint 和 adjacency count
也可能受语法同余或图结构约束。边界确认一律使用 `L/H`；只有证明 `L=N,H=N+1` 时才能简写为 `N/N+1`。

1. **Reachable end-to-end boundary**：closed source 能真实派生 `L/H` 时，完整 constructor 必须在 `L` 处通过该
   容量门并返回完整 closed artifact；该 artifact 还必须通过独立 schema、section hash、stage trace 和冻结语义
   verifier，且无 reject 或 partial side effect。在 `H` 处精确 closed reject `CAPACITY_EXCEEDED:<axis>`；顺序
   反转后结论不变。
2. **Reachable rejection-only boundary**：能真实派生 `H`，但 `L` 因其他冻结语义条件拒绝时，只允许确认 `H`
   无截断且命中该 axis 容量拒绝；不得宣称 exact-success。
3. **Unreachable under R0**：与固定 R0 不变量冲突的轴明确记为 unreachable/NO-GO，不用 helper 或直接构造内部
   count 冒充完整输入。Unreachable 只能由 Lean theorem、完备有限穷举，或冻结的 sound proof checker 接受显式
   不变量证书；第二实现只能复核证书，两个 builder/verifier 都找不到 fixture 不能证明不可达。“builder 构造失败”
   不是不可达证据。

至少包含这些红队 mutation：

- 对每个 reachable axis 建立**只有该轴超限**的 isolated fixture，其他所有轴均在界内；多轴超限 fixture 另行冻结
  failure priority，只验证确定性，不能代替任何单轴确认；
- 把唯一有效且在 canonical identity 下互异的项目放在超界 fixture 最后一位，确认没有先取前缀；manifest 同时记录
  raw count、canonical distinct count、deduplicated count 和最终 derived count；
- 正序、逆序和固定随机置换得到同一 reject 分类；
- `bool`、float、负数和超大整数不能作为合法 count 绕过严格类型；
- `max_raw_assignments` 使用 exact integer 连乘，不使用浮点或对数近似；
- `max_output_bytes` 对 canonical serialized bytes 计数，不对 Python object 大小或压缩后大小计数；
- 每个超界 fixture 必须得到精确 code `CAPACITY_EXCEEDED:<axis>`。source closure、semantic error 或其他 axis 的
  capacity error 都不算该轴通过；reject record 绑定 constructor path/SHA 和命中的 gate axis；
- Constructor 在禁网、默认 write-deny 的隔离 sandbox 中运行，只开放一个全新隔离输出目录；默认禁止子进程，若工具链
  必须启用则冻结 allowlist 并记录完整 process census。执行前后快照全部允许写命名空间（含隐藏文件），同时审计
  syscall/process 记录，防止写 `/tmp`、HOME、仓库兄弟目录或留下后台进程。Constructor 不得返回 partial dict、打印
  candidate section、创建临时/半截文件或覆盖冻结的 observation、candidate、verification、replay；副作用核验完成后，
  外部 harness 才能原子写入具名 reject record。成功发布也必须使用同目录临时文件、`fsync`、atomic rename，且清理
  临时文件。

## 验证器隔离

确认器不得只搜索源码字符串 `capacity_gate(...)`。它需要：

- 验证 `axis_plan` key 集合与 `LIMITS` byte-exact 对应，并在执行前校验 plan commit/SHA；
- 按冻结公式和 measurement point 从 mutation source 独立重算 raw、canonical distinct、deduplicated 和 derived count；
- 记录 source SHA、builder SHA、expected axis/`L/H`、actual failure code、gate trace 和全部副作用 snapshot；
- 检查 constructor 没有 truncate/top-k/partial serialization；
- 对所有可达边界执行至少两次 byte-exact replay；
- 将 reachable、rejection-only、unreachable 三类结果分别序列化并计数。

## GO/NO-GO

Generic reject-only safety 只有在某个 reachable axis 的 isolated `H` mutation 精确命中该 axis 容量拒绝、无截断、
无 partial output、无额外副作用且重放一致时，才可对该**逐轴**结果判 GO。每个 axis 必须单独给 verdict，helper
PASS 永不进入 GO 分母；任何 unreachable/unproven axis 都保持该轴 NO-GO，并留给未来扩大表达域后的新 profile
单独预注册。

聚合结论不得真空成立：`confirmed axis set` 必须非空且完整列出。只有 `LIMITS` 的全部 axis 都得到各自要求的
end-to-end/rejection-only 确认、且没有 unreachable axis 时，才允许写唯一聚合状态
`ALL_AXIS_REJECT_ONLY_SAFETY_GO`。该状态不改变原 exact-success full-capacity 合同的 NO-GO。只确认部分轴时聚合
状态必须写 `PARTIAL/NO_GO_GENERIC`，不能缩写成 generic capacity GO。

即使该安全合同 GO，也只表示资源边界可靠，不表示真实 PNG 可辨识、candidate 完备、UniqueGraph/SDF 正确或模型可训练。
训练继续保持 `training_authorized=false`。
