# Carrier Candidate Family v0.4 协议草案

v0.4 用符号候选族替代“最多 128 个 interpretation 的显式完整列表”。当前只是预注册，不包含真实图片、
proposer、SDF、blind、sealed 或训练代码。

```text
status=draft_for_root_constructor_review
schema_frozen=false
root_constructor_frozen=false
symbolic_verifier_frozen=false
proposer_public_execution_authorized=false
unique_graph_authorized=false
training_authorized=false
```

## 1. 为什么必须升级

v0.3 已证明 Boundary matching 可以独立枚举，但 fragment、intersection、ownership、port、lane 和 Bond 的完整域
不在 root 中。同 root 可以生成不同的结构合法 singleton，因此 v0.3 interpretation completeness 为 NO-GO。

即使把缺失域加入 root，显式列表仍有容量问题。8 个独立二元歧义产生：

```text
2^8 = 256
```

超过 v0.3 的 128 项上限。v0.4 不限制满足赋值的数量，只限制 root object、变量、约束、证明和字节容量。

## 2. 权威语义

设 root canonical objects 为 `R`，独立 verifier 从 `R` 确定性构造有限域 CSP `F_R`：

```text
S(R) = { CanonicalCompile(R, a) | Total(a) and Sat(F_R, a) }
```

必须证明双向精确性：

```text
Sound:
  Total(a) and Sat(F_R,a)
  -> Feasible(R, CanonicalCompile(R,a))

Complete:
  Feasible(R,I)
  -> exists a,
       Total(a) and Sat(F_R,a)
       and CanonicalCompile(R,a) = Canonicalize(I)
```

只证明其中一个方向不能授权候选完整性。

## 3. Producer 与 Verifier 边界

producer 第一版只允许提交匿名 PNG 和冻结 observation/root extractor 产物。producer 不得自由提交：

- semantic variable；
- domain value；
- hard constraint、unit clause 或 CNF；
- auxiliary variable；
- BDD/d-DNNF；
- model count；
- unique graph 声明。

独立 verifier 必须从 root 生成上述权威字节。未来若为缓存允许 producer 携带 formula，其 canonical bytes 必须
与 verifier 重建结果逐字节相等。

## 4. Root 构造层

v0.4 需要冻结以下 root object 或确定性 constructor：

1. Boundary 与 compatible pair；
2. 全部 Boundary matching；
3. base stroke 与 canonical fragment split；
4. crossing/junction/touch intersection option；
5. Atom instance candidate；
6. endpoint attachment edge；
7. canonical port endpoint partition；
8. typed ownership option；
9. fragment successor、gap 和 lane flow/rank object；
10. 1..3 lane Bond grouping object。

每层必须从更低层独立重建完整数组；不能由 proposer 自报 count/hash 代替。任一层构造超过冻结 object/byte
容量，只能 `Reject(outside_capacity)`。

## 5. 变量和约束

首版候选变量至少覆盖：

- matching selection；
- fragment realization 和 intersection mode；
- Atom activation/kind；
- fragment ownership；
- endpoint attachment 与 port equivalence；
- fragment successor、lane membership 和方向；
- lane-to-Bond grouping。

变量 ID、domain 顺序、约束 AST、整数编码和 auxiliary variable 生成必须 canonical。

硬门：

- assignment 必须 total，禁止默认值和部分赋值；
- 多值二进制编码必须排除 invalid code；
- Tseitin auxiliary 必须由 semantic variables 函数唯一决定，且不进入 semantic interpretation count；
- lane/Bond slot 使用 lexicographic first-use 等 symmetry breaking，消除 slot/lane 顺序 gauge；
- 存在跨分量 constraint 时，不得继续用独立分量乘积计数；
- producer 删除 domain value或添加任何额外 hard clause均为 `INVALID`。

## 6. 证书

- 非空：一个 total satisfying assignment，独立求值；
- Ambiguous：两个 satisfying assignment，且 semantic graph projection 不同；
- UNSAT：LRAT 等小型独立 checker 可验证的证明；
- 可选显式列表完整性：逐项满足 `F_R`，并验证 `F_R AND not-in-list` 为 UNSAT；
- 可选 ROBDD/SDD/d-DNNF：只能作为派生缓存，必须证明与 `F_R` 双向等价。

固定变量序下 ROBDD 可 canonical，但最坏仍指数增长。只检查 BDD 的 ordered/reduced 结构不能证明它没有漏解。

## 7. UniqueGraph 仍不授权

首版输出只有：

```text
CandidateFamily
Ambiguous(two checked witnesses)
Reject(reason)
```

`UniqueGraph`/SDF 需要另一个版本，至少验证：

```text
exists witness w, Sat(F_R,w)
and UNSAT(F_R(y) and SemanticGraph(y) != SemanticGraph(w))
```

其中 semantic graph 必须覆盖 Atom identity、Bond order、电荷和立体信息，不能只比较空间骨架 hash。

## 8. 强制 fixture

1. 8 个位置可区分的独立二元歧义，解数精确为 256；
2. 删除任一 domain value必须 `INVALID`；
3. 添加任一 unit clause必须 `INVALID`；
4. root ID、无向端点和 domain 重命名后 canonical 语义不变；
5. auxiliary variable 不得增加 semantic interpretation 数；
6. lane 方向、lane 顺序和 Bond slot permutation 不得制造伪 ambiguity；
7. 跨 component clause 必须破坏错误的乘积计数；
8. solver timeout、unknown、proof checker failure 只能 Reject；
9. glyph/Bond、crossing/junction、double/parallel-singles 保留反事实；
10. root-vs-PNG truth coverage 继续作为独立实验门。

## 9. 当前下一步

先做 root constructor 和约束 AST 的有限微场景设计及容量 census，再写 closed JSON schema。当前 table-CSP 探针
只证明符号表示容量优势，不是最终协议实现。
