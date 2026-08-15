# Owner Candidate Domain v0.5 R0 修订协议

日期：2026-08-15

## 修订原因

原预注册 commit `030376f` 冻结了集合值逆映射方向，但没有完全定义 legal-world predicate、fragment promotion、
residual denotation、canonical baseline 和容量轴。直接实现会允许循环定义的 100% sound/complete，或用 dummy owner
和无条件 residual 得到形式漂亮但没有信息的结果。

本修订在 constructor 和 verifier 实现前提交，只授权一个最小 `R0 known-owner profile`。完整 fragment/residual profile
继续 NO-GO。

## 冻结来源

```text
PNG ownership confirmation commit: 87e1e82
confirmation protocol commit:       61b7779
representative PNG SHA256:
0152efc7791f03515ef554739b545088b36d654aa081da9718d0c31eef3ecb4e
representative witness SHA256:
e00421918b62adc1b76c42ca1a9cb1b5c9981d3ee33d7e6f1f96dcebf63c268d
```

Constructor 不得读取 representative witness 中的 `partition_annotations`、已枚举 count 或任何 oracle owner label。
R0 source 必须是从冻结 renderer/geometry/support 独立重建的 annotation-free closed observation：

```text
schema_version
training_authorized=false
png_sha256
renderer_sha256
geometry_root
parent_supports
```

其中 geometry/support 必须重跑现有 fragment root 与 per-parent support partition；独立 verifier 还要从冻结 renderer
重建三条 carrier support，不能信任 source 自报 RLE。

## Evidence 与 Owner 分离

R0 必须输出完整多值 evidence membership：

```text
pixel -> carrier IDs
pixel, carrier -> exactly one atomic fragment ID
```

同一 pixel 可以同时属于多个 carrier/fragment。fragment membership 只是 provenance，不升级为 concrete semantic owner。

R0 concrete `KnownOwner` 只有：

```text
I_left  = intersection(H,V1)
I_right = intersection(H,V2)
```

Fragment owner option：禁止。Residual concrete owner option：禁止。若出现没有已知 intersection option 的 overlap，
整份 closed reject `UNOWNED_OVERLAP`; 不得加入 residual 后继续宣称 exact GO。

## LegalR0

给定 annotation-free observation `y`：

```text
U(y) = 全部 carrier fiber 大小 >= 2 的 pixel
KnownOwner(y) = {I_left, I_right}
assignment a : U(y) -> KnownOwner(y)
```

`LegalR0(y,a)` 当且仅当：

1. `a` 对 U 是 total exact-one assignment；
2. assigned owner 的 incident carrier IDs 是该 pixel carrier fiber 的 subset；
3. 每个 owner fiber 非空，并包含自己的 exact anchor seed；
4. 每个 owner fiber 在完整 8-neighbor graph 中 connected；
5. 两个 owner fiber disjoint，union exact cover U；
6. Owner ID 来自共享 canonical geometry intersection namespace，不是 world-relative ID。

这只是冻结有限语法，不是真实化学语义。不得用 constructor 输出反向定义 `LegalR0`。

## Candidate CSP

对每个 `p in U`：

```text
D(p) = {I | incident_carriers(I) subseteq carrier_fiber(p)}
```

R0 artifact 的语义解族：

```text
Pointwise(y) = product_p D(p)
Expand(C(y)) = {a in Pointwise(y) | seed + connected + exact-cover constraints}
```

正确性门冻结为：

```text
Sound:    Expand(C(y)) subseteq LegalR0(y)
Complete: LegalR0(y) subseteq Expand(C(y))
Exact:    Expand(C(y)) = LegalR0(y)
Nodup:    canonical expansion 无重复 assignment
```

逐 pixel owner recall 不能代替 whole-world Exact。raw Cartesian count 与 constrained SAT count 必须分开报告。

## 冻结 R0 Truth Table

独立 oracle 必须在 constructor 之前只由上述 `LegalR0` 和 annotation-free source 重建：

```text
pixel variables:                  58
left singleton domains:           24
right singleton domains:          24
multi-intersection domains:        10
raw Cartesian assignments:      1024
LegalR0 assignments:             873
candidate CSP assignments:       873
sound / complete / Nodup:       100%
stored confirmation annotations: 2/2 covered
false singleton count:             0
```

每个 multi-intersection pixel 必须在 LegalR0 解集中分别实现 left 和 right；否则它是 false multi-option，不得用 domain
cardinality 冒充 realized ambiguity。

## Canonical Universal Baseline

R0 baseline 固定为：对 U 中每个 pixel 放入全部且仅有的两个 KnownOwner。禁止 dummy、padding、fragment、residual。

```text
universal memberships:       58 * 2 = 116
candidate memberships:       48 * 1 + 10 * 2 = 68
eliminated memberships:      48
eliminated ratio:            48/116 = 12/29
mean domain size:            68/58 = 34/29
max domain size:             2
singleton ratio:             48/58 = 24/29
multi-intersection ratio:    10/58 = 5/29
residual-only ratio:         0
residual-containing ratio:   0
```

还必须报告：eligible option recall、ineligible option elimination recall、false-singleton rate、每个 domain-size bucket，
不能只报全局均值。

## Oracle 隔离与等变

1. Constructor 只能读 annotation-free observation；读取 GT、stored annotations、已知 873 count、文件名旁路均失败；
2. 同一 observation 配不同 oracle annotation 时，constructor bytes 必须完全相同；
3. 独立 verifier 不 import constructor/domain helper，从 source 重建 artifact 和 `LegalR0` truth set；
4. carrier/intersection 重命名和输入列表 shuffle 后，完整 solution set 必须 up-to-bijection 等变；
5. bridge domain 不得因 ID、seed 顺序或序列化顺序变 singleton；
6. 任何 single-winner constructor 在两份 realized annotation 上失败。

## Closed Output

`candidate_domain` 顶层 fixed keys：

```text
schema_version, status, training_authorized,
source_binding, limits, counts,
owner_options, evidence_memberships,
domain_classes, pixel_variables, adjacency_edges,
constraints, residual_reasons, metrics,
section_hashes, stage_trace
```

`residual_reasons=[]` 是 R0 的强制值。禁止 `winner/unique/accepted/preferred/top_k/Graph/SDF` 字段。Section hash 使用
`u32be(count)||for_each_item(u32be(byte_length)||canonical_item_bytes)`；whole artifact SHA 放外部结果，不自引用。

## 容量

```text
MAX_CARRIERS              = 8
MAX_INTERSECTIONS         = 8
MAX_ATOMIC_FRAGMENTS      = 64
MAX_PIXEL_VARIABLES       = 4096
MAX_OWNER_OPTIONS         = 16
MAX_DOMAIN_CLASSES        = 64
MAX_ADJACENCY_EDGES       = 32768
MAX_CONSTRAINTS           = 32768
MAX_TOTAL_RLE_RUNS        = 65536
MAX_AMBIGUOUS_VARIABLES   = 20
MAX_RAW_ASSIGNMENTS       = 1048576
MAX_OUTPUT_BYTES          = 16777216
```

每轴 `N` 必须成功、`N+1` 整份 closed reject；禁止 truncate、top-k 或只保留先出现的 owner。大整数 assignment count 必须
用 exact integer，不能浮点近似。

## 最小红队反例

- 四 pixel 链 `Lseed-a-b-Rseed`：局部 product 4 项，其中 `a=R,b=L` 断连，合法解只有 3 项；
- baseline padding：向 `{L,R}` 加 dummy 再删除，必须被 closed baseline schema 拒绝；
- residual saturation：R0 任一 residual option 必须 reject，不能以 coverage GO；
- ID/seed winner：交换 L/R 名称和输入顺序，bridge 解域仍为双值；
- oracle leakage：同 observation 的两份 stored annotation 不得改变 constructor bytes；
- capacity `K+1`：真实 option 放最后时也必须整份 reject，不能前 K 截断；
- partial carrier fiber：相邻 pixel 分别 `{H,V1}`、`{H,V2}`，不能用 region carrier union 给二者都加两个 owner。

## 裁决边界

R0 GO 只授权 annotation-free observation 到 factorized known-intersection CSP 的 research constructor。以下继续
NO-GO：fragment owner promotion、residual concretization、真实 PNG semantic fiber、异图碰撞、UniqueGraph、SDF 和训练。
