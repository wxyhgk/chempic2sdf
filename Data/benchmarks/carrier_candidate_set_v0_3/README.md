# Carrier Candidate Set v0.3

这个目录起草 B' 路线的数据协议。它只保存协议和 closed schema，不包含 proposer、评分器、训练代码、
图片、样本身份、SDF、分子标签、blind、sealed 或 GT 数据。对应的数据无关 Validator、公开合成 fixture
和 mutation tests 位于 `formal/chemgraph/audit/carrier_candidate_set_v0_3/`。

当前状态：

```text
schema_frozen_for_implementation_review=true
status=structural_validator_frozen_not_complete
unique_graph_output_allowed=false
blind_execution_authorized=false
training_authorized=false
```

## 1. 为什么不能继续输出单一解释

Blind Carrier v0.2 能验证一份物理 segment、terminal、lane 和 bond 描述在结构上自洽，但还存在两个
无法由单 winner 修补的问题：

1. v0.2 输出没有保留上游 AtomPair 身份。物理载体实际位于 A-B 之间时，上游即使误声称 A-C，
   也可能编译出一份几何上自洽的 v0.2 文档。
2. greedy boundary pairing、最近 Atom anchor、文字/键硬分类都会提前删除其他可行解释。同一像素可能
   同时支持 crossing/junction、glyph/bond 或不同 boundary matching，模型置信度不能消除这种逻辑歧义。

因此 v0.3 的正常输出只有 `candidate_set`，没有 `unique_graph`、`winner`、`selected_interpretation` 或
任何等价字段。候选完整性尚未由独立 verifier 证明时，不得从候选集中挑一个结果进入训练。

## 2. 目录内容

- `protocol/protocol_v0_3.json`：冻结容量、枚举、规范化、拒绝和权限合同；
- `schemas/input.schema.json`：匿名 768x768 规范灰度 PNG 输入；
- `schemas/candidate_set.schema.json`：仅允许完整候选集或 closed reject 的输出 schema。

schema 使用 JSON Schema Draft 2020-12，所有 object 均设置 `additionalProperties: false`。

## 3. 输入边界

输入是一份 canonical JSON，其中只包含：

- `schema_version`；
- 768x768 PNG 的 base64 字节；
- 固定的 `training_authorized=false`。

PNG 必须是非交错、8-bit、灰度、只含 `IHDR/IDAT/IEND` 的规范重编码。schema 只能检查 JSON 和 base64
外形；后续独立 validator 必须复用 Blind Carrier v0.2 的严格 PNG 解码规则，并检查字节级规范性。
proposer 不得收到文件名、路径、样本 ID、分子 ID、SDF、SMILES 或任何 sidecar。

## 4. CandidateSet 结构

每个 interpretation 是自足的局部解释，包含：

- `boundaries`：从最终 PNG 得到的原始边界线；
- `boundary_matching`：本解释选择的边界配对和未配对边界；
- `stroke_fragments`：物理笔画分片、前景 support、boundary member 和视觉 primitive kind；
- `intersection_regions`：具名的 crossing / junction / touch 区域、support 和成员引用；
- `ownership`：fragment 到 bond lane、text glyph、marker、charge 或 residual owner 的显式归属；
- `atoms`：不带元素身份的结构 Atom 候选及其可观察 anchor；
- `endpoint_atom_relations`：所有门内 fragment endpoint 到 Atom/port 的显式几何关系；
- `ports`：Atom 与物理 fragment endpoint 的显式连接及可观察半径；
- `lanes`：一条 lane 的有序 fragment chain；
- `bonds`：1--3 条 lane 构成的 Bond，并显式保留规范无序 `atom_pair`；
- `white_gap_certificates`：绑定最终 PNG、包含 support 与跨度的**同一 lane 内 fragment gap**证书；
- `lane_separation_certificates`：order-2/order-3 Bond 的相邻 lane 之间独立白隙证书。

根级候选宇宙不是计数摘要，必须实际保存：

- `boundary_universe`：一次 boundary detection 得到的完整根级边界成员；
- `compatible_boundary_pairs`：引用根级 boundary ID 的全部通过固定 predicate 的 pair 及其整数观测；
- `atom_proposal_universe`：全部有界 Atom proposal 成员；
- `direct_pair_universe`：这些 proposal 的全部 Direct AtomPair 成员，包括 negative 和 uncertain；
- `stage_trace`：固定 12 阶段、连续 index、组件 SHA256、前一阶段输出到后一阶段输入的 hash 链，以及各阶段根级实际产物 hash。chain framing 同时包含 `component_sha256`，不能只绑定自报的输入输出。

boundary matching 只能引用根级 compatible-pair ID，并与 unmatched boundary 一起形成完整 partition。
Direct-positive 取交已被本协议明确废止：presence decision 只作为证据，negative/uncertain pair 也必须保留在
候选宇宙中，不能因此删除有物理载体的解释。

根级 `image` 还必须保存 final foreground、bond mask 和 text mask 的完整 canonical RLE及三份独立 hash；boundary 记录必须
分别列出其 final-foreground support 与 bond-mask support，不能只给线段坐标。根级
`graph_equivalence` 使用冻结算法把全部 interpretation 分成空间图等价类，但即使 class count 为 1，
本协议也不授权输出唯一图。

不同 interpretation 可以引用同一输入前景的不同分解。support 唯一性只在单个 interpretation 内检查，
不能像 v0.2 那样用全局 duplicate-support 门错误删除互斥解释。

`atoms/ports/endpoint_atom_relations/lanes/bonds` 均允许为空，从而让纯 glyph 或 charge ownership 与
非空空间图解释公平竞争。当前 primitive 只支持 `straight/glyph/charge/unknown`；楔键、虚线整体对象、
芳香圆和曲线必须 closed reject 为 `unsupported_primitive`，不能降级成 straight 后继续。

## 5. 枚举完整性

正常输出必须同时满足：

```text
enumeration.complete = true
enumeration.pruned_candidate_count = 0
enumeration.all_candidates_emitted = true
enumeration.capacity_hit = false
```

需要枚举而不能 greedy 选择的关系包括：

1. 全部允许的 boundary matching，包括合法 unmatched boundary；
2. 每个 carrier endpoint 到门内全部 Atom 候选的关系，不用 nearest argmin；
3. bond lane / text glyph / marker / charge / residual 的全部合法 ownership；
4. crossing 的 through-line 与 split-junction 解释；
5. 多 fragment lane 的全部合法 white-gap 连接。

Direct AtomPair、P1-A 或其他视觉分数可以量化后保存在 evidence 字段中，但不得在本阶段删除候选。
如果完整枚举超过容量，只能返回 `outside_capacity`，不能 top-k、beam prune、静默截断或改阈值。

## 6. 固定容量

| 对象 | 上限 |
|---|---:|
| 输入 PNG | 768x768 |
| 输出 canonical JSON | 67,108,864 bytes |
| raw boundaries / interpretation | 128 |
| root boundary universe | 128 |
| compatible boundary pairs | 8,128 |
| Atom proposal universe | 128 |
| Direct pair universe | 8,128 |
| boundary matchings | 128 |
| ownership components / hypotheses | 128 / 128 |
| interpretations | 128 |
| atoms / interpretation | 128 |
| endpoint-to-Atom alternatives / endpoint | 8 |
| endpoint-to-Atom relations / interpretation | 4,096 |
| ports / interpretation | 512 |
| stroke fragments / interpretation | 512 |
| intersection regions / interpretation | 128 |
| lanes / interpretation | 512 |
| bonds / interpretation | 512 |
| white-gap certificates / interpretation | 512 |
| lane-separation certificates / interpretation | 1,024 |
| ordered stage trace records | 12 |
| fragment endpoint refs / port | 64 |
| fragments / lane | 64 |
| lanes / bond | 3 |
| support RLE runs / fragment | 100,000 |
| support pixels / fragment | 589,824 |

这些是 closed 安全容量，不是能力结论。公开 census 若频繁命中容量，应新建协议版本；不得原地放宽。

## 7. 规范化

- JSON 使用 UTF-8、ASCII escape、key 字典序、紧凑分隔符和单个结尾换行；禁止 BOM、重复 key、
  `NaN` 和无限值。
- 坐标使用 `Q4-pixel`：像素中心 `(x,y)` 为 `(16*x+8,16*y+8)`，范围 `0..12288`。
- 非负实数转 Q4 使用 half-up；落在实现误差带内的半格 tie 必须拒绝，不能依赖平台 round。
- 概率使用 `q24 = floor(p*16777216 + 0.5)`，范围 `0..16777216`；比例使用整数 ppm。
- 所有 ID 从 0 连续；引用数组升序且唯一。
- boundary 两端按 `(x,y)` 字典序规范；`atom_pair` 严格升序；Bond 的 begin port 属于 atom_pair[0]。
- interpretation 先按不含 `id` 的规范内容字节排序、去除逐字节重复项，再赋连续 ID；不得按分数排序。
- 三份根级 RLE 分别按冻结 framing 求 SHA256，不能用一个未说明 framing 的合并 hash 替代。

ambiguity taxonomy 使用冻结代码 `A03/A08/A10/A11/A12/A13/A15/A16`。每个 ambiguity entry 必须引用
至少两个 graph class，并以 tagged `{interpretation_id, object_kind, object_id}` 指向 boundary、fragment、
intersection、port、lane、white-gap 或 lane-separation 证据。

## 8. Reject

Reject 也是 closed canonical JSON，只允许以下原因：

- `invalid_png`：输入 JSON/base64/PNG 不合法或 PNG 非规范；
- `outside_capacity`：完整枚举或输出超过冻结容量；
- `unsupported_geometry`：几何退化、量化 tie、引用冲突或无法形成合同内直线/分片；
- `no_supported_carrier`：没有任何带真实前景 support 的 carrier interpretation；
- `enumeration_incomplete`：实现无法证明没有剪枝或无法给出全部枚举 hash；
- `unsupported_primitive`：出现楔键、芳香圆、曲线或其他 v0.3 未建模 primitive；

`internal_error` 不属于 v0.3 candidate-set schema。runner 若崩溃，只能在协议外的执行台账中记录失败，
不得生成一份看似合法的 proposer reject。

Reject 不得携带输入 ID、路径、异常堆栈或模型输出。

blind 每张图片只允许一次 authoritative run，不允许失败后重放。文档中的 byte-exact replay 只指冻结前在
公开输入上的可复现性测试，不授权对 blind 输入执行第二次 proposer run。

## 9. 后续门禁

closed schema 和结构 Validator 已冻结，但状态只有 `STRUCTURALLY_VALID_NOT_COMPLETE`。在独立 completeness
verifier 冻结前，即使是公开或已暴露图片也不得执行真实 proposer；本目录的公开合成 fixture builder 不属于
图片 proposer。

允许实现 proposer 之前，需先为 schema 和 canonicalization 添加独立 validator 与正负 fixture。之后也只能
在公开合成图和已暴露图片上检查：

- truth interpretation 是否被候选集覆盖；
- crossing/junction、double-vs-parallel-single、glyph-vs-bond 是否保留多个非同构类；
- `false singleton = 0`；
- 输入排列、无向交换和重复运行是否字节一致；
- support precision、carrier recall、AtomPair provenance 和 white-gap 证书是否通过独立审计。

这些门通过只允许进入 blind 前评审，不允许输出唯一图或训练。唯一图接口必须另开版本，并同时具备独立
候选完备性证书和“全部 surviving interpretations 属于同一图等价类”的证明。
