# Observation Closure v0.5 预注册

日期：2026-08-15

状态：`preregistered_before_v0_5_implementation`  
训练授权：`false`

机器可读的冻结合同：

- `formal/chemgraph/challenge_v0_5/prereg/p0_contract_v0_5.json`

合同 raw bytes SHA256：

```text
55ddcaacdf851166554ccd586b2a924c0e793f86cc7a6319f98132ef0ffb6b65
```

该值在 P0 提交时还必须由 freeze record 重算；文档中的路径、hash 与 Git tree 三者必须一致，不能用
工作区中的同名未提交文件替换。

本文与该合同必须在实现、pilot、closure 和 Candidate 之前一并提交。冲突时采用更严格的解释；如果两者
无法同时满足，版本直接 NO-GO，不能在 P1 解释性修订。

v0.4 已证明同一 canonical PNG 可以同时来自 S8 strict scene 与 C4 text/lane scene；旧 generator 用
`seen_png` 丢弃后到的观测，因而标签依赖生成顺序。v0.5 不继续调 Candidate，而是先把真值改为完整
observation fiber 的闭包。

## 1. 数学对象与类型

冻结以下不同类型，禁止混用：

```text
Dq          合同列出的 36,088,416 个 quotient render tuple
I(q)        tuple q 的完整有限合法 interpretation 集合
S           从属和 {(q,i) | q in Dq, i in I(q)}，即 semantic Scene 域
T           [-32,32] x [-32,32] 的 4,225 个整数 delivery translation
Draw        Dq x T，仅用于证明/审计 delivery lift
centered    Dq -> RawGray8(768,768)，baseline 在绝对中心 (384,384) 的 raster
Shift       RawGray8 x T -> RawGray8，整数像素平移、255 填充、不 wrap
rawRender(q,t) = Shift(centered(q),t)
P           RawGray8 -> Result CanonicalGray8 PreprocessFailure
embed       CanonicalGray8 -> RawGray8，逐字节嵌入且不携带 metadata
R(q)        P(rawRender(q,(0,0))) 的成功值
E           CanonicalGray8 -> Result Evidence ExtractFailure
M(q,i)      Scene interpretation -> finite labeled graph
~=          合同冻结的带标签简单图同构
```

`T` 不是群，v0.5 不再称它为 nuisance group。它只是有限 delivery lift。必须证明并执行审计：

```text
forall q in Dq, t in T,
  P(rawRender(q,t)) = Success(R(q))
```

任一空图、clipping、border 失败、preprocess failure 或不相等，整个 v0.5 NO-GO；不得把该 tuple 标 invalid、
Reject 或从 Dq 删除。闭包只枚举一个 centered representative 的前提正是这个合同。随机平移 pack 也必须
走相同的 `PNG -> decoder -> P -> E` 路径。

`rawRender(q,t)` 对应绝对平移 `(384+t.x, 384+t.y)` px，不允许实现自行选择原点。官方 delivery bytes
由 centered raster 做 `Shift` 得到；独立 reference 在冻结 witness 上核验“几何重渲染 = byte shift”。因此
全称 Draw 合同约化为：对 36M 个 centered raster 验证四边白边至少 33 px，再用 Lean/独立 byte checker
证明 `|t.x|,|t.y|<=32` 时 `P(Shift(x,t))=P(x)`。不展开 152B 次，也不把 Python rasterizer 等变性当作
未证假设。

对成功的可达值定义：

```text
RasterGraphSet(r) =
  {[M(q,i)]~= | q in Dq, i in I(q), R(q) = r}

EvidenceGraphSet(e) =
  union {RasterGraphSet(r) | r in R(Dq), E(r) = Success(e)}
```

`E` 的 failure 不是 Evidence bytes，也不进入 fiber。P4 后若任何 `r in R(Dq)` 使 E failure，版本 NO-GO。
域外 Reject 属于后续生产协议，不混入本轮闭包。

对可达 Evidence：

```text
|EvidenceGraphSet| = 1        Graph
2 <= |EvidenceGraphSet| <= 64 Ambiguous + 完整显式候选集合
|EvidenceGraphSet| = 0        在 Dq 内不可达
```

任何 class 超过 64 个非同构图，closure builder 必须保留并提交第 65 个 witness 后整版 NO-GO。Oracle
内部不得先截断；禁止 top-k、选一个 GT、改成 Reject 或提高 cap。

## 2. 声明域 Dq

Dq 已在 P0 合同中完整冻结，不留到 P1 决定。它是以下 tagged Cartesian product；每个 q 再由冻结的
`I(q)` 给出其全部 semantic interpretations：

```text
51 half_length
x 52 motif ordinal
x 9 width
x 3 cap
x 3 antialias
x 3 reflection
x 8 rotation
x 7 scale
= 36,088,416 tuples
```

唯一长度轴是 `half_length=60..110`；旧 `length=120..220` 不再存在，因此没有两个 length alias 到同一
`length//2`。52 个 motif ordinal 明确覆盖 S1--S8、C1--C4 及其 shape/gap/spacing/glyph 变体。tuple ID
公式、轴顺序、全部有理旋转和缩放写在合同中。

delivery lift 为 `Dq x T`，大小精确为：

```text
36,088,416 x 4,225 = 152,473,557,600
```

这不是要直接渲染 152B 次，而是要求 P1 的数学/独立 executable translation contract 允许每个等价类只
枚举 centered representative。PRNG 不定义 Dq，P1 不得增加 `valid`、attempt limit、rejection sampling
或 support gate 来缩域。若冻结几何在某组合上失败，只能 NO-GO 或开 v0.6 重新预注册。

基线几何和 rasterizer 绑定到合同内的路径、Git commit 与 SHA256。P1 可重构实现，但输出必须由独立
reference 对照；不能改变 motif 语义。

## 3. Raster preprocessor

真实输入和 closure representative 使用同一个 P：

1. 严格解码为 768x768、row-major Gray8；其他色型或尺寸在本 finite protocol 中失败；
2. foreground 精确定义为 `pixel != 255`；
3. 空 foreground 为整版 NO-GO；
4. tight bbox 使用 foreground 的 inclusive `min/max`；
5. crop 尺寸为 `max-min+1`；
6. 要求 raw foreground 到四边至少保留 1 px 全白边；
7. 新 768x768 白底，paste left/top 分别为 `floor((768-w)/2)`、`floor((768-h)/2)`；
8. 输出恰好 589,824 个 Gray8 bytes。

centered raster 必须保留至少 33 px 白边，Shift 后必须保留至少 1 px。1 px 是 clipping sentinel，不是
风格阈值。独立 clipping certificate 还必须把同一绝对几何渲染到 1024x1024
画布（原 768 区域位于 offset `(128,128)`），证明 padded foreground 全部落在该区域内部；仅看 768 bbox
不能证明没有被边界裁切。已知最小边距为 44 px 的合法反例因此不再被任意 64 px 门误杀，但 clipping 仍
不能通过。

奇偶 crop 因 floor 规则唯一。class identity 是完整 Gray8 bytes，不包括 crop origin、profile、源 style 或
PNG encoder 元数据。PNG 只是存储格式；canonical PNG 固定用 v0.4 Gray8 encoder，但不作为 class equality。

Extractor host 每次只向 stdin 写 589,824 个 canonical Gray8 bytes。argv、cwd、文件名、delivery ID、tuple/
class ID、crop profile/origin、manifest、family/style/cardinality 均不可见；环境变量采用空 allowlist，网络
关闭。审计 sidecar 只在 verifier 侧。P4 必须用 syscall/file-open audit 证明 E 没有 metadata host channel。

## 4. 闭包和 root

禁止单 pass `seen_png`。完整流程必须是：

```text
枚举全部 tuple
-> 生成 canonical full bytes + 全部 interpretation
-> 按 full bytes 分组
-> union 全部 interpretation graph
-> 图同构去重
-> 最后才分配 class ordinal、quota 和存储去重
```

SHA256 只作索引和 commitment。digest 相同的桶必须比较长度和全部 bytes；digest 相同但 bytes 不同即
NO-GO。Raster 与 Evidence 均使用该规则。

所有 stream root 使用合同冻结的 framing：

```text
u32be(domain_length) || domain_ascii ||
u64be(record_length) || record_bytes
```

records 按完整 canonical key 的字典序排序后串联取 SHA256。以下对象分别使用不同 domain separator：

- D tuple records 与 coverage bitset；
- canonical Raster bytes；
- tuple-to-RasterClass membership；
- RasterClass-to-GraphSet；
- canonical Evidence bytes；
- RasterClass-to-EvidenceClass membership；
- EvidenceClass-to-GraphSet；
- overflow witness；
- public/train/exclusion manifests；
- pack、GT 和 prediction records。

Raster/Evidence/Graph class ID 都是在完整 canonical bytes 排序后的 ordinal，不能 first-seen 分配。相同
seed 在 forward/reverse/shuffled shard 下必须选择相同 class bytes，而不只是得到相同总 root。

Builder A、Reference B 不共享 class builder、serialization 或图 canonicalizer。第三 verifier 从随机
witness 重渲染/重提取。两层 closure 都必须通过 forward、reverse、随机 shard、style bucket ID 置换和
provenance 置换；roots、memberships、GraphSets 和 `seed -> selected class bytes` 全部一致。

已知 S8/C4 八个 style witness 必须进入同一 RasterClass，并得到 `{two_lanes, text_stroke}` 两个非同构图。
所有 cross-family collision 都自动 union，并报告完整 family-pair collision matrix；已知 witness 不是特例。

## 5. 图同构

v0.5 图是有限无向简单带标签图：

- node 保留 `chemical_label` 与 `formal_charge`；
- edge 保留无序端点和 order 1/2/3；
- 孤立节点、断连分量和重复同标签节点都参与；
- 坐标与 source ID 不参与；
- self-loop 禁止；
- 同一无序节点对的 parallel edge 禁止，即使 order 不同也不能 merge。

任何 interpretation 出现禁止结构，整版 NO-GO。禁止在看完 closure 后选择 merge/reject/multiset 策略。
等价关系是存在节点双射，同时保留上述全部字段与邻接键级。

实现 A 使用颜色分桶后的穷举 permutation；实现 B 使用独立 partition refinement + backtracking。
adversarial corpus 必须覆盖置换、断连、孤立、键级、`TEXT_STROKE` label、同原子数异拓扑、自环和重复
边，结果 100% 一致。primitive role 本身不是 graph 字段并应被同构忽略；文字语义只通过
`chemical_label=TEXT_STROKE` 参与。

## 6. Lean 门必须早于 closure

P1 先冻结 Lean 定义与全部 theorem statements，P2 完成证明；P3 才可构造 Raster closure。不得先看 closure
再改变关系。无 `sorry/admit/axiom`，关键定理 `#print axioms` 为空。

必须形式化：

1. `RasterGraphSet`、成功 Evidence fiber 与双 closure；
2. EvidenceGraphSet 是相同 Evidence 的 RasterGraphSet 并集；
3. Raster fiber 到 Evidence fiber 的包含与 coarsening 单调性；
4. sampled/full 在显式 `Dsample subset Dfull` 且 R/E/M 限制一致时的单调性；
5. sampled Ambiguous 推出 full Ambiguous，sampled Graph 不推出 full Graph；
6. strict 保持的充要 fiber 条件；
7. quotient candidate 的 soundness 与 completeness 等价于预测/Oracle quotient coverage 相等；
8. 再加 `NodupUpTo(~=)` 或 canonical representative 前提，才推出每个同构类恰一代表；
9. finite/reachable 且图等价可判定时，singleton/multiple/empty 三分互斥且穷尽；
10. 全域左逆存在的必要条件和 strict fiber 下 selector 的条件充分性；
11. first-wins `seen_png` 反例；
12. 带标签简单图同构是等价关系，Lean reference Boolean checker sound/complete；
13. delivery translation contract，以及 `P(embed(c)) = Success(c)` 的 typed canonical identity。

Lean 不声称验证 Python、rasterizer、extractor、JSON、完整枚举、syscall 隔离或 Git 时序。Python A/B checker
只由 executable audit 验证，不能写成被 Lean 证明。

## 7. 文件与 CLI 计划

```text
formal/chemgraph/challenge_v0_5/
  prereg/p0_contract_v0_5.json
  schemas/p0_contract.schema.json
  schemas/domain_tuple.schema.json
  schemas/raster_class.schema.json
  schemas/evidence_class.schema.json
  schemas/closure_manifest.schema.json
  enumerate_domain_v0_5.py
  enumerate_domain_reference_v0_5.py
  canonical_raster_v0_5.py
  canonical_graph_v0_5.py
  canonical_graph_reference_v0_5.py
  canonical_evidence_v0_5.py
  build_raster_closure_v0_5.py
  build_raster_closure_reference_v0_5.py
  verify_raster_closure_v0_5.py
  build_evidence_closure_v0_5.py
  build_evidence_closure_reference_v0_5.py
  verify_evidence_closure_v0_5.py
  sample_pack_v0_5.py
```

enumerator CLI 固定为 `--contract PATH --expected-contract-sha256 HEX --output DIR --workers 16 --order
forward|reverse|shuffled --shard-index I --shard-count N`。builder 固定为 `--contract PATH
--expected-contract-sha256 HEX --input-manifest PATH --expected-input-root HEX --output DIR --workers 16`。
verifier 固定接收 `--contract/--expected-contract-sha256`、A/B manifest path 与 expected root、witness manifest
path 与 expected root、`--output`。root 不能代替输入路径，也不能省略 expected root。默认值、canonical
JSON profile 和 framing 不得由实现另定。

每类 stream 的精确 domain ASCII 已在机器合同冻结；record 是 P1 对应 closed schema 的 canonical JSON。
third verifier 必含所有公开 collision/axis-extreme witness，并按合同中的 raster-root/tuple-id hash 函数抽取
其余 witness，不能人工选择。

## 8. 资源门

P1 先冻结 P 的源码、strict decoder、依赖、schema、canonical graph bytes 和 commitment，再运行 raster pilot。
pilot 样本已冻结为 `tuple_id mod 100 == 0`，不是看结果抽样。它只报告 tuple/raster/crop、
translation audit、external sort、吞吐和资源；此时 E 尚未冻结，禁止报告或据此选择 EvidenceClass。

硬上限：

```text
pilot wall             4 h
peak RSS               80 GiB
temporary disk         2 TiB
projection safety      2x
full Raster wall       96 h
full Evidence wall     168 h
```

pilot 必须覆盖每个声明 axis value；否则 P1 NO-GO。超过任一资源门只能 NO-GO 或新版本重新声明更小的完整
domain，不能在 v0.5 抽样后签发 Graph。

pilot/full 均固定 16 个 OS process 并绑定 16 个 physical core。wall time 从首 worker 启动计到最终 fsync/root
写完；RSS 是整个 cgroup/process tree 的 aggregate peak；temporary bytes 是空 work filesystem 上 external
sort 在内的 peak allocation。wall/disk full projection 均为 `pilot_measure*(full_count/pilot_count)*2`。

E 在 P4 freeze 后，选择“最小 member tuple ID `%100==0`”的 distinct RasterClass 做独立 Evidence pilot。
wall projection 为 `pilot_wall*(all_raster_classes/pilot_raster_classes)*2`，worker 同样固定 16。该 pilot 只能
验证冻结 E 的资源与 determinism，不能修改 E；失败即 NO-GO。

不得保存 36M 个完整 768x768 raster。保存 tuple、framed digest、membership、graph signature 和 class
representative；collision 桶由 tuple witness 确定性重渲染比较 full bytes。

## 9. Candidate 隔离

Candidate 在完整 Evidence closure plaintext 产生前冻结。作者只能看到：P0/P1/P2/P3 公共合同、Evidence
schema、E 源码、手写 interpretation rules、Raster aggregate 和无 Evidence-to-GraphSet 映射的性能统计。

Candidate stdin 恰为一条 canonical Evidence，stdout 为一个完整 CandidateSet。禁止 closure/member/root
plaintext、Raster、PNG、family/style、tuple/class ID、generator、GT、cache、网络和非空环境。源码审计拒绝
大常量、生成 lookup、Evidence byte comparison table、嵌入 manifest 和运行时额外文件读取。

完整 Evidence closure 由未参与 Candidate 的 custodian 在隔离账户/volume 构建，plaintext membership、
Evidence bytes 与 GraphSets 密封到 P9。Candidate 作者在 P9 前只能看到 commitment 和以下聚合：总 class
数、strict/multiple 数、cardinality histogram、资源上限；不能看到 class 映射、样本或排序 ordinal。访问
日志和 artifact ACL 是硬门。当前共享 workspace 不构成这种隔离，正式 P6 必须另建权限边界。

由于 Dq、E 和 interpretation rules 公开，恶意作者理论上仍可独立重建整个有限闭包。ACL 与 source constant
audit 证明的是官方 plaintext 未泄漏、冻结后未适配和运行输入隔离，不证明作者在认识论上“从未见过该
有限域”。因此 U/X 是 seed-selected reproducibility test，不是现实分布的 hidden generalization 证明。

每个域内样本合取要求：candidate precision/completeness、CandidateSet quotient exact、decode exact 全部
100%，Reject=0。Ambiguous 必须携带 2..64 个显式去同构图；singleton 上输出 Ambiguous 必然多候选，
multiple 上输出 Graph 必然漏候选。

## 10. 固定 U/X 配额与反真空

U 和 X 各 512 个不同 EvidenceClass，且 U/X 零重叠；每包固定：

- 256 singleton、256 multiple；
- crossing、parallel multiplicity、text/lane taxonomy 各至少 32；
- 81 个 style bucket 各至少 4；
- 168 个 reflection/rotation/scale bucket 各至少 2；
- 每个非空 cardinality bucket `2`、`3..8`、`9..24`、`25..64` 至少 8；
- 全域最大 observed cardinality 所在桶至少 32。

一个 EvidenceClass 可覆盖多个来源；style/transform/taxonomy 归属和实际交付都选择该 class 中 full-byte
排序最小的 RasterClass，再选择该 RasterClass 的最小 tuple ID。不能为满足 quota 用一个 provenance，实际
却交付另一个容易视图。抽样器按机器合同的 HMAC seed rank 排序，在满足全部硬约束的 assignment 中取
字典序最小者。无可行 assignment 直接 domain adequacy NO-GO，禁止降 quota 或换 seed 找容易 pack。

对第 `delivery_ordinal` 个 class，平移由冻结的 HMAC counter stream 生成：拒绝 byte `>=195`，前两个接受
byte 分别映射为 `(byte mod 65)-32` 的 x/y。delivery ID、顺序和碰撞失败规则均在合同中。这样同一 closure、
seed 与 sampler root 只能生成一份 PNG pack。

完整 closure 还必须至少有 128 singleton、32 multiple，且三个 ambiguity taxonomy 均非空。Graph 占比不
按旧 family 意图强制；不足时判 domain 不适合作为该测试，而不是改真值。

manifest 输入数、实际 invocation 数、prediction records 数和 verifier denominator 必须逐包逐 replay
都恰为 512。缺一条、超时、crash 或 parse failure 都使整包 NO-GO，不能从分母删除。

## 11. commit/reveal 与泄漏

public/train/diagnostic 与 U/X 按三层隔离：完整 Raster bytes/class、完整 Evidence bytes/class、base motif/
scaffold provenance。P4 freeze E 后，P5a 先由 custodian 生成并提交 public/train/exclusion manifest roots，
同时冻结 `sample_pack_v0_5.py`、quota solver、PNG encoder、依赖和 sampler root；P5b 才原子 freeze
Candidate。Candidate 作者只能看到 exclusion commitment，不能看到 plaintext 映射。
U/X custodian 对 full bytes 执行排除，不能只比较 digest。

Sampler 必须早于 P6 完整 Evidence closure plaintext 冻结，不能让 custodian 看完 closure 再适配 member
selection、U/X 联合 assignment 或 tie-break。该 sampler root 进入 HKDF salt 与 artifact DAG。禁止在看到
closure、share 或 seed 后修改 member、witness、平移或 delivery-order 规则。

commit framing、两个 32-byte share commitment、RFC5869 HKDF salt/info 和 sealed GT HMAC 公式已写入 P0
合同。严格 artifact DAG：

```text
P0 contract commit
-> Raster/Candidate/exclusion/sampler roots freeze
-> Evidence closure root freeze
-> 两名 custodian share commitments
-> 派生域分离的 U/X seeds
-> image manifest commitment + hidden GT root HMAC commitment
-> 发布 opaque PNG pack
-> U/X 各两次 replay
-> prediction record roots commitment
-> reveal 两份 share 与 GT HMAC key
-> 双重建、核验 seed/pack/GT/prediction roots
```

delivery filename/ID 是随机 opaque 值且不进入 E/Candidate stdin；manifest 顺序不能编码 class/family。
prediction 前公开 image commitment，prediction 后才能 reveal。任一 seed/GT/root 不一致，seed 永久退役，
只有新版本可以重测。

## 12. 冻结顺序

```text
P0   本文 + machine-readable contract commit
P1   Lean definitions/statements + P/decoder/graph-bytes freeze + D 双 census + Raster pilot
P2   Lean proofs + 双图同构 + executable translation/iso/P gates
P3   双 Raster closure + third verifier + public commitment
P4   引用既有 P commitment，freeze E/Evidence schema/host sandbox + frozen-E resource pilot
P5a  custodian freeze train/public exclusions + sampler/quota/PNG/dependency roots；plaintext 密封
P5b  freeze Candidate/chemical decoder/container + source constant-table audit
P6   独立 custodian 双 Evidence closure；plaintext 密封，只公开 commitment/aggregate
P7a  确认 P3/P5/P6 全部 roots 与 artifact DAG，不允许修改 sampler
P7b  提交两份新 share commitment
P8a  commit/publish U/X image manifests and sealed-GT commitments
P8b  U/X 各两次 prediction replay
P8c  prediction commitments
P9   reveal + 双重建 + 独立 verifier
```

P3 前禁止构造 Raster closure；P5b 前禁止构造完整 Evidence closure；P9 前 Candidate 作者不得看到 closure
plaintext。任一步乱序失去 hidden 资格。

## 13. GO / NO-GO

以下全部合取：

1. Dq 36,088,416 个连续 ID 双 census exact，Draw translation contract 成立；
2. full-byte equality，不以 digest 代替；
3. Raster closure A/B、顺序扰动、third verifier exact；
4. 所有 duplicate 先 union，S8/C4 回归为 Ambiguous；
5. 图同构双实现、adversarial corpus 与 Lean 合同通过；
6. P/E/schema/host interface frozen、deterministic、域内 failure=0；
7. Evidence closure A/B、顺序扰动和 overflow detector exact；
8. 全域 cardinality 1..64；
9. CandidateSet/decode 100%，Reject=0，分母无缺失；
10. train/public/U/X 三层 full-byte 零泄漏；
11. 固定 quota 可行且 U/X 零重叠；
12. 两次 replay byte exact；
13. pack commitment 早于 prediction，prediction commitment 早于 reveal；
14. 至少两名独立 reviewer 批准 P0 与最终结果；
15. `training_authorized=false`。

任一失败只能 NO-GO，或新版本缩小并明确声明有限域。禁止改 cap、删 class、换 seed、改 taxonomy、用全
Ambiguous/Reject 或把 sampled diagnostic 写成 closure。

## 14. 不声称

- 不声称有限 Dq 覆盖真实化学图片；
- 不声称 Lean 验证 Python/OpenCV、完整枚举或操作隔离；
- 不声称 Candidate exact 等于 OCR/SDF exact；
- 不声称公开有限 Dq 对 Candidate 作者构成不可枚举的秘密分布；
- 不声称已解决楔键、虚线、颜色、多笔画文字、缩写或立体化学；
- 不声称 v0.5 PASS 自动授权训练；
- 不声称尚未执行的 P1--P9 已通过。
