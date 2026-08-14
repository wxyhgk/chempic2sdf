# Hidden Generator v0.4 Oracle 与审计契约

日期：2026-08-15

状态：`P1_SECOND_CROSS_IMPLEMENTATION_REVIEW_APPROVED_FOR_PUBLIC_SMOKE_ONLY`。初版
generator/rasterizer 来自 commit `55a459499a5f72775e2796c7bd0c0ff012eaf244`，第一轮独立审计拒绝后已
修复有限 grammar；本文件与
`formal/chemgraph/challenge_v0_4/schemas/` 冻结数据边界、可见性和 verifier 义务，不读取或约束
未来 extractor 的内部算法，不授权训练。

## 1. 目标与非目标

v0.4 先建立一个可由独立 verifier 从 seed/spec 完整重建的隐藏挑战。它解决的是“测试输入和
Oracle 是否可信、是否在预测前保持密封”，不是先写模型再围绕输出补标签。

当前 schema bundle 同时表达 GraphSet strict/ambiguity 与 C4 `stroke_role_ambiguity`。C4 已缩域为
一个与 chemical carrier 像素 support 分离的单 stroke 字形；因此该有限 grammar 中只有“整个 stroke
是 glyph”或“整个 stroke 是一根 bond lane”两种 role。它不覆盖多 stroke 文字、stroke 内交叉或现实
OCR/text-role hypothesis，不能据此启动 extractor 训练。

必须建立：

```text
seed + frozen spec + frozen generator/rasterizer
  -> InkProgram
  -> Transform
  -> canonical PNG bytes
  -> hidden Interpretations
  -> independently reconstructed GraphSet
  -> deterministic rejection log and pack manifest
```

预测程序只能接收公共 image pack。GT 解封后，独立 verifier 才可把预测 CandidateSet 与重建
GraphSet 比较。

本契约不声称：

- generator 已通过正式 full hidden pack、真实 seed、加密隔离或现实分布验证；
- hidden pack 已经生成；
- future extractor 已经设计或通过；
- GraphSet 对现实化学空间完备；
- 通过有限 hidden profile 即可生产部署；
- 本周期允许训练。

## 2. Schema 清单

运行期 JSON/JSONL 使用 UTF-8、key 字典序、无多余空格、单个结尾换行的 canonical profile。
Schema 源文件自身保留便于 code review 的 pretty JSON，commitment 对其 raw bytes 计算 SHA256；
`schema_bundle_manifest` 是 canonical JSON。Schema 采用 JSON Schema draft 2020-12 的小子集，文件
如下：

| Schema | 对象 | 默认可见时刻 |
|---|---|---|
| `common.schema.json` | SHA、artifact ref、有理数、Q16 点等公共定义 | 预测前 |
| `schema_bundle_manifest.schema.json` | schema 文件及 SHA256 清单 | 预测前 |
| `challenge_spec.schema.json` | generator 写出的 `config.json`：PRNG、配额、变换和 style | 预测前 |
| `transform.schema.json` | 精确反射、旋转、缩放和平移 | 解封后 |
| `ink_program.schema.json` | 无语义可见墨迹程序 | 解封后 |
| `interpretation.schema.json` | InkProgram 的一个隐藏化学/角色解释 | 解封后 |
| `graph_set.schema.json` | 一个 observation 的完整隐藏 GraphSet | 解封后 |
| `observation_gt.schema.json` | delivery image 到全部 GT 对象的连接 | 解封后 |
| `rejection_log.schema.json` | 每次接受/拒绝尝试 | 解封后 |
| `image_pack_manifest.schema.json` | 只含 opaque delivery ID 与 PNG 完整性信息 | 预测前 |
| `commitment.schema.json` | config、代码、seed、pack 和 sealed GT digest 承诺 | 预测前 |
| `sealed_gt_manifest.schema.json` | 密文 artifact 与加密元数据 | 预测前 |
| `sealed_gt_index.schema.json` | 解密后 GT JSON/JSONL artifact 索引 | 解封后 |
| `freeze_record.schema.json` | commit、容器、角色、全部冻结 artifact | 预测前 |
| `reveal.schema.json` | 预测冻结证明、seed shares、解密材料 | 预测冻结后 |
| `candidate_set.schema.json` | 预测侧候选图集合 | 预测冻结后 |
| `prediction_record.schema.json` | 一张图的 Evidence/Candidate/Decode artifact | 预测冻结后 |
| `prediction_run_manifest.schema.json` | Pack/replay 的运行冻结记录 | 预测冻结后 |
| `audit_validation.schema.json` | 分母、epsilon/Delta、全部合取门 | 最终验证后 |

Schema 只验证局部形状。引用完整性、集合相等、时间顺序和数学不变量由本文件定义，不能因为 JSON
Schema 接受就判定挑战通过。

## 3. 可见性矩阵

### 3.1 预测前公开

以下对象可在 predictor 冻结前公开：

- 主预注册、本文和完整 schema bundle；
- `challenge_spec`，包括 family 总配额和 style 桶，但不含 image 到 family 的映射；
- generator/rasterizer/PNG encoder/oracle verifier 源码与容器的 commitment；
- `freeze_record` 中公开角色和 artifact SHA；
- `commitment`；
- `image_pack_manifest` 与 canonical PNG files；
- `sealed_gt_manifest` 与 GT ciphertext；
- prediction interface/schema。

公共 `image_pack_manifest.images[]` 只能包含：

```text
delivery_id
relative_path
png_sha256
byte_length
media_type
width/height
```

不得包含 family、style、transform、seed counter、collision membership、duplicate class、GraphSet
cardinality、ambiguity kind、InkProgram ID 或 Interpretation ID。数组顺序是 opaque delivery 顺序，
不能编码 family 顺序。

### 3.2 预测前仅 custodian 可见

以下内容在四份 prediction run 全部冻结前不得交给 prediction author、runner 或模型进程：

- seed shares 和最终派生 seed；
- GT 解密 key；
- decrypted `sealed_gt_index`；
- InkProgram、Transform、Interpretation、GraphSet、observation GT；
- rejection log；
- image 与 family/style/collision 的映射；
- exact-collision 同组关系；
- generator attempt/rejection 次数。

即使某个对象将来公开，其 plaintext SHA 也不能在预测前单独发布，除非它已经由高熵随机材料
域分离承诺且不存在小空间枚举风险。密文 SHA 可以公开。

### 3.3 预测进程实际输入

预测进程逐图只接收 canonical PNG bytes。Host 可以用 `delivery_id` 调度和落盘结果，但
`delivery_id`、manifest 路径、pack family 和 GT 不进入模型输入协议。

### 3.4 预测冻结后、GT reveal 前公开

必须先冻结四个 run：

```text
run-a-1  Pack A replay 1
run-a-2  Pack A replay 2
run-b-1  Pack B replay 1
run-b-2  Pack B replay 2
```

每个 run 公开 `prediction_run_manifest`、prediction records JSONL、Evidence/Candidate/Decode artifact
SHA 以及 prediction pipeline/container freeze。之后才允许发布 `reveal`。

### 3.5 Reveal 后公开

reveal 验证成功后可公开 seed shares、GT 解密材料、sealed GT plaintext artifacts、rejection log 和
family/style 映射。原始 ciphertext、commitment 和 prediction freeze 永久保留，不能被解封产物覆盖。

## 4. 严格状态机

Challenge 状态只允许单向转换：

```text
SCHEMA_FROZEN
  -> GENERATOR_AND_VERIFIER_FROZEN
  -> SEED_COMMITTED
  -> PACK_AND_CIPHERTEXT_COMMITTED
  -> PACK_PUBLISHED
  -> FOUR_PREDICTION_RUNS_FROZEN
  -> SEED_AND_GT_REVEALED
  -> INDEPENDENTLY_REBUILT
  -> FINAL_VALIDATION_FROZEN
```

每次转换生成一个 canonical record 和 SHA256。缺少任一前驱、Git/source/container/schema SHA
变化、预测后换 seed、reveal 后重跑预测或删除 rejection 都使整个版本 `NO-GO`，不能降级成部分
通过。

## 5. Commitment 算法

commitment record 使用：

```text
SHA256(
  u64be(len(domain_separator)) || domain_separator ||
  u64be(len(payload))          || payload
)
```

`domain_separator` 必须以 `chemgraph-hidden-v0.4/` 开头，每个 subject 唯一。Artifact payload 是
canonical bytes；seed-share payload 是 32-byte 原始 share，不是 hex 文本。

独立 verifier 必须：

1. 验证 commitment records 唯一、按 `record_id` 排序；
2. 重算所有公开 artifact commitment；
3. reveal 后重算每个 seed share commitment；
4. 按 custodian ID 排序 shares，执行冻结的 HKDF-SHA256 得到 pack master seed；
5. 对 Pack A/Pack B 使用不同 domain-separated stream，禁止复用 counter stream；
6. 验证 sealed ciphertext SHA 与预测前 commitment 一致；
7. 验证 prediction freeze artifact 早于 reveal 且 SHA 未改变。

时间戳只辅助审计，不能单独证明先后；先后关系由不可变 commit/artifact DAG 证明。

## 6. JSON 与路径规范

stdlib verifier 必须使用 `json.loads(..., object_pairs_hook=...)` 拒绝重复 key，并显式实现 Schema
所用的小子集：

```text
type, const, enum, required, additionalProperties=false,
pattern, minLength, minimum/maximum, minItems/maxItems, uniqueItems,
items, prefixItems, $ref, anyOf
```

它不能依赖 generator 的 dataclass/parser，也不能因为系统未安装第三方 `jsonschema` 而跳过验证。

Canonical JSON/JSONL 规则：

- UTF-8，无 BOM；
- `ensure_ascii=true`；
- key 字典序；
- separators 为 `(',', ':')`；
- 禁止 float、NaN、Infinity 和 `-0`；
- JSON 恰好一个结尾 `\n`；
- JSONL 每行一个 object，最后一行也有 `\n`；
- 空行、重复 key、unknown key 均失败。

所有 artifact path 必须是 POSIX 相对路径；逐 segment 拒绝空段、`.`、`..`，拒绝绝对路径、反斜杠
和 symlink。Schema 的正则不是完整路径安全证明，verifier 必须额外执行这些检查。

有理数 `[numerator, denominator]` 必须 denominator 正、最大公约数为 1；零只能写 `[0,1]`。Q16.16
坐标是有符号整数，禁止 float。

## 7. 跨对象不变量

### 7.1 Transform

verifier 用 `Fraction` 精确检查：

```text
cos^2 + sin^2 = 1
scale > 0
reflection in {none,x,y}
composition order 固定
最终 affine determinant != 0
```

它从 reflection、rotation、scale、translation 自行构造 matrix 和 inverse；generator 不提供一个
可直接信任的“最终矩阵”。所有变换后点必须仍满足 spec 中的画布/margin 条件。

### 7.2 InkProgram

- `style_id`、`primitive_id` 在每个 program 内唯一并按数值 ID 排序；
- 每个 primitive 的 style 必须存在；
- line 恰好两个不同点，polyline 至少两个相邻不同点，polygon 至少三个点且 `closed=true`；
- 只有 `glyph_template` 可以有非 null `glyph_template_id`；
- InkProgram 不得包含 Atom、Bond、graph、family、ambiguity 或 GT owner 字段；
- 同一个 InkProgram 只 rasterize 一次，再供多个 Interpretation 共享，不能分别渲染 semantic truth。

### 7.3 Interpretation

- Atom/Bond/role ID 在 Interpretation 内唯一且按编号排序；
- Bond 两端 Atom 必须存在、不同，端点无向规范排序；
- `primitive_ids` 必须存在于对应 InkProgram；
- 每个 Ink primitive 必须恰好有一条 `primitive_roles` 记录；
- `bond_lane.owner_id` 必须引用 Bond，`atom_glyph.owner_id` 必须引用 Atom；
- decoration/unassigned 的 owner 必须为 null；
- Interpretation 不携带 canonical graph signature；该签名只能由 verifier 重算。

### 7.4 GraphSet

独立 verifier 从每个 Interpretation 的 Atom/Bond 构图，忽略 Atom ID、Bond 顺序和无向端点顺序，
按 node chemical label/formal charge 与 edge order 独立做图同构。GraphSet 必须恰好等于所有
Interpretation graph 的同构商集：

```text
GraphSet = { Graph(interpretation) } / graph_isomorphism
```

这个等式只检查“已列 Interpretation 的内部闭包”，不能证明 Interpretation 列表本身对 InkProgram 的
允许语义完备。独立 verifier 还必须从 family grammar 和精确几何枚举可能解释，检查：

```text
Interpretations = EnumerateAllowedMeanings(InkProgram, family grammar)
```

若只从 generator 已写出的 Interpretation 反推 GraphSet，会形成同源自证。C4 必须额外考虑 stroke
共享端点和内部交叉所形成的 junction/crossing 图；C2/C3 必须先证明每条 lane 在最终变换与线宽下
仍可分离，否则固定的“bundle 或独立 singles”二候选没有完备性依据。

- `graphs` 中不得有同构重复；
- 每个 Interpretation 恰好映射一个 graph；
- 每个 graph 至少有一个 Interpretation witness；
- `ambiguity_kind=none` 当且仅当 GraphSet cardinality 为 1；
- ambiguity kind 必须由 family/spec 与几何关系独立重算，不能只信字段；
- strict family 的 GraphSet 必须恰好一个；collision family 必须达到 spec 冻结 cardinality。

### 7.5 Observation 与 image pack

- 每个公共 `delivery_id` 在 manifest 中唯一；PNG path/hash/byte length 与实际 bytes 一致；
- 解封后每个 delivery ID 恰好连接一个 observation GT，反之亦然；
- observation 的 PNG SHA 必须等于公共 manifest，production gray SHA 必须由 PNG 解码重算；
- `ink_program_id`、`transform_id`、Interpretation IDs、GraphSet ID 均存在且互相引用同一 observation；
- 不同 Interpretation 若属于同一 observation，必须共享完全相同 InkProgram 与 PNG bytes；
- exact collision 不通过重复渲染证明，而由同一 observation 的多 Interpretation/GraphSet 表达。

### 7.6 Rejection log

每个 `(pack,family,sample_ordinal)` 从 attempt 0 连续记录到第一次 accepted 或最大尝试数：

- rejected row：`accepted=false`、reasons 非空、terminal observation 为 null；
- accepted row：`accepted=true`、reasons 为空、terminal observation 非 null；
- accepted 后不得再有该 sample 的 attempt；
- stream counter 严格按 spec 派生，不得跳号或复用；
- 每个最终 observation 必须有且仅有一个 accepted row；
- 无 accepted row 时整个 pack 失败，不能缩小 family 配额；
- reveal 前不得根据 prediction 输出追加、删除或改写 rejection。

这些条件需 stdlib verifier 手写，JSON Schema 无法表达。

## 8. 独立 rebuild 契约

独立 verifier 不导入 generator、rasterizer、oracle enumerator 或它们的 graph canonicalizer。至少两
个实现角色分离：generator author 与 evaluator reviewer 不得相同；role 冲突直接 `NO-GO`。

从 reveal seed、冻结 spec 和 source/container commitment，verifier 必须重建并逐字节比较：

1. 每个 PRNG domain/counter 的抽样结果；
2. 完整 rejection log canonical JSON；
3. Transform JSONL；
4. InkProgram JSONL；
5. Interpretation JSONL；
6. independently canonicalized GraphSet JSONL；
7. observation GT JSONL；
8. 每张 `768x768 Gray8` canonical PNG bytes；
9. image pack manifest JSON；
10. 采用冻结 tar profile 时的 image pack archive bytes；
11. sealed GT plaintext index 与所有 plaintext artifact bytes；
12. commitment/reveal/freeze DAG 中所有可重算 digest。

加密 ciphertext 不要求通过重新加密获得相同 bytes；verifier 必须验证预测前 committed ciphertext，
用 reveal key 解密，并确认 plaintext 与独立重建 bytes 一致。若协议选择确定性加密，则必须另行说明
nonce/key domain separation，不能默认为可重放。

GraphSet 重建必须有第二套实现。可接受做法是 stdlib 图同构枚举/partition refinement；不可接受
做法是读取 generator 已写的 graph signature 或 import generator canonicalizer。

任一 artifact 只有 hash、没有可读取 committed bytes，视为证据缺失，不得写 `rebuild_passed=true`。

## 9. A/B 与 replay

Pack A、Pack B 是两个独立隐藏 pack，不是同一 pack 的两个目录副本。两者使用相同 spec/配额，但
seed stream 域分离，必须各自独立达到全部门。

每个 pack 各执行两次 prediction replay：

- 同一 committed PNG bytes；
- 同一 prediction source/container freeze；
- 不同预注册输入 permutation；
- 不同 Host 临时路径和 worker 调度；
- 每图独立 invocation；
- 输出按 `delivery_id` 重排后比较。

Pack 内以下 canonical artifact 必须逐字节相同：

```text
prediction records JSONL
Evidence artifacts
CandidateSet artifacts
DecodeResult artifacts
规范化 failure records
```

runtime、PID 和 wall-clock 不进入 byte equality，但必须单独报告。Pack A 与 Pack B 图片不同，因此
不要求跨 pack prediction bytes 相同；要求的是两个 pack 的所有 family/style 分桶分别 100% 通过。

Generator 也必须独立 replay 两次，从同一 seed/spec 重建全部 bytes。Generator replay 与 prediction
replay 是两个不同门，不得合并成一个 `ab_match=true`。

## 10. CandidateSet 与 Decode 审计

解封前 predictor 输出 `candidate_set.schema.json`。解封后 independent verifier 对匿名 graph 重新做
图同构，逐 observation 检查：

```text
PredSet == RebuiltOracleGraphSet
```

必须报告：

```text
candidate_set_exact_count / observation_count
missing_graph_count
extra_graph_count
duplicate_isomorphic_candidate_count
empty_candidate_count
```

集合相等 100% 是硬门；precision、recall 或 top-1 都不能替代。

Conservative decode 规则固定：

```text
|PredSet| = 0  -> Reject
|PredSet| = 1  -> Graph(the only candidate)
|PredSet| > 1  -> Ambiguous(rebuilt ambiguity kind)
```

strict observation 的 Ambiguous/Reject，ambiguous observation 的 Graph/Reject，以及任意 wrong Graph
均为失败。`Reject` 不能算作 strict correct。

## 11. Exact collision 门

每个 exact collision observation 必须由一个 InkProgram 和至少两个 Interpretation 表达。verifier
检查：

- 每个 Interpretation 连接同一 PNG SHA 和 InkProgram；
- rebuilt GraphSet cardinality 等于 family spec；
- prediction replay 的 Evidence canonical bytes 在同一 PNG 上一致；
- PredSet 与 rebuilt GraphSet 完全相等；
- DecodeResult 为正确 Ambiguous kind；
- Graph/Reject 数为 0。

按 family、style、transform bucket 分别要求 100%，禁止合并平均。公开旧 collision fixture 不进入
hidden 分母，只可作 generator/verifier 单元测试。

## 12. epsilon / Delta

只在 rebuilt GraphSet cardinality 为 1 的 strict finite nuisance shell 上计算。像素对象固定为完整
`768 x 768` Gray8 数组：

```text
d(X,Y) = sum[y,x] abs(int(X[y,x]) - int(Y[y,x]))
```

对每个冻结 bucket：

1. 完整枚举 spec 定义的 strict scenes 与 nuisance renders；
2. `epsilon` 是同一 strict graph/motif 的每个 nuisance render 到其 canonical render 的最大距离；
3. `Delta` 是该 bucket 内两个非同构 strict graph 的任意 nuisance render 之间最小距离；
4. 保存 epsilon 最大 witness 与 Delta 最小 witness delivery IDs；
5. 报告完整 strict scene、nuisance render 和 nonisomorphic pair 分母；
6. 检查严格整数不等式 `2 * epsilon < Delta`。

空 strict 域、少于两个非同构图、未完整枚举、只有采样最小值、缺 witness 或等号都失败。每个 bucket
单独通过，不能用全局平均掩盖失败 bucket。

Pack A/B 分别计算。两个 pack 都通过只说明该有限 profile 的稳定性，不形成现实图像的全局间隔。

## 13. 100% 合取门

最终 `audit_validation` 的所有 gate 必须为 true，且 verifier 从原始 artifact 重算，而非信任字段：

1. `schema_bundle_valid`：全部 schema canonical bytes、SHA、closed-world 字段通过；
2. `commitment_and_reveal_valid`：所有 commitment、share、ciphertext、plaintext 与 DAG 一致；
3. `prediction_frozen_before_reveal`：四个 prediction freeze 先于 reveal 且未修改；
4. `generator_rebuild_byte_exact_100_percent`：第 8 节全部可重建 artifact 逐字节一致；
5. `hidden_pack_quota_exact`：A/B 每 family/style/transform 配额精确；
6. `rejection_log_exact`：所有 attempt 连续、完整且可重放；
7. `replay_byte_exact_100_percent`：generator replay 和每个 pack prediction replay mismatch 为 0；
8. `exact_collision_evidence_equal_100_percent`：全部 hidden collision observation 通过；
9. `candidate_set_exact_100_percent`：每个 observation PredSet 等于 rebuilt Oracle GraphSet；
10. `conservative_decode_exact_100_percent`：strict/ambiguous 分类逐项正确；
11. `stability_all_buckets_strict`：每个非空有效 bucket 满足 `2 epsilon < Delta`；
12. `ab_each_pack_passed`：Pack A 和 Pack B 单独满足全部适用门；
13. `independent_verifier_passed`：stdlib verifier 原始报告已冻结并由非实现者批准。

除 gate 外，以下计数必须显式为 0：

```text
schema_failure_count
commitment_mismatch_count
rebuild_byte_mismatch_count
quota_mismatch_count
rejection_log_mismatch_count
prediction_replay_mismatch_count
missing_graph_count
extra_graph_count
duplicate_isomorphic_candidate_count
strict_wrong_or_abstain_count
ambiguous_graph_or_reject_count
wrong_graph_count
collision_png_mismatch_count
collision_evidence_mismatch_count
stability_bucket_failure_count
```

Schema 已把上述逐项计数设为 required，并要求 `bucket_metrics` 按 Pack/family/style/transform 分桶
报告。实现 audit runner 时仍必须从原始 artifact 重算，不能只把这些字段填成 0。所有分母非零并
分别报告。

任一 gate false、任一失败计数非零、artifact 缺失或 verifier 无法重建：

```text
overall_gate_status = no_go
training_authorized = false
```

即使全部 gate true，`overall_gate_status` 最多为
`go_limited_hidden_profile_only`，`training_authorized` 在 v0.4 仍固定为 false。是否训练 neutral
Evidence predictor 必须进入下一个独立预注册周期。

## 14. Freeze record 与角色分离

freeze record 必须承诺 schema/spec、generator、rasterizer、PNG encoder、oracle enumerator、independent
verifier、prediction pipeline 及容器。每个 role 只能由明确 agent 承担。

最低分离要求：

- seed custodian 不参与 prediction 实现；
- generator author 不担任唯一 generator reviewer；
- oracle enumerator author 不担任唯一 evaluator reviewer；
- prediction author 在 reveal 前不能读取 custodian artifact；
- 至少一名 evaluator reviewer 未参与 generator、enumerator 和 prediction 实现。

Schema 只能检查 role records 存在；agent 是否重复、审批是否独立由 verifier 和人工双重检查。

## 15. 实现前门

开始写 generator 前必须完成：

1. 所有 schema 通过 JSON 解析、`$id/$ref` 闭包和 canonicalization 自检；
2. 生成 `schema_bundle_manifest`，冻结每个 schema SHA；
3. 独立 reviewer 审批本契约中的可见性和跨对象不变量；
4. 审批第 13 节 validation 失败计数和分桶分母是否完整；
5. 冻结 generator source tree、container 和 verifier interface；
6. 确认 seed custodian 与实现者角色分离。

在这些条件完成前：

```text
generator_implementation_authorized = false
hidden_pack_generation_authorized = false
prediction_execution_authorized = false
training_authorized = false
```

## 16. P1 独立交叉审计结果

### 16.1 第一轮：拒绝

审计器：`formal/chemgraph/challenge_v0_4/audit_generated_pack_v0_4.py`。它只依赖 Python 标准库，
不 import generator、rasterizer 或 generator 的图规范化代码。公开全零 seed 的两个 smoke pack
逐字节相同；48 个 observation、72 次 invocation、Gray8 PNG/CRC、closed schema、artifact digest、
引用双射、rejection log 和从 Interpretation 独立重建 GraphSet 均通过。

第一轮结论为 `NO-GO`，原因是候选完备性没有成立：

1. C3 的 style 为 9px。四个 variant 的相邻 lane 中心最小距离依次为
   `10 / 6.25 / 7.5 / 10 px`。以保守的可见白隙充分条件
   `distance > width + 2px` 检查时 4/4 失败，其中 `6.25px` 和 `7.5px` 两项墨迹明确重叠。
   generator 既未拒绝/重采样这些变换，也未枚举 lane 合并后可能出现的解释。
2. C4 的 4/4 smoke observation 都存在 glyph stroke 共享端点或内部相交。现有 GraphSet 只含
   “一个文字 Atom”与“每 stroke 一根互不相连单键”两个图，缺少把相交/共端 stroke 合并为
   junction/crossing 的图，因此不能称为完整 Oracle GraphSet。

因此第一轮只能声称“协议字节与已列 witness 自洽”，不能声称“有限 grammar 的所有解释已经枚举
完毕”。该结论触发 generator/spec 修复，没有用放宽审计器或增加训练掩盖问题。

### 16.2 第二轮：公开 smoke 有限批准

修复包括：

- C2/C3/C4 在 accept 前按正式 fixed-point 采样规则逐 primitive 重算前景 support；任意像素交集或
  8-neighbor adjacency 都以 `invalid_exact_collision` 拒绝并重采样，固定分母不变；
- C4 缩为单 stroke `I/l/1/slash`，候选图标签恒为 `TEXT_STROKE`，模板 ID 仅作 sealed 视觉诊断；
  chemical primitive 与 glyph primitive 必须通过同一 support 门；
- style axis 冻结为真实实现的 `round join`，不再把未实现的 miter/bevel 声称为不同 bucket；
- commitment 新增递归 protocol bundle digest，覆盖 `.py`、README、根 schema 和独立 schema bundle。

在 protocol 闭包稳定后，从空目录用公开全零 seed 生成 Pack A 两次、Pack B 一次。独立 auditor 不
import generator/rasterizer，重新实现 Q16 segment/cap/round-join sampling、Gray8 合成、support
certificate、图同构签名与 commitment framing。结果：

```text
Pack A observation / invocation       48 / 72
Pack B observation / invocation       48 / 72
Pack A byte-exact replay              PASS
Pack A/B PNG SHA overlap              0
C2/C3/C4 support certificates         12/12 independently exact
parallel lane separation failures     0
C4 unenumerated junction witnesses    0
C4 single-stroke contract failures    0
artifact / schema / reference errors  0
semantic blockers                     0
Lean build targets                    15/15
generator + oracle tests              28/28
```

最终可复核 hash：

```text
schema bundle SHA256       0ec53107573e482bd7c9b3942d2ce89974648c903fc93ef01b3fb8b9773bea49
Pack A commitment SHA256   5eb7882e9096a933695fc8cbfd5fe2184213685c1b66b37c94895b532992df99
Pack B commitment SHA256   2303ae97b29f853628d9a559c3ca13746ec62e1555f5d62bb519da630891912b
generator source SHA256    362afbf9c19d8b9d4c87cb38666868cd2141969920a1055996cfcbc2195b421c
protocol bundle SHA256     7d90c89617bcae70bacd6d6013df3316122b6b2563f45a0ff4f3eb811fcd129f
```

第二轮结论为：`P1 APPROVE for the frozen public smoke profile`。这是 generator/oracle 协议的有限批准，
不是模型路线或训练批准。尚未覆盖 full 5,632 observation pack、正式隐藏 seed 与 custodian 流程、
sealed GT 加密、miter/bevel join、多 stroke/多字符 OCR、现实化学图片、epsilon/Delta 全量稳定性、
prediction replay 或 USPTO/MR-TADF benchmark。因此仍保持：

```text
overall_gate_status = no_go
training_authorized = false
```
