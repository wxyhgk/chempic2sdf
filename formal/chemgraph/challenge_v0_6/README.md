# Observation Recovery v0.6

本目录用于定义下一轮训练前的集合值恢复协议。v0.6 不把 OCR、BondObject 或化学先验当成
天然正确的中间真值，而是先回答两个问题：最终图片究竟支持哪些解释，以及证据不足时系统
应输出候选集合还是拒识。

当前状态是 `P0_CONTRACT_PRESENT`。权威机器合同和 closed schema 已落盘：

```text
formal/chemgraph/challenge_v0_6/prereg/p0_contract_v0_6.json
SHA256 e326fc744903b998793746f299232c4225ce3b8648e1abdda7b4e065f06a9402

formal/chemgraph/challenge_v0_6/schemas/p0_contract_v0_6.schema.json
SHA256 730ff356732fda5ef236d97bfcd3770af1d16a433dabb3d6c5ddf5977c682402
```

合同 schema version 为 `chemgraph-observation-closure-p0-v0.6`。源码冻结记录和正式结果尚未
出现；任何合同、schema、claim wording、资源、witness、sample、quotient bound 或 domain binding
变更都必须开新协议版本，不能在结果出来后静默修改。

`training_authorized=false`。v0.6 的形式化、生产像素绑定、分层诊断和可选全商闭包全部完成前，
不得继续模型训练，也不得用已有 checkpoint 的表现替代本协议。

P0 还固定继承完整 v0.5 父域：36,088,416 个 tuple、ID `0..36,088,415`，每个 `q` 的全部
`I(q)` interpretation 均在域内；`allow_exclusions=false`、`exclusion_count=0`。v0.6 不复制或
重定义父域，只按父合同 SHA
`55ddcaacdf851166554ccd586b2a924c0e793f86cc7a6319f98132ef0ffb6b65`
引用。不存在 active/post-hoc validity predicate。

## 1. v0.5 的正式结论

v0.5 的正式 P1 绑定源码提交
`fa796de982e9de63d4cff3015fbfa45317365e38` 和 source freeze
`148dc8b7a3bcb6c846b102dd1b0c3826fc60d3a562fcc4a898f182767ac8cba5`。

三路 census 各完整枚举 36,088,416 个 tuple，A-forward、A-reverse 和独立 B-shuffled 的
`records_root`、`coverage_root` 一致。1% Raster pilot 对固定规则
`tuple_id mod 100 == 0` 选出的 360,885 个 tuple 完成：

- 360,885 份 padded clipping certificate；
- 360,885 份 canonical frame；
- 360,885 次独立重渲染 full-byte replay；
- 351,759 个 distinct raster class；
- 16,699 个位于重复桶中的样本；
- 9,126 个超出每桶首成员的重复成员；
- 最大 raster class 大小为 7。

v0.5 P1 的正式状态是 `NO-GO`。失败门只有：

- pilot wall 上界 16,876 秒，大于 14,400 秒；
- 全量 wall 双倍投影 3,375,193 秒，大于 345,600 秒。

权威 artifact 位于 root-owned 只读目录
`/opt/chemgraph/results/p1-raster-pilot-fa796de`。仓库中的
`challenge_v0_5/p1_frozen_fa796de/pilot_receipt.json` 只是绑定索引，不替代权威
`validation.json` 和 `resource_validation.json`。

这些结果只证明冻结采样域中存在 exact-raster 重复，并证明 v0.5 的全量计算路线超过资源门。
它们没有构造完整 RasterGraphSet，不把“不同 tuple”自动等同为“不同 Graph”，也不允许从
sampled singleton 推出 full singleton。因此 v0.5 既没有签发 full Graph 结论，也没有授权训练。

## 2. 第一性原理对象

设：

```text
D                 合法生产场景域
q : Q             决定最终可见绘制的 render key
i : Interpretation(q)
                  同一个 q 下允许的语义解释
R(q)              最终 production raster/full bytes
G(q, i)           对应的带标签化学图
~=                冻结的图同构/化学等价关系
E(R)              只从最终 raster 提取的集合值视觉证据
C(E)              evidence-only 候选图同构类集合
```

OCR text mask、token、anchor 和 BondObject 只要在生产中可用，就必须是最终 raster 的确定性输出
或集合值输出。若它们读取 SDF、Render JSON、源 Atom/Bond ID、GT 坐标、family、tuple ID 或
collision class，它们是 Oracle，不是可观察证据。

即使 OCR 和 BondObject 都是确定性函数，增强观测仍只有：

```text
O(q) = (R(q), OCR(R(q)), BondObjects(R(q))) = F(R(q))
```

所以若存在同一合法 `q` 的两个解释 `i1`、`i2`，且
`G(q,i1) !~= G(q,i2)`，任何 `O -> Graph` 的确定性函数都不可能同时正确。增加参数、训练图片、
MoE expert 或后处理规则都不能恢复已经不在观测中的信息。

## 3. 集合值恢复协议

对 exact raster，定义真实商候选集：

```text
Q0(r) = { [G(q, i)]/~= |
          q 属于合法域，R(q) = r，i 属于 Interpretation(q) }
```

对带扰动输入，可在冻结距离 `d` 和半径 `epsilon` 下定义：

```text
Qepsilon(o) = { [G(q, i)]/~= |
                q 属于合法域，d(R(q), o) <= epsilon，
                i 属于 Interpretation(q) }
```

生产解码必须遵守：

```text
|Q| = 0          -> Reject(out_of_domain 或 extract_failure)
|Q| = 1          -> Graph(唯一同构类代表)
2 <= |Q| <= K    -> Ambiguous(显式去同构候选集合和 taxonomy)
|Q| > K          -> Reject(candidate_overflow)
```

这里的“最优”不是最高 softmax 或最常见先验，而是在集合包含意义下最小且同时 sound、complete 的
商候选集合。若业务必须从多个候选中给出单答案，只能另行冻结 prior、损失和 tie-break，把结果称为
prior-resolved 或 Bayes decision；不得称为普适精确逆映射。

相同最终 raster 或相同 EvidenceClass 必须得到相同候选集合和相同三态决策。禁止按源数据集、
样本 ID、GT Graph 或模型是否恰好答对来选择性 Reject。

## 4. 四条固定验证 lane

v0.6 固定 E1、E2、S、Q 四条 lane。它们回答不同问题，不能互相替代。

| Lane | `claim_kind` | 允许声称的范围 |
|---|---|---|
| E1 | `exact_full_domain_existential_same_tuple` | 完整冻结 scene 域至少存在一个同 q 异图解释对，因此不存在全域精确确定性 decoder |
| E2 | `exact_full_domain_existential_cross_tuple_full_bytes` | 完整冻结 tuple 域包含 8 个预注册跨 tuple full-byte 异图碰撞对，因此 observation map 非单射 |
| S | `preregistered_stratified_finite_population_statistics` | 只报告冻结 predicate、strata 和 simultaneous confidence 下的有限总体统计界，不是 full closure 或全局碰撞率 |
| Q | `exact_full_closure_over_proved_quotient` | 仅在全部 Q obligation 和资源门通过后，报告完整父域经已证明 K 得到的 Raster/Evidence/Graph closure |

合同中的英文 `permitted_wording` 是最终权威。任何 lane 失败只使该 lane `NO-GO`，
`cross_lane_promotion_forbidden=true`；不得用 E1/E2 的存在反例、S 的统计结果或 Q 的可选状态提升
另一个 lane 的结论。

### 4.1 E1：Lean 构造性全域存在证明

E1 在 Lean 中构造：

```text
存在合法 q、i1、i2，满足
i1 != i2
G(q, i1) !~= G(q, i2)
且 render 只依赖 q，不读取 interpretation。
```

P0 已冻结具体 witness：

```text
family                  c1
half_length             85
motif_ordinal           25
width_px                5
cap                     round
antialias_profile       box_2x2
reflection              none
rotation_index          0
scale_index             3
interpretation_indices  [0, 1]
```

由此证明不存在对该合法域全正确的确定性 `raster -> Graph` 左逆，并证明任何可观察 OCR/Bond
field 都必须在该碰撞上相同，或承认它不能 factor through raster。

E1 必须是构造性 witness，不得通过空域、不可达 `q`、恒假 validity、退化图等价关系或把
`Interpretation` 偷塞进 `Raster` 来真空通过。Lean 证明接口与逻辑结论，不证明 Python producer
真的生成了相同 PNG，也不证明现实图片遵守该有限语法；后者由 E2 绑定。

### 4.2 E2：生产像素绑定

E2 预注册 8 个 S8/C4 跨 tuple full-byte collision witness。每个 witness 必须由：

1. 冻结 producer 生成；
2. 独立 verifier A 从源 tuple 重建并比较 full bytes；
3. 不导入 producer/verifier A 的独立 verifier B 再次重建。

8 对 witness 由以下冻结 Cartesian product 组成：

```text
共同 antialias_profile  binary_1x
caps                    [butt, square]
width_px                [2, 4, 6, 8]

left                    half_length=62, motif_ordinal=11,
                        reflection=none, rotation_index=0,
                        scale_index=2, interpretation_index=0
right                   half_length=93, motif_ordinal=48,
                        reflection=none, rotation_index=4,
                        scale_index=0, interpretation_index=0
```

三者必须同时确认：

- 两个 tuple 都属于冻结合法域；
- tuple 不同且 interpretation/GraphSet 要求不同；
- final Gray bytes、canonical PNG bytes 和 production preprocess bytes 全部相同；
- hash 仅作索引，相同 hash 桶最终比较长度和全部 bytes；
- OCR text/lane 语义冲突被并入同一个 raster observation class；
- 没有 first-seen、retry、去重或 family filter 丢弃后到解释。

8/8 通过只绑定这 8 个生产像素反例，不表示已枚举全部碰撞。任一个 witness 失败，E2 为
`NO-GO`；不得删除 witness、改变坐标、线宽、字体或预处理来保住结论。

E2 与 Q 至少各需要两个独立 verifier，producer 不得与 verifier 共享实现；至少一个 verifier
必须使用不同于 producer 的实现语言或 runtime。运行时禁止网络。共享冻结合同是唯一允许的共享
语义来源。

### 4.3 S：分层统计诊断

S lane 固定 108 个 strata：

```text
12 motif family x 3 width bucket x 3 antialias bucket = 108 strata
```

每层无放回抽取 64 个 tuple，总计：

```text
108 x 64 = 6,912
```

抽样必须使用 commit/reveal seed 和冻结 hash-priority。顺序固定为先提交生成器、schema、分层映射、
hash framing 和 seed commitment，再 reveal seed 并选择每层优先级最低的 64 个不同 tuple。层不足、
重复、拒绝后补抽、换 seed 或在看结果后改 bucket 都是 `NO-GO`。

seed derivation 必须由父合同 hash、v0.6 合同 hash 和两份 32-byte custodian share 做带 framing 的
SHA256；两份 share 在 source freeze 前承诺，sample implementation 冻结后才 reveal。统计置信度固定
为 99%（`confidence_level_ppm=990000`），使用对每个预注册 predicate 和每个 stratum 的
Bonferroni simultaneous finite-population bounds。

S 必须逐层报告：

- invocation、成功、failure、Reject 数；
- distinct raster/evidence class；
- duplicate member、超首成员、最大桶；
- Graph/Ambiguous/Reject 与候选集基数；
- OCR token/mask/anchor 和 BondObject candidate 的 soundness/completeness；
- 文字/键重叠、crossing、multiplicity、stereo、charge、unknown label 分桶；
- A/B replay 和 full-byte collision 证书。

当前 P0 只冻结了 seed 成分和统计边界，没有在 JSON 中枚举 hash-priority 的精确 framing、rank、
tie-break，也没有枚举上述 OCR/Bond/candidate predicate 的机器 ID、分母和公式。因此 S 在执行前
仍需一个早于 sample implementation/source freeze 的 closed sampler/predicate preregistration，并绑定
到 artifact DAG。若该后续记录缺失，不能用自然语言列表代替，S 必须保持 `not_executed`。后续记录
不得改变 P0 的 strata、样本量、seed 成分、置信度或 familywise method。

S 是诊断，不是 full quotient closure。S 中发现一个同像异图 witness 可以否定全域单射；S 中没有
发现碰撞不能证明全域单射，S 中的 singleton 也不能签发 full Graph。禁止用均值掩盖任一 strata
失败，禁止把 6,912 个样本称为完整域。

### 4.4 Q：可选 full quotient closure

Q lane 不是默认必跑。只有先证明下列全部前提，才允许执行：

```text
pi : D -> K 是满射；
每个 cell(k) 非空；
不同 cell 互斥；
全部 cell 覆盖 D；
render 在 pi 上满足冻结 factorization；
每个 cell 的 Interpretation union 完备；
每个 cell 的 Graph quotient union 完备。
```

P0 冻结 `maximum_quotient_cardinality=100000`。若证明或枚举得到的 `|K|` 超过该值，必须在代表
raster 渲染前拒绝 Q；不得合并 cell、采样 K、缩轴或在 v0.6 提高上限。

Q 的 builder 必须按完整 raster bytes 建类，union 同类中的全部 interpretation，再按冻结图等价关系
去同构。A/B builder 不共享 class builder、序列化或图 canonicalizer，第三 verifier 独立重建固定
witness 和 hash-priority 样本。

若上述任一证明缺失、cell 数量超资源、candidate cardinality 溢出或 full projection 超预算，Q
直接 `NO-GO/not_executed`。禁止用 S 的 sampled closure、E2 的 8 个 witness 或某个 family 的 closure
冒充 Q。

## 5. 必须覆盖的最小反例族

v0.6 的公开和密封挑战至少覆盖以下反例。它们不是可以通过训练忽略的“噪声”，而是用于检查
表示是否诚实。

### 5.1 Crossing 与 junction

两条 Bond 在内部交叉可以与四条 Bond 在中心隐式 C 终止得到同一墨迹。前者为 4 Atom/2 Bond，
后者为 5 Atom/4 Bond。没有可见 crossing marker、terminal 或公开缩域规则时，必须输出
`Ambiguous(crossing_or_junction)`。

### 5.2 双/三键与平行单键

两条或三条平行 lane 可解释为一个 order 2/3 Bond，也可解释为两/三根独立单键。BondObject
必须是集合值 grouping hypothesis；“投影对齐就合并”只能作为预注册 domain prior，不能成为全局真理。

### 5.3 Glyph 与 bond lane

`I`、`l`、`1`、`/`、`-`、`+` 的笔画可以与 Bond lane、charge 或 crossing 产生相同像素。
OCR mask 与 bond mask 不得强制 one-hot owner；同一像素允许多个候选归属。E2 的 S8/C4 witness
专门绑定这一反例。

### 5.4 量化和模糊导致的键级塌缩

当 lane gap 小于最终分辨率，单、双、三键可能在 resize、抗锯齿、阈值或模糊后得到同一 bytes。
没有 final-preprocess 白隙、方向和 multiplicity 稳定证书时，不得静默选择键级。

### 5.5 共线隐式碳

一根长 Bond 与两根首尾相接 Bond 加一个不可见 180 度二价 C 可以得到相同墨迹。若 production
grammar 没有预先禁止不可见共线二价点，就必须保留两个图候选。

### 5.6 文字遮挡和裁剪

不透明文字背景或图片边界可以隐藏 Bond terminal 和任意子图。若 ink 触边、文字 mask 内允许未绑定
terminal，或隐藏拓扑没有冻结上界，真实 fiber 可能超过 K 甚至无界，只能 Reject，不能截断 top-K
后声称 complete。

### 5.7 缩写、元素与未知标签

正确识别字符串仍不等于确定化学语义。例如：

- `B` 可表示硼或 Markush 占位符；
- `Ac`、`Pr` 可表示元素或缩写；
- `Ph`、`Me`、`CH3` 可表示 opaque alias 或展开子图；
- `R`、`X`、`A` 及任意字符串可能是 query/unknown label。

v0.6 默认输出 `OpaqueLabel(token)`，把缩写展开留给另一个有版本的协议。若 SDF dialect 无法无损
表达 opaque label，则该样本必须 Reject，不能猜成最接近元素。

### 5.8 Charge 与归属

`N-` 的减号可能与短 Bond 相同，`+` 可能与 crossing 相同，位于多个 Atom 之间的 charge 也可能有
多个 owner。token、owner relation 和 formal charge 必须分别建模；owner 不唯一时输出候选。

### 5.9 多字符 anchor

`Cl` 可与 `C` 加一条形似 `l` 的线冲突；Bond 进入 `CH3`、`NH2`、`Ph` mask 后，bbox 中心不是
化学连接点。anchor 应是 `BondPort <-> TextObject` 的候选 relation，不得直接使用源 SDF Atom 坐标。

### 5.10 Stereo

窄 wedge 可能退化为单线，hashed wedge 可能与短 dash 或文字冲突，文字还可能遮住 wedge 的宽窄端。
若图等价关系保留 stereo，候选必须同时保留方向、相关端口和邻居顺序；证据不足时输出
`Ambiguous(stereo)` 或 unspecified，不能静默当作普通单键。

### 5.11 表达协议等价

芳香键与 Kekule、同位素、形式电荷、断连分量、unknown label 和 stereo 是否参与 `~=` 必须在结果前
冻结。等价关系过粗会把真实错误藏掉，过细会把 Atom ID、Bond 顺序或等价 Kekule 形式误算为错误。

## 6. 必须拒识或输出 Ambiguous 的条件

满足任一条件时不得输出无条件 Graph：

1. 同 raster/evidence fiber 含两个或更多非同构 Graph；
2. OCR token、owner、anchor 的候选会改变 Graph；
3. Bond grouping、endpoint、order 或 stereo 的候选会改变 Graph；
4. ink 触边或遮挡区域允许不可枚举的隐藏拓扑；
5. 完备候选类数超过 K；
6. extractor failure、schema overflow、unsupported label/SDF dialect；
7. 扰动半径与异图最小间隔未满足 `2 * epsilon < Delta`；
8. 输入不属于冻结合法域；
9. 候选 completeness 未证明；
10. evidence 不能证明 factor through final raster。

Reject 必须带机器可验证 reason。Ambiguous 必须携带显式去同构候选或绑定候选 artifact，不允许用
一个置信度数字代替。全 Reject 虽可能满足弱 soundness，但不能通过 v0.6 的 coverage 和反真空门。

## 7. 禁止的范围偷换

以下做法一律使当前版本 `NO-GO`：

- 把 SDF/Render JSON 的 text、anchor、Bond owner 当成 raster observable；
- 先看 collision，再修改 valid domain、prior、taxonomy、K 或 equivalence；
- first-seen、hash-only 去重或 duplicate retry，导致后到 interpretation 丢失；
- 只在一个 motif family 内计算 completeness，不 union 跨 family observation class；
- 用 digest 相等代替 full-byte 相等；
- 忽略 bond order、charge、isotope、stereo、component 或 opaque label 来制造 singleton；
- 用 Atom/Bond 列表顺序或 source ID 制造伪不同图；
- 从 sampled singleton 推出 full singleton；
- 把 E1 的接口定理写成 Python/PNG 正确性；
- 把 E2 的 8 个 witness 写成完整 collision census；
- 把 S 的 6,912 样本写成 Q closure；
- 固定 K 截断后仍声称 CandidateSet complete；
- 把 MAP、softmax top-1 或 RDKit sanitize 成功称为 exact inverse；
- 删除 failure/crash/timeout 样本或从 denominator 移除 Reject；
- 在 sealed test 后选 threshold、prior、checkpoint 或 seed。

## 8. Lean 可以与不可以证明什么

Lean 可以证明：

- E1 的合法 `q`、两个 interpretation 与异图 witness；
- render 只依赖 `q` 时的 deterministic inverse 不可能性；
- 同 raster 的确定性 extractor 必须输出相同 evidence；
- candidate soundness/completeness 与真实 quotient coverage 的关系；
- singleton/multiple/empty 三态的互斥和条件穷尽；
- sampled ambiguity 到 full ambiguity 的单调性；
- sampled singleton 不推出 full singleton 的反例；
- `2 * epsilon < Delta` 下稳定唯一性的条件定理；
- Q lane 中有限映射、cell partition、union 和图同构关系的逻辑规格。

Lean 不能单独证明：

- Python/C++ producer、OCR、PNG decoder 或 rasterizer 实现正确；
- 两个真实 PNG/full tensor 的 bytes 相等；
- source freeze、Git 时序、容器、cgroup、ACL 或无网络隔离正确；
- S 抽样器真的使用了承诺 seed；
- 某个神经网络在未知图片上具有经验准确率；
- 有限 grammar 覆盖现实 OCSR 分布；
- Oracle text/bond field 能从 raster 恢复，除非另给 factorization 证明。

因此每个 Lean 定理必须列出假设和可执行绑定；不得把 `oracleField -> Graph` 左逆误写成
`raster -> oracleField -> Graph` 已解决。

## 9. GO / NO-GO

v0.6 的 GO 是合取门，不使用平均分补偿失败项。

1. 机器合同和 closed schema 先于 producer/extractor 实现提交；
2. E1 构造性 witness 通过，关键定理无 `sorry/admit/axiom`；
3. E2 的 8/8 跨 tuple witness 经 producer 和两个独立 verifier full-byte 一致；
4. S 的 108 strata 各恰好 64 个不同 tuple，总计 6,912，commit/reveal 可重放；
5. S 中所有同 observation 解释先 union，错误 Graph 输出为 0；
6. Evidence 只读 final raster，禁止字段、依赖闭包和 deterministic replay 全通过；
7. CandidateSet 按图同构类同时 sound、complete，overflow 明确 Reject；
8. Graph/Ambiguous/Reject 三态与候选基数一致，Reject 不从 denominator 删除；
9. crossing、multiplicity、glyph/lane、occlusion、abbreviation、charge、anchor、stereo 反例均通过；
10. 图等价、canonical bytes 和两套独立 graph verifier 一致；
11. 资源、wall、memory、temporary、FD、进程和 artifact publication 门全部通过；
12. 至少两名未参与实现的 reviewer 批准源码依赖、结果和范围声明；
13. `training_authorized=false`。

上述第 5--10 项中涉及 OCR/BondObject/Candidate/三态输出的执行字段，尚未由当前 P0 JSON 的 closed
schema 机器化；它们是集合值恢复系统进入后续版本前必须满足的要求，不得声称是 P0 已经验证的
claim lane。v0.6 P0 当前只授权验证 E1、E2、S、Q 四种合同 wording。

Q lane 只有在第 4.4 节的全部 partition/factorization/union 前提和独立资源门通过后，才能另行报告
`full_quotient_closure_passed`。S GO 不能自动让 Q GO；v0.6 的反例确认也不等于真实分子 OCR 已经可用。

任一合取项失败，本版本只能 `NO-GO` 或保持 `not_executed`。不得在同一版本缩域、降 quota、换 seed、
删反例、放宽 K 或追加训练后重测。

## 10. 资源预算

P0 已逐 lane 冻结以下硬上限：

| Lane | Worker | Wall | Peak RSS | Temporary | Authoritative artifact | OOM |
|---|---:|---:|---:|---:|---:|---:|
| E1 | 4 | 600 秒 | 4 GiB | 2 GiB | 2 GiB | 0 |
| E2 | 4 | 1,800 秒 | 8 GiB | 16 GiB | 2 GiB | 0 |
| S | 16 | 3,600 秒 | 16 GiB | 64 GiB | 16 GiB | 0 |
| Q | 16 | 14,400 秒 | 32 GiB | 256 GiB | 128 GiB | 0 |

所有 lane 禁止 GPU，worker 数必须与所选 lane 完全一致。任一超限、任一 OOM event、非零退出、
缺失 selected record、verifier disagreement 或 unsafe artifact tree 都使该 lane `NO-GO`。

资源计量必须覆盖完整进程树和 publication：专用 cgroup aggregate peak、OOM events、空文件系统上的
peak allocation、从首 worker 到最终 fsync/atomic receipt 的 wall。hash、replay、verifier 和失败日志都计入。

v0.5 已实证 36,088,416 tuple 的原路线双倍投影约 937.6 小时，既远超 v0.5 的 96 小时门，
也远超 v0.6 Q 的 4 小时门。因此 v0.6 不得原样重跑该全域再期待硬件解决。Q 若要执行，必须先用
有证明的 quotient key `pi` 降低重复计算，同时保持满射、
cell partition、render factorization 和 interpretation/Graph union 完备；否则 Q 维持 `not_executed`。

资源 pilot 通过只表示计算可承受，不表示图可辨识。资源 pilot 失败也不能通过抽掉困难 strata 或 collision
witness 修复，只能开新版本重新声明域和预算。

## 11. 当前决定

当前只允许：

- 完成 v0.6 机器合同、schema 和 preregistration；
- 完成 E1 Lean 定理与无公理审计；
- 实现 E2 producer/两个 verifier；
- 实现 S 的 commit/reveal 分层抽样和只读诊断；
- 证明或否决 Q 的前置条件与资源可行性。

当前禁止：

- 训练 OCR、BondObject、Graph decoder 或端到端模型；
- 生成大规模新训练图片；
- 用 MR-TADF、USPTO 或已暴露 benchmark 选择 v0.6 规则；
- 把 Oracle layer 成功写成生产恢复成功；
- 发布“已经解决化学 OCR”或“可开始训练”的结论。

P0 明确规定：v0.6 的 E1、E2、S、Q 任何结果都不授权训练，且不同 lane 的 PASS 不得提升另一个
lane 的状态。训练若要恢复，必须由 v0.6 之后的新协议版本另行预注册并明确授权；本文自身以及
v0.6 的任何 PASS 都不具备授权效力。
