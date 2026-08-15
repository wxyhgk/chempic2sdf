# ChemGraph 形式化工程

这个目录用 Lean 4 检查“图片到化学图”路线中最基础的数学前提。它不是模型实现，也不以通过几个定理来替代真实 benchmark。

本轮机器验收见 `validation.json`。当前状态是 `evidence_core_v0_4_p3_public_no_go`：v0.4 已实现
有界中性 EvidenceCore、stdlib-only Candidate、独立图同构 verifier 和三态解码合同。公开 primitive
geometry 为 48/48，但 CandidateSet 只有 44/48；C4 与 S8 存在同 raster 异标签反例，因此没有运行
正式 hidden challenge，`training_authorized=false`。详细结果见
`docs/model/formal/20260815-evidence-core-v0_4-p3-results.md`。

## 当前结论

第一版先回答唯一性问题。把不同语义对象的墨迹直接做像素并集后，中心四价原子的十字和两根互不相连键的交叉可以得到完全相同的观测，但两者化学图不同。因此：

```text
存在 s1 != s2，使 Render(s1) = Render(s2) 且 Graph(s1) != Graph(s2)
=> 不存在对所有场景都正确的确定性 Decode
```

这个结论与参数量、训练图片数量和优化器无关。遇到这种样本只能：

1. 从渲染中保留更多可观察证据；
2. 用明确先验缩小允许场景集合；
3. 输出多候选或“不可判定”；
4. 从 strict exact 测试集剔除，并保留冲突审计记录。

`Recovery.lean` 还验证了一个很小的 Oracle 结果：对于只含“十字交叉或中心 junction”的有限语法，如果外部已经给出每个语义 Bond 的墨迹分组，查表解码器能够区分两者。这里的 `BondGroupingOracle` 已包含关键语义分组，不是模型可直接读取的图片观测，也不是恢复成功的证明。

相反，`owned_observation_not_factor_through_opaque` 证明：在当前不透明四臂模型中，不存在一个只读 `Ink` 的确定性函数，能够同时为 crossing 和 junction 生成正确分组。`grouping_oracle_not_sufficient_for_general_graph` 又证明，仅有分组仍会忘掉键级，不能恢复一般 ChemicalGraph。Two-Port 路线若要成立，必须从真实最终 PNG 中找到这个四臂抽象尚未表达的额外可观察证据，并显式预测 order、可见端口和文字实例；否则必须登记为歧义。

## 文件说明

- `ChemGraph/Scene.lean`：化学图、语义 Bond 和十字墨迹的有限模型。
- `ChemGraph/Render.lean`：丢弃所有权的普通渲染，以及保留 Bond 分组的观测规格。
- `ChemGraph/Identifiability.lean`：可辨识性、先验消歧、左逆充分条件和碰撞不可能定理。
- `ChemGraph/Stability.lean`：异图最小间隔、噪声半径与唯一解码定理。
- `ChemGraph/Grammar.lean`：精确线段几何、三层 grammar 信号与分类定理。
- `ChemGraph/ObservationRecovery.lean`：Graph/Ambiguous/Reject 解码协议、strict/ambiguity soundness、先验唯一性和 Oracle field 左逆边界。
- `ChemGraph/RasterEvidence.lean`：raster factorization、candidate exactness、保守解码和 epsilon/Delta 的条件式接口定理。
- `ChemGraph/EvidenceCore.lean`：P3 bounded Evidence、Evidence-only Candidate、图等价类 exactness 和三态 decoder 合同。
- `ChemGraph/Counterexamples.lean`：crossing 与 junction 的机器检查反例。
- `ChemGraph/Multiplicity.lean`：双键与两根平行单键的墨迹分组反例。
- `ChemGraph/Recovery.lean`：受限十字语法上的恢复定理。
- `ChemGraph/FiniteAudit.lean`：最小可执行碰撞分类器和 strict/ambiguity 回归测试。
- `ChemGraph/ObservationClosure.lean`：v0.5 的从属 tuple/interpretation 域、8 轴 Cartesian product 的无公理结构双射与基数乘积、Raster/Evidence 双层闭包、sampled/full 单调性、三态、候选 coverage 与 `NodupUpTo`、带标签简单图同构，以及 typed preprocess/Shift 合同。
- `ChemGraph/ConstrainedRecovery.lean`：笔画精确归属、Bond lane 成组、carrier 支撑边和 `2 * epsilon` 鲁棒分数间隔。
- `ChemGraph/ConstrainedRecoveryV02.lean`：具名 support、lane 级端点与 provenance、coverage/calibration 双证书和缺证拒绝边界。
- `ChemGraph/BlindCarrierTripleExpressivityV01.lean`：形式化 Blind Carrier v0.1 三条平行 lane 在固定 terminal 半径与最小间距下的合同不相容性。
- `ChemGraph/BlindCarrierTerminalRegionV02.lean`：形式化 v0.2 terminal region 修复，证明三条平行 lane 的相邻 endpoint disk 相交且交图 `0-1-2` 连通，不要求三圆公共交点。
- `ChemGraph/BlindCarrierTerminalComponentsV02.lean`：证明 v0.2 endpoint-disk 相交图在 terminal 内部连通且跨 terminal 无边时，terminal fiber 恰好是连通分量，且该分区唯一。
- `ChemGraph/CarrierCandidateSetV03.lean`：形式化 PNG proposer v0.3 的根级 boundary/pair/Atom/DirectPair 有限宇宙及 pair 端点归属、通用 interpretation 完备枚举、空图 interpretation、boundary/stroke 与 lane/Bond 精确 fiber、bond/text/charge ownership、`1..3` lane 键级、具名 reason/stage Reject，以及绑定组件 SHA256、实际 root artifact、前后输入输出和整链摘要的固定 12 阶段 trace。v0.3 序列化仅有 candidate set/reject，均不发出 graph；JSON 自签 `complete=true` 不代替 root-relative completeness 证书。Lean 把 schema 指定的 SHA256 framing 建模为抽象函数，具体哈希实现仍由独立 validator 审计。文件内 accept/abstain 定理只供未来升级，不属于当前序列化。
- `ChemGraph/CarrierCompletenessVerifierV03.lean`：给出独立 completeness verifier 的正反边界。正向证明 root compatible-pair list 的全部 canonical sublist 经 pairwise endpoint-disjoint 过滤后，恰好是全部规范 boundary matching；每个 matching 唯一决定 root boundary 顺序下的 unmatched complement。反向证明：若 fragment split、intersection、ownership、port 或 lane universe 没有进入 root，则同一 root record 可以对应不同 interpretation universe，任何只读 root 的 verifier 都不可能同时 sound 且 complete。该 NO-GO 不否定单独 completeness；文件同时证明“输出全部 Interpretation”的超集 verifier 总是 complete，说明没有 soundness 的 completeness 是真空门禁。
- `ChemGraph/SymbolicCandidateFamilyV04.lean`：形式化 v0.4 factorized symbolic candidate family 的容量动机。8 个无约束 Boolean variable domain 展开为 `2^8 = 256` 个无重复 assignment；完整 canonical materialized list 因此不可能满足 128 项容量，而 factorized CSP 只保存 8 个 `[false,true]` 域即可表示并展开为同一解族。该定理只证明表示容量差异，不证明真实图片必然出现 8 个相互独立歧义，也不证明 symbolic constraint extractor 正确。
- `ChemGraph/FragmentRootConstructorV04.lean`：形式化 root 中已经物化的 canonical ordered Cut 列表到 FragmentCell 的最小有限构造器。它在每个 Carrier 的 `start/cut/finish` delimiter 列表上取全部相邻 pair，证明 `n` 个 Cut 恰好产生 `n+1` 个 Cell、输出 Cell 当且仅当来自独立 `ListAdjacent` 关系中的相邻 delimiter、每个 root Carrier 的 fiber 非空，以及零 Cut 时唯一 Cell 为 `start -> finish`。这里的 completeness 只相对于输入 root delimiter 列表；该模块不包含 PNG、像素坐标、intersection extractor、support、primitive kind、ownership 或化学语义，不证明 Cut root 对真实图像完备。
- `ChemGraph/FragmentSupportPartitionV04.lean`：形式化已经给定的有限 parent Pixel 列表和 total `Pixel -> Fin childCount` bucket 所诱导的 filter fiber 分区。证明 Pixel 属于 child fiber 当且仅当它属于 parent 且 bucket 指向该 child；每个 parent Pixel 恰属一个 child；不同 child fiber 不相交；全部 child fiber 的集合并集精确覆盖 parent。该模块只证明给定 total bucket 的有限分区，不定义或证明 projection、PNG、坐标、support extractor、几何或语义正确性。
- `ChemGraph/GlobalOwnershipNoGoV04.lean`：把 Observation、World、total OwnerPartition 和包含 intersection support 的 OwnershipInterpretation 分开建模。条件式证明：若两份 legal World 具有同一 Observation、却要求不同 owner partition，则不存在对两者都 exact 的 root-only unique-owner constructor；若 intersection support 不同，也不能 exact 构造完整 OwnershipInterpretation。文件中的有限 Boolean witness 只证明该逻辑前提可满足，不是真实 PNG 或化学样本 witness；模块不定义 projection/support extractor，也不证明 `legal` predicate 与真实图像一致。

## 工具链与构建

工程固定 Lean `v4.33.0`，版本记录在 `lean-toolchain`，不跟随浮动版本。官方 Linux 工具链压缩包大小为 `574882764` 字节，官方 SHA256 为 `4b3fb03c29a1e0a253fb1d11f9bae3725f19a0dc6fc09b3ea16d2c9df3349e2c`。

```bash
source "$HOME/.elan/env"
cd /home/zhenghe/GK/Project/chempic2sdf/formal/chemgraph
lake build
```

## 数学门

在训练 Two-Port Ink Field 前，形式化和可执行验证必须依次通过：

1. 定义最终栅格输入上的等价关系，而不只比较矢量源文件。
2. 自动枚举空图、文字、单/双/三键、共线链、fan、环、交叉和 junction。
3. 找出所有“同一最终输入、非同构化学图”的碰撞并移入 ambiguity 集。
4. 在 strict 集上证明 Oracle 中间表示到 production decoder 为左逆。
5. 对端口交换、Bond 顺序、平移、旋转和缩放验证不变性或等变性。
6. 由未参与实现的 reviewer 使用未知 seed 连续审批两轮。

Grammar v0.2 还明确区分两类“prior”。`ResolvedByPrior` 中的 scene predicate
可以筛掉同一 observation 的某些候选解释；但 Lean 已证明，如果这个 predicate
在整个 observation class 内恒定且仍覆盖每个 base observation，它就不可能恢复
原本不可辨识的域。因此生产协议中的 prior 必须是对候选解释执行统一规则的
observation 级解析器，不能是按 Scene ID 或 GT 图查表的标签。对逻辑碰撞类，未激活
这种公开规则时只能输出 `Ambiguous` 或 `Reject`。

只满足 `AbstainingDecoderSound` 仍允许“全部 Reject”真空通过，因此策略 A 的实际门是
`ConservativeAbstainingDecoderCorrect`：strict class 必须输出其唯一 Graph，ambiguity
class 必须输出 `Ambiguous(kind)`，且 `kind` 必须满足单独给出的 taxonomy relation。

`ObservationPriorResolver := Observation -> Graph -> Prop` 是本轮采用的 class-level
抽象：相同 observation 必然得到相同候选规则；`ResolvedByObservationPrior` 另外要求
每个支持 observation 至少保留一个真实候选，并且只保留一个图签名。它仍然只是数学
协议，不意味着可以用 observation hash 查 GT，也不证明具体 lane/bundle/port 字段已可预测。

`ProductionPriorChooser` 进一步冻结生产接口为
`extract(observation) + candidates(evidence) + choose(evidence, candidates)`；其类型
不接收 Scene ID 或 `meaning`。`KeptByProductionChooser` 仅在事后用
`meaning(scene) = choose(...)` 检验保留了哪个解释。Lean 同时证明 factorization 门：若
两个碰撞场景要求不同 Oracle evidence，就不存在任何 `extract : Observation -> Evidence`
满足 `oracleEvidence = extract ∘ render`。这类字段不能作为“可学习真值”直接送入训练。
但函数类型本身仍无法禁止 `extract` 的实现内部按 observation hash 查 GT；这一禁止项必须
由可执行代码、数据依赖和独立 reviewer 审计，而不能声称已由 Lean 单独证明。
`ProductionCandidateSetExact` 还要求 evidence 派生的候选集合与该 observation 的真实图
候选恰好相等，防止 chooser 通过遗漏候选制造“唯一性”；这仍是待可执行审计验证的规格。

`ObservableOracleField` 把 factorization 与 field strict 串联起来：只有存在统一
`extract : Observation -> Field`，fieldStrict 才能推出 production observation 上的
strict（up to graph equivalence）。生产中的 `Observation` 必须实例化为完整 tensor/raster
对象，不能只用 SHA digest 代替；从 digest 提取字段除了查表没有可验证的视觉含义。

## 稳定恢复条件

仅仅没有发现完全相同的输入还不够。`Stability.lean` 把不同图的渲染当作观测空间中的码字。令异图之间的最小距离为 `Δ`，实际输入扰动上界为 `ε`；只有满足：

```text
2 * ε < Δ
```

时，Lean 定理 `robust_ball_has_unique_graph` 才保证同一个噪声球内的全部合法候选属于同一化学图。后续 production tensor 审计必须估计每个样本族的 `Δ`，并用小于 `Δ/2` 的拒识半径；否则只能输出多解或不确定，不能把“hash 没撞”误称为稳定。

`ObservationClosure.lean` 已形式化有限无向带标签简单图，以及由节点双射保持节点标签和边标签定义的图同构，并证明该关系为等价关系。可执行 Boolean checker 仍通过显式 `Decidable` 接口接入；Lean 证明 checker 的 sound/complete 规格，但不声称已验证 Python canonicalizer 的具体实现。

当前 Lean 模型仍只证明冻结接口之间的逻辑事实。它没有执行 3608 万 tuple 枚举，也不验证真实 rasterizer、extractor、PNG/JSON、syscall 隔离或 Git 时序。真实抗锯齿 PNG、文字遮挡、短键、双键 lane、虚线和任意分子的恢复也尚未证明。因此当前状态仍是 `FORMALIZATION_IN_PROGRESS`，不能只凭 Lean 接口定理开始大规模训练。

Carrier Candidate Set v0.3 最新加入的整数几何也仍在这个边界之外。`CarrierCandidateSetV03.lean`
证明 compatible/direct pair 的端点属于相应 root universe，但没有在 Lean 中重算以下具体公式；它们当前只由
`formal/chemgraph/audit/carrier_candidate_set_v0_3/validate_candidate_set.py` 和 mutation tests 保证：

- endpoint-to-Atom evidence 必须重复真实 Q4 endpoint/anchor；以“实际 endpoint 减另一端”为 outward、
  “Atom anchor 减实际 endpoint”为 attachment，精确重算 Q8 dot 与 absolute cross，并要求 dot 非负；
- attachment gate 使用 Q4 坐标的平方距离比较 `d2 <= radius_q4^2`，不能用先开方取整的距离代替；
  `distance_q4` 另按整数平方根向下取整，acute angle 按 half-up millidegrees 重算；
- compatible boundary pair 的 acute angle、对应端点逐坐标 half-up 中线、两端整数距离的 half-up 平均 separation、
  dominant-axis overlap 及相对较短 boundary 的 half-up ppm overlap 均必须从根级 boundary 重算；
- white-gap 必须绑定最终 PNG，引用有向相邻 fragment endpoint，span/长度与引用严格相等；全部 support pixel
  必须为白色并位于冻结宽度的 span 邻域内，且证书与 lane 的非接触相邻项形成精确 fiber；
- lane-separation 必须绑定 order-2/order-3 Bond 的规范相邻 lane，span 是两条 lane 外端点中点之间的线段，
  min/max 是两对外端点距离；每个相邻 lane pair 恰有一份证书，support 同样必须为白色且靠近该 span。

所以 Lean 当前没有证明 Python 的浮点角度实现跨平台一致，也没有证明 Q4/Q8 数值不溢出、RLE support
确实来自目标 PNG、gap 像素确实为空白或 compatible predicate 对真实图像具有召回率。这些仍是独立
validator、fixture 和真实公开图片实验的责任。

## 参考原则

- 逆问题必须先检查解的唯一性和稳定性；只有“存在预测函数”不够。
- 若渲染映射在语义图上不是单射，增加训练数据不能恢复已经被观测丢掉的信息。
- 图上的 crossing 和 vertex 是不同的组合结构；像素相交本身不能定义 Atom。
- Lean 官方文档：<https://lean-lang.org/documentation/>
- Arridge 等关于数据驱动逆问题的综述：<https://doi.org/10.1017/S0962492919000059>
- 关于多笔画图像中歧义区分析的工作：<https://doi.org/10.1016/j.eswa.2016.08.017>
