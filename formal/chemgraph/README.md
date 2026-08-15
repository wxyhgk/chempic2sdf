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

## 参考原则

- 逆问题必须先检查解的唯一性和稳定性；只有“存在预测函数”不够。
- 若渲染映射在语义图上不是单射，增加训练数据不能恢复已经被观测丢掉的信息。
- 图上的 crossing 和 vertex 是不同的组合结构；像素相交本身不能定义 Atom。
- Lean 官方文档：<https://lean-lang.org/documentation/>
- Arridge 等关于数据驱动逆问题的综述：<https://doi.org/10.1017/S0962492919000059>
- 关于多笔画图像中歧义区分析的工作：<https://doi.org/10.1016/j.eswa.2016.08.017>
