import ChemGraph.Identifiability

namespace ChemGraph

/-!
Grammar v0.2 不再要求每个 observation 都产生一张图。`Ambiguity` 和
`RejectReason` 是协议层类型；这里不预先固定其枚举值。
-/
inductive DecodeResult (Graph Ambiguity RejectReason : Type) where
  | graph (value : Graph)
  | ambiguous (kind : Ambiguity)
  | reject (reason : RejectReason)
deriving Repr

def ObservationSupported {Scene Observation : Type}
    (valid : Scene -> Prop) (render : Scene -> Observation)
    (observation : Observation) : Prop :=
  Exists fun scene => valid scene ∧ render scene = observation

def GraphSupportedAt {Scene Observation Graph : Type}
    (valid : Scene -> Prop) (render : Scene -> Observation)
    (meaning : Scene -> Graph) (observation : Observation) (graph : Graph) : Prop :=
  Exists fun scene =>
    valid scene ∧ render scene = observation ∧ meaning scene = graph

/-! `StrictObservation` 同时要求 observation class 非空且图签名唯一。 -/
def StrictObservation {Scene Observation Graph : Type}
    (valid : Scene -> Prop) (render : Scene -> Observation)
    (meaning : Scene -> Graph) (observation : Observation) : Prop :=
  Exists fun graph =>
    GraphSupportedAt valid render meaning observation graph ∧
      forall scene, valid scene -> render scene = observation -> meaning scene = graph

def AmbiguousObservationExact {Scene Observation Graph : Type}
    (valid : Scene -> Prop) (render : Scene -> Observation)
    (meaning : Scene -> Graph) (observation : Observation) : Prop :=
  Exists fun left => Exists fun right =>
    valid left ∧ valid right ∧
      render left = observation ∧ render right = observation ∧
        meaning left ≠ meaning right

def StrictObservationUpTo {Scene Observation Graph : Type}
    (valid : Scene -> Prop) (equivalent : Graph -> Graph -> Prop)
    (render : Scene -> Observation) (meaning : Scene -> Graph)
    (observation : Observation) : Prop :=
  Exists fun graph =>
    GraphSupportedAt valid render meaning observation graph ∧
      forall scene, valid scene -> render scene = observation ->
        equivalent (meaning scene) graph

theorem strict_observation_not_ambiguous
    {Scene Observation Graph : Type}
    (valid : Scene -> Prop) (render : Scene -> Observation)
    (meaning : Scene -> Graph) (observation : Observation)
    (strict : StrictObservation valid render meaning observation) :
    Not (AmbiguousObservationExact valid render meaning observation) := by
  intro ambiguous
  rcases strict with ⟨graph, _supported, unique⟩
  rcases ambiguous with ⟨left, right, leftValid, rightValid,
    leftObservation, rightObservation, different⟩
  apply different
  calc
    meaning left = graph := unique left leftValid leftObservation
    _ = meaning right := (unique right rightValid rightObservation).symm

/-!
无 prior 的 soundness 要求：一旦输出 `graph g`，该 observation class 中的
每个合法场景都必须有图 `g`。这比“输出图恰好是某个候选”更强，并正是禁止
在歧义类里偷偷挑一个 GT 的条件。
-/
structure AbstainingDecoderSound
    {Scene Observation Graph Ambiguity RejectReason : Type}
    (valid : Scene -> Prop) (render : Scene -> Observation)
    (meaning : Scene -> Graph)
    (decode : Observation -> DecodeResult Graph Ambiguity RejectReason) : Prop where
  graphSupported : forall observation graph,
    decode observation = .graph graph ->
      GraphSupportedAt valid render meaning observation graph
  graphUniversal : forall observation graph,
    decode observation = .graph graph ->
      forall scene, valid scene -> render scene = observation -> meaning scene = graph
  ambiguityHonest : forall observation kind,
    decode observation = .ambiguous kind ->
      AmbiguousObservationExact valid render meaning observation

/-!
仅有 soundness 允许 `decode := Reject` 真空通过。策略 A 还必须满足 strict
completeness，并为每个 ambiguity class 输出带正确 taxonomy 的 `Ambiguous`。
-/
structure ConservativeAbstainingDecoderCorrect
    {Scene Observation Graph Ambiguity RejectReason : Type}
    (valid : Scene -> Prop) (render : Scene -> Observation)
    (meaning : Scene -> Graph)
    (ambiguityKindAt : Observation -> Ambiguity -> Prop)
    (decode : Observation -> DecodeResult Graph Ambiguity RejectReason) : Prop where
  sound : AbstainingDecoderSound valid render meaning decode
  strictComplete : forall observation,
    StrictObservation valid render meaning observation ->
      Exists fun graph => decode observation = .graph graph
  ambiguityComplete : forall observation,
    AmbiguousObservationExact valid render meaning observation ->
      Exists fun kind => decode observation = .ambiguous kind
  ambiguityKindSound : forall observation kind,
    decode observation = .ambiguous kind -> ambiguityKindAt observation kind

theorem conservative_strict_decodes_unique_graph
    {Scene Observation Graph Ambiguity RejectReason : Type}
    (valid : Scene -> Prop) (render : Scene -> Observation)
    (meaning : Scene -> Graph)
    (ambiguityKindAt : Observation -> Ambiguity -> Prop)
    (decode : Observation -> DecodeResult Graph Ambiguity RejectReason)
    (correct : ConservativeAbstainingDecoderCorrect
      valid render meaning ambiguityKindAt decode)
    (observation : Observation)
    (strict : StrictObservation valid render meaning observation) :
    Exists fun uniqueGraph =>
      decode observation = .graph uniqueGraph ∧
        forall scene, valid scene -> render scene = observation ->
          meaning scene = uniqueGraph := by
  rcases strict with
    ⟨uniqueGraph, ⟨witness, witnessValid, witnessObservation, witnessMeaning⟩,
      unique⟩
  have strictAgain : StrictObservation valid render meaning observation :=
    ⟨uniqueGraph,
      ⟨witness, witnessValid, witnessObservation, witnessMeaning⟩, unique⟩
  rcases correct.strictComplete observation strictAgain with ⟨graph, decoded⟩
  have graphIsUnique : graph = uniqueGraph := by
    calc
      graph = meaning witness :=
        (correct.sound.graphUniversal observation graph decoded witness
          witnessValid witnessObservation).symm
      _ = uniqueGraph := witnessMeaning
  refine ⟨uniqueGraph, ?_, unique⟩
  simpa [graphIsUnique] using decoded

theorem conservative_ambiguity_decodes_honest_kind
    {Scene Observation Graph Ambiguity RejectReason : Type}
    (valid : Scene -> Prop) (render : Scene -> Observation)
    (meaning : Scene -> Graph)
    (ambiguityKindAt : Observation -> Ambiguity -> Prop)
    (decode : Observation -> DecodeResult Graph Ambiguity RejectReason)
    (correct : ConservativeAbstainingDecoderCorrect
      valid render meaning ambiguityKindAt decode)
    (observation : Observation)
    (ambiguous : AmbiguousObservationExact valid render meaning observation) :
    Exists fun kind =>
      decode observation = .ambiguous kind ∧ ambiguityKindAt observation kind := by
  rcases correct.ambiguityComplete observation ambiguous with ⟨kind, decoded⟩
  exact ⟨kind, decoded, correct.ambiguityKindSound observation kind decoded⟩

theorem strict_graph_output_is_unique
    {Scene Observation Graph Ambiguity RejectReason : Type}
    (valid : Scene -> Prop) (render : Scene -> Observation)
    (meaning : Scene -> Graph)
    (decode : Observation -> DecodeResult Graph Ambiguity RejectReason)
    (sound : AbstainingDecoderSound valid render meaning decode)
    (observation : Observation)
    (strict : StrictObservation valid render meaning observation)
    (graph : Graph) (decoded : decode observation = .graph graph) :
    Exists fun uniqueGraph =>
      graph = uniqueGraph ∧
        forall scene, valid scene -> render scene = observation ->
          meaning scene = uniqueGraph := by
  rcases strict with
    ⟨uniqueGraph, ⟨witness, witnessValid, witnessObservation, witnessMeaning⟩,
      unique⟩
  refine ⟨uniqueGraph, ?_, unique⟩
  calc
    graph = meaning witness :=
      (sound.graphUniversal observation graph decoded witness
        witnessValid witnessObservation).symm
    _ = uniqueGraph := witnessMeaning

theorem ambiguous_observation_forbids_graph_output
    {Scene Observation Graph Ambiguity RejectReason : Type}
    (valid : Scene -> Prop) (render : Scene -> Observation)
    (meaning : Scene -> Graph)
    (decode : Observation -> DecodeResult Graph Ambiguity RejectReason)
    (sound : AbstainingDecoderSound valid render meaning decode)
    (observation : Observation)
    (ambiguous : AmbiguousObservationExact valid render meaning observation) :
    forall graph, decode observation ≠ .graph graph := by
  intro graph decoded
  rcases ambiguous with ⟨left, right, leftValid, rightValid,
    leftObservation, rightObservation, different⟩
  apply different
  calc
    meaning left = graph :=
      sound.graphUniversal observation graph decoded left leftValid leftObservation
    _ = meaning right :=
      (sound.graphUniversal observation graph decoded right rightValid rightObservation).symm

theorem ambiguous_observation_must_abstain
    {Scene Observation Graph Ambiguity RejectReason : Type}
    (valid : Scene -> Prop) (render : Scene -> Observation)
    (meaning : Scene -> Graph)
    (decode : Observation -> DecodeResult Graph Ambiguity RejectReason)
    (sound : AbstainingDecoderSound valid render meaning decode)
    (observation : Observation)
    (ambiguous : AmbiguousObservationExact valid render meaning observation) :
    (Exists fun kind => decode observation = .ambiguous kind) ∨
      Exists fun reason => decode observation = .reject reason := by
  cases decoded : decode observation with
  | graph graph =>
      exact False.elim
        (ambiguous_observation_forbids_graph_output
          valid render meaning decode sound observation ambiguous graph decoded)
  | ambiguous kind => exact Or.inl ⟨kind, rfl⟩
  | reject reason => exact Or.inr ⟨reason, rfl⟩

/-! 同一输入只有一个函数值，因此两个异图答案不可能同时正确。 -/
theorem same_observation_different_graph_forbids_two_graph_answers
    {Scene Observation Graph Ambiguity RejectReason : Type}
    (render : Scene -> Observation) (meaning : Scene -> Graph)
    (left right : Scene)
    (sameObservation : render left = render right)
    (differentMeaning : meaning left ≠ meaning right) :
    Not (Exists fun decode : Observation -> DecodeResult Graph Ambiguity RejectReason =>
      decode (render left) = .graph (meaning left) ∧
        decode (render right) = .graph (meaning right)) := by
  intro existsDecoder
  rcases existsDecoder with ⟨decode, leftCorrect, rightCorrect⟩
  apply differentMeaning
  have sameResult : decode (render left) = decode (render right) :=
    congrArg decode sameObservation
  rw [leftCorrect, rightCorrect] at sameResult
  injection sameResult

/-!
旧版 `ResolvedByPrior` 已把两项真正需要的门写入定义：每个 base observation
至少保留一个解释，以及所有被保留解释在同一 observation 下图签名唯一。
下面把这两项展开成可直接用于解码证明的 witness。
-/
theorem resolvedByPrior_has_nonempty_unique_survivor
    {Scene Observation Graph : Type}
    (base prior : Scene -> Prop) (render : Scene -> Observation)
    (meaning : Scene -> Graph)
    (resolved : ResolvedByPrior base prior render meaning)
    (source : Scene) (sourceBase : base source) :
    Exists fun kept =>
      base kept ∧ prior kept ∧ render kept = render source ∧
        forall other, base other -> prior other -> render other = render source ->
          meaning other = meaning kept := by
  rcases resolved with ⟨_baseAmbiguous, coverage, unique⟩
  rcases coverage source sourceBase with
    ⟨kept, keptBase, keptPrior, keptObservation⟩
  refine ⟨kept, keptBase, keptPrior, keptObservation, ?_⟩
  intro other otherBase otherPrior otherObservation
  exact unique other kept ⟨otherBase, otherPrior⟩ ⟨keptBase, keptPrior⟩
    (otherObservation.trans keptObservation.symm)

theorem resolvedByPrior_roundtrip_is_sound
    {Scene Observation Graph Ambiguity RejectReason : Type}
    (base prior : Scene -> Prop) (render : Scene -> Observation)
    (meaning : Scene -> Graph)
    (decode : Observation -> DecodeResult Graph Ambiguity RejectReason)
    (resolved : ResolvedByPrior base prior render meaning)
    (roundtrip : forall scene, base scene -> prior scene ->
      decode (render scene) = .graph (meaning scene))
    (source : Scene) (sourceBase : base source) :
    Exists fun graph =>
      decode (render source) = .graph graph ∧
        forall other, base other -> prior other -> render other = render source ->
          meaning other = graph := by
  rcases resolvedByPrior_has_nonempty_unique_survivor
      base prior render meaning resolved source sourceBase with
    ⟨kept, keptBase, keptPrior, keptObservation, unique⟩
  refine ⟨meaning kept, ?_, ?_⟩
  · rw [← keptObservation]
    exact roundtrip kept keptBase keptPrior
  · intro other otherBase otherPrior otherObservation
    exact unique other otherBase otherPrior otherObservation

/-!
这是一个重要的建模边界：如果 scene predicate `prior` 在整个 observation
class 内恒定，那么 coverage 会迫使每个 base scene 都满足 prior，它便不可能
通过删除碰撞解释来恢复可辨识性。真正的“只依赖 observation 的 prior”应建模
为 observation 到候选图集合的统一解析规则，而不是伪装成 class 内恒定的
scene 布尔标签。
-/
def ScenePriorObservationInvariant {Scene Observation : Type}
    (prior : Scene -> Prop) (render : Scene -> Observation) : Prop :=
  forall left right, render left = render right -> (prior left ↔ prior right)

theorem observation_invariant_prior_with_coverage_keeps_base
    {Scene Observation : Type}
    (base prior : Scene -> Prop) (render : Scene -> Observation)
    (coverage : PriorCoversBaseObservations base prior render)
    (invariant : ScenePriorObservationInvariant prior render) :
    forall scene, base scene -> prior scene := by
  intro scene sceneBase
  rcases coverage scene sceneBase with
    ⟨kept, _keptBase, keptPrior, keptObservation⟩
  exact (invariant kept scene keptObservation).mp keptPrior

theorem resolvedByPrior_cannot_be_scene_observation_invariant
    {Scene Observation Graph : Type}
    (base prior : Scene -> Prop) (render : Scene -> Observation)
    (meaning : Scene -> Graph)
    (resolved : ResolvedByPrior base prior render meaning) :
    Not (ScenePriorObservationInvariant prior render) := by
  intro invariant
  rcases resolved with ⟨baseNotIdentifiable, coverage, keptIdentifiable⟩
  apply baseNotIdentifiable
  intro left right leftBase rightBase sameObservation
  have leftPrior := observation_invariant_prior_with_coverage_keeps_base
    base prior render coverage invariant left leftBase
  have rightPrior := observation_invariant_prior_with_coverage_keeps_base
    base prior render coverage invariant right rightBase
  exact keptIdentifiable left right
    ⟨leftBase, leftPrior⟩ ⟨rightBase, rightPrior⟩ sameObservation

/-!
生产 prior 的正确抽象是 observation 到“允许候选图”的函数。它对同一
observation 自动给出同一规则，却仍可在该 class 的多个候选图之间作统一选择。
-/
abbrev ObservationPriorResolver (Observation Graph : Type) :=
  Observation -> Graph -> Prop

def ResolvedByObservationPrior {Scene Observation Graph : Type}
    (valid : Scene -> Prop) (render : Scene -> Observation)
    (meaning : Scene -> Graph)
    (resolver : ObservationPriorResolver Observation Graph) : Prop :=
  forall observation, ObservationSupported valid render observation ->
    Exists fun graph =>
      GraphSupportedAt valid render meaning observation graph ∧
        resolver observation graph ∧
          forall otherGraph,
            GraphSupportedAt valid render meaning observation otherGraph ->
              resolver observation otherGraph -> otherGraph = graph

theorem observation_prior_is_class_level
    {Scene Observation Graph : Type}
    (resolver : ObservationPriorResolver Observation Graph)
    (render : Scene -> Observation) (left right : Scene)
    (sameObservation : render left = render right) :
    resolver (render left) = resolver (render right) :=
  congrArg resolver sameObservation

theorem resolvedByObservationPrior_nonempty_unique
    {Scene Observation Graph : Type}
    (valid : Scene -> Prop) (render : Scene -> Observation)
    (meaning : Scene -> Graph)
    (resolver : ObservationPriorResolver Observation Graph)
    (resolved : ResolvedByObservationPrior valid render meaning resolver)
    (observation : Observation)
    (supported : ObservationSupported valid render observation) :
    Exists fun graph =>
      GraphSupportedAt valid render meaning observation graph ∧
        resolver observation graph ∧
          forall otherGraph,
            GraphSupportedAt valid render meaning observation otherGraph ->
              resolver observation otherGraph -> otherGraph = graph :=
  resolved observation supported

theorem resolved_observation_prior_graph_output_sound
    {Scene Observation Graph Ambiguity RejectReason : Type}
    (valid : Scene -> Prop) (render : Scene -> Observation)
    (meaning : Scene -> Graph)
    (resolver : ObservationPriorResolver Observation Graph)
    (resolved : ResolvedByObservationPrior valid render meaning resolver)
    (decode : Observation -> DecodeResult Graph Ambiguity RejectReason)
    (decoderSelects : forall observation graph,
      decode observation = .graph graph ->
        GraphSupportedAt valid render meaning observation graph ∧
          resolver observation graph)
    (observation : Observation) (graph : Graph)
    (decoded : decode observation = .graph graph) :
    forall otherGraph,
      GraphSupportedAt valid render meaning observation otherGraph ->
        resolver observation otherGraph -> otherGraph = graph := by
  have selected := decoderSelects observation graph decoded
  have supported : ObservationSupported valid render observation := by
    rcases selected.1 with ⟨scene, sceneValid, sceneObservation, _sceneMeaning⟩
    exact ⟨scene, sceneValid, sceneObservation⟩
  rcases resolved observation supported with
    ⟨uniqueGraph, uniqueSupported, uniqueSelected, unique⟩
  have graphIsUnique : graph = uniqueGraph :=
    unique graph selected.1 selected.2
  intro otherGraph otherSupported otherSelected
  exact (unique otherGraph otherSupported otherSelected).trans graphIsUnique.symm

/-!
`ProductionPriorChooser` 把可执行 prior 再拆成三步。`resolve` 的类型中没有
`Scene`、Scene ID 或 `meaning` 参数；候选集合也必须由 observation 生成。
这能阻止接口层直接读取 GT，但不能单靠类型系统排除实现内部用 observation hash
查表，因此生成器仍需接受独立代码审计。
-/
abbrev CandidateSet (Graph : Type) := Graph -> Prop

structure ProductionPriorChooser (Observation Evidence Graph : Type) where
  extract : Observation -> Evidence
  candidates : Evidence -> CandidateSet Graph
  choose : Evidence -> CandidateSet Graph -> Option Graph

def ProductionPriorChooser.candidateSet
    {Observation Evidence Graph : Type}
    (chooser : ProductionPriorChooser Observation Evidence Graph)
    (observation : Observation) : CandidateSet Graph :=
  chooser.candidates (chooser.extract observation)

def ProductionPriorChooser.resolve
    {Observation Evidence Graph : Type}
    (chooser : ProductionPriorChooser Observation Evidence Graph)
    (observation : Observation) : Option Graph :=
  let evidence := chooser.extract observation
  chooser.choose evidence (chooser.candidates evidence)

theorem production_chooser_is_observation_class_level
    {Scene Observation Evidence Graph : Type}
    (chooser : ProductionPriorChooser Observation Evidence Graph)
    (render : Scene -> Observation) (left right : Scene)
    (sameObservation : render left = render right) :
    chooser.resolve (render left) = chooser.resolve (render right) :=
  congrArg chooser.resolve sameObservation

/-!
`meaning` 只出现在验证谓词右侧；它不是 chooser 的输入。因而相同 observation
对应异图时，确定性 chooser 至多保留其中一种解释。
-/
def KeptByProductionChooser
    {Scene Observation Evidence Graph : Type}
    (chooser : ProductionPriorChooser Observation Evidence Graph)
    (render : Scene -> Observation) (meaning : Scene -> Graph)
    (scene : Scene) : Prop :=
  chooser.resolve (render scene) = some (meaning scene)

theorem production_chooser_cannot_keep_two_colliding_graphs
    {Scene Observation Evidence Graph : Type}
    (chooser : ProductionPriorChooser Observation Evidence Graph)
    (render : Scene -> Observation) (meaning : Scene -> Graph)
    (left right : Scene)
    (sameObservation : render left = render right)
    (differentMeaning : meaning left ≠ meaning right) :
    Not (KeptByProductionChooser chooser render meaning left ∧
      KeptByProductionChooser chooser render meaning right) := by
  intro bothKept
  apply differentMeaning
  have sameResolution := production_chooser_is_observation_class_level
    chooser render left right sameObservation
  rw [bothKept.1, bothKept.2] at sameResolution
  injection sameResolution

def ProductionChooserResolved
    {Scene Observation Evidence Graph : Type}
    (valid : Scene -> Prop) (render : Scene -> Observation)
    (meaning : Scene -> Graph)
    (chooser : ProductionPriorChooser Observation Evidence Graph) : Prop :=
  forall observation, ObservationSupported valid render observation ->
    Exists fun graph =>
      chooser.resolve observation = some graph ∧
        chooser.candidateSet observation graph ∧
          GraphSupportedAt valid render meaning observation graph

def ProductionCandidateSetExact
    {Scene Observation Evidence Graph : Type}
    (valid : Scene -> Prop) (render : Scene -> Observation)
    (meaning : Scene -> Graph)
    (chooser : ProductionPriorChooser Observation Evidence Graph) : Prop :=
  forall observation graph,
    chooser.candidateSet observation graph ↔
      GraphSupportedAt valid render meaning observation graph

structure CertifiedProductionChooser
    {Scene Observation Evidence Graph : Type}
    (valid : Scene -> Prop) (render : Scene -> Observation)
    (meaning : Scene -> Graph)
    (chooser : ProductionPriorChooser Observation Evidence Graph) : Prop where
  candidateSetExact : ProductionCandidateSetExact valid render meaning chooser
  resolved : ProductionChooserResolved valid render meaning chooser

theorem production_chooser_resolved_output_is_supported
    {Scene Observation Evidence Graph : Type}
    (valid : Scene -> Prop) (render : Scene -> Observation)
    (meaning : Scene -> Graph)
    (chooser : ProductionPriorChooser Observation Evidence Graph)
    (resolved : ProductionChooserResolved valid render meaning chooser)
    (observation : Observation)
    (supported : ObservationSupported valid render observation) :
    Exists fun graph =>
      chooser.resolve observation = some graph ∧
        chooser.candidateSet observation graph ∧
          GraphSupportedAt valid render meaning observation graph :=
  resolved observation supported

/-!
Oracle evidence 只有在显式 `extract ∘ render` 分解成立时才是可观察字段。
若两个场景的 production observation 已碰撞而 Oracle evidence 不同，任何模型
容量和训练量都无法补出这个 factorization。
-/
def OracleEvidenceFactorsThrough {Scene Observation Evidence : Type}
    (render : Scene -> Observation) (oracleEvidence : Scene -> Evidence)
    (extract : Observation -> Evidence) : Prop :=
  forall scene, extract (render scene) = oracleEvidence scene

theorem oracle_evidence_factorization_equation
    {Scene Observation Evidence : Type}
    (render : Scene -> Observation) (oracleEvidence : Scene -> Evidence)
    (extract : Observation -> Evidence)
    (factors : OracleEvidenceFactorsThrough render oracleEvidence extract)
    (scene : Scene) :
    extract (render scene) = oracleEvidence scene :=
  factors scene

theorem oracle_evidence_collision_forbids_extractor
    {Scene Observation Evidence : Type}
    (render : Scene -> Observation) (oracleEvidence : Scene -> Evidence)
    (left right : Scene)
    (sameObservation : render left = render right)
    (differentEvidence : oracleEvidence left ≠ oracleEvidence right) :
    Not (Exists fun extract : Observation -> Evidence =>
      OracleEvidenceFactorsThrough render oracleEvidence extract) :=
  collision_forbids_factorization render oracleEvidence left right
    sameObservation differentEvidence

structure ObservableOracleField {Scene Observation Field : Type}
    (render : Scene -> Observation) (oracleField : Scene -> Field) where
  extract : Observation -> Field
  factors : OracleEvidenceFactorsThrough render oracleField extract

/-!
Oracle field 的左逆必须带 `fieldStrict` 前提。这里的 strict 使用图等价关系，
因此既可实例化为相等，也可实例化为规范图同构。
-/
def FieldStrict {Scene Field Graph : Type}
    (valid : Scene -> Prop) (equivalent : Graph -> Graph -> Prop)
    (oracleField : Scene -> Field) (meaning : Scene -> Graph)
    (scene : Scene) : Prop :=
  valid scene ∧ forall other, valid other -> oracleField other = oracleField scene ->
    equivalent (meaning other) (meaning scene)

def OracleFieldLeftInverseOnStrict {Scene Field Graph : Type}
    (valid : Scene -> Prop) (equivalent : Graph -> Graph -> Prop)
    (oracleField : Scene -> Field) (meaning : Scene -> Graph)
    (decodeField : Field -> Graph) : Prop :=
  forall scene, FieldStrict valid equivalent oracleField meaning scene ->
    equivalent (decodeField (oracleField scene)) (meaning scene)

theorem observable_field_strict_implies_observation_strict
    {Scene Observation Field Graph : Type}
    (valid : Scene -> Prop) (equivalent : Graph -> Graph -> Prop)
    (render : Scene -> Observation) (oracleField : Scene -> Field)
    (meaning : Scene -> Graph)
    (observable : ObservableOracleField render oracleField)
    (scene : Scene)
    (strict : FieldStrict valid equivalent oracleField meaning scene) :
    StrictObservationUpTo valid equivalent render meaning (render scene) := by
  refine ⟨meaning scene, ?_, ?_⟩
  · exact ⟨scene, strict.1, rfl, rfl⟩
  · intro other otherValid sameObservation
    apply strict.2 other otherValid
    calc
      oracleField other = observable.extract (render other) :=
        (observable.factors other).symm
      _ = observable.extract (render scene) := congrArg observable.extract sameObservation
      _ = oracleField scene := observable.factors scene

theorem oracle_field_left_inverse_statement
    {Scene Field Graph : Type}
    (valid : Scene -> Prop) (equivalent : Graph -> Graph -> Prop)
    (oracleField : Scene -> Field) (meaning : Scene -> Graph)
    (decodeField : Field -> Graph)
    (roundtrip : OracleFieldLeftInverseOnStrict
      valid equivalent oracleField meaning decodeField)
    (scene : Scene)
    (strict : FieldStrict valid equivalent oracleField meaning scene) :
    equivalent (decodeField (oracleField scene)) (meaning scene) :=
  roundtrip scene strict

theorem global_oracle_field_left_inverse_implies_field_strict
    {Scene Field Graph : Type}
    (valid : Scene -> Prop) (equivalent : Graph -> Graph -> Prop)
    (equivalenceLaw : Equivalence equivalent)
    (oracleField : Scene -> Field) (meaning : Scene -> Graph)
    (decodeField : Field -> Graph)
    (roundtrip : forall scene, valid scene ->
      equivalent (decodeField (oracleField scene)) (meaning scene)) :
    forall scene, valid scene ->
      FieldStrict valid equivalent oracleField meaning scene := by
  intro scene sceneValid
  refine ⟨sceneValid, ?_⟩
  intro other otherValid sameField
  exact equivalenceLaw.trans
    (equivalenceLaw.symm (roundtrip other otherValid))
    (sameField ▸ roundtrip scene sceneValid)

theorem oracle_field_collision_forbids_global_left_inverse
    {Scene Field Graph : Type}
    (equivalent : Graph -> Graph -> Prop)
    (equivalenceLaw : Equivalence equivalent)
    (oracleField : Scene -> Field) (meaning : Scene -> Graph)
    (left right : Scene)
    (sameField : oracleField left = oracleField right)
    (differentMeaning : Not (equivalent (meaning left) (meaning right))) :
    Not (Exists fun decodeField : Field -> Graph =>
      forall scene, equivalent (decodeField (oracleField scene)) (meaning scene)) :=
  collision_forbids_decoder_up_to equivalent equivalenceLaw oracleField meaning
    left right sameField differentMeaning

end ChemGraph
