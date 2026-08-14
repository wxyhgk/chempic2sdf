import ChemGraph.HiddenChallenge

namespace ChemGraph

/-!
P3 EvidenceCore 只形式化隔离边界与候选/解码协议。`primitiveCount`、
`eventCount` 和 `serializedBytes` 是外部实现提供并审计的度量；Lean 不读取
JSON，不计算真实序列化长度，也不证明某个 Python payload 没有携带 raster。
-/
structure EvidenceCoreLimits where
  maxPrimitives : Nat
  maxEvents : Nat
  maxSerializedBytes : Nat

structure EvidenceCoreExternalMetrics (Evidence : Type) where
  primitiveCount : Evidence -> Nat
  eventCount : Evidence -> Nat
  serializedBytes : Evidence -> Nat

def EvidenceCoreWithinBounds {Evidence : Type}
    (limits : EvidenceCoreLimits)
    (metrics : EvidenceCoreExternalMetrics Evidence)
    (evidence : Evidence) : Prop :=
  metrics.primitiveCount evidence <= limits.maxPrimitives ∧
    metrics.eventCount evidence <= limits.maxEvents ∧
      metrics.serializedBytes evidence <= limits.maxSerializedBytes

/-!
`NoRasterPayload` 是外部审计谓词接口，不是对内存布局或序列化器的 Lean
实现。生产验证必须独立给出该谓词在每个 extractor 输出上的见证。
-/
abbrev NoRasterPayload (Evidence : Type) := Evidence -> Prop

structure BoundedEvidenceCoreExtractor (Raster Evidence : Type) where
  extract : Raster -> Evidence
  limits : EvidenceCoreLimits
  metrics : EvidenceCoreExternalMetrics Evidence
  noRasterPayload : NoRasterPayload Evidence
  withinBounds : forall raster,
    EvidenceCoreWithinBounds limits metrics (extract raster)
  excludesRasterPayload : forall raster, noRasterPayload (extract raster)

/-!
一个 raster 级公开输出只有在存在纯 Evidence consumer 时才称为经
EvidenceCore 分解。该定义不限制 consumer 的实现，只冻结它的输入边界。
-/
def ExtractorFactorsThroughEvidence {Raster Evidence Output : Type}
    (extract : Raster -> Evidence) (output : Raster -> Output) : Prop :=
  Exists fun consume : Evidence -> Output =>
    forall raster, output raster = consume (extract raster)

theorem factorized_output_equal_on_same_evidence
    {Raster Evidence Output : Type}
    (extract : Raster -> Evidence) (output : Raster -> Output)
    (factors : ExtractorFactorsThroughEvidence extract output)
    (left right : Raster)
    (sameEvidence : extract left = extract right) :
    output left = output right := by
  rcases factors with ⟨consume, equation⟩
  calc
    output left = consume (extract left) := equation left
    _ = consume (extract right) := congrArg consume sameEvidence
    _ = output right := (equation right).symm

/-!
候选进程的外部接口可以按 raster 索引用于审计，但 `CandidateReadsEvidenceOnly`
要求存在一个只接收 Evidence 的实现。因而相同 Evidence 必须得到相同候选集合。
-/
def CandidateReadsEvidenceOnly {Raster Evidence Graph : Type}
    (extract : Raster -> Evidence)
    (candidateAt : Raster -> CandidateSet Graph) : Prop :=
  Exists fun fromEvidence : Evidence -> CandidateSet Graph =>
    forall raster graph,
      candidateAt raster graph ↔ fromEvidence (extract raster) graph

theorem evidence_only_candidate_equal_on_same_evidence
    {Raster Evidence Graph : Type}
    (extract : Raster -> Evidence)
    (candidateAt : Raster -> CandidateSet Graph)
    (readsOnly : CandidateReadsEvidenceOnly extract candidateAt)
    (left right : Raster)
    (sameEvidence : extract left = extract right) :
    forall graph, candidateAt left graph ↔ candidateAt right graph := by
  rcases readsOnly with ⟨fromEvidence, equation⟩
  intro graph
  calc
    candidateAt left graph ↔ fromEvidence (extract left) graph := equation left graph
    _ ↔ fromEvidence (extract right) graph := by rw [sameEvidence]
    _ ↔ candidateAt right graph := (equation right graph).symm

/-! 候选集合按图等价类而非匿名节点编号比较。 -/
def CandidateClassSingletonAtUpTo {Raster Graph : Type}
    (equivalent : Graph -> Graph -> Prop)
    (candidateAt : Raster -> CandidateSet Graph)
    (raster : Raster) (representative : Graph) : Prop :=
  candidateAt raster representative ∧
    forall other, candidateAt raster other -> equivalent other representative

def CandidateMultipleClassesAtUpTo {Raster Graph : Type}
    (equivalent : Graph -> Graph -> Prop)
    (candidateAt : Raster -> CandidateSet Graph)
    (raster : Raster) : Prop :=
  Exists fun left => Exists fun right =>
    candidateAt raster left ∧ candidateAt raster right ∧
      Not (equivalent left right)

def CandidateEmptyAt {Raster Graph : Type}
    (candidateAt : Raster -> CandidateSet Graph) (raster : Raster) : Prop :=
  forall graph, Not (candidateAt raster graph)

/-!
Soundness 要求每个候选都等价于当前 raster 的某个有效解释；completeness
要求每个有效解释都至少有一个等价候选。它允许同一等价类有多个匿名编号表示。
-/
def CandidateSoundUpTo
    {Scene Raster Graph : Type}
    (valid : Scene -> Prop) (equivalent : Graph -> Graph -> Prop)
    (render : Scene -> Raster) (meaning : Scene -> Graph)
    (candidateAt : Raster -> CandidateSet Graph) : Prop :=
  forall raster graph, candidateAt raster graph ->
    Exists fun scene =>
      valid scene ∧ render scene = raster ∧ equivalent graph (meaning scene)

def CandidateCompleteUpTo
    {Scene Raster Graph : Type}
    (valid : Scene -> Prop) (equivalent : Graph -> Graph -> Prop)
    (render : Scene -> Raster) (meaning : Scene -> Graph)
    (candidateAt : Raster -> CandidateSet Graph) : Prop :=
  forall raster scene, valid scene -> render scene = raster ->
    Exists fun graph => candidateAt raster graph ∧ equivalent graph (meaning scene)

structure CandidatesExactUpTo
    {Scene Raster Graph : Type}
    (valid : Scene -> Prop) (equivalent : Graph -> Graph -> Prop)
    (render : Scene -> Raster) (meaning : Scene -> Graph)
    (candidateAt : Raster -> CandidateSet Graph) : Prop where
  sound : CandidateSoundUpTo valid equivalent render meaning candidateAt
  complete : CandidateCompleteUpTo valid equivalent render meaning candidateAt

def AmbiguousObservationUpTo
    {Scene Raster Graph : Type}
    (valid : Scene -> Prop) (equivalent : Graph -> Graph -> Prop)
    (render : Scene -> Raster) (meaning : Scene -> Graph)
    (raster : Raster) : Prop :=
  Exists fun left => Exists fun right =>
    valid left ∧ valid right ∧
      render left = raster ∧ render right = raster ∧
        Not (equivalent (meaning left) (meaning right))

/-!
若候选只依赖 Evidence，则两个产生相同 Evidence 的有效场景会共享整个候选集。
当它们的图不等价时，不可能同时保持 scene-level soundness 又输出单一等价类。
-/
def CandidateSingletonSoundOnScenes
    {Scene Raster Graph : Type}
    (valid : Scene -> Prop) (equivalent : Graph -> Graph -> Prop)
    (render : Scene -> Raster) (meaning : Scene -> Graph)
    (candidateAt : Raster -> CandidateSet Graph) : Prop :=
  forall scene, valid scene -> forall graph,
    CandidateClassSingletonAtUpTo equivalent candidateAt (render scene) graph ->
      equivalent graph (meaning scene)

theorem exact_candidates_are_singleton_sound_on_scenes
    {Scene Raster Graph : Type}
    (valid : Scene -> Prop) (equivalent : Graph -> Graph -> Prop)
    (equivalenceLaw : Equivalence equivalent)
    (render : Scene -> Raster) (meaning : Scene -> Graph)
    (candidateAt : Raster -> CandidateSet Graph)
    (exact : CandidatesExactUpTo valid equivalent render meaning candidateAt) :
    CandidateSingletonSoundOnScenes valid equivalent render meaning candidateAt := by
  intro scene sceneValid graph singleton
  rcases exact.complete (render scene) scene sceneValid rfl with
    ⟨representative, representativeCandidate, representativeToMeaning⟩
  have representativeToGraph := singleton.2 representative representativeCandidate
  exact equivalenceLaw.trans
    (equivalenceLaw.symm representativeToGraph) representativeToMeaning

theorem same_evidence_inequivalent_graphs_forbid_sound_single_graph
    {Scene Raster Evidence Graph : Type}
    (valid : Scene -> Prop) (equivalent : Graph -> Graph -> Prop)
    (equivalenceLaw : Equivalence equivalent)
    (render : Scene -> Raster) (meaning : Scene -> Graph)
    (extract : Raster -> Evidence)
    (candidateAt : Raster -> CandidateSet Graph)
    (readsOnly : CandidateReadsEvidenceOnly extract candidateAt)
    (singletonSound : CandidateSingletonSoundOnScenes
      valid equivalent render meaning candidateAt)
    (left right : Scene) (leftValid : valid left) (rightValid : valid right)
    (sameEvidence : extract (render left) = extract (render right))
    (differentMeaning : Not (equivalent (meaning left) (meaning right))) :
    Not (Exists fun graph =>
      CandidateClassSingletonAtUpTo equivalent candidateAt (render left) graph) := by
  intro existsSingleton
  rcases existsSingleton with ⟨graph, singletonLeft⟩
  have candidatesAgree := evidence_only_candidate_equal_on_same_evidence
    extract candidateAt readsOnly (render left) (render right) sameEvidence
  have singletonRight : CandidateClassSingletonAtUpTo
      equivalent candidateAt (render right) graph := by
    refine ⟨(candidatesAgree graph).mp singletonLeft.1, ?_⟩
    intro other otherCandidate
    exact singletonLeft.2 other ((candidatesAgree other).mpr otherCandidate)
  have graphToLeft := singletonSound left leftValid graph singletonLeft
  have graphToRight := singletonSound right rightValid graph singletonRight
  apply differentMeaning
  exact equivalenceLaw.trans (equivalenceLaw.symm graphToLeft) graphToRight

theorem same_evidence_inequivalent_graphs_forbid_exact_single_graph
    {Scene Raster Evidence Graph : Type}
    (valid : Scene -> Prop) (equivalent : Graph -> Graph -> Prop)
    (equivalenceLaw : Equivalence equivalent)
    (render : Scene -> Raster) (meaning : Scene -> Graph)
    (extract : Raster -> Evidence)
    (candidateAt : Raster -> CandidateSet Graph)
    (readsOnly : CandidateReadsEvidenceOnly extract candidateAt)
    (exact : CandidatesExactUpTo valid equivalent render meaning candidateAt)
    (left right : Scene) (leftValid : valid left) (rightValid : valid right)
    (sameEvidence : extract (render left) = extract (render right))
    (differentMeaning : Not (equivalent (meaning left) (meaning right))) :
    Not (Exists fun graph =>
      CandidateClassSingletonAtUpTo equivalent candidateAt (render left) graph) :=
  same_evidence_inequivalent_graphs_forbid_sound_single_graph
    valid equivalent equivalenceLaw render meaning extract candidateAt readsOnly
      (exact_candidates_are_singleton_sound_on_scenes
        valid equivalent equivalenceLaw render meaning candidateAt exact)
      left right leftValid rightValid sameEvidence differentMeaning

theorem exact_candidates_turn_strict_into_single_class
    {Scene Raster Graph : Type}
    (valid : Scene -> Prop) (equivalent : Graph -> Graph -> Prop)
    (equivalenceLaw : Equivalence equivalent)
    (render : Scene -> Raster) (meaning : Scene -> Graph)
    (candidateAt : Raster -> CandidateSet Graph)
    (exact : CandidatesExactUpTo valid equivalent render meaning candidateAt)
    (raster : Raster)
    (strict : StrictObservationUpTo valid equivalent render meaning raster) :
    Exists fun graph =>
      CandidateClassSingletonAtUpTo equivalent candidateAt raster graph := by
  rcases strict with ⟨anchor, supported, unique⟩
  rcases supported with ⟨source, sourceValid, sourceRaster, sourceMeaning⟩
  rcases exact.complete raster source sourceValid sourceRaster with
    ⟨graph, graphCandidate, graphToSource⟩
  refine ⟨graph, graphCandidate, ?_⟩
  intro other otherCandidate
  rcases exact.sound raster other otherCandidate with
    ⟨otherScene, otherValid, otherRaster, otherToMeaning⟩
  have sourceToAnchor : equivalent (meaning source) anchor :=
    sourceMeaning ▸ unique source sourceValid sourceRaster
  have otherMeaningToAnchor : equivalent (meaning otherScene) anchor :=
    unique otherScene otherValid otherRaster
  have graphToAnchor := equivalenceLaw.trans graphToSource sourceToAnchor
  have otherToAnchor := equivalenceLaw.trans otherToMeaning otherMeaningToAnchor
  exact equivalenceLaw.trans otherToAnchor (equivalenceLaw.symm graphToAnchor)

theorem exact_candidates_turn_ambiguity_into_multiple_classes
    {Scene Raster Graph : Type}
    (valid : Scene -> Prop) (equivalent : Graph -> Graph -> Prop)
    (equivalenceLaw : Equivalence equivalent)
    (render : Scene -> Raster) (meaning : Scene -> Graph)
    (candidateAt : Raster -> CandidateSet Graph)
    (exact : CandidatesExactUpTo valid equivalent render meaning candidateAt)
    (raster : Raster)
    (ambiguous : AmbiguousObservationUpTo
      valid equivalent render meaning raster) :
    CandidateMultipleClassesAtUpTo equivalent candidateAt raster := by
  rcases ambiguous with ⟨left, right, leftValid, rightValid,
    leftRaster, rightRaster, differentMeaning⟩
  rcases exact.complete raster left leftValid leftRaster with
    ⟨leftGraph, leftCandidate, leftGraphToMeaning⟩
  rcases exact.complete raster right rightValid rightRaster with
    ⟨rightGraph, rightCandidate, rightGraphToMeaning⟩
  refine ⟨leftGraph, rightGraph, leftCandidate, rightCandidate, ?_⟩
  intro graphEquivalent
  apply differentMeaning
  exact equivalenceLaw.trans (equivalenceLaw.symm leftGraphToMeaning)
    (equivalenceLaw.trans graphEquivalent rightGraphToMeaning)

/-!
三态 decoder 合同同时覆盖 Graph、Ambiguous 和 Reject。Reject 只允许在候选
为空时出现；taxonomy/reason 的真实性仍由外部谓词给出。
-/
structure CandidateDecoderCorrectUpTo
    {Raster Graph Ambiguity RejectReason : Type}
    (equivalent : Graph -> Graph -> Prop)
    (candidateAt : Raster -> CandidateSet Graph)
    (ambiguityKindAt : Raster -> Ambiguity -> Prop)
    (rejectReasonAt : Raster -> RejectReason -> Prop)
    (decode : Raster -> DecodeResult Graph Ambiguity RejectReason) : Prop where
  singletonComplete : forall raster representative,
    CandidateClassSingletonAtUpTo equivalent candidateAt raster representative ->
      Exists fun output =>
        candidateAt raster output ∧ equivalent output representative ∧
          decode raster = .graph output
  graphSound : forall raster graph,
    decode raster = .graph graph ->
      CandidateClassSingletonAtUpTo equivalent candidateAt raster graph
  multipleComplete : forall raster,
    CandidateMultipleClassesAtUpTo equivalent candidateAt raster ->
      Exists fun kind => decode raster = .ambiguous kind ∧ ambiguityKindAt raster kind
  ambiguitySound : forall raster kind,
    decode raster = .ambiguous kind ->
      CandidateMultipleClassesAtUpTo equivalent candidateAt raster ∧
        ambiguityKindAt raster kind
  emptyComplete : forall raster,
    CandidateEmptyAt candidateAt raster ->
      Exists fun reason => decode raster = .reject reason ∧ rejectReasonAt raster reason
  rejectSound : forall raster reason,
    decode raster = .reject reason ->
      CandidateEmptyAt candidateAt raster ∧ rejectReasonAt raster reason

theorem exact_candidates_and_correct_decoder_decode_strict
    {Scene Raster Graph Ambiguity RejectReason : Type}
    (valid : Scene -> Prop) (equivalent : Graph -> Graph -> Prop)
    (equivalenceLaw : Equivalence equivalent)
    (render : Scene -> Raster) (meaning : Scene -> Graph)
    (candidateAt : Raster -> CandidateSet Graph)
    (exact : CandidatesExactUpTo valid equivalent render meaning candidateAt)
    (ambiguityKindAt : Raster -> Ambiguity -> Prop)
    (rejectReasonAt : Raster -> RejectReason -> Prop)
    (decode : Raster -> DecodeResult Graph Ambiguity RejectReason)
    (decoderCorrect : CandidateDecoderCorrectUpTo
      equivalent candidateAt ambiguityKindAt rejectReasonAt decode)
    (raster : Raster)
    (strict : StrictObservationUpTo valid equivalent render meaning raster) :
    Exists fun graph =>
      decode raster = .graph graph ∧ candidateAt raster graph ∧
        forall scene, valid scene -> render scene = raster ->
          equivalent graph (meaning scene) := by
  rcases exact_candidates_turn_strict_into_single_class
      valid equivalent equivalenceLaw render meaning candidateAt exact raster strict with
    ⟨representative, singleton⟩
  rcases decoderCorrect.singletonComplete raster representative singleton with
    ⟨output, outputCandidate, _outputToRepresentative, decoded⟩
  refine ⟨output, decoded, outputCandidate, ?_⟩
  intro scene sceneValid sceneRaster
  rcases exact.sound raster output outputCandidate with
    ⟨witness, witnessValid, witnessRaster, outputToWitness⟩
  rcases strict with ⟨anchor, _supported, unique⟩
  have witnessToAnchor := unique witness witnessValid witnessRaster
  have sceneToAnchor := unique scene sceneValid sceneRaster
  exact equivalenceLaw.trans
    (equivalenceLaw.trans outputToWitness witnessToAnchor)
    (equivalenceLaw.symm sceneToAnchor)

theorem exact_candidates_and_correct_decoder_decode_ambiguity
    {Scene Raster Graph Ambiguity RejectReason : Type}
    (valid : Scene -> Prop) (equivalent : Graph -> Graph -> Prop)
    (equivalenceLaw : Equivalence equivalent)
    (render : Scene -> Raster) (meaning : Scene -> Graph)
    (candidateAt : Raster -> CandidateSet Graph)
    (exact : CandidatesExactUpTo valid equivalent render meaning candidateAt)
    (ambiguityKindAt : Raster -> Ambiguity -> Prop)
    (rejectReasonAt : Raster -> RejectReason -> Prop)
    (decode : Raster -> DecodeResult Graph Ambiguity RejectReason)
    (decoderCorrect : CandidateDecoderCorrectUpTo
      equivalent candidateAt ambiguityKindAt rejectReasonAt decode)
    (raster : Raster)
    (ambiguous : AmbiguousObservationUpTo
      valid equivalent render meaning raster) :
    Exists fun kind => decode raster = .ambiguous kind ∧ ambiguityKindAt raster kind :=
  decoderCorrect.multipleComplete raster
    (exact_candidates_turn_ambiguity_into_multiple_classes
      valid equivalent equivalenceLaw render meaning candidateAt exact raster ambiguous)

theorem correct_decoder_reject_is_empty_and_allowed
    {Raster Graph Ambiguity RejectReason : Type}
    (equivalent : Graph -> Graph -> Prop)
    (candidateAt : Raster -> CandidateSet Graph)
    (ambiguityKindAt : Raster -> Ambiguity -> Prop)
    (rejectReasonAt : Raster -> RejectReason -> Prop)
    (decode : Raster -> DecodeResult Graph Ambiguity RejectReason)
    (decoderCorrect : CandidateDecoderCorrectUpTo
      equivalent candidateAt ambiguityKindAt rejectReasonAt decode)
    (raster : Raster) (reason : RejectReason)
    (decoded : decode raster = .reject reason) :
    CandidateEmptyAt candidateAt raster ∧ rejectReasonAt raster reason :=
  decoderCorrect.rejectSound raster reason decoded

/-!
S8/C4 的具体 PNG 等同性由外部 audit 证明。Lean 这里只表达其最小逻辑后果：
若 strict S8 与 ambiguous C4 进入相同 Evidence，任何只读 Evidence 的确定性
decoder 都不可能同时满足 `.graph` 与 `.ambiguous` 两个要求。
-/
def S8StrictGraphDemand
    {Evidence Graph Ambiguity RejectReason : Type}
    (decode : Evidence -> DecodeResult Graph Ambiguity RejectReason)
    (evidence : Evidence) (graph : Graph) : Prop :=
  decode evidence = .graph graph

def C4AmbiguityDemand
    {Evidence Graph Ambiguity RejectReason : Type}
    (decode : Evidence -> DecodeResult Graph Ambiguity RejectReason)
    (evidence : Evidence) (kind : Ambiguity) : Prop :=
  decode evidence = .ambiguous kind

theorem same_evidence_s8_graph_and_c4_ambiguity_incompatible
    {Evidence Graph Ambiguity RejectReason : Type}
    (decode : Evidence -> DecodeResult Graph Ambiguity RejectReason)
    (s8Evidence c4Evidence : Evidence)
    (sameEvidence : s8Evidence = c4Evidence)
    (s8Graph : Graph) (c4Kind : Ambiguity) :
    Not (S8StrictGraphDemand decode s8Evidence s8Graph ∧
      C4AmbiguityDemand decode c4Evidence c4Kind) := by
  intro demands
  have sameDecode := congrArg decode sameEvidence
  rw [demands.1, demands.2] at sameDecode
  cases sameDecode

end ChemGraph
