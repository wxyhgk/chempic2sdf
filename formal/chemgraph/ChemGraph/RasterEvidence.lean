import ChemGraph.ObservationRecovery
import ChemGraph.Stability

namespace ChemGraph

/-!
Raster Evidence v0.3 只形式化接口条件。`Raster` 应在生产实例中取完整的
tensor/raster 对象；本文件既不实现 Python extractor，也不证明任何具体 extractor
满足这些条件。
-/
structure RasterEvidencePipeline (Raster Evidence Graph : Type) where
  extract : Raster -> Evidence
  candidates : Evidence -> CandidateSet Graph

def RasterEvidencePipeline.candidateAt
    {Raster Evidence Graph : Type}
    (pipeline : RasterEvidencePipeline Raster Evidence Graph)
    (raster : Raster) (graph : Graph) : Prop :=
  pipeline.candidates (pipeline.extract raster) graph

/-!
Oracle evidence 仅用于审计 factorization。`valid` 限定声明域；若没有提供
`factors` 的实现见证，就不能称 Oracle 字段由 raster 可计算。
-/
def RasterEvidenceFactorization
    {Scene Raster Evidence : Type}
    (valid : Scene -> Prop) (render : Scene -> Raster)
    (oracleEvidence : Scene -> Evidence) (extract : Raster -> Evidence) : Prop :=
  forall scene, valid scene -> extract (render scene) = oracleEvidence scene

theorem same_raster_has_same_extracted_evidence
    {Scene Raster Evidence : Type}
    (render : Scene -> Raster) (extract : Raster -> Evidence)
    (left right : Scene)
    (sameRaster : render left = render right) :
    extract (render left) = extract (render right) :=
  congrArg extract sameRaster

theorem factorized_evidence_equal_on_raster_collision
    {Scene Raster Evidence : Type}
    (valid : Scene -> Prop) (render : Scene -> Raster)
    (oracleEvidence : Scene -> Evidence) (extract : Raster -> Evidence)
    (factors : RasterEvidenceFactorization valid render oracleEvidence extract)
    (left right : Scene) (leftValid : valid left) (rightValid : valid right)
    (sameRaster : render left = render right) :
    oracleEvidence left = oracleEvidence right := by
  calc
    oracleEvidence left = extract (render left) := (factors left leftValid).symm
    _ = extract (render right) := congrArg extract sameRaster
    _ = oracleEvidence right := factors right rightValid

theorem different_oracle_evidence_forbids_raster_factorization
    {Scene Raster Evidence : Type}
    (valid : Scene -> Prop) (render : Scene -> Raster)
    (oracleEvidence : Scene -> Evidence)
    (left right : Scene) (leftValid : valid left) (rightValid : valid right)
    (sameRaster : render left = render right)
    (differentEvidence : oracleEvidence left ≠ oracleEvidence right) :
    Not (Exists fun extract : Raster -> Evidence =>
      RasterEvidenceFactorization valid render oracleEvidence extract) := by
  intro existsExtractor
  rcases existsExtractor with ⟨extract, factors⟩
  apply differentEvidence
  exact factorized_evidence_equal_on_raster_collision
    valid render oracleEvidence extract factors left right
    leftValid rightValid sameRaster

/-!
候选 soundness 和 completeness 分开陈述。二者同时成立才叫 exact；只证明
chosen candidate 正确，不能替代 completeness。
-/
def RasterCandidateSound
    {Scene Raster Evidence Graph : Type}
    (valid : Scene -> Prop) (render : Scene -> Raster)
    (meaning : Scene -> Graph)
    (pipeline : RasterEvidencePipeline Raster Evidence Graph) : Prop :=
  forall raster graph, pipeline.candidateAt raster graph ->
    GraphSupportedAt valid render meaning raster graph

def RasterCandidateComplete
    {Scene Raster Evidence Graph : Type}
    (valid : Scene -> Prop) (render : Scene -> Raster)
    (meaning : Scene -> Graph)
    (pipeline : RasterEvidencePipeline Raster Evidence Graph) : Prop :=
  forall raster graph, GraphSupportedAt valid render meaning raster graph ->
    pipeline.candidateAt raster graph

structure RasterCandidatesExact
    {Scene Raster Evidence Graph : Type}
    (valid : Scene -> Prop) (render : Scene -> Raster)
    (meaning : Scene -> Graph)
    (pipeline : RasterEvidencePipeline Raster Evidence Graph) : Prop where
  sound : RasterCandidateSound valid render meaning pipeline
  complete : RasterCandidateComplete valid render meaning pipeline

def CandidateSingletonAt
    {Raster Evidence Graph : Type}
    (pipeline : RasterEvidencePipeline Raster Evidence Graph)
    (raster : Raster) (graph : Graph) : Prop :=
  pipeline.candidateAt raster graph ∧
    forall other, pipeline.candidateAt raster other -> other = graph

def CandidateMultipleAt
    {Raster Evidence Graph : Type}
    (pipeline : RasterEvidencePipeline Raster Evidence Graph)
    (raster : Raster) : Prop :=
  Exists fun left => Exists fun right =>
    pipeline.candidateAt raster left ∧
      pipeline.candidateAt raster right ∧ left ≠ right

/-!
严格/歧义解码协议是双向规格：singleton 必须且只能输出对应 Graph；multiple
必须且只能输出带合法 taxonomy 的 Ambiguous。这里不构造具体 decoder。
-/
structure RasterCandidateDecoderCorrect
    {Raster Evidence Graph Ambiguity RejectReason : Type}
    (pipeline : RasterEvidencePipeline Raster Evidence Graph)
    (ambiguityKindAt : Raster -> Ambiguity -> Prop)
    (decode : Raster -> DecodeResult Graph Ambiguity RejectReason) : Prop where
  singletonComplete : forall raster graph,
    CandidateSingletonAt pipeline raster graph -> decode raster = .graph graph
  graphSound : forall raster graph,
    decode raster = .graph graph -> CandidateSingletonAt pipeline raster graph
  multipleComplete : forall raster,
    CandidateMultipleAt pipeline raster ->
      Exists fun kind => decode raster = .ambiguous kind ∧ ambiguityKindAt raster kind
  ambiguitySound : forall raster kind,
    decode raster = .ambiguous kind ->
      CandidateMultipleAt pipeline raster ∧ ambiguityKindAt raster kind

theorem exact_singleton_candidates_decode_sound_graph
    {Scene Raster Evidence Graph Ambiguity RejectReason : Type}
    (valid : Scene -> Prop) (render : Scene -> Raster)
    (meaning : Scene -> Graph)
    (pipeline : RasterEvidencePipeline Raster Evidence Graph)
    (exact : RasterCandidatesExact valid render meaning pipeline)
    (ambiguityKindAt : Raster -> Ambiguity -> Prop)
    (decode : Raster -> DecodeResult Graph Ambiguity RejectReason)
    (decoderCorrect : RasterCandidateDecoderCorrect
      pipeline ambiguityKindAt decode)
    (raster : Raster) (graph : Graph)
    (singleton : CandidateSingletonAt pipeline raster graph) :
    decode raster = .graph graph ∧
      GraphSupportedAt valid render meaning raster graph ∧
        forall scene, valid scene -> render scene = raster -> meaning scene = graph := by
  refine ⟨decoderCorrect.singletonComplete raster graph singleton,
    exact.sound raster graph singleton.1, ?_⟩
  intro scene sceneValid sceneRaster
  apply singleton.2 (meaning scene)
  apply exact.complete raster (meaning scene)
  exact ⟨scene, sceneValid, sceneRaster, rfl⟩

theorem multiple_candidates_must_decode_ambiguous
    {Raster Evidence Graph Ambiguity RejectReason : Type}
    (pipeline : RasterEvidencePipeline Raster Evidence Graph)
    (ambiguityKindAt : Raster -> Ambiguity -> Prop)
    (decode : Raster -> DecodeResult Graph Ambiguity RejectReason)
    (decoderCorrect : RasterCandidateDecoderCorrect
      pipeline ambiguityKindAt decode)
    (raster : Raster)
    (multiple : CandidateMultipleAt pipeline raster) :
    Exists fun kind => decode raster = .ambiguous kind ∧ ambiguityKindAt raster kind :=
  decoderCorrect.multipleComplete raster multiple

theorem exact_candidates_turn_observation_ambiguity_into_multiple
    {Scene Raster Evidence Graph : Type}
    (valid : Scene -> Prop) (render : Scene -> Raster)
    (meaning : Scene -> Graph)
    (pipeline : RasterEvidencePipeline Raster Evidence Graph)
    (exact : RasterCandidatesExact valid render meaning pipeline)
    (raster : Raster)
    (ambiguous : AmbiguousObservationExact valid render meaning raster) :
    CandidateMultipleAt pipeline raster := by
  rcases ambiguous with ⟨left, right, leftValid, rightValid,
    leftRaster, rightRaster, different⟩
  refine ⟨meaning left, meaning right, ?_, ?_, different⟩
  · exact exact.complete raster (meaning left)
      ⟨left, leftValid, leftRaster, rfl⟩
  · exact exact.complete raster (meaning right)
      ⟨right, rightValid, rightRaster, rfl⟩

theorem exact_candidates_turn_unique_supported_graph_into_singleton
    {Scene Raster Evidence Graph : Type}
    (valid : Scene -> Prop) (render : Scene -> Raster)
    (meaning : Scene -> Graph)
    (pipeline : RasterEvidencePipeline Raster Evidence Graph)
    (exact : RasterCandidatesExact valid render meaning pipeline)
    (raster : Raster) (graph : Graph)
    (strict : GraphSupportedAt valid render meaning raster graph ∧
      forall scene, valid scene -> render scene = raster -> meaning scene = graph) :
    CandidateSingletonAt pipeline raster graph := by
  refine ⟨exact.complete raster graph strict.1, ?_⟩
  intro other otherCandidate
  rcases exact.sound raster other otherCandidate with
    ⟨scene, sceneValid, sceneRaster, sceneMeaning⟩
  calc
    other = meaning scene := sceneMeaning.symm
    _ = graph := strict.2 scene sceneValid sceneRaster

theorem exact_candidates_turn_strict_observation_into_singleton
    {Scene Raster Evidence Graph : Type}
    (valid : Scene -> Prop) (render : Scene -> Raster)
    (meaning : Scene -> Graph)
    (pipeline : RasterEvidencePipeline Raster Evidence Graph)
    (exact : RasterCandidatesExact valid render meaning pipeline)
    (raster : Raster)
    (strict : StrictObservation valid render meaning raster) :
    Exists fun graph => CandidateSingletonAt pipeline raster graph := by
  rcases strict with ⟨graph, supported, unique⟩
  exact ⟨graph,
    exact_candidates_turn_unique_supported_graph_into_singleton
      valid render meaning pipeline exact raster graph ⟨supported, unique⟩⟩

theorem exact_ambiguous_observation_must_decode_ambiguous
    {Scene Raster Evidence Graph Ambiguity RejectReason : Type}
    (valid : Scene -> Prop) (render : Scene -> Raster)
    (meaning : Scene -> Graph)
    (pipeline : RasterEvidencePipeline Raster Evidence Graph)
    (exact : RasterCandidatesExact valid render meaning pipeline)
    (ambiguityKindAt : Raster -> Ambiguity -> Prop)
    (decode : Raster -> DecodeResult Graph Ambiguity RejectReason)
    (decoderCorrect : RasterCandidateDecoderCorrect
      pipeline ambiguityKindAt decode)
    (raster : Raster)
    (ambiguous : AmbiguousObservationExact valid render meaning raster) :
    Exists fun kind => decode raster = .ambiguous kind ∧ ambiguityKindAt raster kind :=
  multiple_candidates_must_decode_ambiguous pipeline ambiguityKindAt decode
    decoderCorrect raster
      (exact_candidates_turn_observation_ambiguity_into_multiple
        valid render meaning pipeline exact raster ambiguous)

/-!
`separationLowerBound` 是异图 raster 距离的下界 Delta，不声称它已经由数据估计。
稳定门明确要求 `2 * epsilon < Delta`；自然数中写成
`epsilon + epsilon < separationLowerBound`。
-/
def InterGraphSeparationAtLeast
    {Scene Raster Graph : Type}
    (metric : NatDistance Raster)
    (valid : Scene -> Prop) (equivalent : Graph -> Graph -> Prop)
    (render : Scene -> Raster) (meaning : Scene -> Graph)
    (separationLowerBound : Nat) : Prop :=
  forall left right, valid left -> valid right ->
    Not (equivalent (meaning left) (meaning right)) ->
      separationLowerBound <= metric.dist (render left) (render right)

structure EpsilonDeltaStabilityGate
    {Scene Raster Graph : Type}
    (metric : NatDistance Raster)
    (valid : Scene -> Prop) (equivalent : Graph -> Graph -> Prop)
    (render : Scene -> Raster) (meaning : Scene -> Graph)
    (noiseRadius separationLowerBound : Nat) : Prop where
  twiceNoiseBelowSeparation : noiseRadius + noiseRadius < separationLowerBound
  interGraphSeparation : InterGraphSeparationAtLeast
    metric valid equivalent render meaning separationLowerBound

/-!
纯数学稳定门在空域或单一图域上可以真空成立。若要把它用作可执行 GO 门，
还必须提交域非空和至少一对异图的 witness。该结构仍不证明 Delta 已被完整枚举；
`interGraphSeparation` 的全域量词必须由独立审计兑现。
-/
structure AuditableEpsilonDeltaStabilityGate
    {Scene Raster Graph : Type}
    (metric : NatDistance Raster)
    (valid : Scene -> Prop) (equivalent : Graph -> Graph -> Prop)
    (render : Scene -> Raster) (meaning : Scene -> Graph)
    (noiseRadius separationLowerBound : Nat) : Prop where
  validDomainNonempty : Exists fun scene => valid scene
  inequivalentPairExists : Exists fun left => Exists fun right =>
    valid left ∧ valid right ∧ Not (equivalent (meaning left) (meaning right))
  stability : EpsilonDeltaStabilityGate metric valid equivalent render meaning
    noiseRadius separationLowerBound

theorem auditable_stability_gate_implies_stability_gate
    {Scene Raster Graph : Type}
    (metric : NatDistance Raster)
    (valid : Scene -> Prop) (equivalent : Graph -> Graph -> Prop)
    (render : Scene -> Raster) (meaning : Scene -> Graph)
    (noiseRadius separationLowerBound : Nat)
    (gate : AuditableEpsilonDeltaStabilityGate
      metric valid equivalent render meaning noiseRadius separationLowerBound) :
    EpsilonDeltaStabilityGate metric valid equivalent render meaning
      noiseRadius separationLowerBound :=
  gate.stability

theorem epsilon_delta_gate_implies_robust_separation
    {Scene Raster Graph : Type}
    (metric : NatDistance Raster)
    (valid : Scene -> Prop) (equivalent : Graph -> Graph -> Prop)
    (render : Scene -> Raster) (meaning : Scene -> Graph)
    (noiseRadius separationLowerBound : Nat)
    (gate : EpsilonDeltaStabilityGate metric valid equivalent render meaning
      noiseRadius separationLowerBound) :
    RobustlySeparated metric valid equivalent render meaning noiseRadius := by
  intro left right leftValid rightValid differentGraph
  exact Nat.lt_of_lt_of_le gate.twiceNoiseBelowSeparation
    (gate.interGraphSeparation left right leftValid rightValid differentGraph)

/-! 该定理直接复用 v0.1 的三角不等式唯一恢复边界。 -/
theorem epsilon_delta_ball_has_unique_graph
    {Scene Raster Graph : Type}
    (metric : NatDistance Raster)
    (valid : Scene -> Prop) (equivalent : Graph -> Graph -> Prop)
    (render : Scene -> Raster) (meaning : Scene -> Graph)
    (noiseRadius separationLowerBound : Nat)
    (gate : EpsilonDeltaStabilityGate metric valid equivalent render meaning
      noiseRadius separationLowerBound)
    (observed : Raster) (left right : Scene)
    (leftValid : valid left) (rightValid : valid right)
    (nearLeft : metric.dist observed (render left) <= noiseRadius)
    (nearRight : metric.dist observed (render right) <= noiseRadius) :
    equivalent (meaning left) (meaning right) :=
  robust_ball_has_unique_graph metric valid equivalent render meaning noiseRadius
    (epsilon_delta_gate_implies_robust_separation metric valid equivalent
      render meaning noiseRadius separationLowerBound gate)
    observed left right leftValid rightValid nearLeft nearRight

/-!
若候选生成器声称候选图由 epsilon-ball 中的合法 scene 支撑，则稳定门保证任意
两个候选图等价。这里只证明唯一性，不证明候选生成器具备 soundness/completeness。
-/
def BallCandidateSound
    {Scene Raster Graph : Type}
    (metric : NatDistance Raster) (valid : Scene -> Prop)
    (render : Scene -> Raster) (meaning : Scene -> Graph)
    (noiseRadius : Nat) (candidates : Raster -> CandidateSet Graph) : Prop :=
  forall observed graph, candidates observed graph ->
    Exists fun scene => valid scene ∧
      metric.dist observed (render scene) <= noiseRadius ∧ meaning scene = graph

theorem stable_ball_candidates_have_one_graph_up_to_equivalence
    {Scene Raster Graph : Type}
    (metric : NatDistance Raster)
    (valid : Scene -> Prop) (equivalent : Graph -> Graph -> Prop)
    (render : Scene -> Raster) (meaning : Scene -> Graph)
    (noiseRadius separationLowerBound : Nat)
    (gate : EpsilonDeltaStabilityGate metric valid equivalent render meaning
      noiseRadius separationLowerBound)
    (candidates : Raster -> CandidateSet Graph)
    (candidateSound : BallCandidateSound
      metric valid render meaning noiseRadius candidates)
    (observed : Raster) (leftGraph rightGraph : Graph)
    (leftCandidate : candidates observed leftGraph)
    (rightCandidate : candidates observed rightGraph) :
    equivalent leftGraph rightGraph := by
  rcases candidateSound observed leftGraph leftCandidate with
    ⟨left, leftValid, nearLeft, leftMeaning⟩
  rcases candidateSound observed rightGraph rightCandidate with
    ⟨right, rightValid, nearRight, rightMeaning⟩
  rw [← leftMeaning, ← rightMeaning]
  exact epsilon_delta_ball_has_unique_graph metric valid equivalent render meaning
    noiseRadius separationLowerBound gate observed left right
    leftValid rightValid nearLeft nearRight

end ChemGraph
