import Std

namespace ChemGraph

namespace CarrierCandidateSetV03

/-!
PNG proposer v0.3 的有限候选集合合同。

本模块只形式化有限 boundary pairing、boundary-to-stroke 与 lane-to-bond 商映射、
stroke ownership 排他性，以及 interpretation 有限完备枚举。v0.3 序列化
只允许 candidate set 或 reject，两者都不发出 graph；文件后部的
accept/abstain 决策只为未来升级保留，不属于 v0.3 序列化。
它不声明候选能够从真实 PNG 完备或准确地提取。
-/

def boundaryPairDomain
    (boundaries : List Nat) : List (Nat × Nat) :=
  boundaries.flatMap fun left =>
    boundaries.map fun right => (left, right)

/-- 在有限 boundary 域中枚举全部满足冻结谓词的有序规范 pair。 -/
def enumerateBoundaryPairs
    (boundaries : List Nat)
    (feasiblePair : Nat -> Nat -> Bool) : List (Nat × Nat) :=
  (boundaryPairDomain boundaries).filter fun pair =>
    decide (pair.1 < pair.2) && feasiblePair pair.1 pair.2

theorem pairEnumerator_sound
    {boundaries : List Nat}
    {feasiblePair : Nat -> Nat -> Bool}
    {left right : Nat}
    (member : (left, right) ∈ enumerateBoundaryPairs boundaries feasiblePair) :
    left ∈ boundaries /\ right ∈ boundaries /\
      left < right /\ feasiblePair left right = true := by
  have reduced :
      (left ∈ boundaries /\ right ∈ boundaries) /\
        left < right /\ feasiblePair left right = true := by
    simpa [enumerateBoundaryPairs, boundaryPairDomain] using member
  exact ⟨reduced.1.1, reduced.1.2, reduced.2⟩

theorem pairEnumerator_complete
    {boundaries : List Nat}
    {feasiblePair : Nat -> Nat -> Bool}
    {left right : Nat}
    (leftMember : left ∈ boundaries)
    (rightMember : right ∈ boundaries)
    (ordered : left < right)
    (feasible : feasiblePair left right = true) :
    (left, right) ∈ enumerateBoundaryPairs boundaries feasiblePair := by
  simp [enumerateBoundaryPairs, boundaryPairDomain, leftMember, rightMember,
    ordered, feasible]

/-- 可序列化的 boundary-to-stroke 商映射及其精确 fiber 证书。 -/
structure BoundaryStrokeQuotient (Boundary Stroke : Type) where
  strokeOf : Boundary -> Stroke
  samePhysicalStroke : Boundary -> Boundary -> Prop
  exactFiber : forall left right,
    samePhysicalStroke left right <-> strokeOf left = strokeOf right
  fiberNonempty : forall stroke, Exists fun boundary => strokeOf boundary = stroke

theorem boundary_sameStroke_iff_same_id
    {Boundary Stroke : Type}
    (quotient : BoundaryStrokeQuotient Boundary Stroke)
    (left right : Boundary) :
    quotient.samePhysicalStroke left right <->
      quotient.strokeOf left = quotient.strokeOf right :=
  quotient.exactFiber left right

theorem stroke_fiber_nonempty
    {Boundary Stroke : Type}
    (quotient : BoundaryStrokeQuotient Boundary Stroke)
    (stroke : Stroke) :
    Exists fun boundary => quotient.strokeOf boundary = stroke :=
  quotient.fiberNonempty stroke

inductive StrokeOwner (Lane Glyph Marker Charge : Type) where
  | bondLane (lane : Lane)
  | textGlyph (glyph : Glyph)
  | marker (marker : Marker)
  | charge (charge : Charge)
  | residual
deriving DecidableEq, Repr

structure OwnershipPartition
    (Stroke Lane Glyph Marker Charge : Type) where
  owner : Stroke -> StrokeOwner Lane Glyph Marker Charge

theorem bond_text_ownership_exclusive
    {Stroke Lane Glyph Marker Charge : Type}
    (ownership : OwnershipPartition Stroke Lane Glyph Marker Charge)
    {stroke : Stroke} {lane : Lane} {glyph : Glyph}
    (bondOwner : ownership.owner stroke = .bondLane lane)
    (textOwner : ownership.owner stroke = .textGlyph glyph) : False := by
  have impossible :
      StrokeOwner.bondLane lane = (StrokeOwner.textGlyph glyph :
        StrokeOwner Lane Glyph Marker Charge) :=
    Eq.trans bondOwner.symm textOwner
  cases impossible

/-- 一个 semantic Bond 是其 1..3 条 lane 的精确且无重复 fiber。 -/
structure LaneBondQuotient (Lane Bond : Type) where
  bondOf : Lane -> Bond
  lanesOf : Bond -> List Lane
  lanesNodup : forall bond, (lanesOf bond).Nodup
  exactFiber : forall lane bond, lane ∈ lanesOf bond <-> bondOf lane = bond
  fiberNonempty : forall bond, Not (lanesOf bond = [])
  order : Bond -> Nat
  orderExact : forall bond, order bond = (lanesOf bond).length
  orderLower : forall bond, 1 <= order bond
  orderUpper : forall bond, order bond <= 3

theorem lane_member_iff_bond_id
    {Lane Bond : Type}
    (quotient : LaneBondQuotient Lane Bond)
    (lane : Lane) (bond : Bond) :
    lane ∈ quotient.lanesOf bond <-> quotient.bondOf lane = bond :=
  quotient.exactFiber lane bond

theorem bond_lane_fiber_nonempty
    {Lane Bond : Type}
    (quotient : LaneBondQuotient Lane Bond)
    (bond : Bond) :
    Not (quotient.lanesOf bond = []) :=
  quotient.fiberNonempty bond

theorem bond_order_eq_lane_count
    {Lane Bond : Type}
    (quotient : LaneBondQuotient Lane Bond)
    (bond : Bond) :
    quotient.order bond = (quotient.lanesOf bond).length :=
  quotient.orderExact bond

theorem bond_order_between_one_and_three
    {Lane Bond : Type}
    (quotient : LaneBondQuotient Lane Bond)
    (bond : Bond) :
    1 <= quotient.order bond /\ quotient.order bond <= 3 :=
  ⟨quotient.orderLower bond, quotient.orderUpper bond⟩

def allEquivalent
    {Interpretation : Type}
    (candidates : List Interpretation)
    (equivalent : Interpretation -> Interpretation -> Bool) : Bool :=
  candidates.all fun left =>
    candidates.all fun right => equivalent left right

inductive CandidateDecision (Interpretation : Type) where
  | accept (equivalentClass : List Interpretation)
  | abstain (candidates : List Interpretation)
  | reject

/-- 在任意有限 candidate/interpretation 域上枚举全部 feasible 项。 -/
def enumerateFeasibleCandidates
    {Interpretation : Type}
    (domain : List Interpretation)
    (feasible : Interpretation -> Bool) : List Interpretation :=
  domain.filter feasible

theorem candidateEnumerator_sound
    {Interpretation : Type}
    {domain : List Interpretation}
    {feasible : Interpretation -> Bool}
    {candidate : Interpretation}
    (member : candidate ∈ enumerateFeasibleCandidates domain feasible) :
    candidate ∈ domain /\ feasible candidate = true := by
  simpa [enumerateFeasibleCandidates] using member

theorem candidateEnumerator_complete
    {Interpretation : Type}
    {domain : List Interpretation}
    {feasible : Interpretation -> Bool}
    {candidate : Interpretation}
    (domainMember : candidate ∈ domain)
    (isFeasible : feasible candidate = true) :
    candidate ∈ enumerateFeasibleCandidates domain feasible := by
  simp [enumerateFeasibleCandidates, domainMember, isFeasible]

/-- 与 v0.3 schema 对齐的四个根级有限候选宇宙。 -/
structure RootFiniteUniverses
    (Boundary CompatiblePair AtomProposal DirectPair : Type) where
  boundaryUniverse : List Boundary
  boundaryNodup : boundaryUniverse.Nodup
  boundaryNonempty : Not (boundaryUniverse = [])
  boundaryWithinCapacity : boundaryUniverse.length <= 128
  compatiblePairUniverse : List CompatiblePair
  compatiblePairNodup : compatiblePairUniverse.Nodup
  compatiblePairWithinCapacity : compatiblePairUniverse.length <= 8128
  compatiblePairEndpoints : CompatiblePair -> Boundary × Boundary
  compatiblePairEndpointsRooted : forall pair,
    pair ∈ compatiblePairUniverse ->
      (compatiblePairEndpoints pair).1 ∈ boundaryUniverse /\
      (compatiblePairEndpoints pair).2 ∈ boundaryUniverse
  atomProposalUniverse : List AtomProposal
  atomProposalNodup : atomProposalUniverse.Nodup
  atomProposalWithinCapacity : atomProposalUniverse.length <= 128
  directPairUniverse : List DirectPair
  directPairNodup : directPairUniverse.Nodup
  directPairWithinCapacity : directPairUniverse.length <= 8128
  directPairEndpoints : DirectPair -> AtomProposal × AtomProposal
  directPairEndpointsRooted : forall pair,
    pair ∈ directPairUniverse ->
      (directPairEndpoints pair).1 ∈ atomProposalUniverse /\
      (directPairEndpoints pair).2 ∈ atomProposalUniverse

theorem compatible_pair_endpoints_belong_to_root
    {Boundary CompatiblePair AtomProposal DirectPair : Type}
    (roots : RootFiniteUniverses
      Boundary CompatiblePair AtomProposal DirectPair)
    {pair : CompatiblePair}
    (member : pair ∈ roots.compatiblePairUniverse) :
    (roots.compatiblePairEndpoints pair).1 ∈ roots.boundaryUniverse /\
      (roots.compatiblePairEndpoints pair).2 ∈ roots.boundaryUniverse :=
  roots.compatiblePairEndpointsRooted pair member

theorem direct_pair_endpoints_belong_to_root
    {Boundary CompatiblePair AtomProposal DirectPair : Type}
    (roots : RootFiniteUniverses
      Boundary CompatiblePair AtomProposal DirectPair)
    {pair : DirectPair}
    (member : pair ∈ roots.directPairUniverse) :
    (roots.directPairEndpoints pair).1 ∈ roots.atomProposalUniverse /\
      (roots.directPairEndpoints pair).2 ∈ roots.atomProposalUniverse :=
  roots.directPairEndpointsRooted pair member

/-- 正常 candidate-set 输出中冻结的 12 个执行阶段。 -/
inductive StageTraceStage where
  | preprocess
  | semanticInk
  | boundaryDetection
  | boundaryCompatibility
  | boundaryMatching
  | atomProposals
  | directPairs
  | strokeOwnership
  | endpointAtomRelations
  | interpretations
  | graphEquivalence
  | ambiguityTaxonomy
deriving DecidableEq, Repr

def stageTraceOrder : List StageTraceStage :=
  [.preprocess, .semanticInk, .boundaryDetection, .boundaryCompatibility,
    .boundaryMatching, .atomProposals, .directPairs, .strokeOwnership,
    .endpointAtomRelations, .interpretations, .graphEquivalence,
    .ambiguityTaxonomy]

def StageTraceStage.index : StageTraceStage -> Nat
  | .preprocess => 0
  | .semanticInk => 1
  | .boundaryDetection => 2
  | .boundaryCompatibility => 3
  | .boundaryMatching => 4
  | .atomProposals => 5
  | .directPairs => 6
  | .strokeOwnership => 7
  | .endpointAtomRelations => 8
  | .interpretations => 9
  | .graphEquivalence => 10
  | .ambiguityTaxonomy => 11

/-- 每个阶段必须绑定的实际 root artifact，而不是自由填写的摘要名称。 -/
inductive StageOutputKind where
  | finalForegroundRuns
  | bondAndTextMaskRuns
  | boundaryUniverseMembers
  | compatibleBoundaryPairMembers
  | boundaryMatchings
  | atomProposalUniverseMembers
  | directPairUniverseMembers
  | ownershipHypotheses
  | endpointAtomRelationRecords
  | interpretations
  | graphEquivalence
  | ambiguityTaxonomy
deriving DecidableEq, Repr

def StageTraceStage.outputKind : StageTraceStage -> StageOutputKind
  | .preprocess => .finalForegroundRuns
  | .semanticInk => .bondAndTextMaskRuns
  | .boundaryDetection => .boundaryUniverseMembers
  | .boundaryCompatibility => .compatibleBoundaryPairMembers
  | .boundaryMatching => .boundaryMatchings
  | .atomProposals => .atomProposalUniverseMembers
  | .directPairs => .directPairUniverseMembers
  | .strokeOwnership => .ownershipHypotheses
  | .endpointAtomRelations => .endpointAtomRelationRecords
  | .interpretations => .interpretations
  | .graphEquivalence => .graphEquivalence
  | .ambiguityTaxonomy => .ambiguityTaxonomy

structure StageArtifact (Hash : Type) where
  outputHash : Hash
  recordCount : Nat
  recordCountWithinCapacity : recordCount <= 589824

/-- 与 schema provenance 字段一致；重复使用同一组件的阶段共享同一字段。 -/
structure StageComponentBindings (Hash : Type) where
  preprocess : Hash
  semanticInkExtractor : Hash
  boundaryExtractor : Hash
  boundaryPairing : Hash
  enumerator : Hash
  atomSourceCheckpoint : Hash
  directAtomPairCheckpoint : Hash
  candidatePredicate : Hash
  structuralVerifier : Hash

def StageComponentBindings.forStage
    {Hash : Type} (components : StageComponentBindings Hash) :
    StageTraceStage -> Hash
  | .preprocess => components.preprocess
  | .semanticInk => components.semanticInkExtractor
  | .boundaryDetection => components.boundaryExtractor
  | .boundaryCompatibility => components.boundaryPairing
  | .boundaryMatching => components.enumerator
  | .atomProposals => components.atomSourceCheckpoint
  | .directPairs => components.directAtomPairCheckpoint
  | .strokeOwnership => components.enumerator
  | .endpointAtomRelations => components.candidatePredicate
  | .interpretations => components.enumerator
  | .graphEquivalence => components.structuralVerifier
  | .ambiguityTaxonomy => components.structuralVerifier

structure StageArtifacts (Hash : Type) where
  finalForegroundRuns : StageArtifact Hash
  bondAndTextMaskRuns : StageArtifact Hash
  boundaryUniverseMembers : StageArtifact Hash
  compatibleBoundaryPairMembers : StageArtifact Hash
  boundaryMatchings : StageArtifact Hash
  atomProposalUniverseMembers : StageArtifact Hash
  directPairUniverseMembers : StageArtifact Hash
  ownershipHypotheses : StageArtifact Hash
  endpointAtomRelationRecords : StageArtifact Hash
  interpretations : StageArtifact Hash
  graphEquivalence : StageArtifact Hash
  ambiguityTaxonomy : StageArtifact Hash

def StageArtifacts.forOutputKind
    {Hash : Type} (artifacts : StageArtifacts Hash) :
    StageOutputKind -> StageArtifact Hash
  | .finalForegroundRuns => artifacts.finalForegroundRuns
  | .bondAndTextMaskRuns => artifacts.bondAndTextMaskRuns
  | .boundaryUniverseMembers => artifacts.boundaryUniverseMembers
  | .compatibleBoundaryPairMembers => artifacts.compatibleBoundaryPairMembers
  | .boundaryMatchings => artifacts.boundaryMatchings
  | .atomProposalUniverseMembers => artifacts.atomProposalUniverseMembers
  | .directPairUniverseMembers => artifacts.directPairUniverseMembers
  | .ownershipHypotheses => artifacts.ownershipHypotheses
  | .endpointAtomRelationRecords => artifacts.endpointAtomRelationRecords
  | .interpretations => artifacts.interpretations
  | .graphEquivalence => artifacts.graphEquivalence
  | .ambiguityTaxonomy => artifacts.ambiguityTaxonomy

def StageArtifacts.forStage
    {Hash : Type} (artifacts : StageArtifacts Hash)
    (stage : StageTraceStage) : StageArtifact Hash :=
  artifacts.forOutputKind stage.outputKind

def StageArtifacts.inputForStage
    {Hash : Type} (inputPngHash : Hash) (artifacts : StageArtifacts Hash) :
    StageTraceStage -> Hash
  | .preprocess => inputPngHash
  | .semanticInk => artifacts.finalForegroundRuns.outputHash
  | .boundaryDetection => artifacts.bondAndTextMaskRuns.outputHash
  | .boundaryCompatibility => artifacts.boundaryUniverseMembers.outputHash
  | .boundaryMatching => artifacts.compatibleBoundaryPairMembers.outputHash
  | .atomProposals => artifacts.boundaryMatchings.outputHash
  | .directPairs => artifacts.atomProposalUniverseMembers.outputHash
  | .strokeOwnership => artifacts.directPairUniverseMembers.outputHash
  | .endpointAtomRelations => artifacts.ownershipHypotheses.outputHash
  | .interpretations => artifacts.endpointAtomRelationRecords.outputHash
  | .graphEquivalence => artifacts.interpretations.outputHash
  | .ambiguityTaxonomy => artifacts.graphEquivalence.outputHash

structure StageRecord (Hash : Type) where
  index : Nat
  stage : StageTraceStage
  componentHash : Hash
  inputHash : Hash
  outputHash : Hash
  recordCount : Nat
deriving Repr

def expectedStageRecord
    {Hash : Type}
    (inputPngHash : Hash)
    (components : StageComponentBindings Hash)
    (artifacts : StageArtifacts Hash)
    (stage : StageTraceStage) : StageRecord Hash :=
  { index := stage.index
    stage := stage
    componentHash := components.forStage stage
    inputHash := artifacts.inputForStage inputPngHash stage
    outputHash := (artifacts.forStage stage).outputHash
    recordCount := (artifacts.forStage stage).recordCount }

structure StageTrace (Hash : Type) where
  stages : List (StageRecord Hash)
  orderedChainHash : Hash
deriving Repr

/-- `chainDigest` 抽象 schema 指定的 SHA256 framing；密码学实现留给 validator。 -/
def buildStageTrace
    {Hash : Type}
    (inputPngHash : Hash)
    (components : StageComponentBindings Hash)
    (artifacts : StageArtifacts Hash)
    (chainDigest : List (StageRecord Hash) -> Hash) : StageTrace Hash :=
  let stages := stageTraceOrder.map
    (expectedStageRecord inputPngHash components artifacts)
  { stages := stages, orderedChainHash := chainDigest stages }

def StageTrace.Valid
    {Hash : Type}
    (inputPngHash : Hash)
    (components : StageComponentBindings Hash)
    (artifacts : StageArtifacts Hash)
    (chainDigest : List (StageRecord Hash) -> Hash)
    (trace : StageTrace Hash) : Prop :=
  trace.stages = stageTraceOrder.map
      (expectedStageRecord inputPngHash components artifacts) /\
    trace.orderedChainHash = chainDigest trace.stages

theorem built_stage_trace_valid
    {Hash : Type}
    (inputPngHash : Hash)
    (components : StageComponentBindings Hash)
    (artifacts : StageArtifacts Hash)
    (chainDigest : List (StageRecord Hash) -> Hash) :
    (buildStageTrace inputPngHash components artifacts chainDigest).Valid
      inputPngHash components artifacts chainDigest := by
  simp [StageTrace.Valid, buildStageTrace]

theorem built_stage_trace_has_twelve_stages
    {Hash : Type}
    (inputPngHash : Hash)
    (components : StageComponentBindings Hash)
    (artifacts : StageArtifacts Hash)
    (chainDigest : List (StageRecord Hash) -> Hash) :
    (buildStageTrace inputPngHash components artifacts chainDigest).stages.length = 12 := by
  rfl

theorem built_stage_trace_has_frozen_order
    {Hash : Type}
    (inputPngHash : Hash)
    (components : StageComponentBindings Hash)
    (artifacts : StageArtifacts Hash)
    (chainDigest : List (StageRecord Hash) -> Hash) :
    (buildStageTrace inputPngHash components artifacts chainDigest).stages.map
      StageRecord.stage = stageTraceOrder := by
  simp [buildStageTrace, stageTraceOrder, expectedStageRecord]

theorem expected_stage_record_count_within_capacity
    {Hash : Type}
    (inputPngHash : Hash)
    (components : StageComponentBindings Hash)
    (artifacts : StageArtifacts Hash)
    (stage : StageTraceStage) :
    (expectedStageRecord inputPngHash components artifacts stage).recordCount <=
      589824 :=
  (artifacts.forStage stage).recordCountWithinCapacity

/--
枚举器必须把相对于同一份 root universe 的全部 feasible interpretation
放入有限 domain；这个证书不由 JSON 中自签的 `complete=true` 代替。
-/
structure RootedInterpretationUniverse
    (Root Interpretation : Type) where
  roots : Root
  domain : List Interpretation
  domainNodup : domain.Nodup
  feasible : Root -> Interpretation -> Bool
  sound : forall interpretation, interpretation ∈ domain ->
    feasible roots interpretation = true
  complete : forall interpretation, feasible roots interpretation = true ->
    interpretation ∈ domain

abbrev CarrierRootedInterpretationUniverse
    (Boundary CompatiblePair AtomProposal DirectPair Interpretation : Type) :=
  RootedInterpretationUniverse
    (RootFiniteUniverses Boundary CompatiblePair AtomProposal DirectPair)
    Interpretation

def enumerateRootedInterpretations
    {Root Interpretation : Type}
    (rootedUniverse : RootedInterpretationUniverse Root Interpretation) :
    List Interpretation :=
  enumerateFeasibleCandidates rootedUniverse.domain
    (rootedUniverse.feasible rootedUniverse.roots)

theorem rooted_interpretation_enumerator_sound
    {Root Interpretation : Type}
    (rootedUniverse : RootedInterpretationUniverse Root Interpretation)
    {interpretation : Interpretation}
    (member : interpretation ∈ enumerateRootedInterpretations rootedUniverse) :
    rootedUniverse.feasible rootedUniverse.roots interpretation = true := by
  exact (candidateEnumerator_sound member).2

theorem rooted_interpretation_enumerator_complete
    {Root Interpretation : Type}
    (rootedUniverse : RootedInterpretationUniverse Root Interpretation)
    {interpretation : Interpretation}
    (isFeasible : rootedUniverse.feasible rootedUniverse.roots interpretation = true) :
    interpretation ∈ enumerateRootedInterpretations rootedUniverse := by
  exact candidateEnumerator_complete
    (rootedUniverse.complete interpretation isFeasible) isFeasible

/-- 空图只要求结构图对象为空；fragment ownership 仍可以表达 glyph/charge。 -/
structure InterpretationShape where
  atomCount : Nat
  portCount : Nat
  endpointRelationCount : Nat
  laneCount : Nat
  bondCount : Nat
deriving DecidableEq, Repr

def InterpretationShape.IsEmptyGraph (shape : InterpretationShape) : Prop :=
  shape.atomCount = 0 /\ shape.portCount = 0 /\
    shape.endpointRelationCount = 0 /\ shape.laneCount = 0 /\
    shape.bondCount = 0

theorem feasible_empty_graph_interpretation_is_enumerated
    {Root Interpretation : Type}
    (rootedUniverse : RootedInterpretationUniverse Root Interpretation)
    (shapeOf : Interpretation -> InterpretationShape)
    {interpretation : Interpretation}
    (_emptyGraph : (shapeOf interpretation).IsEmptyGraph)
    (isFeasible : rootedUniverse.feasible rootedUniverse.roots interpretation = true) :
    interpretation ∈ enumerateRootedInterpretations rootedUniverse :=
  rooted_interpretation_enumerator_complete rootedUniverse isFeasible

/-- 保留旧内部名称；它与通用完备枚举器完全相同。 -/
def feasibleCandidates
    {Interpretation : Type}
    (domain : List Interpretation)
    (feasible : Interpretation -> Bool) : List Interpretation :=
  enumerateFeasibleCandidates domain feasible

/-- v0.3 closed-schema Reject 原因。 -/
inductive RejectReason where
  | invalidPng
  | outsideCapacity
  | unsupportedGeometry
  | noSupportedCarrier
  | enumerationIncomplete
  | unsupportedPrimitive
deriving DecidableEq, Repr

/-- v0.3 closed-schema Reject 阶段。 -/
inductive RejectStage where
  | input
  | semanticInk
  | boundaryDetection
  | boundaryEnumeration
  | ownershipEnumeration
  | atomEnumeration
  | portEnumeration
  | laneEnumeration
  | bondEnumeration
  | graphEquivalence
  | ambiguityTaxonomy
  | serialization
deriving DecidableEq, Repr

/-- v0.3 序列化边界：只发布 candidate set 或 closed reject。 -/
inductive SerializedCandidateOutput (Interpretation : Type) where
  | candidateSet (candidates : List Interpretation)
  | reject (reason : RejectReason) (stage : RejectStage)

def serializeCandidateSet
    {Interpretation : Type}
    (domain : List Interpretation)
    (feasible : Interpretation -> Bool) :
    SerializedCandidateOutput Interpretation :=
  let candidates := enumerateFeasibleCandidates domain feasible
  if candidates = [] then
    .reject .noSupportedCarrier .bondEnumeration
  else
    .candidateSet candidates

def SerializedCandidateOutput.EmitsGraph {Interpretation : Type} :
    SerializedCandidateOutput Interpretation -> Prop
  | .candidateSet _ => False
  | .reject _ _ => False

theorem serialized_candidateSet_emits_no_graph
    {Interpretation : Type} (candidates : List Interpretation) :
    Not (SerializedCandidateOutput.candidateSet candidates).EmitsGraph := by
  intro emits
  exact emits

theorem serialized_reject_emits_no_graph
    {Interpretation : Type} (reason : RejectReason) (stage : RejectStage) :
    Not (SerializedCandidateOutput.reject reason stage :
      SerializedCandidateOutput Interpretation).EmitsGraph := by
  intro emits
  exact emits

theorem serialized_output_never_emits_graph
    {Interpretation : Type}
    (output : SerializedCandidateOutput Interpretation) :
    Not output.EmitsGraph := by
  cases output with
  | candidateSet candidates =>
      exact serialized_candidateSet_emits_no_graph candidates
  | reject reason stage =>
      exact serialized_reject_emits_no_graph reason stage

theorem serialized_candidate_set_nonempty
    {Interpretation : Type}
    {domain candidates : List Interpretation}
    {feasible : Interpretation -> Bool}
    (serialized : serializeCandidateSet domain feasible =
      SerializedCandidateOutput.candidateSet candidates) :
    Not (candidates = []) := by
  by_cases emptyEnumerated : enumerateFeasibleCandidates domain feasible = []
  · simp [serializeCandidateSet, emptyEnumerated] at serialized
  · have candidateIdentity :
        enumerateFeasibleCandidates domain feasible = candidates := by
      simpa [serializeCandidateSet, emptyEnumerated] using serialized
    intro candidatesEmpty
    exact emptyEnumerated (candidateIdentity.trans candidatesEmpty)

theorem serialized_reject_iff_no_feasible_candidates
    {Interpretation : Type}
    {domain : List Interpretation}
    {feasible : Interpretation -> Bool} :
    serializeCandidateSet domain feasible =
      SerializedCandidateOutput.reject .noSupportedCarrier .bondEnumeration <->
    enumerateFeasibleCandidates domain feasible = [] := by
  by_cases emptyEnumerated : enumerateFeasibleCandidates domain feasible = []
  · simp [serializeCandidateSet, emptyEnumerated]
  · simp [serializeCandidateSet, emptyEnumerated]

theorem serialized_candidate_set_contains_every_feasible
    {Interpretation : Type}
    {domain : List Interpretation}
    {feasible : Interpretation -> Bool}
    {candidate : Interpretation}
    (domainMember : candidate ∈ domain)
    (isFeasible : feasible candidate = true) :
    Exists fun candidates =>
      serializeCandidateSet domain feasible =
        SerializedCandidateOutput.candidateSet candidates /\
      candidate ∈ candidates := by
  have member : candidate ∈ enumerateFeasibleCandidates domain feasible :=
    candidateEnumerator_complete domainMember isFeasible
  have nonempty : Not (enumerateFeasibleCandidates domain feasible = []) := by
    intro empty
    rw [empty] at member
    cases member
  exact ⟨enumerateFeasibleCandidates domain feasible, by
    simp [serializeCandidateSet, nonempty], member⟩

theorem serialized_feasible_empty_graph_is_emitted
    {Root Interpretation : Type}
    (rootedUniverse : RootedInterpretationUniverse Root Interpretation)
    (shapeOf : Interpretation -> InterpretationShape)
    {interpretation : Interpretation}
    (_emptyGraph : (shapeOf interpretation).IsEmptyGraph)
    (isFeasible : rootedUniverse.feasible rootedUniverse.roots interpretation = true) :
    Exists fun candidates =>
      serializeCandidateSet rootedUniverse.domain
          (rootedUniverse.feasible rootedUniverse.roots) =
        SerializedCandidateOutput.candidateSet candidates /\
      interpretation ∈ candidates := by
  exact serialized_candidate_set_contains_every_feasible
    (rootedUniverse.complete interpretation isFeasible) isFeasible

def decideInterpretations
    {Interpretation : Type}
    (domain : List Interpretation)
    (feasible : Interpretation -> Bool)
    (equivalent : Interpretation -> Interpretation -> Bool) :
    CandidateDecision Interpretation :=
  let candidates := feasibleCandidates domain feasible
  if candidates = [] then
    .reject
  else if allEquivalent candidates equivalent = true then
    .accept candidates
  else
    .abstain candidates

theorem inequivalent_feasible_interpretations_force_abstain
    {Interpretation : Type}
    {domain : List Interpretation}
    {feasible : Interpretation -> Bool}
    {equivalent : Interpretation -> Interpretation -> Bool}
    {left right : Interpretation}
    (leftMember : left ∈ domain)
    (rightMember : right ∈ domain)
    (leftFeasible : feasible left = true)
    (rightFeasible : feasible right = true)
    (inequivalent : equivalent left right = false) :
    decideInterpretations domain feasible equivalent =
      .abstain (feasibleCandidates domain feasible) := by
  let candidates := feasibleCandidates domain feasible
  have leftCandidate : left ∈ candidates := by
    exact candidateEnumerator_complete leftMember leftFeasible
  have rightCandidate : right ∈ candidates := by
    exact candidateEnumerator_complete rightMember rightFeasible
  have candidatesNonempty : Not (candidates = []) := by
    intro empty
    rw [empty] at leftCandidate
    cases leftCandidate
  have notOneClass : Not (allEquivalent candidates equivalent = true) := by
    intro oneClass
    have outer := (List.all_eq_true.mp oneClass) left leftCandidate
    have pairTrue := (List.all_eq_true.mp outer) right rightCandidate
    rw [inequivalent] at pairTrue
    exact Bool.noConfusion pairTrue
  simp [decideInterpretations, candidates, candidatesNonempty, notOneClass]

def CandidateDecision.EmitsGraph {Interpretation : Type} :
    CandidateDecision Interpretation -> Prop
  | .accept _ => True
  | .abstain _ => False
  | .reject => False

theorem reject_emits_no_graph {Interpretation : Type} :
    Not (CandidateDecision.reject : CandidateDecision Interpretation).EmitsGraph := by
  intro emits
  exact emits

theorem abstain_emits_no_graph
    {Interpretation : Type} (candidates : List Interpretation) :
    Not (CandidateDecision.abstain candidates).EmitsGraph := by
  intro emits
  exact emits

#print axioms pairEnumerator_sound
#print axioms pairEnumerator_complete
#print axioms boundary_sameStroke_iff_same_id
#print axioms stroke_fiber_nonempty
#print axioms bond_text_ownership_exclusive
#print axioms lane_member_iff_bond_id
#print axioms bond_lane_fiber_nonempty
#print axioms bond_order_eq_lane_count
#print axioms bond_order_between_one_and_three
#print axioms candidateEnumerator_sound
#print axioms candidateEnumerator_complete
#print axioms compatible_pair_endpoints_belong_to_root
#print axioms direct_pair_endpoints_belong_to_root
#print axioms built_stage_trace_valid
#print axioms built_stage_trace_has_twelve_stages
#print axioms built_stage_trace_has_frozen_order
#print axioms expected_stage_record_count_within_capacity
#print axioms rooted_interpretation_enumerator_sound
#print axioms rooted_interpretation_enumerator_complete
#print axioms feasible_empty_graph_interpretation_is_enumerated
#print axioms serialized_candidateSet_emits_no_graph
#print axioms serialized_reject_emits_no_graph
#print axioms serialized_output_never_emits_graph
#print axioms serialized_candidate_set_nonempty
#print axioms serialized_reject_iff_no_feasible_candidates
#print axioms serialized_candidate_set_contains_every_feasible
#print axioms serialized_feasible_empty_graph_is_emitted
#print axioms inequivalent_feasible_interpretations_force_abstain
#print axioms reject_emits_no_graph
#print axioms abstain_emits_no_graph

end CarrierCandidateSetV03

end ChemGraph
