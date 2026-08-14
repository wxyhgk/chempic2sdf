import ChemGraph.ObservationRecovery
import Std

namespace ChemGraph

/-!
Observation Closure v0.5 的纯数学内核。

本文件只证明冻结接口之间的逻辑关系。它不声称 Lean 已执行 Python
rasterizer/extractor、完整枚举、JSON 序列化、syscall 隔离或 Git 冻结审计。
-/

namespace ObservationClosure

structure Bijection (Alpha Beta : Type) where
  toFun : Alpha -> Beta
  invFun : Beta -> Alpha
  leftInverse : forall value, invFun (toFun value) = value
  rightInverse : forall value, toFun (invFun value) = value

def Bijection.refl (Alpha : Type) : Bijection Alpha Alpha where
  toFun := id
  invFun := id
  leftInverse := fun _ => rfl
  rightInverse := fun _ => rfl

def Bijection.symm {Alpha Beta : Type}
    (bijection : Bijection Alpha Beta) : Bijection Beta Alpha where
  toFun := bijection.invFun
  invFun := bijection.toFun
  leftInverse := bijection.rightInverse
  rightInverse := bijection.leftInverse

def Bijection.trans {Alpha Beta Gamma : Type}
    (first : Bijection Alpha Beta) (second : Bijection Beta Gamma) :
    Bijection Alpha Gamma where
  toFun := fun value => second.toFun (first.toFun value)
  invFun := fun value => first.invFun (second.invFun value)
  leftInverse := by
    intro value
    rw [second.leftInverse, first.leftInverse]
  rightInverse := by
    intro value
    rw [first.rightInverse, second.rightInverse]

def Bijection.prodCongr {Alpha Beta Gamma Delta : Type}
    (first : Bijection Alpha Beta) (second : Bijection Gamma Delta) :
    Bijection (Alpha × Gamma) (Beta × Delta) where
  toFun := fun value => (first.toFun value.1, second.toFun value.2)
  invFun := fun value => (first.invFun value.1, second.invFun value.2)
  leftInverse := by
    intro value
    cases value
    rw [first.leftInverse, second.leftInverse]
  rightInverse := by
    intro value
    cases value
    rw [first.rightInverse, second.rightInverse]

/-! ## 冻结的有限 tuple 域和从属 semantic scene -/

structure RenderTuple where
  halfLength : Fin 51
  motif : Fin 52
  width : Fin 9
  cap : Fin 3
  antialias : Fin 3
  reflection : Fin 3
  rotation : Fin 8
  scale : Fin 7
deriving DecidableEq, Repr

abbrev DeliveryTranslation := Fin 65 × Fin 65

abbrev RenderTupleAxes :=
  Fin 51 × (Fin 52 × (Fin 9 × (Fin 3 × (Fin 3 × (Fin 3 × (Fin 8 × Fin 7))))))

def encodeRenderTupleAxes (tuple : RenderTuple) : RenderTupleAxes :=
  (tuple.halfLength, (tuple.motif, (tuple.width, (tuple.cap,
    (tuple.antialias, (tuple.reflection, (tuple.rotation, tuple.scale)))))))

def decodeRenderTupleAxes (axes : RenderTupleAxes) : RenderTuple := {
  halfLength := axes.1
  motif := axes.2.1
  width := axes.2.2.1
  cap := axes.2.2.2.1
  antialias := axes.2.2.2.2.1
  reflection := axes.2.2.2.2.2.1
  rotation := axes.2.2.2.2.2.2.1
  scale := axes.2.2.2.2.2.2.2
}

theorem decodeRenderTupleAxes_encodeRenderTupleAxes (tuple : RenderTuple) :
    decodeRenderTupleAxes (encodeRenderTupleAxes tuple) = tuple := by
  cases tuple
  rfl

theorem encodeRenderTupleAxes_decodeRenderTupleAxes (axes : RenderTupleAxes) :
    encodeRenderTupleAxes (decodeRenderTupleAxes axes) = axes := by
  rcases axes with ⟨halfLength, motif, width, cap, antialias, reflection,
    rotation, scale⟩
  rfl

def renderTupleAxesBijection : Bijection RenderTuple RenderTupleAxes where
  toFun := encodeRenderTupleAxes
  invFun := decodeRenderTupleAxes
  leftInverse := decodeRenderTupleAxes_encodeRenderTupleAxes
  rightInverse := encodeRenderTupleAxes_decodeRenderTupleAxes

abbrev DeliveryLiftAxes := RenderTupleAxes × DeliveryTranslation

def deliveryLiftAxesBijection :
    Bijection (RenderTuple × DeliveryTranslation) DeliveryLiftAxes :=
  Bijection.prodCongr renderTupleAxesBijection (Bijection.refl DeliveryTranslation)

def renderTupleCard : Nat := 51 * 52 * 9 * 3 * 3 * 3 * 8 * 7
def deliveryTranslationCard : Nat := 65 * 65
def deliveryLiftCard : Nat := renderTupleCard * deliveryTranslationCard

theorem renderTuple_card : renderTupleCard = 36088416 := by
  decide

theorem deliveryTranslation_card : deliveryTranslationCard = 4225 := by
  decide

theorem deliveryLift_card : deliveryLiftCard = 152473557600 := by
  decide

structure DependentDomain where
  Q : Type
  Interpretation : Q -> Type

def DependentDomain.Scene (domain : DependentDomain) : Type :=
  (q : domain.Q) × domain.Interpretation q

/-! `Result` 保持 success 和 failure 的类型不同，避免把 failure 当 Evidence bytes。 -/
inductive Result (Success Failure : Type) where
  | success (value : Success)
  | failure (error : Failure)
deriving Repr

/-!
`ClosureModel` 共享同一组 `R/E/M` 函数。`active` 只选择 tuple；因此 sampled
和 full 的“一致限制”由类型保证，而不是另设一个可能不一致的副本。
-/
structure ClosureModel where
  Q : Type
  Interpretation : Q -> Type
  Raster : Type
  Evidence : Type
  Graph : Type
  ExtractFailure : Type
  active : Q -> Prop
  raster : Q -> Raster
  extract : Raster -> Result Evidence ExtractFailure
  meaning : (q : Q) -> Interpretation q -> Graph
  equivalent : Graph -> Graph -> Prop

def ClosureModel.DependentScene (model : ClosureModel) : Type :=
  (q : model.Q) × model.Interpretation q

def RasterGraphSet (model : ClosureModel) (raster : model.Raster)
    (graph : model.Graph) : Prop :=
  Exists fun q => Exists fun interpretation : model.Interpretation q =>
    model.active q ∧ model.raster q = raster ∧
      model.equivalent graph (model.meaning q interpretation)

def SuccessfulEvidenceFiber (model : ClosureModel) (evidence : model.Evidence)
    (raster : model.Raster) : Prop :=
  (Exists fun q => model.active q ∧ model.raster q = raster) ∧
    model.extract raster = .success evidence

def EvidenceGraphSet (model : ClosureModel) (evidence : model.Evidence)
    (graph : model.Graph) : Prop :=
  Exists fun raster =>
    SuccessfulEvidenceFiber model evidence raster ∧
      RasterGraphSet model raster graph

theorem evidenceGraphSet_is_union_of_rasterGraphSets
    (model : ClosureModel) (evidence : model.Evidence) (graph : model.Graph) :
    EvidenceGraphSet model evidence graph ↔
      Exists fun raster =>
        SuccessfulEvidenceFiber model evidence raster ∧
          RasterGraphSet model raster graph := by
  rfl

theorem raster_fiber_included_in_successful_evidence_fiber
    (model : ClosureModel) (raster : model.Raster) (evidence : model.Evidence)
    (success : model.extract raster = .success evidence) :
    forall graph, RasterGraphSet model raster graph ->
      EvidenceGraphSet model evidence graph := by
  intro graph member
  rcases member with ⟨q, interpretation, active, rendered, equivalent⟩
  exact ⟨raster, ⟨⟨q, active, rendered⟩, success⟩,
    ⟨q, interpretation, active, rendered, equivalent⟩⟩

theorem same_evidence_coarsens_raster_fibers
    (model : ClosureModel)
    (left right : model.Raster) (evidence : model.Evidence)
    (leftReachable : Exists fun q => model.active q ∧ model.raster q = left)
    (rightReachable : Exists fun q => model.active q ∧ model.raster q = right)
    (leftSuccess : model.extract left = .success evidence)
    (rightSuccess : model.extract right = .success evidence) :
    (forall graph, RasterGraphSet model left graph ->
      EvidenceGraphSet model evidence graph) ∧
    (forall graph, RasterGraphSet model right graph ->
      EvidenceGraphSet model evidence graph) := by
  constructor
  · intro graph member
    exact ⟨left, ⟨leftReachable, leftSuccess⟩, member⟩
  · intro graph member
    exact ⟨right, ⟨rightReachable, rightSuccess⟩, member⟩

/-! ## sampled/full 单调性 -/

def Restrict (model : ClosureModel) (active : model.Q -> Prop) : ClosureModel where
  Q := model.Q
  Interpretation := model.Interpretation
  Raster := model.Raster
  Evidence := model.Evidence
  Graph := model.Graph
  ExtractFailure := model.ExtractFailure
  active := active
  raster := model.raster
  extract := model.extract
  meaning := model.meaning
  equivalent := model.equivalent

theorem rasterGraphSet_mono
    (model : ClosureModel) (sample full : model.Q -> Prop)
    (subset : forall q, sample q -> full q)
    (raster : model.Raster) (graph : model.Graph) :
    RasterGraphSet (Restrict model sample) raster graph ->
      RasterGraphSet (Restrict model full) raster graph := by
  rintro ⟨q, interpretation, sampled, rendered, equivalent⟩
  exact ⟨q, interpretation, subset q sampled, rendered, equivalent⟩

theorem evidenceGraphSet_mono
    (model : ClosureModel) (sample full : model.Q -> Prop)
    (subset : forall q, sample q -> full q)
    (evidence : model.Evidence) (graph : model.Graph) :
    EvidenceGraphSet (Restrict model sample) evidence graph ->
      EvidenceGraphSet (Restrict model full) evidence graph := by
  rintro ⟨raster, ⟨⟨q, sampled, rendered⟩, extracted⟩, graphMember⟩
  refine ⟨raster, ⟨⟨q, subset q sampled, rendered⟩, extracted⟩, ?_⟩
  exact rasterGraphSet_mono model sample full subset raster graph graphMember

def MultipleClasses {Graph : Type} (equivalent : Graph -> Graph -> Prop)
    (members : Graph -> Prop) : Prop :=
  Exists fun left => Exists fun right =>
    members left ∧ members right ∧ Not (equivalent left right)

def SingletonClass {Graph : Type} (equivalent : Graph -> Graph -> Prop)
    (members : Graph -> Prop) : Prop :=
  Exists fun representative =>
    members representative ∧ forall graph, members graph -> equivalent graph representative

def EmptyClass {Graph : Type} (members : Graph -> Prop) : Prop :=
  forall graph, Not (members graph)

theorem sampled_ambiguous_implies_full_ambiguous
    (model : ClosureModel) (sample full : model.Q -> Prop)
    (subset : forall q, sample q -> full q) (evidence : model.Evidence) :
    MultipleClasses model.equivalent
        (EvidenceGraphSet (Restrict model sample) evidence) ->
      MultipleClasses model.equivalent
        (EvidenceGraphSet (Restrict model full) evidence) := by
  rintro ⟨left, right, leftMember, rightMember, different⟩
  exact ⟨left, right,
    evidenceGraphSet_mono model sample full subset evidence left leftMember,
    evidenceGraphSet_mono model sample full subset evidence right rightMember,
    different⟩

/-!
单元素 sample 不能推出 full 仍为单元素。这个可执行二点模型固定了该逻辑反例，
避免把 sampled Graph 当作 full Graph 的证明。
-/
private def sampleCounterModel : ClosureModel where
  Q := Bool
  Interpretation := fun _ => Unit
  Raster := Unit
  Evidence := Unit
  Graph := Bool
  ExtractFailure := Empty
  active := fun _ => True
  raster := fun _ => ()
  extract := fun _ => .success ()
  meaning := fun q _ => q
  equivalent := Eq

private def sampleOnlyFalse : Bool -> Prop := fun q => q = false
private def sampleBoth : Bool -> Prop := fun _ => True

theorem sampled_graph_does_not_imply_full_graph :
    SingletonClass Eq
        (EvidenceGraphSet (Restrict sampleCounterModel sampleOnlyFalse) ()) ∧
      MultipleClasses Eq
        (EvidenceGraphSet (Restrict sampleCounterModel sampleBoth) ()) := by
  constructor
  · refine ⟨false, ?_, ?_⟩
    · exact ⟨(), ⟨⟨false, rfl, rfl⟩, rfl⟩,
        ⟨false, (), rfl, rfl, rfl⟩⟩
    · intro graph member
      rcases member with
        ⟨_, _, q, interpretation, active, _, graphEq⟩
      have qFalse : q = false := active
      change graph = q at graphEq
      rw [qFalse] at graphEq
      exact graphEq
  · refine ⟨false, true, ?_, ?_, ?_⟩
    · exact ⟨(), ⟨⟨false, trivial, rfl⟩, rfl⟩,
        ⟨false, (), trivial, rfl, rfl⟩⟩
    · exact ⟨(), ⟨⟨true, trivial, rfl⟩, rfl⟩,
        ⟨true, (), trivial, rfl, rfl⟩⟩
    · intro equal
      cases equal

/-! ## strict fiber、候选 coverage 与 representative 唯一性 -/

theorem singletonClass_iff_nonempty_pairwise
    {Graph : Type} (equivalent : Graph -> Graph -> Prop)
    (law : Equivalence equivalent) (members : Graph -> Prop) :
    SingletonClass equivalent members ↔
      (Exists fun graph => members graph) ∧
        forall left right, members left -> members right -> equivalent left right := by
  constructor
  · rintro ⟨representative, representativeMember, allEquivalent⟩
    refine ⟨⟨representative, representativeMember⟩, ?_⟩
    intro left right leftMember rightMember
    exact law.trans (allEquivalent left leftMember)
      (law.symm (allEquivalent right rightMember))
  · rintro ⟨⟨representative, representativeMember⟩, pairwise⟩
    exact ⟨representative, representativeMember,
      fun graph graphMember => pairwise graph representative graphMember representativeMember⟩

def CandidateSound {Graph : Type} (oracle predicted : Graph -> Prop) : Prop :=
  forall graph, predicted graph -> oracle graph

def CandidateComplete {Graph : Type} (oracle predicted : Graph -> Prop) : Prop :=
  forall graph, oracle graph -> predicted graph

theorem sound_and_complete_iff_coverage_equal
    {Graph : Type} (oracle predicted : Graph -> Prop) :
    (CandidateSound oracle predicted ∧ CandidateComplete oracle predicted) ↔
      forall graph, predicted graph ↔ oracle graph := by
  constructor
  · rintro ⟨sound, complete⟩ graph
    exact ⟨sound graph, complete graph⟩
  · intro equal
    exact ⟨fun graph => (equal graph).mp, fun graph => (equal graph).mpr⟩

def NodupUpTo {Graph : Type} (equivalent : Graph -> Graph -> Prop)
    (predicted : Graph -> Prop) : Prop :=
  forall left right, predicted left -> predicted right -> equivalent left right -> left = right

def ExactlyOneRepresentativePerClass {Graph : Type}
    (equivalent : Graph -> Graph -> Prop)
    (oracle predicted : Graph -> Prop) : Prop :=
  forall oracleGraph, oracle oracleGraph -> Exists fun representative =>
    predicted representative ∧ oracle representative ∧
      equivalent representative oracleGraph ∧
      forall other, predicted other -> equivalent other oracleGraph -> other = representative

theorem exact_coverage_and_nodup_gives_one_representative
    {Graph : Type} (equivalent : Graph -> Graph -> Prop)
    (law : Equivalence equivalent) (oracle predicted : Graph -> Prop)
    (sound : CandidateSound oracle predicted)
    (completeUpTo : forall graph, oracle graph -> Exists fun candidate =>
      predicted candidate ∧ equivalent candidate graph)
    (nodup : NodupUpTo equivalent predicted) :
    ExactlyOneRepresentativePerClass equivalent oracle predicted := by
  intro oracleGraph oracleMember
  rcases completeUpTo oracleGraph oracleMember with
    ⟨representative, representativeMember, representativeEquivalent⟩
  refine ⟨representative, representativeMember,
    sound representative representativeMember, representativeEquivalent, ?_⟩
  intro other otherMember otherEquivalent
  apply nodup other representative otherMember representativeMember
  exact law.trans otherEquivalent (law.symm representativeEquivalent)

/-! 三态的定义互斥；穷尽定理由有限枚举 witness 明确携带。 -/
structure FiniteCoverage (Graph : Type) (members : Graph -> Prop) where
  values : List Graph
  complete : forall graph, members graph -> graph ∈ values

theorem empty_not_singleton
    {Graph : Type} {equivalent : Graph -> Graph -> Prop} {members : Graph -> Prop} :
    EmptyClass members -> Not (SingletonClass equivalent members) := by
  intro empty singleton
  rcases singleton with ⟨graph, member, _⟩
  exact empty graph member

theorem empty_not_multiple
    {Graph : Type} {equivalent : Graph -> Graph -> Prop} {members : Graph -> Prop} :
    EmptyClass members -> Not (MultipleClasses equivalent members) := by
  intro empty multiple
  rcases multiple with ⟨left, _right, leftMember, _rightMember, _different⟩
  exact empty left leftMember

theorem singleton_not_multiple
    {Graph : Type} (equivalent : Graph -> Graph -> Prop)
    (law : Equivalence equivalent) (members : Graph -> Prop) :
    SingletonClass equivalent members -> Not (MultipleClasses equivalent members) := by
  rintro ⟨representative, representativeMember, allEquivalent⟩
    ⟨left, right, leftMember, rightMember, different⟩
  exact different (law.trans (allEquivalent left leftMember)
    (law.symm (allEquivalent right rightMember)))

theorem finite_three_way_classification
    {Graph : Type} [DecidableEq Graph]
    (equivalent : Graph -> Graph -> Prop) [DecidableRel equivalent]
    (_law : Equivalence equivalent) (members : Graph -> Prop)
    [DecidablePred members]
    [Decidable (Exists fun graph => members graph)]
    [Decidable (MultipleClasses equivalent members)]
    (_finite : FiniteCoverage Graph members) :
    EmptyClass members ∨ SingletonClass equivalent members ∨
      MultipleClasses equivalent members := by
  by_cases nonempty : Exists fun graph => members graph
  · rcases nonempty with ⟨anchor, anchorMember⟩
    by_cases multiple : MultipleClasses equivalent members
    · exact Or.inr (Or.inr multiple)
    · have singleton : SingletonClass equivalent members := by
        refine ⟨anchor, anchorMember, ?_⟩
        intro graph graphMember
        by_cases same : equivalent graph anchor
        · exact same
        · exact False.elim
            (multiple ⟨graph, anchor, graphMember, anchorMember, same⟩)
      exact Or.inr (Or.inl singleton)
  · have empty : EmptyClass members := by
      intro graph member
      exact nonempty ⟨graph, member⟩
    exact Or.inl empty

/-! ## 左逆与 strict selector -/

def LeftInverseUpTo {Scene Observation Graph : Type}
    (equivalent : Graph -> Graph -> Prop)
    (render : Scene -> Observation) (meaning : Scene -> Graph)
    (decode : Observation -> Graph) : Prop :=
  forall scene, equivalent (decode (render scene)) (meaning scene)

theorem global_left_inverse_requires_strict_fibers
    {Scene Observation Graph : Type}
    (equivalent : Graph -> Graph -> Prop) (law : Equivalence equivalent)
    (render : Scene -> Observation) (meaning : Scene -> Graph)
    (decode : Observation -> Graph)
    (leftInverse : LeftInverseUpTo equivalent render meaning decode) :
    forall observation left right,
      render left = observation -> render right = observation ->
        equivalent (meaning left) (meaning right) := by
  intro observation left right leftObservation rightObservation
  exact law.trans (law.symm (leftInverse left))
    (by simpa [leftObservation, rightObservation] using leftInverse right)

theorem strict_selector_is_sufficient
    {Scene Observation Graph : Type}
    (equivalent : Graph -> Graph -> Prop)
    (render : Scene -> Observation) (meaning : Scene -> Graph)
    (selector : Observation -> Graph)
    (strictFibers : forall observation left right,
      render left = observation -> render right = observation ->
        equivalent (meaning left) (meaning right))
    (selectedFromFiber : forall scene, Exists fun witness =>
      render witness = render scene ∧ selector (render scene) = meaning witness) :
    LeftInverseUpTo equivalent render meaning selector := by
  intro scene
  rcases selectedFromFiber scene with
    ⟨witness, witnessObservation, selectedMeaning⟩
  rw [selectedMeaning]
  exact strictFibers (render scene) witness scene witnessObservation rfl

/-! ## first-wins 反例 -/

inductive FirstWinsScene where
  | first
  | second
deriving DecidableEq, Repr

def firstWinsRender : FirstWinsScene -> Unit := fun _ => ()
def firstWinsMeaning : FirstWinsScene -> Bool
  | .first => false
  | .second => true

def firstWinsDecode (_ : Unit) : Bool := false

theorem first_wins_loses_a_valid_interpretation :
    firstWinsDecode (firstWinsRender .first) = firstWinsMeaning .first ∧
      firstWinsDecode (firstWinsRender .second) ≠ firstWinsMeaning .second := by
  decide

/-! ## 带标签简单图同构与可判定 checker -/

structure LabeledSimpleGraph (NodeLabel EdgeLabel : Type) where
  nodeCount : Nat
  nodeLabel : Fin nodeCount -> NodeLabel
  edgeLabel : Fin nodeCount -> Fin nodeCount -> Option EdgeLabel
  noSelfLoop : forall node, edgeLabel node node = none
  symmetric : forall left right, edgeLabel left right = edgeLabel right left

structure LabeledGraphIso {NodeLabel EdgeLabel : Type}
    (left right : LabeledSimpleGraph NodeLabel EdgeLabel) where
  rename : Bijection (Fin left.nodeCount) (Fin right.nodeCount)
  preservesNode : forall node,
    right.nodeLabel (rename.toFun node) = left.nodeLabel node
  preservesEdge : forall a b,
    right.edgeLabel (rename.toFun a) (rename.toFun b) = left.edgeLabel a b

def LabeledGraphEquivalent {NodeLabel EdgeLabel : Type}
    (left right : LabeledSimpleGraph NodeLabel EdgeLabel) : Prop :=
  Nonempty (LabeledGraphIso left right)

def LabeledGraphIso.refl {NodeLabel EdgeLabel : Type}
    (graph : LabeledSimpleGraph NodeLabel EdgeLabel) : LabeledGraphIso graph graph where
  rename := Bijection.refl _
  preservesNode := fun _ => rfl
  preservesEdge := fun _ _ => rfl

def LabeledGraphIso.symm {NodeLabel EdgeLabel : Type}
    {left right : LabeledSimpleGraph NodeLabel EdgeLabel}
    (iso : LabeledGraphIso left right) : LabeledGraphIso right left where
  rename := iso.rename.symm
  preservesNode := by
    intro node
    change left.nodeLabel (iso.rename.invFun node) = right.nodeLabel node
    rw [← iso.preservesNode (iso.rename.invFun node)]
    rw [iso.rename.rightInverse]
  preservesEdge := by
    intro a b
    change left.edgeLabel (iso.rename.invFun a) (iso.rename.invFun b) =
      right.edgeLabel a b
    rw [← iso.preservesEdge (iso.rename.invFun a) (iso.rename.invFun b)]
    rw [iso.rename.rightInverse, iso.rename.rightInverse]

def LabeledGraphIso.trans {NodeLabel EdgeLabel : Type}
    {left middle right : LabeledSimpleGraph NodeLabel EdgeLabel}
    (first : LabeledGraphIso left middle)
    (second : LabeledGraphIso middle right) : LabeledGraphIso left right where
  rename := first.rename.trans second.rename
  preservesNode := by
    intro node
    exact (second.preservesNode (first.rename.toFun node)).trans
      (first.preservesNode node)
  preservesEdge := by
    intro a b
    exact (second.preservesEdge (first.rename.toFun a) (first.rename.toFun b)).trans
      (first.preservesEdge a b)

theorem labeledGraphEquivalent_equivalence
    {NodeLabel EdgeLabel : Type} :
    Equivalence (@LabeledGraphEquivalent NodeLabel EdgeLabel) := by
  refine ⟨?_, ?_, ?_⟩
  · intro graph
    exact ⟨LabeledGraphIso.refl graph⟩
  · intro left right equivalent
    rcases equivalent with ⟨iso⟩
    exact ⟨iso.symm⟩
  · intro left middle right leftMiddle middleRight
    rcases leftMiddle with ⟨first⟩
    rcases middleRight with ⟨second⟩
    exact ⟨first.trans second⟩

def decisionToBool {proposition : Prop} : Decidable proposition -> Bool
  | .isTrue _ => true
  | .isFalse _ => false

def labeledGraphIsoBool
    {NodeLabel EdgeLabel : Type}
    (decision : forall left right : LabeledSimpleGraph NodeLabel EdgeLabel,
      Decidable (LabeledGraphEquivalent left right))
    (left right : LabeledSimpleGraph NodeLabel EdgeLabel) : Bool :=
  decisionToBool (decision left right)

theorem labeledGraphIsoBool_sound_complete
    {NodeLabel EdgeLabel : Type}
    (decision : forall left right : LabeledSimpleGraph NodeLabel EdgeLabel,
      Decidable (LabeledGraphEquivalent left right))
    (left right : LabeledSimpleGraph NodeLabel EdgeLabel) :
    labeledGraphIsoBool decision left right = true ↔
      LabeledGraphEquivalent left right := by
  unfold labeledGraphIsoBool
  cases decided : decision left right with
  | isTrue proof =>
      constructor
      · intro _
        exact proof
      · intro _
        rfl
  | isFalse refute =>
      constructor
      · intro impossible
        cases impossible
      · intro proof
        exact False.elim (refute proof)

/-! ## typed canonical embedding 与 delivery Shift 合同 -/

structure CanonicalEmbeddingContract
    (Raw Canonical PreprocessFailure : Type) where
  preprocess : Raw -> Result Canonical PreprocessFailure
  embed : Canonical -> Raw
  canonicalIdentity : forall canonical,
    preprocess (embed canonical) = .success canonical

theorem preprocess_embed_success
    {Raw Canonical PreprocessFailure : Type}
    (contract : CanonicalEmbeddingContract Raw Canonical PreprocessFailure)
    (canonical : Canonical) :
    contract.preprocess (contract.embed canonical) = .success canonical :=
  contract.canonicalIdentity canonical

structure DeliveryShiftContract
    (Q Translation Raw Canonical PreprocessFailure : Type) where
  centered : Q -> Raw
  shift : Raw -> Translation -> Raw
  zero : Translation
  preprocess : Raw -> Result Canonical PreprocessFailure
  canonical : Q -> Canonical
  rawRender : Q -> Translation -> Raw
  rawRender_eq_shift : forall q translation,
    rawRender q translation = shift (centered q) translation
  translationInvariant : forall q translation,
    preprocess (rawRender q translation) = .success (canonical q)

theorem delivery_translation_contract
    {Q Translation Raw Canonical PreprocessFailure : Type}
    (contract : DeliveryShiftContract Q Translation Raw Canonical PreprocessFailure)
    (q : Q) (translation : Translation) :
    contract.preprocess (contract.shift (contract.centered q) translation) =
      .success (contract.canonical q) := by
  rw [← contract.rawRender_eq_shift q translation]
  exact contract.translationInvariant q translation

/-!
Canonical bytes/graph IDs 可通过一个稳定 canonical key 冻结。这里仅证明 key
合同诱导等价关系；具体 bytes、排序和独立 canonicalizer 仍由 executable audit 验证。
-/
structure CanonicalKeyContract (Object Key : Type) where
  key : Object -> Key

def CanonicalKeyContract.Equivalent {Object Key : Type}
    (contract : CanonicalKeyContract Object Key) (left right : Object) : Prop :=
  contract.key left = contract.key right

theorem canonicalKey_equivalence {Object Key : Type}
    (contract : CanonicalKeyContract Object Key) :
    Equivalence contract.Equivalent := by
  refine ⟨?_, ?_, ?_⟩
  · intro object
    rfl
  · intro left right equal
    exact equal.symm
  · intro left middle right leftMiddle middleRight
    exact leftMiddle.trans middleRight

end ObservationClosure

end ChemGraph
