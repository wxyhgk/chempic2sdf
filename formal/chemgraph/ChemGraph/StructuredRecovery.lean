import ChemGraph.ObservationClosure
import ChemGraph.Counterexamples

namespace ChemGraph

/-!
Structured Recovery v0.6 的最小数学内核。

结构化文字/键对象必须由 raster 决定；它们可以改善计算过程，但不能消除
已经发生的 exact-raster 信息碰撞。本文件不实现 OCR、bond extractor 或模型。
-/

namespace StructuredRecovery

open ObservationClosure

structure StructuredObservation
    (Raster TextRegion BondObject : Type) where
  raster : Raster
  textRegions : List TextRegion
  bondObjects : List BondObject

def RasterProjection
    {Raster TextRegion BondObject : Type}
    (extract : Raster -> StructuredObservation Raster TextRegion BondObject) : Prop :=
  forall raster, (extract raster).raster = raster

def FactorsThroughRaster
    {Scene Raster Structured : Type}
    (render : Scene -> Raster)
    (observeScene : Scene -> Structured)
    (extract : Raster -> Structured) : Prop :=
  forall scene, observeScene scene = extract (render scene)

theorem same_raster_same_structured_observation
    {Raster Structured : Type}
    (extract : Raster -> Structured) (left right : Raster)
    (sameRaster : left = right) :
    extract left = extract right :=
  congrArg extract sameRaster

theorem raster_projection_makes_extractor_injective
    {Raster TextRegion BondObject : Type}
    (extract : Raster -> StructuredObservation Raster TextRegion BondObject)
    (projects : RasterProjection extract)
    (left right : Raster)
    (sameObservation : extract left = extract right) : left = right := by
  calc
    left = (extract left).raster := (projects left).symm
    _ = (extract right).raster := congrArg StructuredObservation.raster sameObservation
    _ = right := projects right

theorem raster_collision_lifts_through_factorization
    {Scene Raster Structured : Type}
    (render : Scene -> Raster)
    (observeScene : Scene -> Structured)
    (extract : Raster -> Structured)
    (factors : FactorsThroughRaster render observeScene extract)
    (left right : Scene)
    (sameRaster : render left = render right) :
    observeScene left = observeScene right := by
  calc
    observeScene left = extract (render left) := factors left
    _ = extract (render right) := congrArg extract sameRaster
    _ = observeScene right := (factors right).symm

theorem factorized_collision_forbids_universal_single_decoder
    {Scene Raster Structured Graph : Type}
    (equivalent : Graph -> Graph -> Prop)
    (equivalenceLaw : Equivalence equivalent)
    (render : Scene -> Raster) (meaning : Scene -> Graph)
    (observeScene : Scene -> Structured) (extract : Raster -> Structured)
    (factors : FactorsThroughRaster render observeScene extract)
    (left right : Scene)
    (sameRaster : render left = render right)
    (differentMeaning : Not (equivalent (meaning left) (meaning right))) :
    Not (Exists fun decode : Structured -> Graph =>
      forall scene, equivalent (decode (observeScene scene)) (meaning scene)) := by
  exact collision_forbids_decoder_up_to equivalent equivalenceLaw
    observeScene meaning left right
      (raster_collision_lifts_through_factorization
        render observeScene extract factors left right sameRaster)
      differentMeaning

/-! ## 同一合法 q 的多个 interpretation -/

theorem same_q_inequivalent_interpretations_forbid_universal_decoder
    {Q Observation Graph : Type}
    (Interpretation : Q -> Type)
    (valid : Q -> Prop)
    (equivalent : Graph -> Graph -> Prop)
    (equivalenceLaw : Equivalence equivalent)
    (render : Q -> Observation)
    (meaning : (q : Q) -> Interpretation q -> Graph)
    (q : Q) (qValid : valid q)
    (left right : Interpretation q)
    (different : Not (equivalent (meaning q left) (meaning q right))) :
    Not (Exists fun decode : Observation -> Graph =>
      forall q, valid q -> forall interpretation,
        equivalent (decode (render q)) (meaning q interpretation)) := by
  intro existsDecoder
  rcases existsDecoder with ⟨decode, correct⟩
  apply different
  exact equivalenceLaw.trans
    (equivalenceLaw.symm (correct q qValid left))
    (correct q qValid right)

theorem graphEquivalent_equivalence : Equivalence GraphEquivalent := by
  refine ⟨?_, ?_, ?_⟩
  · intro graph
    exact ⟨rfl, List.Perm.refl graph.bonds⟩
  · intro left right equivalent
    exact ⟨equivalent.left.symm, equivalent.right.symm⟩
  · intro left middle right leftMiddle middleRight
    exact ⟨leftMiddle.left.trans middleRight.left,
      leftMiddle.right.trans middleRight.right⟩

inductive ChemicalCollisionInterpretation where
  | crossing
  | junction
deriving DecidableEq, Repr

def chemicalCollisionScene : ChemicalCollisionInterpretation -> Scene
  | .crossing => crossingScene
  | .junction => junctionScene

def chemicalCollisionRender (_ : Unit) : Ink :=
  renderOpaque crossingScene

def chemicalCollisionMeaning
    (_ : Unit) : ChemicalCollisionInterpretation -> ChemicalGraph
  | .crossing => crossingScene.graph
  | .junction => junctionScene.graph

theorem chemicalCollision_render_realizes_interpretation
    (interpretation : ChemicalCollisionInterpretation) :
    renderOpaque (chemicalCollisionScene interpretation) = chemicalCollisionRender () := by
  cases interpretation with
  | crossing => rfl
  | junction => exact crossing_and_junction_same_opaque_render.symm

/-! ## 完整 observation fiber 与集合候选 -/

def StructuredGraphFiber
    {Scene Structured Graph : Type}
    (valid : Scene -> Prop) (equivalent : Graph -> Graph -> Prop)
    (observe : Scene -> Structured) (meaning : Scene -> Graph)
    (observation : Structured) (graph : Graph) : Prop :=
  Exists fun scene =>
    valid scene ∧ observe scene = observation ∧ equivalent graph (meaning scene)

def CandidateSound
    {Graph : Type} (oracle predicted : Graph -> Prop) : Prop :=
  forall graph, predicted graph -> oracle graph

def CandidateComplete
    {Graph : Type} (oracle predicted : Graph -> Prop) : Prop :=
  forall graph, oracle graph -> predicted graph

structure CandidatesExact
    {Graph : Type} (oracle predicted : Graph -> Prop) : Prop where
  sound : CandidateSound oracle predicted
  complete : CandidateComplete oracle predicted

theorem candidates_exact_iff_coverage_equal
    {Graph : Type} (oracle predicted : Graph -> Prop) :
    CandidatesExact oracle predicted ↔
      forall graph, predicted graph ↔ oracle graph := by
  constructor
  · intro exact graph
    exact ⟨exact.sound graph, exact.complete graph⟩
  · intro coverage
    exact {
      sound := fun graph => (coverage graph).mp
      complete := fun graph => (coverage graph).mpr
    }

theorem exact_candidates_preserve_multiple_classes
    {Graph : Type} (equivalent : Graph -> Graph -> Prop)
    (oracle predicted : Graph -> Prop)
    (exact : CandidatesExact oracle predicted)
    (multiple : MultipleClasses equivalent oracle) :
    MultipleClasses equivalent predicted := by
  rcases multiple with ⟨left, right, leftMember, rightMember, different⟩
  exact ⟨left, right,
    exact.complete left leftMember,
    exact.complete right rightMember,
    different⟩

inductive RecoveryResult (Graph Reason : Type) where
  | candidates (members : Graph -> Prop)
  | abstain (reason : Reason)

def CandidateOutputsExact
    {Structured Graph Reason : Type}
    (oracle : Structured -> Graph -> Prop)
    (decode : Structured -> RecoveryResult Graph Reason) : Prop :=
  forall observation predicted,
    decode observation = .candidates predicted ->
      CandidatesExact (oracle observation) predicted

theorem ambiguous_must_return_multiple_or_abstain
    {Structured Graph Reason : Type}
    (equivalent : Graph -> Graph -> Prop)
    (oracle : Structured -> Graph -> Prop)
    (decode : Structured -> RecoveryResult Graph Reason)
    (candidateExact : CandidateOutputsExact oracle decode)
    (observation : Structured)
    (ambiguous : MultipleClasses equivalent (oracle observation)) :
    (Exists fun predicted =>
      decode observation = .candidates predicted ∧
        MultipleClasses equivalent predicted) ∨
      Exists fun reason => decode observation = .abstain reason := by
  cases decoded : decode observation with
  | candidates predicted =>
      exact Or.inl ⟨predicted, rfl,
        exact_candidates_preserve_multiple_classes equivalent
          (oracle observation) predicted
          (candidateExact observation predicted decoded) ambiguous⟩
  | abstain reason =>
      exact Or.inr ⟨reason, rfl⟩

/-! ## sampled/full：共享函数，只改变 active 域 -/

structure StructuredDomain where
  Q : Type
  Interpretation : Q -> Type
  Structured : Type
  Graph : Type
  active : Q -> Prop
  observe : Q -> Structured
  meaning : (q : Q) -> Interpretation q -> Graph
  equivalent : Graph -> Graph -> Prop

def chemicalCollisionDomain : StructuredDomain where
  Q := Unit
  Interpretation := fun _ => ChemicalCollisionInterpretation
  Structured := Ink
  Graph := ChemicalGraph
  active := fun _ => True
  observe := chemicalCollisionRender
  meaning := chemicalCollisionMeaning
  equivalent := GraphEquivalent

theorem concrete_dependent_domain_has_inequivalent_interpretations :
    Exists fun q : chemicalCollisionDomain.Q =>
      chemicalCollisionDomain.active q ∧
        Exists fun left : chemicalCollisionDomain.Interpretation q =>
          Exists fun right : chemicalCollisionDomain.Interpretation q =>
            Not (chemicalCollisionDomain.equivalent
              (chemicalCollisionDomain.meaning q left)
              (chemicalCollisionDomain.meaning q right)) := by
  exact ⟨(), trivial, .crossing, .junction,
    crossing_and_junction_not_graph_equivalent⟩

theorem concrete_dependent_domain_forbids_universal_decoder :
    Not (Exists fun decode : chemicalCollisionDomain.Structured ->
        chemicalCollisionDomain.Graph =>
      forall q, chemicalCollisionDomain.active q ->
        forall interpretation : chemicalCollisionDomain.Interpretation q,
          chemicalCollisionDomain.equivalent
            (decode (chemicalCollisionDomain.observe q))
            (chemicalCollisionDomain.meaning q interpretation)) := by
  exact same_q_inequivalent_interpretations_forbid_universal_decoder
    chemicalCollisionDomain.Interpretation chemicalCollisionDomain.active
    chemicalCollisionDomain.equivalent graphEquivalent_equivalence
    chemicalCollisionDomain.observe chemicalCollisionDomain.meaning
    () trivial .crossing .junction
    crossing_and_junction_not_graph_equivalent

def StructuredDomainFiber
    (domain : StructuredDomain)
    (observation : domain.Structured) (graph : domain.Graph) : Prop :=
  Exists fun q => Exists fun interpretation : domain.Interpretation q =>
    domain.active q ∧ domain.observe q = observation ∧
      domain.equivalent graph (domain.meaning q interpretation)

def StructuredDomain.restrict
    (domain : StructuredDomain) (active : domain.Q -> Prop) : StructuredDomain where
  Q := domain.Q
  Interpretation := domain.Interpretation
  Structured := domain.Structured
  Graph := domain.Graph
  active := active
  observe := domain.observe
  meaning := domain.meaning
  equivalent := domain.equivalent

theorem structuredDomainFiber_mono
    (domain : StructuredDomain) (sample full : domain.Q -> Prop)
    (subset : forall q, sample q -> full q)
    (observation : domain.Structured) (graph : domain.Graph) :
    StructuredDomainFiber (domain.restrict sample) observation graph ->
      StructuredDomainFiber (domain.restrict full) observation graph := by
  rintro ⟨q, interpretation, sampled, observed, equivalent⟩
  exact ⟨q, interpretation, subset q sampled, observed, equivalent⟩

theorem sampled_ambiguous_implies_full_ambiguous
    (domain : StructuredDomain) (sample full : domain.Q -> Prop)
    (subset : forall q, sample q -> full q)
    (observation : domain.Structured) :
    MultipleClasses domain.equivalent
        (StructuredDomainFiber (domain.restrict sample) observation) ->
      MultipleClasses domain.equivalent
        (StructuredDomainFiber (domain.restrict full) observation) := by
  rintro ⟨left, right, leftMember, rightMember, different⟩
  exact ⟨left, right,
    structuredDomainFiber_mono domain sample full subset observation left leftMember,
    structuredDomainFiber_mono domain sample full subset observation right rightMember,
    different⟩

private def singletonCounterDomain : StructuredDomain where
  Q := Bool
  Interpretation := fun _ => Unit
  Structured := Unit
  Graph := Bool
  active := fun _ => True
  observe := fun _ => ()
  meaning := fun q _ => q
  equivalent := Eq

private def onlyFalse : Bool -> Prop := fun q => q = false
private def bothValues : Bool -> Prop := fun _ => True

theorem sampled_singleton_does_not_imply_full_singleton :
    SingletonClass Eq
        (StructuredDomainFiber (singletonCounterDomain.restrict onlyFalse) ()) ∧
      MultipleClasses Eq
        (StructuredDomainFiber (singletonCounterDomain.restrict bothValues) ()) := by
  constructor
  · refine ⟨false, ?_, ?_⟩
    · exact ⟨false, (), rfl, rfl, rfl⟩
    · intro graph member
      rcases member with ⟨q, interpretation, active, observed, equivalent⟩
      have qFalse : q = false := active
      change graph = q at equivalent
      rw [qFalse] at equivalent
      exact equivalent
  · refine ⟨false, true, ?_, ?_, ?_⟩
    · exact ⟨false, (), trivial, rfl, rfl⟩
    · exact ⟨true, (), trivial, rfl, rfl⟩
    · intro equal
      cases equal

/-! 公开关键定理的 kernel 公理审计。 -/

#print axioms same_raster_same_structured_observation
#print axioms raster_projection_makes_extractor_injective
#print axioms raster_collision_lifts_through_factorization
#print axioms factorized_collision_forbids_universal_single_decoder
#print axioms same_q_inequivalent_interpretations_forbid_universal_decoder
#print axioms graphEquivalent_equivalence
#print axioms chemicalCollision_render_realizes_interpretation
#print axioms concrete_dependent_domain_has_inequivalent_interpretations
#print axioms concrete_dependent_domain_forbids_universal_decoder
#print axioms candidates_exact_iff_coverage_equal
#print axioms exact_candidates_preserve_multiple_classes
#print axioms ambiguous_must_return_multiple_or_abstain
#print axioms structuredDomainFiber_mono
#print axioms sampled_ambiguous_implies_full_ambiguous
#print axioms sampled_singleton_does_not_imply_full_singleton

end StructuredRecovery

end ChemGraph
