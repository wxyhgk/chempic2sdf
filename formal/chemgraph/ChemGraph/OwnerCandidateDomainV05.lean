import ChemGraph.GlobalOwnershipNoGoV04
import Std

namespace ChemGraph

namespace OwnerCandidateDomainV05

/-!
Owner Candidate Domain v0.5 的纯有限 CSP 内核。

本模块区分多值 evidence membership、逐像素 Cartesian product 和经过全局
constraint 过滤的 feasible assignment。所有 coverage、soundness 和 legal world
结论都相对于调用者提供的 `legal` predicate；本模块不定义 PNG、renderer、RLE、
extractor，也不形式化任何 873 解的外部 artifact。
-/

open GlobalOwnershipNoGoV04

/-! ## 多值 evidence membership -/

structure EvidenceMembership (Pixel Carrier Fragment : Type) where
  carrierFiber : Pixel -> List Carrier
  fragmentOf : Pixel -> Carrier -> Option Fragment
  fragmentDefined : forall pixel carrier,
    carrier ∈ carrierFiber pixel ->
      Exists fun fragment => fragmentOf pixel carrier = some fragment
  fragmentSound : forall pixel carrier fragment,
    fragmentOf pixel carrier = some fragment ->
      carrier ∈ carrierFiber pixel

def EvidenceMembership.FragmentMember
    {Pixel Carrier Fragment : Type}
    (evidence : EvidenceMembership Pixel Carrier Fragment)
    (pixel : Pixel) (carrier : Carrier) (fragment : Fragment) : Prop :=
  evidence.fragmentOf pixel carrier = some fragment

theorem fragment_membership_unique_within_carrier
    {Pixel Carrier Fragment : Type}
    (evidence : EvidenceMembership Pixel Carrier Fragment)
    (pixel : Pixel) (carrier : Carrier) (left right : Fragment)
    (leftMember : evidence.FragmentMember pixel carrier left)
    (rightMember : evidence.FragmentMember pixel carrier right) :
    left = right := by
  exact Option.some.inj (leftMember.symm.trans rightMember)

def sharedPixelEvidence : EvidenceMembership Unit Bool Bool where
  carrierFiber := fun _ => [false, true]
  fragmentOf := fun _ carrier => some carrier
  fragmentDefined := by
    intro pixel carrier member
    exact ⟨carrier, rfl⟩
  fragmentSound := by
    intro pixel carrier fragment equal
    cases carrier <;> simp

theorem evidence_membership_can_be_multi_carrier :
    false ∈ sharedPixelEvidence.carrierFiber () /\
      true ∈ sharedPixelEvidence.carrierFiber () /\
      false ≠ true /\
      sharedPixelEvidence.FragmentMember () false false /\
      sharedPixelEvidence.FragmentMember () true true := by
  simp [sharedPixelEvidence, EvidenceMembership.FragmentMember]

/-! ## Factorized owner candidate CSP -/

inductive OwnerOption (Intersection Fragment : Type) where
  | intersection (id : Intersection)
  | fragment (id : Fragment)
  | residual
deriving DecidableEq, Repr

structure OwnerCandidateCSP
    (Observation Pixel Owner : Type) where
  variables : Observation -> List Pixel
  universal : Observation -> Pixel -> List Owner
  domain : Observation -> Pixel -> List Owner
  constraint : Observation -> (Pixel -> Owner) -> Prop

def PointwiseAllowed
    {Observation Pixel Owner : Type}
    (csp : OwnerCandidateCSP Observation Pixel Owner)
    (observation : Observation) (assignment : Pixel -> Owner) : Prop :=
  forall pixel, pixel ∈ csp.variables observation ->
    assignment pixel ∈ csp.domain observation pixel

def FeasibleAssignment
    {Observation Pixel Owner : Type}
    (csp : OwnerCandidateCSP Observation Pixel Owner)
    (observation : Observation) (assignment : Pixel -> Owner) : Prop :=
  PointwiseAllowed csp observation assignment /\
    csp.constraint observation assignment

def AgreesOn
    {Pixel Owner : Type}
    (pixels : List Pixel) (left right : Pixel -> Owner) : Prop :=
  forall pixel, pixel ∈ pixels -> left pixel = right pixel

def CandidateComplete
    {RootEvidence Pixel Owner : Type}
    (legal : World RootEvidence Pixel Owner -> Prop)
    (csp : OwnerCandidateCSP (Observation RootEvidence) Pixel Owner) : Prop :=
  forall world, legal world ->
    FeasibleAssignment csp world.observation world.ownerPartition.ownerOf

def CandidateSound
    {RootEvidence Pixel Owner : Type}
    (legal : World RootEvidence Pixel Owner -> Prop)
    (csp : OwnerCandidateCSP (Observation RootEvidence) Pixel Owner) : Prop :=
  forall observation assignment, FeasibleAssignment csp observation assignment ->
    Exists fun world =>
      legal world /\ world.observation = observation /\
        AgreesOn (csp.variables observation)
          assignment world.ownerPartition.ownerOf

theorem complete_implies_true_owner_membership
    {RootEvidence Pixel Owner : Type}
    (legal : World RootEvidence Pixel Owner -> Prop)
    (csp : OwnerCandidateCSP (Observation RootEvidence) Pixel Owner)
    (complete : CandidateComplete legal csp)
    (world : World RootEvidence Pixel Owner) (worldLegal : legal world)
    (pixel : Pixel) (pixelListed : pixel ∈ csp.variables world.observation) :
    world.ownerPartition.ownerOf pixel ∈ csp.domain world.observation pixel := by
  exact (complete world worldLegal).1 pixel pixelListed

def PixelSingletonAt
    {Observation Pixel Owner : Type}
    (csp : OwnerCandidateCSP Observation Pixel Owner)
    (observation : Observation) (pixel : Pixel) (owner : Owner) : Prop :=
  csp.domain observation pixel = [owner]

theorem covered_singleton_fixes_local_owner
    {RootEvidence Pixel Owner : Type}
    (legal : World RootEvidence Pixel Owner -> Prop)
    (csp : OwnerCandidateCSP (Observation RootEvidence) Pixel Owner)
    (complete : CandidateComplete legal csp)
    (world : World RootEvidence Pixel Owner) (worldLegal : legal world)
    (pixel : Pixel) (pixelListed : pixel ∈ csp.variables world.observation)
    (owner : Owner)
    (singleton : PixelSingletonAt csp world.observation pixel owner) :
    world.ownerPartition.ownerOf pixel = owner := by
  have member := complete_implies_true_owner_membership
    legal csp complete world worldLegal pixel pixelListed
  change csp.domain world.observation pixel = [owner] at singleton
  rw [singleton] at member
  simpa using member

/-! ## 局部 singleton 与 realized ambiguity 的边界 -/

def localSingletonCSP : OwnerCandidateCSP Unit Bool Bool where
  variables := fun _ => [false, true]
  universal := fun _ _ => [false, true]
  domain := fun _ pixel => if pixel = false then [false] else [false, true]
  constraint := fun _ _ => True

def allFalseAssignment : Bool -> Bool := fun _ => false

def identityAssignment : Bool -> Bool := fun pixel => pixel

theorem local_singleton_does_not_imply_unique_world :
    PixelSingletonAt localSingletonCSP () false false /\
      FeasibleAssignment localSingletonCSP () allFalseAssignment /\
      FeasibleAssignment localSingletonCSP () identityAssignment /\
      allFalseAssignment ≠ identityAssignment := by
  constructor
  · rfl
  constructor
  · constructor
    · intro pixel listed
      cases pixel <;> simp [localSingletonCSP, allFalseAssignment]
    · trivial
  constructor
  · constructor
    · intro pixel listed
      cases pixel <;> simp [localSingletonCSP, identityAssignment]
    · trivial
  · intro equal
    have atTrue := congrFun equal true
    cases atTrue

def RealizedOptionAt
    {Observation Pixel Owner : Type}
    (csp : OwnerCandidateCSP Observation Pixel Owner)
    (observation : Observation) (pixel : Pixel) (owner : Owner) : Prop :=
  Exists fun assignment : Pixel -> Owner =>
    FeasibleAssignment csp observation assignment /\ assignment pixel = owner

def forcedFalseCSP : OwnerCandidateCSP Unit Unit Bool where
  variables := fun _ => [()]
  universal := fun _ _ => [false, true]
  domain := fun _ _ => [false, true]
  constraint := fun _ assignment => assignment () = false

theorem raw_multi_option_does_not_imply_realized_ambiguity :
    false ∈ forcedFalseCSP.domain () () /\
      true ∈ forcedFalseCSP.domain () () /\
      RealizedOptionAt forcedFalseCSP () () false /\
      Not (RealizedOptionAt forcedFalseCSP () () true) := by
  constructor
  · simp [forcedFalseCSP]
  constructor
  · simp [forcedFalseCSP]
  constructor
  · refine ⟨fun _ => false, ?_, rfl⟩
    constructor
    · intro pixel listed
      simp [forcedFalseCSP]
    · rfl
  · rintro ⟨assignment, feasible, realized⟩
    have forced := feasible.2
    change assignment () = false at forced
    rw [realized] at forced
    cases forced

/-! ## Cartesian product 不是 legal/feasible world fiber -/

def equalOwnerCSP : OwnerCandidateCSP Unit Bool Bool where
  variables := fun _ => [false, true]
  universal := fun _ _ => [false, true]
  domain := fun _ _ => [false, true]
  constraint := fun _ assignment => assignment false = assignment true

theorem universal_domain_covers_every_assignment
    (assignment : Bool -> Bool) :
    PointwiseAllowed equalOwnerCSP () assignment := by
  intro pixel listed
  cases assignment pixel <;> simp [equalOwnerCSP]

theorem pointwise_cartesian_product_need_not_be_feasible :
    PointwiseAllowed equalOwnerCSP () identityAssignment /\
      Not (FeasibleAssignment equalOwnerCSP () identityAssignment) := by
  constructor
  · exact universal_domain_covers_every_assignment identityAssignment
  · intro feasible
    have equalOwners := feasible.2
    cases equalOwners

/-! ## Realized legal ambiguity connects to the existing global NO-GO. -/

def LegalRealizedOwnerAt
    {RootEvidence Pixel Owner : Type}
    (legal : World RootEvidence Pixel Owner -> Prop)
    (observation : Observation RootEvidence) (pixel : Pixel) (owner : Owner) : Prop :=
  Exists fun world =>
    legal world /\ world.observation = observation /\
      world.ownerPartition.ownerOf pixel = owner

theorem different_owner_at_pixel_implies_different_partitions
    {Pixel Owner : Type}
    (left right : OwnerPartition Pixel Owner) (pixel : Pixel)
    (different : left.ownerOf pixel ≠ right.ownerOf pixel) :
    left ≠ right := by
  intro equal
  exact different (congrArg (fun partition => partition.ownerOf pixel) equal)

theorem realized_ambiguity_forbids_root_only_single_owner
    {RootEvidence Pixel Owner : Type}
    (legal : World RootEvidence Pixel Owner -> Prop)
    (observation : Observation RootEvidence) (pixel : Pixel)
    (leftOwner rightOwner : Owner) (differentOwner : leftOwner ≠ rightOwner)
    (leftRealized : LegalRealizedOwnerAt legal observation pixel leftOwner)
    (rightRealized : LegalRealizedOwnerAt legal observation pixel rightOwner) :
    Not (Exists fun construct : Observation RootEvidence -> OwnerPartition Pixel Owner =>
      RootOnlyExactUniqueOwnerConstructor legal construct) := by
  rcases leftRealized with ⟨left, leftLegal, leftObservation, leftAtPixel⟩
  rcases rightRealized with ⟨right, rightLegal, rightObservation, rightAtPixel⟩
  apply same_observation_different_owner_partitions_forbid_unique_owner_constructor
    legal left right leftLegal rightLegal
  · exact leftObservation.trans rightObservation.symm
  · apply different_owner_at_pixel_implies_different_partitions
      left.ownerPartition right.ownerPartition pixel
    intro equal
    apply differentOwner
    exact leftAtPixel.symm.trans (equal.trans rightAtPixel)

/-! ## 非真空与 SAFE_BUT_VACUOUS 门 -/

def NoOptionEliminatedAt
    {Observation Pixel Owner : Type}
    (csp : OwnerCandidateCSP Observation Pixel Owner)
    (observation : Observation) : Prop :=
  forall pixel, pixel ∈ csp.variables observation ->
    csp.domain observation pixel = csp.universal observation pixel

def ResidualSaturatedAt
    {Observation Pixel Owner : Type}
    (csp : OwnerCandidateCSP Observation Pixel Owner)
    (observation : Observation) (residual : Owner) : Prop :=
  forall pixel, pixel ∈ csp.variables observation ->
    residual ∈ csp.domain observation pixel /\
      1 < (csp.domain observation pixel).length

def SafeButVacuousAt
    {RootEvidence Pixel Owner : Type}
    (legal : World RootEvidence Pixel Owner -> Prop)
    (csp : OwnerCandidateCSP (Observation RootEvidence) Pixel Owner)
    (observation : Observation RootEvidence) (residual : Owner) : Prop :=
  CandidateComplete legal csp /\
    (Exists fun pixel => pixel ∈ csp.variables observation) /\
    (NoOptionEliminatedAt csp observation \/
      ResidualSaturatedAt csp observation residual)

theorem complete_universal_domain_is_safe_but_vacuous
    {RootEvidence Pixel Owner : Type}
    (legal : World RootEvidence Pixel Owner -> Prop)
    (csp : OwnerCandidateCSP (Observation RootEvidence) Pixel Owner)
    (complete : CandidateComplete legal csp)
    (observation : Observation RootEvidence) (residual : Owner)
    (nonemptyVariables : Exists fun pixel => pixel ∈ csp.variables observation)
    (universal : NoOptionEliminatedAt csp observation) :
    SafeButVacuousAt legal csp observation residual := by
  exact ⟨complete, nonemptyVariables, Or.inl universal⟩

theorem complete_residual_saturated_is_safe_but_vacuous
    {RootEvidence Pixel Owner : Type}
    (legal : World RootEvidence Pixel Owner -> Prop)
    (csp : OwnerCandidateCSP (Observation RootEvidence) Pixel Owner)
    (complete : CandidateComplete legal csp)
    (observation : Observation RootEvidence) (residual : Owner)
    (nonemptyVariables : Exists fun pixel => pixel ∈ csp.variables observation)
    (saturated : ResidualSaturatedAt csp observation residual) :
    SafeButVacuousAt legal csp observation residual := by
  exact ⟨complete, nonemptyVariables, Or.inr saturated⟩

/-! ## 冻结 R0 profile 与声明 safety ceiling 的不可达性 -/

structure R0CapacityCounts where
  intersectionCount : Nat
  ownerOptionCount : Nat

def FixedTwoIntersectionR0 (counts : R0CapacityCounts) : Prop :=
  counts.intersectionCount = 2 /\ counts.ownerOptionCount = 2

def ReachesOriginalDeclaredCeilings (counts : R0CapacityCounts) : Prop :=
  counts.intersectionCount = 8 /\ counts.ownerOptionCount = 16

/-- 固定 two-intersection R0 不可能同时作为命中 8/16 ceiling 的 full-constructor
boundary fixture。Ceiling 可以是 reject-only safety gate，但不能声称该固定 grammar 已
端到端验证 ceiling exact-success。 -/
theorem fixed_two_intersection_r0_cannot_reach_original_declared_ceilings
    (counts : R0CapacityCounts)
    (fixed : FixedTwoIntersectionR0 counts) :
    Not (ReachesOriginalDeclaredCeilings counts) := by
  intro reaches
  unfold FixedTwoIntersectionR0 at fixed
  unfold ReachesOriginalDeclaredCeilings at reaches
  rcases fixed with ⟨fixedIntersections, fixedOwners⟩
  rcases reaches with ⟨ceilingIntersections, ceilingOwners⟩
  omega

#print axioms fragment_membership_unique_within_carrier
#print axioms evidence_membership_can_be_multi_carrier
#print axioms complete_implies_true_owner_membership
#print axioms covered_singleton_fixes_local_owner
#print axioms local_singleton_does_not_imply_unique_world
#print axioms raw_multi_option_does_not_imply_realized_ambiguity
#print axioms universal_domain_covers_every_assignment
#print axioms pointwise_cartesian_product_need_not_be_feasible
#print axioms different_owner_at_pixel_implies_different_partitions
#print axioms realized_ambiguity_forbids_root_only_single_owner
#print axioms complete_universal_domain_is_safe_but_vacuous
#print axioms complete_residual_saturated_is_safe_but_vacuous
#print axioms fixed_two_intersection_r0_cannot_reach_original_declared_ceilings

end OwnerCandidateDomainV05

end ChemGraph
