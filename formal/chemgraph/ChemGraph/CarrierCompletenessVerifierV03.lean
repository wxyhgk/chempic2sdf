import ChemGraph.CarrierCandidateSetV03
import Std

namespace ChemGraph

namespace CarrierCompletenessVerifierV03

/-!
Carrier Candidate Set v0.3 独立 completeness verifier 的数学边界。

根级 compatible pair 列表足以完全枚举规范 boundary matching：枚举其全部
sublist，再只保留端点两两不交的项。相反，若 fragment split、intersection、
ownership、port 或 lane interpretation universe 没有进入 root record，则相同 root
可以对应不同的真实 interpretation universe；只读 root 的 verifier 不能同时对两者
sound 且 complete。本模块不把“只要求 complete”的超集输出误称为不可能。
-/

/-! ## 全部 canonical sublist -/

def enumerateSublists {Alpha : Type} : List Alpha -> List (List Alpha)
  | [] => [[]]
  | head :: tail =>
      let rest := enumerateSublists tail
      rest ++ rest.map (fun selected => head :: selected)

theorem enumerateSublists_sound
    {Alpha : Type} {candidate source : List Alpha}
    (member : candidate ∈ enumerateSublists source) :
    candidate.Sublist source := by
  induction source generalizing candidate with
  | nil =>
      simpa [enumerateSublists] using member
  | cons head tail ih =>
      simp only [enumerateSublists, List.mem_append, List.mem_map] at member
      rcases member with member | ⟨selected, selectedMember, selectedEq⟩
      · exact List.Sublist.cons head (ih member)
      · rw [← selectedEq]
        exact List.Sublist.cons_cons head (ih selectedMember)

theorem enumerateSublists_complete
    {Alpha : Type} {candidate source : List Alpha}
    (sublist : candidate.Sublist source) :
    candidate ∈ enumerateSublists source := by
  induction sublist with
  | slnil =>
      simp [enumerateSublists]
  | cons head sublist ih =>
      simp only [enumerateSublists, List.mem_append, List.mem_map]
      exact Or.inl ih
  | cons_cons head sublist ih =>
      simp only [enumerateSublists, List.mem_append, List.mem_map]
      exact Or.inr ⟨_, ih, rfl⟩

/-! ## Boundary matching -/

def EndpointDisjoint
    {Boundary Pair : Type}
    (endpoints : Pair -> Boundary × Boundary)
    (left right : Pair) : Prop :=
  (endpoints left).1 ≠ (endpoints right).1 /\
    (endpoints left).1 ≠ (endpoints right).2 /\
    (endpoints left).2 ≠ (endpoints right).1 /\
    (endpoints left).2 ≠ (endpoints right).2

instance endpointDisjointDecidable
    {Boundary Pair : Type} [DecidableEq Boundary]
    (endpoints : Pair -> Boundary × Boundary) (left right : Pair) :
    Decidable (EndpointDisjoint endpoints left right) := by
  unfold EndpointDisjoint
  infer_instance

/-- schema 的 pair ID 升序要求在抽象中对应 root pair list 的 sublist。 -/
def CanonicalMatching
    {Boundary Pair : Type}
    (rootPairs : List Pair)
    (endpoints : Pair -> Boundary × Boundary)
    (candidate : List Pair) : Prop :=
  candidate.Sublist rootPairs /\
    candidate.Pairwise (EndpointDisjoint endpoints)

def enumerateMatchings
    {Boundary Pair : Type} [DecidableEq Boundary]
    (rootPairs : List Pair)
    (endpoints : Pair -> Boundary × Boundary) : List (List Pair) :=
  (enumerateSublists rootPairs).filter fun candidate =>
    decide (candidate.Pairwise (EndpointDisjoint endpoints))

theorem matchingEnumerator_sound
    {Boundary Pair : Type} [DecidableEq Boundary]
    {rootPairs : List Pair}
    {endpoints : Pair -> Boundary × Boundary}
    {candidate : List Pair}
    (member : candidate ∈ enumerateMatchings rootPairs endpoints) :
    CanonicalMatching rootPairs endpoints candidate := by
  have reduced := List.mem_filter.mp member
  exact ⟨enumerateSublists_sound reduced.1,
    of_decide_eq_true reduced.2⟩

theorem matchingEnumerator_complete
    {Boundary Pair : Type} [DecidableEq Boundary]
    {rootPairs : List Pair}
    {endpoints : Pair -> Boundary × Boundary}
    {candidate : List Pair}
    (valid : CanonicalMatching rootPairs endpoints candidate) :
    candidate ∈ enumerateMatchings rootPairs endpoints := by
  apply List.mem_filter.mpr
  exact ⟨enumerateSublists_complete valid.1,
    decide_eq_true valid.2⟩

theorem matchingEnumerator_exact
    {Boundary Pair : Type} [DecidableEq Boundary]
    {rootPairs : List Pair}
    {endpoints : Pair -> Boundary × Boundary}
    {candidate : List Pair} :
    candidate ∈ enumerateMatchings rootPairs endpoints <->
      CanonicalMatching rootPairs endpoints candidate :=
  ⟨matchingEnumerator_sound, matchingEnumerator_complete⟩

theorem matching_pair_endpoints_are_rooted
    {Boundary Pair : Type}
    {rootBoundaries : List Boundary}
    {rootPairs matching : List Pair}
    {endpoints : Pair -> Boundary × Boundary}
    (pairRooted : forall pair, pair ∈ rootPairs ->
      (endpoints pair).1 ∈ rootBoundaries /\
        (endpoints pair).2 ∈ rootBoundaries)
    (matchingRooted : matching.Sublist rootPairs)
    {pair : Pair} (pairMember : pair ∈ matching) :
    (endpoints pair).1 ∈ rootBoundaries /\
      (endpoints pair).2 ∈ rootBoundaries :=
  pairRooted pair (matchingRooted.mem pairMember)

def enumerateRootMatchings
    {Boundary CompatiblePair AtomProposal DirectPair : Type}
    [DecidableEq Boundary]
    (roots : CarrierCandidateSetV03.RootFiniteUniverses
      Boundary CompatiblePair AtomProposal DirectPair) :
    List (List CompatiblePair) :=
  enumerateMatchings roots.compatiblePairUniverse
    roots.compatiblePairEndpoints

theorem rootMatchingEnumerator_exact
    {Boundary CompatiblePair AtomProposal DirectPair : Type}
    [DecidableEq Boundary]
    (roots : CarrierCandidateSetV03.RootFiniteUniverses
      Boundary CompatiblePair AtomProposal DirectPair)
    {candidate : List CompatiblePair} :
    candidate ∈ enumerateRootMatchings roots <->
      CanonicalMatching roots.compatiblePairUniverse
        roots.compatiblePairEndpoints candidate :=
  matchingEnumerator_exact

theorem root_matching_pair_endpoints_are_rooted
    {Boundary CompatiblePair AtomProposal DirectPair : Type}
    (roots : CarrierCandidateSetV03.RootFiniteUniverses
      Boundary CompatiblePair AtomProposal DirectPair)
    {matching : List CompatiblePair}
    (matchingRooted : matching.Sublist roots.compatiblePairUniverse)
    {pair : CompatiblePair} (pairMember : pair ∈ matching) :
    (roots.compatiblePairEndpoints pair).1 ∈ roots.boundaryUniverse /\
      (roots.compatiblePairEndpoints pair).2 ∈ roots.boundaryUniverse :=
  matching_pair_endpoints_are_rooted
    roots.compatiblePairEndpointsRooted matchingRooted pairMember

/-! ## Matching 决定唯一的 canonical unmatched complement -/

def BoundaryUsed
    {Boundary Pair : Type}
    (endpoints : Pair -> Boundary × Boundary)
    (matching : List Pair)
    (boundary : Boundary) : Prop :=
  Exists fun pair => pair ∈ matching /\
    (boundary = (endpoints pair).1 ∨ boundary = (endpoints pair).2)

instance boundaryUsedDecidable
    {Boundary Pair : Type} [DecidableEq Boundary] [DecidableEq Pair]
    (endpoints : Pair -> Boundary × Boundary)
    (matching : List Pair) (boundary : Boundary) :
    Decidable (BoundaryUsed endpoints matching boundary) := by
  unfold BoundaryUsed
  infer_instance

def unmatchedBoundaries
    {Boundary Pair : Type} [DecidableEq Boundary] [DecidableEq Pair]
    (rootBoundaries : List Boundary)
    (endpoints : Pair -> Boundary × Boundary)
    (matching : List Pair) : List Boundary :=
  rootBoundaries.filter fun boundary =>
    decide (Not (BoundaryUsed endpoints matching boundary))

theorem unmatched_boundary_iff
    {Boundary Pair : Type} [DecidableEq Boundary] [DecidableEq Pair]
    {rootBoundaries : List Boundary}
    {endpoints : Pair -> Boundary × Boundary}
    {matching : List Pair} {boundary : Boundary} :
    boundary ∈ unmatchedBoundaries rootBoundaries endpoints matching <->
      boundary ∈ rootBoundaries /\
        Not (BoundaryUsed endpoints matching boundary) := by
  simp [unmatchedBoundaries]

theorem root_boundary_used_xor_unmatched
    {Boundary Pair : Type} [DecidableEq Boundary] [DecidableEq Pair]
    {rootBoundaries : List Boundary}
    {endpoints : Pair -> Boundary × Boundary}
    {matching : List Pair} {boundary : Boundary}
    (rooted : boundary ∈ rootBoundaries) :
    (BoundaryUsed endpoints matching boundary /\
      boundary ∉ unmatchedBoundaries rootBoundaries endpoints matching) ∨
    (Not (BoundaryUsed endpoints matching boundary) /\
      boundary ∈ unmatchedBoundaries rootBoundaries endpoints matching) := by
  by_cases used : BoundaryUsed endpoints matching boundary
  · exact Or.inl ⟨used, fun unmatched =>
      (unmatched_boundary_iff.mp unmatched).2 used⟩
  · exact Or.inr ⟨used,
      unmatched_boundary_iff.mpr ⟨rooted, used⟩⟩

/-- 规范 unmatched 列表按 root boundary 顺序过滤，因此其选择谓词唯一。 -/
theorem canonical_unmatched_partition_unique
    {Boundary Pair : Type} [DecidableEq Boundary] [DecidableEq Pair]
    (rootBoundaries : List Boundary)
    (endpoints : Pair -> Boundary × Boundary)
    (matching : List Pair)
    (keep : Boundary -> Bool)
    (exactKeep : forall boundary,
      keep boundary = decide (Not (BoundaryUsed endpoints matching boundary))) :
    rootBoundaries.filter keep =
      unmatchedBoundaries rootBoundaries endpoints matching := by
  unfold unmatchedBoundaries
  congr 1
  funext boundary
  exact exactKeep boundary

/-! ## 当前 root record 对完整 interpretation universe 的 NO-GO -/

structure RootOnlyExactVerifier
    (World Root Interpretation : Type)
    (rootOf : World -> Root)
    (interpretationAt : World -> Interpretation -> Prop)
    (verify : Root -> Interpretation -> Prop) : Prop where
  sound : forall world interpretation,
    verify (rootOf world) interpretation ->
      interpretationAt world interpretation
  complete : forall world interpretation,
    interpretationAt world interpretation ->
      verify (rootOf world) interpretation

theorem same_root_different_interpretation_domains_forbid_root_only_exact_verifier
    {World Root Interpretation : Type}
    (rootOf : World -> Root)
    (interpretationAt : World -> Interpretation -> Prop)
    (left right : World)
    (sameRoot : rootOf left = rootOf right)
    (witness : Interpretation)
    (leftMember : interpretationAt left witness)
    (rightAbsent : Not (interpretationAt right witness)) :
    Not (Exists fun verify : Root -> Interpretation -> Prop =>
      RootOnlyExactVerifier World Root Interpretation
        rootOf interpretationAt verify) := by
  rintro ⟨verify, exact⟩
  apply rightAbsent
  apply exact.sound right witness
  rw [← sameRoot]
  exact exact.complete left witness leftMember

def RootOnlyComplete
    {World Root Interpretation : Type}
    (rootOf : World -> Root)
    (interpretationAt : World -> Interpretation -> Prop)
    (verify : Root -> Interpretation -> Prop) : Prop :=
  forall world interpretation,
    interpretationAt world interpretation ->
      verify (rootOf world) interpretation

def universalSupersetVerifier
    {Root Interpretation : Type} : Root -> Interpretation -> Prop :=
  fun _ _ => True

/-- completeness 单独成立没有辨识力；全体 Interpretation 超集总能通过。 -/
theorem universal_superset_is_root_only_complete
    {World Root Interpretation : Type}
    (rootOf : World -> Root)
    (interpretationAt : World -> Interpretation -> Prop) :
    RootOnlyComplete rootOf interpretationAt universalSupersetVerifier := by
  intro world interpretation member
  trivial

inductive MissingRootAxis where
  | fragmentSplit
  | intersection
  | ownership
  | port
  | lane
deriving DecidableEq, Repr

inductive HiddenWorld where
  | base
  | expanded
deriving DecidableEq, Repr

inductive HiddenInterpretation where
  | base
  | hiddenAlternative (axis : MissingRootAxis)
deriving DecidableEq, Repr

/-- 两个 world 的 v0.3 root record 按构造逐字相同。 -/
def hiddenRoot (_ : HiddenWorld) : Unit := ()

def hiddenInterpretationAt : HiddenWorld -> HiddenInterpretation -> Prop
  | .base, .base => True
  | .base, .hiddenAlternative _ => False
  | .expanded, _ => True

/--
对任一未进入 root 的隐藏轴，都存在同 root、不同 interpretation universe 的有限 witness。
-/
theorem concrete_missing_axis_forbids_root_only_exact_verifier
    (axis : MissingRootAxis) :
    Not (Exists fun verify : Unit -> HiddenInterpretation -> Prop =>
      RootOnlyExactVerifier HiddenWorld Unit HiddenInterpretation
        hiddenRoot hiddenInterpretationAt verify) := by
  exact same_root_different_interpretation_domains_forbid_root_only_exact_verifier
    hiddenRoot hiddenInterpretationAt .expanded .base rfl
      (.hiddenAlternative axis) trivial (by simp [hiddenInterpretationAt])

#print axioms enumerateSublists_sound
#print axioms enumerateSublists_complete
#print axioms matchingEnumerator_sound
#print axioms matchingEnumerator_complete
#print axioms matchingEnumerator_exact
#print axioms matching_pair_endpoints_are_rooted
#print axioms rootMatchingEnumerator_exact
#print axioms root_matching_pair_endpoints_are_rooted
#print axioms unmatched_boundary_iff
#print axioms root_boundary_used_xor_unmatched
#print axioms canonical_unmatched_partition_unique
#print axioms same_root_different_interpretation_domains_forbid_root_only_exact_verifier
#print axioms universal_superset_is_root_only_complete
#print axioms concrete_missing_axis_forbids_root_only_exact_verifier

end CarrierCompletenessVerifierV03

end ChemGraph
