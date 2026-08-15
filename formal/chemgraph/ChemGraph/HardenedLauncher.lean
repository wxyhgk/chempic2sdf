import Std

namespace ChemGraph

namespace HardenedLauncher

/-!
Pure logical kernel for a hardened launcher publication boundary.

The operating system supplies observations and proofs of the predicates below.
Lean only proves that an authority marker cannot be constructed or reached unless
all front-door obligations are present.  In particular, this module does not
claim to inspect cgroups, namespaces, clocks, files, or process tables.
-/

structure ResourceLimits where
  workers : Nat
  maxWallNanoseconds : Nat
  maxTotalMemoryBytes : Nat
  maxTemporaryAllocatedBytes : Nat
  maxPublishedLogicalBytes : Nat
  maxPublishedAllocatedBytes : Nat
  maxOomEvents : Nat
deriving DecidableEq, Repr

structure ResourceObservation where
  workers : Nat
  wallNanoseconds : Nat
  totalMemoryBytes : Nat
  temporaryAllocatedBytes : Nat
  publishedLogicalBytes : Nat
  publishedAllocatedBytes : Nat
  oomEvents : Nat
  nonzeroExitCount : Nat
deriving DecidableEq, Repr

def ResourceWithinLimits
    (limits : ResourceLimits) (observed : ResourceObservation) : Prop :=
  observed.workers = limits.workers ∧
    observed.wallNanoseconds <= limits.maxWallNanoseconds ∧
      observed.totalMemoryBytes <= limits.maxTotalMemoryBytes ∧
        observed.temporaryAllocatedBytes <= limits.maxTemporaryAllocatedBytes ∧
          observed.publishedLogicalBytes <= limits.maxPublishedLogicalBytes ∧
            observed.publishedAllocatedBytes <= limits.maxPublishedAllocatedBytes ∧
              observed.oomEvents <= limits.maxOomEvents ∧
                observed.nonzeroExitCount = 0

theorem resourceWithinLimits_iff_all
    (limits : ResourceLimits) (observed : ResourceObservation) :
    ResourceWithinLimits limits observed ↔
      observed.workers = limits.workers ∧
        observed.wallNanoseconds <= limits.maxWallNanoseconds ∧
          observed.totalMemoryBytes <= limits.maxTotalMemoryBytes ∧
            observed.temporaryAllocatedBytes <= limits.maxTemporaryAllocatedBytes ∧
              observed.publishedLogicalBytes <= limits.maxPublishedLogicalBytes ∧
                observed.publishedAllocatedBytes <= limits.maxPublishedAllocatedBytes ∧
                  observed.oomEvents <= limits.maxOomEvents ∧
                    observed.nonzeroExitCount = 0 := by
  rfl

theorem resource_gate_rejects_worker_mismatch
    (limits : ResourceLimits) (observed : ResourceObservation)
    (mismatch : observed.workers ≠ limits.workers) :
    ¬ ResourceWithinLimits limits observed := by
  intro gate
  exact mismatch gate.1

theorem resource_gate_rejects_nonzero_exit
    (limits : ResourceLimits) (observed : ResourceObservation)
    (failed : observed.nonzeroExitCount ≠ 0) :
    ¬ ResourceWithinLimits limits observed := by
  intro gate
  exact failed gate.2.2.2.2.2.2.2

structure Policy (Run Bytes Digest : Type) where
  semanticOK : Run -> Prop
  isolationOK : Run -> Prop
  limits : ResourceLimits
  resources : Run -> ResourceObservation
  quiescenceOK : Run -> Prop
  publicationOK : Run -> Prop
  actualBytes : Run -> Bytes
  expectedBytes : Run -> Bytes
  digest : Run -> Digest
  hash : Bytes -> Digest

structure FrontDoor
    {Run Bytes Digest : Type}
    (policy : Policy Run Bytes Digest) (run : Run) : Prop where
  semantic : policy.semanticOK run
  isolation : policy.isolationOK run
  resource : ResourceWithinLimits policy.limits (policy.resources run)
  quiescence : policy.quiescenceOK run
  publication : policy.publicationOK run
  fullBytes : policy.actualBytes run = policy.expectedBytes run
  digestIndex : policy.digest run = policy.hash (policy.actualBytes run)

structure AuthorityMarker
    {Run Bytes Digest : Type}
    (policy : Policy Run Bytes Digest) (run : Run) : Prop where
  gate : FrontDoor policy run

theorem marker_iff_frontDoor
    {Run Bytes Digest : Type}
    {policy : Policy Run Bytes Digest} {run : Run} :
    AuthorityMarker policy run ↔ FrontDoor policy run := by
  constructor
  · intro marker
    exact marker.gate
  · intro gate
    exact ⟨gate⟩

theorem marker_implies_all_front_doors
    {Run Bytes Digest : Type}
    {policy : Policy Run Bytes Digest} {run : Run}
    (marker : AuthorityMarker policy run) :
    policy.semanticOK run ∧
      policy.isolationOK run ∧
        ResourceWithinLimits policy.limits (policy.resources run) ∧
          policy.quiescenceOK run ∧
            policy.publicationOK run ∧
              policy.actualBytes run = policy.expectedBytes run ∧
                policy.digest run = policy.hash (policy.actualBytes run) := by
  exact ⟨marker.gate.semantic, marker.gate.isolation, marker.gate.resource,
    marker.gate.quiescence, marker.gate.publication, marker.gate.fullBytes,
    marker.gate.digestIndex⟩

theorem no_marker_of_semantic_failure
    {Run Bytes Digest : Type}
    {policy : Policy Run Bytes Digest} {run : Run}
    (failed : ¬ policy.semanticOK run) :
    ¬ AuthorityMarker policy run := by
  intro marker
  exact failed marker.gate.semantic

theorem no_marker_of_isolation_failure
    {Run Bytes Digest : Type}
    {policy : Policy Run Bytes Digest} {run : Run}
    (failed : ¬ policy.isolationOK run) :
    ¬ AuthorityMarker policy run := by
  intro marker
  exact failed marker.gate.isolation

theorem no_marker_of_resource_failure
    {Run Bytes Digest : Type}
    {policy : Policy Run Bytes Digest} {run : Run}
    (failed : ¬ ResourceWithinLimits policy.limits (policy.resources run)) :
    ¬ AuthorityMarker policy run := by
  intro marker
  exact failed marker.gate.resource

theorem no_marker_of_quiescence_failure
    {Run Bytes Digest : Type}
    {policy : Policy Run Bytes Digest} {run : Run}
    (failed : ¬ policy.quiescenceOK run) :
    ¬ AuthorityMarker policy run := by
  intro marker
  exact failed marker.gate.quiescence

theorem no_marker_of_publication_failure
    {Run Bytes Digest : Type}
    {policy : Policy Run Bytes Digest} {run : Run}
    (failed : ¬ policy.publicationOK run) :
    ¬ AuthorityMarker policy run := by
  intro marker
  exact failed marker.gate.publication

theorem no_marker_of_full_byte_failure
    {Run Bytes Digest : Type}
    {policy : Policy Run Bytes Digest} {run : Run}
    (failed : policy.actualBytes run ≠ policy.expectedBytes run) :
    ¬ AuthorityMarker policy run := by
  intro marker
  exact failed marker.gate.fullBytes

theorem no_marker_of_digest_index_failure
    {Run Bytes Digest : Type}
    {policy : Policy Run Bytes Digest} {run : Run}
    (failed : policy.digest run ≠ policy.hash (policy.actualBytes run)) :
    ¬ AuthorityMarker policy run := by
  intro marker
  exact failed marker.gate.digestIndex

inductive LauncherState
    {Run Bytes Digest : Type}
    (policy : Policy Run Bytes Digest) (run : Run)
    (Failure Pid : Type) where
  | running
  | candidate
  | eligible (gate : FrontDoor policy run)
  | authoritative (marker : AuthorityMarker policy run)
  | failed (reason : Failure)
  | timedOut
  | residual (pids : List Pid) (nonempty : pids ≠ [])
  | quarantined

inductive Step
    {Run Bytes Digest Failure Pid : Type}
    {policy : Policy Run Bytes Digest} {run : Run} :
    LauncherState policy run Failure Pid ->
      LauncherState policy run Failure Pid -> Prop where
  | stage : Step .running .candidate
  | certify (gate : FrontDoor policy run) : Step .candidate (.eligible gate)
  | publish (gate : FrontDoor policy run) :
      Step (.eligible gate) (.authoritative ⟨gate⟩)
  | failRunning (reason : Failure) : Step .running (.failed reason)
  | failCandidate (reason : Failure) : Step .candidate (.failed reason)
  | failEligible (gate : FrontDoor policy run) (reason : Failure) :
      Step (.eligible gate) (.failed reason)
  | timeoutRunning : Step .running .timedOut
  | timeoutCandidate : Step .candidate .timedOut
  | timeoutEligible (gate : FrontDoor policy run) : Step (.eligible gate) .timedOut
  | residualRunning (pids : List Pid) (nonempty : pids ≠ []) :
      Step .running (.residual pids nonempty)
  | residualCandidate (pids : List Pid) (nonempty : pids ≠ []) :
      Step .candidate (.residual pids nonempty)
  | residualEligible (gate : FrontDoor policy run)
      (pids : List Pid) (nonempty : pids ≠ []) :
      Step (.eligible gate) (.residual pids nonempty)
  | quarantineCandidate : Step .candidate .quarantined
  | quarantineEligible (gate : FrontDoor policy run) :
      Step (.eligible gate) .quarantined

inductive Reach {State : Type} (step : State -> State -> Prop) :
    State -> State -> Prop where
  | refl (state : State) : Reach step state state
  | next {source middle target : State} :
      step source middle -> Reach step middle target -> Reach step source target

def Terminal {State : Type} (step : State -> State -> Prop) (state : State) : Prop :=
  forall target, ¬ step state target

theorem step_to_authority_only_from_eligible
    {Run Bytes Digest Failure Pid : Type}
    {policy : Policy Run Bytes Digest} {run : Run}
    {source : LauncherState policy run Failure Pid}
    {marker : AuthorityMarker policy run}
    (transition : Step source (.authoritative marker)) :
    Exists fun gate : FrontDoor policy run => source = .eligible gate := by
  cases transition with
  | publish gate => exact ⟨gate, rfl⟩

theorem running_cannot_bypass_to_authority
    {Run Bytes Digest Failure Pid : Type}
    {policy : Policy Run Bytes Digest} {run : Run}
    {marker : AuthorityMarker policy run} :
    ¬ Step (Failure := Failure) (Pid := Pid) .running (.authoritative marker) := by
  intro transition
  cases transition

theorem candidate_cannot_bypass_to_authority
    {Run Bytes Digest Failure Pid : Type}
    {policy : Policy Run Bytes Digest} {run : Run}
    {marker : AuthorityMarker policy run} :
    ¬ Step (Failure := Failure) (Pid := Pid) .candidate (.authoritative marker) := by
  intro transition
  cases transition

theorem failed_terminal
    {Run Bytes Digest Failure Pid : Type}
    {policy : Policy Run Bytes Digest} {run : Run} (reason : Failure) :
    Terminal (Step (policy := policy) (run := run) (Failure := Failure) (Pid := Pid))
      (.failed reason) := by
  intro target transition
  cases transition

theorem timedOut_terminal
    {Run Bytes Digest Failure Pid : Type}
    {policy : Policy Run Bytes Digest} {run : Run} :
    Terminal (Step (policy := policy) (run := run) (Failure := Failure) (Pid := Pid))
      .timedOut := by
  intro target transition
  cases transition

theorem residual_terminal
    {Run Bytes Digest Failure Pid : Type}
    {policy : Policy Run Bytes Digest} {run : Run}
    (pids : List Pid) (nonempty : pids ≠ []) :
    Terminal (Step (policy := policy) (run := run) (Failure := Failure) (Pid := Pid))
      (.residual pids nonempty) := by
  intro target transition
  cases transition

theorem quarantined_terminal
    {Run Bytes Digest Failure Pid : Type}
    {policy : Policy Run Bytes Digest} {run : Run} :
    Terminal (Step (policy := policy) (run := run) (Failure := Failure) (Pid := Pid))
      .quarantined := by
  intro target transition
  cases transition

theorem terminal_reach_eq
    {State : Type} {step : State -> State -> Prop} {source target : State}
    (terminal : Terminal step source) (reachable : Reach step source target) :
    target = source := by
  cases reachable with
  | refl => rfl
  | next first _ => exact False.elim (terminal _ first)

theorem failed_cannot_reach_authority
    {Run Bytes Digest Failure Pid : Type}
    {policy : Policy Run Bytes Digest} {run : Run}
    (reason : Failure) (marker : AuthorityMarker policy run) :
    ¬ Reach (Step (policy := policy) (run := run) (Failure := Failure) (Pid := Pid))
      (.failed reason) (.authoritative marker) := by
  intro reachable
  have impossible := terminal_reach_eq (failed_terminal (Pid := Pid) reason) reachable
  cases impossible

theorem timedOut_cannot_reach_authority
    {Run Bytes Digest Failure Pid : Type}
    {policy : Policy Run Bytes Digest} {run : Run}
    (marker : AuthorityMarker policy run) :
    ¬ Reach (Step (policy := policy) (run := run) (Failure := Failure) (Pid := Pid))
      .timedOut (.authoritative marker) := by
  intro reachable
  have impossible := terminal_reach_eq
    (timedOut_terminal (policy := policy) (run := run) (Failure := Failure) (Pid := Pid))
    reachable
  cases impossible

theorem residual_cannot_reach_authority
    {Run Bytes Digest Failure Pid : Type}
    {policy : Policy Run Bytes Digest} {run : Run}
    (pids : List Pid) (nonempty : pids ≠ [])
    (marker : AuthorityMarker policy run) :
    ¬ Reach (Step (policy := policy) (run := run) (Failure := Failure) (Pid := Pid))
      (.residual pids nonempty) (.authoritative marker) := by
  intro reachable
  have impossible := terminal_reach_eq
    (residual_terminal (policy := policy) (run := run) (Failure := Failure)
      pids nonempty)
    reachable
  cases impossible

theorem quarantined_cannot_reach_authority
    {Run Bytes Digest Failure Pid : Type}
    {policy : Policy Run Bytes Digest} {run : Run}
    (marker : AuthorityMarker policy run) :
    ¬ Reach (Step (policy := policy) (run := run) (Failure := Failure) (Pid := Pid))
      .quarantined (.authoritative marker) := by
  intro reachable
  have impossible := terminal_reach_eq
    (quarantined_terminal (policy := policy) (run := run)
      (Failure := Failure) (Pid := Pid)) reachable
  cases impossible

theorem full_bytes_imply_equal_digest
    {Bytes Digest : Type} (hash : Bytes -> Digest) (left right : Bytes)
    (equalBytes : left = right) : hash left = hash right := by
  exact congrArg hash equalBytes

theorem frontDoor_digest_indexes_expected_bytes
    {Run Bytes Digest : Type}
    {policy : Policy Run Bytes Digest} {run : Run}
    (gate : FrontDoor policy run) :
    policy.digest run = policy.hash (policy.expectedBytes run) := by
  calc
    policy.digest run = policy.hash (policy.actualBytes run) := gate.digestIndex
    _ = policy.hash (policy.expectedBytes run) :=
      full_bytes_imply_equal_digest policy.hash _ _ gate.fullBytes

theorem equal_digest_does_not_imply_equal_bytes :
    Exists fun hash : Bool -> Unit => Exists fun left : Bool => Exists fun right : Bool =>
      hash left = hash right ∧ left ≠ right := by
  exact ⟨fun _ => (), false, true, rfl, by decide⟩

#print axioms marker_iff_frontDoor
#print axioms marker_implies_all_front_doors
#print axioms no_marker_of_semantic_failure
#print axioms no_marker_of_isolation_failure
#print axioms no_marker_of_resource_failure
#print axioms no_marker_of_quiescence_failure
#print axioms no_marker_of_publication_failure
#print axioms no_marker_of_full_byte_failure
#print axioms no_marker_of_digest_index_failure
#print axioms step_to_authority_only_from_eligible
#print axioms running_cannot_bypass_to_authority
#print axioms candidate_cannot_bypass_to_authority
#print axioms failed_cannot_reach_authority
#print axioms timedOut_cannot_reach_authority
#print axioms residual_cannot_reach_authority
#print axioms quarantined_cannot_reach_authority
#print axioms resourceWithinLimits_iff_all
#print axioms resource_gate_rejects_worker_mismatch
#print axioms resource_gate_rejects_nonzero_exit
#print axioms full_bytes_imply_equal_digest
#print axioms frontDoor_digest_indexes_expected_bytes
#print axioms equal_digest_does_not_imply_equal_bytes

end HardenedLauncher

end ChemGraph
