import Std

namespace ChemGraph

namespace SymbolicCandidateFamilyV04

/-!
Candidate Set v0.4 的 factorized symbolic family 容量定理。

这里把一个“相互独立的二元歧义”严格定义为一个无约束 Boolean variable domain。
8 个这样的域的 Cartesian product 有 256 个互不相同的 assignment，因此无法作为
完整 canonical materialized list 放入 128 项容量；同一解族可以由 8 个变量域
factorized 表示。本模块只比较表示容量，不证明任何真实图像含有 8 个独立歧义。
-/

/-- `n` 个独立 Boolean variable domain。 -/
def binaryDomains : Nat -> List (List Bool)
  | 0 => []
  | n + 1 => [false, true] :: binaryDomains n

/-- 有限变量域的显式 Cartesian product。 -/
def expandDomains : List (List Bool) -> List (List Bool)
  | [] => [[]]
  | domain :: domains =>
      domain.flatMap fun value =>
        (expandDomains domains).map fun assignment => value :: assignment

theorem binaryDomains_length (n : Nat) :
    (binaryDomains n).length = n := by
  induction n with
  | zero => rfl
  | succ n ih => simp [binaryDomains, ih]

theorem every_binary_domain_has_two_values
    (n : Nat) {domain : List Bool}
    (member : domain ∈ binaryDomains n) :
    domain = [false, true] := by
  induction n with
  | zero =>
      simp [binaryDomains] at member
  | succ n ih =>
      simp only [binaryDomains, List.mem_cons] at member
      rcases member with rfl | member
      · rfl
      · exact ih member

theorem expand_binary_domains_length (n : Nat) :
    (expandDomains (binaryDomains n)).length = 2 ^ n := by
  induction n with
  | zero => rfl
  | succ n ih =>
      simp [binaryDomains, expandDomains, ih, Nat.pow_succ]
      omega

theorem nodup_map_cons_bool
    (value : Bool) {assignments : List (List Bool)}
    (nodup : assignments.Nodup) :
    (assignments.map fun assignment => value :: assignment).Nodup := by
  unfold List.Nodup at nodup ⊢
  rw [List.pairwise_map]
  apply nodup.imp
  intro left right different equalCons
  apply different
  exact (List.cons.inj equalCons).2

theorem expand_binary_domains_nodup (n : Nat) :
    (expandDomains (binaryDomains n)).Nodup := by
  induction n with
  | zero => simp [binaryDomains, expandDomains]
  | succ n ih =>
      simp only [binaryDomains, expandDomains, List.flatMap_cons,
        List.flatMap_nil, List.append_nil]
      rw [List.nodup_append]
      refine ⟨nodup_map_cons_bool false ih,
        nodup_map_cons_bool true ih, ?_⟩
      intro left leftMember right rightMember equal
      simp only [List.mem_map] at leftMember rightMember
      rcases leftMember with ⟨leftTail, _, leftEq⟩
      rcases rightMember with ⟨rightTail, _, rightEq⟩
      rw [← leftEq, ← rightEq] at equal
      cases equal

theorem eight_binary_assignments_count :
    (expandDomains (binaryDomains 8)).length = 256 := by
  rw [expand_binary_domains_length]

theorem eight_binary_assignments_nodup :
    (expandDomains (binaryDomains 8)).Nodup :=
  expand_binary_domains_nodup 8

theorem eight_binary_assignments_exceed_capacity_128 :
    Not ((expandDomains (binaryDomains 8)).length <= 128) := by
  rw [eight_binary_assignments_count]
  omega

/-- canonical materialization 若完整保留 256 解族，就必然超过 128 项。 -/
theorem capacity_128_cannot_materialize_eight_binary_assignments
    (materialized : List (List Bool))
    (complete : (expandDomains (binaryDomains 8)).Sublist materialized)
    (withinCapacity : materialized.length <= 128) : False := by
  have lower := List.Sublist.length_le complete
  rw [eight_binary_assignments_count] at lower
  omega

structure FactorizedCandidateFamily where
  variableDomains : List (List Bool)
deriving DecidableEq, Repr

def FactorizedCandidateFamily.solutions
    (family : FactorizedCandidateFamily) : List (List Bool) :=
  expandDomains family.variableDomains

def eightIndependentBinaryFamily : FactorizedCandidateFamily where
  variableDomains := binaryDomains 8

theorem factorized_family_uses_eight_variable_domains :
    eightIndependentBinaryFamily.variableDomains.length = 8 :=
  binaryDomains_length 8

theorem factorized_family_expands_to_same_256_assignments :
    eightIndependentBinaryFamily.solutions =
      expandDomains (binaryDomains 8) :=
  rfl

theorem factorized_family_solution_count :
    eightIndependentBinaryFamily.solutions.length = 256 :=
  eight_binary_assignments_count

theorem factorized_family_solutions_nodup :
    eightIndependentBinaryFamily.solutions.Nodup :=
  eight_binary_assignments_nodup

#print axioms binaryDomains_length
#print axioms every_binary_domain_has_two_values
#print axioms expand_binary_domains_length
#print axioms expand_binary_domains_nodup
#print axioms eight_binary_assignments_count
#print axioms eight_binary_assignments_nodup
#print axioms eight_binary_assignments_exceed_capacity_128
#print axioms capacity_128_cannot_materialize_eight_binary_assignments
#print axioms factorized_family_uses_eight_variable_domains
#print axioms factorized_family_expands_to_same_256_assignments
#print axioms factorized_family_solution_count
#print axioms factorized_family_solutions_nodup

end SymbolicCandidateFamilyV04

end ChemGraph
