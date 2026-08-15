import Std

namespace ChemGraph

namespace OwnerCapacityReachabilityV05

/-!
Owner Candidate Domain v0.5 的容量可达性算术内核。

本模块只证明冻结 fixed-R0 计数合同及若干必要上界。它不定义 Python
constructor 的 measurement point、gate 顺序、straight-carrier 几何、PNG、RLE 或
canonical JSON codec。因此这里的条件上界不能冒充某个容量轴已经获得端到端
L/H witness；几何 incidence 与序列化字节上界仍需冻结的 Python proof checker。
-/

/-! ## Fixed R0 owner 与 constraint 计数 -/

structure FixedR0Counts where
  ownerOptionCount : Nat
  constraintCount : Nat

def FixedR0CountLaw (counts : FixedR0Counts) : Prop :=
  counts.ownerOptionCount = 2 /\
    counts.constraintCount = 1 + 3 * counts.ownerOptionCount

theorem fixed_r0_owner_option_count_eq_two
    (counts : FixedR0Counts) (fixed : FixedR0CountLaw counts) :
    counts.ownerOptionCount = 2 :=
  fixed.1

theorem fixed_r0_constraint_count_eq_seven
    (counts : FixedR0Counts) (fixed : FixedR0CountLaw counts) :
    counts.constraintCount = 7 := by
  rcases fixed with ⟨owners, constraints⟩
  omega

/-! ## 两 owner 的规范非空 domain class 只有三种 -/

def canonicalNonemptyTwoOwnerDomains : List (List Bool) :=
  [[false], [true], [false, true]]

def CanonicalTwoOwnerDomainClasses (classes : List (List Bool)) : Prop :=
  classes.Nodup /\ classes ⊆ canonicalNonemptyTwoOwnerDomains

theorem canonical_nonempty_two_owner_domain_classes_le_three
    (classes : List (List Bool))
    (canonical : CanonicalTwoOwnerDomainClasses classes) :
    classes.length <= 3 := by
  have bound := canonical.1.length_le_of_subset canonical.2
  simpa [canonicalNonemptyTwoOwnerDomains] using bound

/-! ## 8-neighbor simple undirected edge 的握手上界 -/

theorem grid8_edges_below_declared_high_boundary
    (vertexCount edgeCount : Nat)
    (degreeSumBound : 2 * edgeCount <= 8 * vertexCount)
    (vertexCapacity : vertexCount <= 4096) :
    edgeCount <= 16384 /\ edgeCount < 32769 := by
  omega

/-! ## 二元 domain 的 raw Cartesian assignment 上界 -/

theorem binary_raw_assignments_le_limit
    (ambiguousCount rawAssignmentCount : Nat)
    (rawCount : rawAssignmentCount = 2 ^ ambiguousCount)
    (ambiguousCapacity : ambiguousCount <= 20) :
    rawAssignmentCount <= 2 ^ 20 := by
  rw [rawCount]
  exact Nat.pow_le_pow_right (by omega) ambiguousCapacity

theorem binary_raw_over_limit_implies_ambiguous_over_limit
    (ambiguousCount rawAssignmentCount : Nat)
    (rawCount : rawAssignmentCount = 2 ^ ambiguousCount)
    (rawOverLimit : 2 ^ 20 < rawAssignmentCount) :
    20 < ambiguousCount := by
  rw [rawCount] at rawOverLimit
  exact (Nat.pow_lt_pow_iff_right (by omega)).mp rawOverLimit

/-! ## Straight fragment 的条件式平方上界 -/

theorem straight_fragment_count_le_sixty_four
    (carrierCount fragmentCount : Nat)
    (straightIncidenceBound : fragmentCount <= carrierCount ^ 2)
    (carrierCapacity : carrierCount <= 8) :
    fragmentCount <= 64 := by
  calc
    fragmentCount <= carrierCount ^ 2 := straightIncidenceBound
    _ <= 8 ^ 2 := Nat.pow_le_pow_left carrierCapacity 2
    _ = 64 := by decide

#print axioms fixed_r0_owner_option_count_eq_two
#print axioms fixed_r0_constraint_count_eq_seven
#print axioms canonical_nonempty_two_owner_domain_classes_le_three
#print axioms grid8_edges_below_declared_high_boundary
#print axioms binary_raw_assignments_le_limit
#print axioms binary_raw_over_limit_implies_ambiguous_over_limit
#print axioms straight_fragment_count_le_sixty_four

end OwnerCapacityReachabilityV05

end ChemGraph
