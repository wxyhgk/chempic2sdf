import ChemGraph.Multiplicity
import ChemGraph.Recovery

namespace ChemGraph

/-!
最小可执行碰撞审计。真实 production tensor 枚举器尚未实现；这里先让
Lean 对有限抽象自动分类，避免只靠手写定理描述预期结果。
-/
inductive ToyKind where
  | empty
  | singleArm
  | corner
  | cornerReordered
  | crossing
  | junction
  | doubleBond
  | parallelSingles
deriving DecidableEq, Repr, BEq

inductive ToyObservation where
  | crossInk (ink : Ink)
  | laneInk (ink : LaneInk)
deriving DecidableEq, Repr, BEq

def emptyScene : Scene := { atomCount := 0, bonds := [] }

def singleArmScene : Scene := {
  atomCount := 2
  bonds := [{
    endpoints := mkEdge 0 1
    trace := Ink.singleton .north
  }]
}

def cornerScene : Scene := {
  atomCount := 3
  bonds := [
    { endpoints := mkEdge 0 2, trace := Ink.singleton .north },
    { endpoints := mkEdge 1 2, trace := Ink.singleton .east }
  ]
}

def cornerReorderedScene : Scene := {
  atomCount := 3
  bonds := cornerScene.bonds.reverse
}

def ToyKind.observation : ToyKind -> ToyObservation
  | .empty => .crossInk (renderOpaque emptyScene)
  | .singleArm => .crossInk (renderOpaque singleArmScene)
  | .corner => .crossInk (renderOpaque cornerScene)
  | .cornerReordered => .crossInk (renderOpaque cornerReorderedScene)
  | .crossing => .crossInk (renderOpaque crossingScene)
  | .junction => .crossInk (renderOpaque junctionScene)
  | .doubleBond => .laneInk (renderLaneOpaque doubleBondScene)
  | .parallelSingles => .laneInk (renderLaneOpaque parallelSinglesScene)

def ToyKind.graph : ToyKind -> ChemicalGraph
  | .empty => emptyScene.graph
  | .singleArm => singleArmScene.graph
  | .corner => cornerScene.graph
  | .cornerReordered => cornerReorderedScene.graph
  | .crossing => crossingScene.graph
  | .junction => junctionScene.graph
  | .doubleBond => doubleBondScene.graph
  | .parallelSingles => parallelSinglesScene.graph

def allToyKinds : List ToyKind := [
  .empty,
  .singleArm,
  .corner,
  .cornerReordered,
  .crossing,
  .junction,
  .doubleBond,
  .parallelSingles
]

def hasGraphCollision (kind : ToyKind) : Bool :=
  allToyKinds.any fun other =>
    (kind.observation == other.observation) &&
      !(graphEquivalentBool kind.graph other.graph)

def strictToyKinds : List ToyKind :=
  allToyKinds.filter fun kind => !hasGraphCollision kind

def ambiguousToyKinds : List ToyKind :=
  allToyKinds.filter hasGraphCollision

theorem finite_audit_strict_expected :
    strictToyKinds = [.empty, .singleArm, .corner, .cornerReordered] := by
  decide

theorem finite_audit_ambiguity_expected :
    ambiguousToyKinds = [.crossing, .junction, .doubleBond, .parallelSingles] := by
  decide

theorem finite_audit_endpoint_swap_invariant (a b : Nat) :
    mkEdge a b = mkEdge b a :=
  mkEdge_swap a b

theorem finite_audit_bond_reorder_is_semantically_equal :
    graphEquivalentBool cornerScene.graph cornerReorderedScene.graph = true := by
  decide

end ChemGraph
