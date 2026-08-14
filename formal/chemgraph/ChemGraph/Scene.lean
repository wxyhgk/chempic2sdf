namespace ChemGraph

/-! 一个只保留十字局部所需信息的有限离散墨迹模型。 -/

inductive Arm where
  | north
  | east
  | south
  | west
deriving DecidableEq, Repr, BEq

structure Ink where
  north : Bool := false
  east : Bool := false
  south : Bool := false
  west : Bool := false
deriving DecidableEq, Repr, BEq

namespace Ink

def empty : Ink := {}

def singleton : Arm -> Ink
  | .north => { north := true }
  | .east => { east := true }
  | .south => { south := true }
  | .west => { west := true }

def merge (a b : Ink) : Ink := {
  north := a.north || b.north
  east := a.east || b.east
  south := a.south || b.south
  west := a.west || b.west
}

def northSouth : Ink := merge (singleton .north) (singleton .south)

def eastWest : Ink := merge (singleton .east) (singleton .west)

def fullCross : Ink := merge northSouth eastWest

def hasInk (ink : Ink) : Bool :=
  ink.north || ink.east || ink.south || ink.west

end Ink

/-! `Edge` 使用规范化的无向端点编号，避免端点顺序成为语义。 -/
structure Edge where
  lo : Nat
  hi : Nat
deriving DecidableEq, Repr, BEq

def mkEdge (a b : Nat) : Edge :=
  { lo := min a b, hi := max a b }

theorem mkEdge_swap (a b : Nat) : mkEdge a b = mkEdge b a := by
  simp [mkEdge, Nat.min_comm, Nat.max_comm]

structure SemanticBond where
  endpoints : Edge
  trace : Ink
  order : Nat := 1
deriving DecidableEq, Repr, BEq

structure GraphBond where
  endpoints : Edge
  order : Nat
deriving DecidableEq, Repr, BEq

structure ChemicalGraph where
  atomCount : Nat
  bonds : List GraphBond
deriving DecidableEq, Repr, BEq

structure Scene where
  atomCount : Nat
  bonds : List SemanticBond
deriving DecidableEq, Repr, BEq

def Scene.graph (scene : Scene) : ChemicalGraph := {
  atomCount := scene.atomCount
  bonds := scene.bonds.map (fun bond => {
    endpoints := bond.endpoints
    order := bond.order
  })
}

/-!
当前有限模型用空间角色给 Atom 规范编号；一般分子还需要再商掉 Atom 重命名。
Bond 列表顺序不属于图语义，因此先显式定义忽略列表排列的等价关系。
-/
def GraphEquivalent (left right : ChemicalGraph) : Prop :=
  And (left.atomCount = right.atomCount) (left.bonds.Perm right.bonds)

def graphEquivalentBool (left right : ChemicalGraph) : Bool :=
  left.atomCount == right.atomCount && left.bonds.isPerm right.bonds

def ValidBondOrder (order : Nat) : Prop :=
  order = 1 ∨ order = 2 ∨ order = 3

private def validBondBool (atomCount : Nat) (bond : SemanticBond) : Bool :=
  decide (bond.endpoints.lo < bond.endpoints.hi) &&
    decide (bond.endpoints.hi < atomCount) &&
    (bond.order == 1 || bond.order == 2 || bond.order == 3) &&
    bond.trace.hasInk

private def noDuplicateEndpoints : List SemanticBond -> Bool
  | [] => true
  | bond :: rest =>
      !(rest.any fun other => other.endpoints == bond.endpoints) &&
        noDuplicateEndpoints rest

def Scene.valid (scene : Scene) : Bool :=
  scene.bonds.all (validBondBool scene.atomCount) &&
    noDuplicateEndpoints scene.bonds &&
    (List.range scene.atomCount).all fun atom =>
      scene.bonds.any fun bond =>
        bond.endpoints.lo == atom || bond.endpoints.hi == atom

def Scene.Valid (scene : Scene) : Prop := scene.valid = true

end ChemGraph
