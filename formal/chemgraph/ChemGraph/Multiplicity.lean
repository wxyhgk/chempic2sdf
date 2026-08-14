import ChemGraph.Identifiability
import ChemGraph.Scene

namespace ChemGraph

/-!
`LaneInk` 只记录两条可见的平行笔画，不记录它们属于一根双键，还是
分别属于两根单键。这是“先检测线，再事后猜键级”必须面对的信息损失。
-/
structure LaneInk where
  primary : Bool := false
  secondary : Bool := false
deriving DecidableEq, Repr, BEq

namespace LaneInk

def empty : LaneInk := {}

def primaryOnly : LaneInk := { primary := true }

def secondaryOnly : LaneInk := { secondary := true }

def both : LaneInk := { primary := true, secondary := true }

def merge (a b : LaneInk) : LaneInk := {
  primary := a.primary || b.primary
  secondary := a.secondary || b.secondary
}

end LaneInk

structure LaneBond where
  endpoints : Edge
  order : Nat
  physicalLanes : LaneInk
deriving DecidableEq, Repr, BEq

structure LaneScene where
  atomCount : Nat
  bonds : List LaneBond
deriving DecidableEq, Repr, BEq

def LaneScene.graph (scene : LaneScene) : ChemicalGraph := {
  atomCount := scene.atomCount
  bonds := scene.bonds.map (fun bond => {
    endpoints := bond.endpoints
    order := bond.order
  })
}

def renderLaneOpaque (scene : LaneScene) : LaneInk :=
  scene.bonds.foldl
    (fun observation bond => LaneInk.merge observation bond.physicalLanes)
    LaneInk.empty

/-! 一根双键：两个原子，一个 order=2 的语义 Bond，两条 physical lane。 -/
def doubleBondScene : LaneScene := {
  atomCount := 2
  bonds := [{
    endpoints := mkEdge 0 1
    order := 2
    physicalLanes := LaneInk.both
  }]
}

/-! 两根平行单键：四个原子，两个 order=1 的语义 Bond。 -/
def parallelSinglesScene : LaneScene := {
  atomCount := 4
  bonds := [
    {
      endpoints := mkEdge 0 1
      order := 1
      physicalLanes := LaneInk.primaryOnly
    },
    {
      endpoints := mkEdge 2 3
      order := 1
      physicalLanes := LaneInk.secondaryOnly
    }
  ]
}

theorem double_and_parallel_singles_same_opaque_lanes :
    renderLaneOpaque doubleBondScene = renderLaneOpaque parallelSinglesScene := by
  decide

theorem double_and_parallel_singles_different_graph :
    doubleBondScene.graph ≠ parallelSinglesScene.graph := by
  decide

theorem double_and_parallel_singles_not_graph_equivalent :
    Not (GraphEquivalent doubleBondScene.graph parallelSinglesScene.graph) := by
  intro equivalent
  have differentAtomCount :
      doubleBondScene.graph.atomCount ≠ parallelSinglesScene.graph.atomCount := by
    decide
  exact differentAtomCount equivalent.left

theorem opaque_lanes_not_identifiable :
    Not (Identifiable renderLaneOpaque LaneScene.graph) := by
  intro identifiable
  apply double_and_parallel_singles_different_graph
  exact identifiable doubleBondScene parallelSinglesScene
    double_and_parallel_singles_same_opaque_lanes

theorem no_universal_opaque_lane_decoder :
    Not (Exists fun decode : LaneInk -> ChemicalGraph =>
      forall scene, decode (renderLaneOpaque scene) = scene.graph) :=
  collision_forbids_exact_decoder renderLaneOpaque LaneScene.graph
    doubleBondScene parallelSinglesScene
    double_and_parallel_singles_same_opaque_lanes
    double_and_parallel_singles_different_graph

/-!
布尔并集还具有幂等性：两根完全重叠但端点不同的合法 Bond，会和只保留
其中一根得到同一墨迹。这个反例独立于 Bond 列表顺序。
-/
def oneOverlaidBondScene : LaneScene := {
  atomCount := 4
  bonds := [{
    endpoints := mkEdge 0 1
    order := 1
    physicalLanes := LaneInk.primaryOnly
  }]
}

def twoOverlaidBondsScene : LaneScene := {
  atomCount := 4
  bonds := [
    {
      endpoints := mkEdge 0 1
      order := 1
      physicalLanes := LaneInk.primaryOnly
    },
    {
      endpoints := mkEdge 2 3
      order := 1
      physicalLanes := LaneInk.primaryOnly
    }
  ]
}

theorem opaque_lane_merge_is_idempotent :
    renderLaneOpaque oneOverlaidBondScene = renderLaneOpaque twoOverlaidBondsScene := by
  decide

theorem overlaid_bonds_have_different_graphs :
    oneOverlaidBondScene.graph ≠ twoOverlaidBondsScene.graph := by
  decide

end ChemGraph
