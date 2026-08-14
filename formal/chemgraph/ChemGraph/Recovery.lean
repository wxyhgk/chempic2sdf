import ChemGraph.Counterexamples

namespace ChemGraph

/-!
这是一个刻意很小的可恢复语法。它不声称覆盖任意分子，只用于验证：
只要保留“哪些墨迹属于同一个 Bond group”，交叉和中心 junction 就能区分。
-/
inductive CrossKind where
  | crossing
  | junction
deriving DecidableEq, Repr, BEq

def CrossKind.scene : CrossKind -> Scene
  | .crossing => crossingScene
  | .junction => junctionScene

def decodeCrossKind (observation : BondGroupingOracle) : Option CrossKind :=
  if observation Ink.northSouth && observation Ink.eastWest then
    some .crossing
  else if observation (Ink.singleton .north) &&
      observation (Ink.singleton .east) &&
      observation (Ink.singleton .south) &&
      observation (Ink.singleton .west) then
    some .junction
  else
    none

theorem owned_observation_roundtrip (kind : CrossKind) :
    decodeCrossKind (renderBondGroupingOracle kind.scene) = some kind := by
  cases kind <;> decide

theorem owned_observation_distinguishes_crossing_and_junction :
    renderBondGroupingOracle crossingScene ≠ renderBondGroupingOracle junctionScene := by
  intro sameObservation
  have sameAtNorthSouth := congrFun sameObservation Ink.northSouth
  have crossingHasPair :
      renderBondGroupingOracle crossingScene Ink.northSouth = true := by
    decide
  have junctionLacksPair :
      renderBondGroupingOracle junctionScene Ink.northSouth = false := by
    decide
  rw [crossingHasPair, junctionLacksPair] at sameAtNorthSouth
  contradiction

/-! 这个定理只批准 Oracle 中间表示到二场景语法的查表恢复。 -/
theorem owned_oracle_roundtrip :
    forall kind, decodeCrossKind (renderBondGroupingOracle kind.scene) = some kind :=
  owned_observation_roundtrip

/-!
更重要的反定理：当前不透明 `Ink` 已经发生信息碰撞，所以不存在一个
只读 `Ink` 的确定性函数，能对所有场景给出正确的 Oracle ownership。
-/
theorem owned_observation_not_factor_through_opaque :
    Not (Exists fun observe : Ink -> BondGroupingOracle =>
      forall scene, observe (renderOpaque scene) = renderBondGroupingOracle scene) :=
  collision_forbids_factorization renderOpaque renderBondGroupingOracle
    crossingScene junctionScene
    crossing_and_junction_same_opaque_render
    owned_observation_distinguishes_crossing_and_junction

def sameTraceScene (order : Nat) : Scene := {
  atomCount := 2
  bonds := [{
    endpoints := mkEdge 0 1
    trace := Ink.singleton .north
    order := order
  }]
}

theorem grouping_oracle_forgets_order :
    renderBondGroupingOracle (sameTraceScene 1) =
      renderBondGroupingOracle (sameTraceScene 2) := by
  rfl

theorem same_grouping_different_order_graph :
    (sameTraceScene 1).graph ≠ (sameTraceScene 2).graph := by
  decide

theorem grouping_oracle_not_sufficient_for_general_graph :
    Not (Exists fun decode : BondGroupingOracle -> ChemicalGraph =>
      forall scene, decode (renderBondGroupingOracle scene) = scene.graph) :=
  collision_forbids_exact_decoder renderBondGroupingOracle Scene.graph
    (sameTraceScene 1) (sameTraceScene 2)
    grouping_oracle_forgets_order
    same_grouping_different_order_graph

def oneTraceInstanceScene : Scene := {
  atomCount := 2
  bonds := [{
    endpoints := mkEdge 0 1
    trace := Ink.singleton .north
  }]
}

def twoTraceInstancesScene : Scene := {
  atomCount := 4
  bonds := [
    {
      endpoints := mkEdge 0 1
      trace := Ink.singleton .north
    },
    {
      endpoints := mkEdge 2 3
      trace := Ink.singleton .north
    }
  ]
}

theorem grouping_oracle_forgets_multiplicity :
    renderBondGroupingOracle oneTraceInstanceScene =
      renderBondGroupingOracle twoTraceInstancesScene := by
  funext trace
  simp [renderBondGroupingOracle, oneTraceInstanceScene, twoTraceInstancesScene]

theorem same_grouping_different_bond_count :
    oneTraceInstanceScene.graph ≠ twoTraceInstancesScene.graph := by
  decide

theorem one_trace_instance_scene_valid : oneTraceInstanceScene.Valid := by
  rfl

theorem two_trace_instances_scene_valid : twoTraceInstancesScene.Valid := by
  rfl

theorem grouping_oracle_not_sufficient_even_on_valid_scenes :
    Not (IdentifiableOn Scene.Valid renderBondGroupingOracle Scene.graph) := by
  intro identifiable
  apply same_grouping_different_bond_count
  exact identifiable oneTraceInstanceScene twoTraceInstancesScene
    one_trace_instance_scene_valid two_trace_instances_scene_valid
    grouping_oracle_forgets_multiplicity

end ChemGraph
