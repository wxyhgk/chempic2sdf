import ChemGraph.Render
import ChemGraph.Identifiability

namespace ChemGraph

private def armBond (outer center : Nat) (arm : Arm) : SemanticBond := {
  endpoints := mkEdge outer center
  trace := Ink.singleton arm
}

/-! 四条键在中心原子终止：五个原子，四根键。 -/
def junctionScene : Scene := {
  atomCount := 5
  bonds := [
    armBond 0 4 .north,
    armBond 1 4 .east,
    armBond 2 4 .south,
    armBond 3 4 .west
  ]
}

/-! 两条键在内部交叉：四个原子，两根键，没有中心原子。 -/
def crossingScene : Scene := {
  atomCount := 4
  bonds := [
    { endpoints := mkEdge 0 2, trace := Ink.northSouth },
    { endpoints := mkEdge 1 3, trace := Ink.eastWest }
  ]
}

theorem crossing_and_junction_same_opaque_render :
    renderOpaque crossingScene = renderOpaque junctionScene := by
  decide

theorem crossing_and_junction_different_graph :
    crossingScene.graph ≠ junctionScene.graph := by
  decide

theorem crossing_and_junction_not_graph_equivalent :
    Not (GraphEquivalent crossingScene.graph junctionScene.graph) := by
  intro equivalent
  have differentAtomCount : crossingScene.graph.atomCount ≠ junctionScene.graph.atomCount := by
    decide
  exact differentAtomCount equivalent.left

theorem crossing_scene_valid : crossingScene.Valid := by
  rfl

theorem junction_scene_valid : junctionScene.Valid := by
  rfl

theorem opaque_render_not_identifiable :
    Not (Identifiable renderOpaque Scene.graph) := by
  intro identifiable
  apply crossing_and_junction_different_graph
  exact identifiable crossingScene junctionScene
    crossing_and_junction_same_opaque_render

theorem opaque_render_not_identifiable_on_valid_scenes :
    Not (IdentifiableOn Scene.Valid renderOpaque Scene.graph) := by
  intro identifiable
  apply crossing_and_junction_different_graph
  exact identifiable crossingScene junctionScene crossing_scene_valid junction_scene_valid
    crossing_and_junction_same_opaque_render

theorem opaque_render_not_identifiable_up_to_graph_equivalence :
    Not (IdentifiableOnUpTo Scene.Valid GraphEquivalent renderOpaque Scene.graph) := by
  intro identifiable
  apply crossing_and_junction_not_graph_equivalent
  exact identifiable crossingScene junctionScene crossing_scene_valid junction_scene_valid
    crossing_and_junction_same_opaque_render

theorem no_universal_opaque_decoder :
    Not (Exists fun decode : Ink -> ChemicalGraph =>
      forall scene, decode (renderOpaque scene) = scene.graph) :=
  collision_forbids_exact_decoder renderOpaque Scene.graph crossingScene junctionScene
    crossing_and_junction_same_opaque_render
    crossing_and_junction_different_graph

end ChemGraph
