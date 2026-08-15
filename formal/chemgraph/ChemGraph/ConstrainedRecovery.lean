import Std

namespace ChemGraph

namespace ConstrainedRecovery

/-!
This module specifies the deterministic boundary of constrained chemical graph
recovery.  A learned component may propose primitives and scores, but it cannot
declare a graph correct unless the observable candidates satisfy these gates.
-/

/-- Every visible primitive has exactly one semantic owner. -/
def ExactOwnership {Primitive Owner : Type}
    (owns : Owner -> Primitive -> Prop) : Prop :=
  forall primitive, exists owner, owns owner primitive /\
    forall other, owns other primitive -> other = owner

theorem exactOwnership_owner_unique {Primitive Owner : Type}
    {owns : Owner -> Primitive -> Prop}
    (ownership : ExactOwnership owns)
    {primitive : Primitive} {left right : Owner}
    (leftOwns : owns left primitive) (rightOwns : owns right primitive) :
    left = right := by
  rcases ownership primitive with ⟨owner, _, unique⟩
  exact (unique left leftOwns).trans (unique right rightOwns).symm

theorem exactOwnership_forbids_bond_and_glyph_overlap
    {Primitive Owner : Type}
    {owns : Owner -> Primitive -> Prop}
    (ownership : ExactOwnership owns)
    {primitive : Primitive} {bondOwner glyphOwner : Owner}
    (different : Not (bondOwner = glyphOwner))
    (bondOwns : owns bondOwner primitive)
    (glyphOwns : owns glyphOwner primitive) : False := by
  exact different (exactOwnership_owner_unique ownership bondOwns glyphOwns)

/--
A multi-line bond is one semantic bond object.  Its physical lanes do not own
independent graph endpoints.
-/
structure BondBundle (Vertex Lane : Type) where
  beginNode : Vertex
  endNode : Vertex
  loopFree : Not (beginNode = endNode)
  order : Nat
  orderLower : 1 <= order
  orderUpper : order <= 3
  lanes : List Lane
  laneCount : lanes.length = order
  lanesNodup : lanes.Nodup

def BondBundle.endpoint {Vertex Lane : Type}
    (bundle : BondBundle Vertex Lane) (_lane : Lane) (atEnd : Bool) : Vertex :=
  if atEnd then bundle.endNode else bundle.beginNode

theorem bundle_lanes_share_endpoints {Vertex Lane : Type}
    (bundle : BondBundle Vertex Lane) (left right : Lane) (atEnd : Bool) :
    bundle.endpoint left atEnd = bundle.endpoint right atEnd := by
  rfl

theorem bundle_has_only_two_semantic_endpoint_roles {Vertex Lane : Type}
    (bundle : BondBundle Vertex Lane) (lane : Lane) :
    bundle.endpoint lane false = bundle.beginNode /\
      bundle.endpoint lane true = bundle.endNode := by
  simp [BondBundle.endpoint]

/-- Every accepted graph edge must be backed by an observable carrier. -/
def EdgeSound {Vertex : Type}
    (supported edge : Vertex -> Vertex -> Prop) : Prop :=
  forall left right, edge left right -> supported left right

theorem unsupported_pair_cannot_be_edge {Vertex : Type}
    {supported edge : Vertex -> Vertex -> Prop}
    (sound : EdgeSound supported edge) {left right : Vertex}
    (unsupported : Not (supported left right)) : Not (edge left right) := by
  intro accepted
  exact unsupported (sound left right accepted)

/-- A two-sided deterministic error bound between clean and observed scores. -/
structure ScoreErrorBound {Hypothesis : Type}
    (clean observed : Hypothesis -> Nat) (epsilon : Nat) : Prop where
  observedUpper : forall hypothesis,
    observed hypothesis <= clean hypothesis + epsilon
  cleanUpper : forall hypothesis,
    clean hypothesis <= observed hypothesis + epsilon

/-- Every feasible inequivalent alternative is separated from the truth. -/
def SeparatedWinner {Hypothesis : Type}
    (feasible : Hypothesis -> Prop)
    (equivalent : Hypothesis -> Hypothesis -> Prop)
    (clean : Hypothesis -> Nat) (epsilon : Nat)
    (truth : Hypothesis) : Prop :=
  feasible truth /\
    forall alternative, feasible alternative ->
      Not (equivalent truth alternative) ->
      clean truth + 2 * epsilon < clean alternative

theorem robust_margin_preserves_winner {Hypothesis : Type}
    {feasible : Hypothesis -> Prop}
    {equivalent : Hypothesis -> Hypothesis -> Prop}
    {clean observed : Hypothesis -> Nat} {epsilon : Nat}
    (error : ScoreErrorBound clean observed epsilon)
    {truth alternative : Hypothesis}
    (separated : SeparatedWinner feasible equivalent clean epsilon truth)
    (alternativeFeasible : feasible alternative)
    (inequivalent : Not (equivalent truth alternative)) :
    observed truth < observed alternative := by
  have margin := separated.2 alternative alternativeFeasible inequivalent
  have truthUpper := error.observedUpper truth
  have alternativeLower := error.cleanUpper alternative
  omega

/-- Rendering-identical hypotheses receive the same purely visual score. -/
def RenderScore {Hypothesis Observation : Type}
    (render : Hypothesis -> Observation)
    (distance : Observation -> Observation -> Nat)
    (observed : Observation) (hypothesis : Hypothesis) : Nat :=
  distance observed (render hypothesis)

theorem same_render_same_visual_score {Hypothesis Observation : Type}
    {render : Hypothesis -> Observation}
    {distance : Observation -> Observation -> Nat}
    {observed : Observation} {left right : Hypothesis}
    (same : render left = render right) :
    RenderScore render distance observed left =
      RenderScore render distance observed right := by
  simp [RenderScore, same]

theorem equal_visual_score_has_no_strict_margin
    {Hypothesis : Type} {score : Hypothesis -> Nat}
    {left right : Hypothesis} (equalScore : score left = score right) :
    Not (score left < score right) := by
  omega

inductive PathNode where
  | a | b | c
deriving DecidableEq, Repr

def pathCarrier : PathNode -> PathNode -> Prop
  | .a, .b | .b, .a | .b, .c | .c, .b => True
  | _, _ => False

example : Not (pathCarrier .a .c) := by simp [pathCarrier]

theorem sound_path_cannot_add_shortcut
    (edge : PathNode -> PathNode -> Prop)
    (sound : EdgeSound pathCarrier edge) : Not (edge .a .c) := by
  exact unsupported_pair_cannot_be_edge sound (by simp [pathCarrier])

inductive FanNode where
  | center | a | b | c
deriving DecidableEq, Repr

def fanCarrier : FanNode -> FanNode -> Prop
  | .center, .a | .a, .center
  | .center, .b | .b, .center
  | .center, .c | .c, .center => True
  | _, _ => False

theorem sound_fan_cannot_join_outer_nodes
    (edge : FanNode -> FanNode -> Prop)
    (sound : EdgeSound fanCarrier edge) : Not (edge .a .c) := by
  exact unsupported_pair_cannot_be_edge sound (by simp [fanCarrier])

end ConstrainedRecovery

end ChemGraph
