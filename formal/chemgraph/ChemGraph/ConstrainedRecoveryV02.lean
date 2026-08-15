import Std

namespace ChemGraph

namespace ConstrainedRecoveryV02

/-!
The v0.2 boundary makes support, provenance, coverage, and calibration explicit.
It is a conditional specification and does not assert that a raster extractor
can produce these certificates on real chemical images.
-/

structure SupportPixel where
  supportId : Nat
  x : Nat
  y : Nat
deriving DecidableEq, Repr

structure SegmentGeometry where
  segmentId : Nat
  beginX : Nat
  beginY : Nat
  endX : Nat
  endY : Nat
  nonzero : Not (beginX = endX /\ beginY = endY)

inductive PrimitiveOwner where
  | bondLane (bondId : Nat)
  | glyphStroke (glyphId : Nat)
  | marker (markerId : Nat)
deriving DecidableEq, Repr

/-- A carrier cannot exist without named pixel and segment support. -/
structure CarrierEvidence where
  carrierId : Nat
  primitiveId : Nat
  supportPixels : List SupportPixel
  supportPixelsNonempty : Not (supportPixels = [])
  supportPixelIdsNodup : (supportPixels.map (fun pixel => pixel.supportId)).Nodup
  supportSegments : List SegmentGeometry
  supportSegmentsNonempty : Not (supportSegments = [])
  supportSegmentIdsNodup :
    (supportSegments.map (fun segment => segment.segmentId)).Nodup
  supportScore : Nat
  minimumSupportScore : Nat
  supportPasses : minimumSupportScore <= supportScore
  owner : PrimitiveOwner

def PrimitiveOwnershipExact (carriers : List CarrierEvidence) : Prop :=
  forall left, left ∈ carriers -> forall right, right ∈ carriers ->
    left.primitiveId = right.primitiveId ->
    left.carrierId = right.carrierId /\ left.owner = right.owner

theorem exact_ownership_forbids_two_owners
    {carriers : List CarrierEvidence}
    (ownership : PrimitiveOwnershipExact carriers)
    {left right : CarrierEvidence}
    (leftMember : left ∈ carriers) (rightMember : right ∈ carriers)
    (samePrimitive : left.primitiveId = right.primitiveId) :
    left.owner = right.owner := by
  exact (ownership left leftMember right rightMember samePrimitive).2

def SupportDisjoint (left right : CarrierEvidence) : Prop :=
  (forall pixel, pixel ∈ left.supportPixels -> pixel ∈ right.supportPixels -> False) /\
  (forall segment, segment ∈ left.supportSegments ->
    segment ∈ right.supportSegments -> False)

/-- A lane carries its own endpoints, geometry, support, primitive, and owner. -/
structure BondLane (Vertex : Type) where
  primitiveId : Nat
  geometry : SegmentGeometry
  support : CarrierEvidence
  supportPrimitive : support.primitiveId = primitiveId
  bondId : Nat
  supportOwner : support.owner = .bondLane bondId
  beginNode : Vertex
  endNode : Vertex
  loopFree : Not (beginNode = endNode)

def BondLane.endpoint {Vertex : Type}
    (lane : BondLane Vertex) (atEnd : Bool) : Vertex :=
  if atEnd then lane.endNode else lane.beginNode

theorem lane_endpoint_is_lane_specific {Vertex : Type}
    (lane : BondLane Vertex) :
    lane.endpoint false = lane.beginNode /\ lane.endpoint true = lane.endNode := by
  simp [BondLane.endpoint]

/-- One semantic bond owns exactly `order` mutually exclusive supported lanes. -/
structure BondBundle (Vertex : Type) where
  bondId : Nat
  beginNode : Vertex
  endNode : Vertex
  loopFree : Not (beginNode = endNode)
  order : Nat
  orderLower : 1 <= order
  orderUpper : order <= 3
  lanes : List (BondLane Vertex)
  exactLaneCount : lanes.length = order
  sharedEndpoints : forall lane, lane ∈ lanes ->
    lane.beginNode = beginNode /\ lane.endNode = endNode
  sharedOwner : forall lane, lane ∈ lanes -> lane.bondId = bondId
  primitiveIdsDistinct :
    (lanes.map (fun lane => lane.primitiveId)).Nodup
  supportExclusive : forall left, left ∈ lanes -> forall right, right ∈ lanes ->
    Not (left.primitiveId = right.primitiveId) -> SupportDisjoint left.support right.support

theorem order_two_has_exactly_two_lanes {Vertex : Type}
    (bundle : BondBundle Vertex) (order : bundle.order = 2) :
    bundle.lanes.length = 2 := by
  exact bundle.exactLaneCount.trans order

theorem order_three_has_exactly_three_lanes {Vertex : Type}
    (bundle : BondBundle Vertex) (order : bundle.order = 3) :
    bundle.lanes.length = 3 := by
  exact bundle.exactLaneCount.trans order

theorem member_lane_has_bundle_endpoints {Vertex : Type}
    (bundle : BondBundle Vertex) (lane : BondLane Vertex)
    (member : lane ∈ bundle.lanes) :
    lane.endpoint false = bundle.beginNode /\
      lane.endpoint true = bundle.endNode := by
  rcases bundle.sharedEndpoints lane member with ⟨beginEqual, endEqual⟩
  simp [BondLane.endpoint, beginEqual, endEqual]

/-- Edge output retains all lane-level evidence needed for an audit. -/
structure ProvenancedEdge (Vertex : Type) where
  beginNode : Vertex
  endNode : Vertex
  order : Nat
  primitiveIds : List Nat
  carrierIds : List Nat
  laneGeometry : List SegmentGeometry
  primitiveCount : primitiveIds.length = order
  carrierCount : carrierIds.length = order
  geometryCount : laneGeometry.length = order
  primitiveIdsNodup : primitiveIds.Nodup

def BondBundle.toProvenancedEdge {Vertex : Type}
    (bundle : BondBundle Vertex) : ProvenancedEdge Vertex where
  beginNode := bundle.beginNode
  endNode := bundle.endNode
  order := bundle.order
  primitiveIds := bundle.lanes.map (fun lane => lane.primitiveId)
  carrierIds := bundle.lanes.map (fun lane => lane.support.carrierId)
  laneGeometry := bundle.lanes.map (fun lane => lane.geometry)
  primitiveCount := by simpa using bundle.exactLaneCount
  carrierCount := by simpa using bundle.exactLaneCount
  geometryCount := by simpa using bundle.exactLaneCount
  primitiveIdsNodup := bundle.primitiveIdsDistinct

theorem edge_provenance_preserves_primitive_ids {Vertex : Type}
    (bundle : BondBundle Vertex) :
    bundle.toProvenancedEdge.primitiveIds =
      bundle.lanes.map (fun lane => lane.primitiveId) := by
  rfl

structure CandidateCoverageCertificate
    {Hypothesis : Type}
    (candidate supported : Hypothesis -> Prop) where
  evidenceId : String
  evidenceIdNonempty : Not (evidenceId = "")
  enumeratorId : String
  enumeratorIdNonempty : Not (enumeratorId = "")
  candidateWitness : exists hypothesis, candidate hypothesis
  complete : forall hypothesis, supported hypothesis -> candidate hypothesis

structure ScoreErrorBound {Hypothesis : Type}
    (clean observed : Hypothesis -> Nat) (epsilon : Nat) : Prop where
  observedUpper : forall hypothesis,
    observed hypothesis <= clean hypothesis + epsilon
  cleanUpper : forall hypothesis,
    clean hypothesis <= observed hypothesis + epsilon

structure CalibrationEvidence {Hypothesis : Type}
    (clean observed : Hypothesis -> Nat) where
  calibrationId : String
  calibrationIdNonempty : Not (calibrationId = "")
  sampleCount : Nat
  sampleCountPositive : 0 < sampleCount
  epsilon : Nat
  errorBound : ScoreErrorBound clean observed epsilon

def OneGraphClass {Hypothesis : Type}
    (candidate : Hypothesis -> Prop)
    (equivalent : Hypothesis -> Hypothesis -> Prop) : Prop :=
  exists representative, candidate representative /\
    forall other, candidate other -> equivalent representative other

def SeparatedWinner {Hypothesis : Type}
    (candidate : Hypothesis -> Prop)
    (equivalent : Hypothesis -> Hypothesis -> Prop)
    (clean : Hypothesis -> Nat) (epsilon : Nat)
    (winner : Hypothesis) : Prop :=
  candidate winner /\ forall alternative, candidate alternative ->
    Not (equivalent winner alternative) ->
    clean winner + 2 * epsilon < clean alternative

inductive GraphAuthority {Hypothesis : Type}
    (candidate supported : Hypothesis -> Prop)
    (equivalent : Hypothesis -> Hypothesis -> Prop)
    (clean observed : Hypothesis -> Nat) : Type where
  | singleton
      (coverage : CandidateCoverageCertificate candidate supported)
      (oneClass : OneGraphClass candidate equivalent)
  | intervalWinner
      (coverage : CandidateCoverageCertificate candidate supported)
      (calibration : CalibrationEvidence clean observed)
      (winner : Hypothesis)
      (separated : SeparatedWinner candidate equivalent clean calibration.epsilon winner)

structure CertifiedGraphOutput
    (Graph Hypothesis : Type)
    (candidate supported : Hypothesis -> Prop)
    (equivalent : Hypothesis -> Hypothesis -> Prop)
    (meaning : Hypothesis -> Graph)
    (clean observed : Hypothesis -> Nat) where
  graph : Graph
  authority : GraphAuthority candidate supported equivalent clean observed
  represented : exists hypothesis, candidate hypothesis /\ meaning hypothesis = graph

inductive DecodeDecision
    (Graph Hypothesis : Type)
    (candidate supported : Hypothesis -> Prop)
    (equivalent : Hypothesis -> Hypothesis -> Prop)
    (meaning : Hypothesis -> Graph)
    (clean observed : Hypothesis -> Nat) where
  | graph (output : CertifiedGraphOutput Graph Hypothesis candidate supported
      equivalent meaning clean observed)
  | ambiguous
  | reject

def DecodeDecision.IsGraph
    {Graph Hypothesis : Type}
    {candidate supported : Hypothesis -> Prop}
    {equivalent : Hypothesis -> Hypothesis -> Prop}
    {meaning : Hypothesis -> Graph}
    {clean observed : Hypothesis -> Nat} :
    DecodeDecision Graph Hypothesis candidate supported equivalent meaning clean observed -> Prop
  | .graph _ => True
  | .ambiguous => False
  | .reject => False

/-- Without a coverage certificate, a nonempty proposal can only be ambiguous. -/
def decideWithoutCoverage
    {Graph Hypothesis : Type}
    {candidate supported : Hypothesis -> Prop}
    {equivalent : Hypothesis -> Hypothesis -> Prop}
    {meaning : Hypothesis -> Graph}
    {clean observed : Hypothesis -> Nat}
    (hasCandidate : Bool) :
    DecodeDecision Graph Hypothesis candidate supported equivalent meaning clean observed :=
  if hasCandidate then .ambiguous else .reject

theorem decideWithoutCoverage_never_graph
    {Graph Hypothesis : Type}
    {candidate supported : Hypothesis -> Prop}
    {equivalent : Hypothesis -> Hypothesis -> Prop}
    {meaning : Hypothesis -> Graph}
    {clean observed : Hypothesis -> Nat}
    (hasCandidate : Bool) :
    Not ((decideWithoutCoverage (Graph := Graph) (Hypothesis := Hypothesis)
      (candidate := candidate) (supported := supported) (equivalent := equivalent)
      (meaning := meaning) (clean := clean) (observed := observed) hasCandidate).IsGraph) := by
  cases hasCandidate <;> simp [decideWithoutCoverage, DecodeDecision.IsGraph]

theorem calibrated_margin_preserves_winner {Hypothesis : Type}
    {candidate : Hypothesis -> Prop}
    {equivalent : Hypothesis -> Hypothesis -> Prop}
    {clean observed : Hypothesis -> Nat}
    (calibration : CalibrationEvidence clean observed)
    {winner alternative : Hypothesis}
    (separated : SeparatedWinner candidate equivalent clean calibration.epsilon winner)
    (alternativeCandidate : candidate alternative)
    (inequivalent : Not (equivalent winner alternative)) :
    observed winner < observed alternative := by
  have margin := separated.2 alternative alternativeCandidate inequivalent
  have winnerUpper := calibration.errorBound.observedUpper winner
  have alternativeLower := calibration.errorBound.cleanUpper alternative
  omega

end ConstrainedRecoveryV02

end ChemGraph
