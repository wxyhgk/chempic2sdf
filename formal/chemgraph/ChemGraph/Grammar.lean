namespace ChemGraph

/-!
Grammar v0.1 的精确整数几何。整数坐标是有理坐标的有限子集；这里不引入
浮点 tolerance、像素半径或近似角度。
-/
structure GridPoint where
  x : Int
  y : Int
deriving DecidableEq, Repr, BEq

structure GridSegment where
  start : GridPoint
  finish : GridPoint
deriving DecidableEq, Repr, BEq

def orient (a b c : GridPoint) : Int :=
  (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)

def directionDot (center left right : GridPoint) : Int :=
  (left.x - center.x) * (right.x - center.x) +
    (left.y - center.y) * (right.y - center.y)

def GridSegment.Nonzero (segment : GridSegment) : Prop :=
  segment.start ≠ segment.finish

def GridSegment.StrictInterior (segment : GridSegment) (point : GridPoint) : Prop :=
  orient segment.start segment.finish point = 0 ∧
    ((point.x - segment.start.x) * (point.x - segment.finish.x) +
      (point.y - segment.start.y) * (point.y - segment.finish.y) < 0)

def openIntervalsOverlap (a b c d : Int) : Prop :=
  max (min a b) (min c d) < min (max a b) (max c d)

def GridSegment.Collinear (left right : GridSegment) : Prop :=
  orient left.start left.finish right.start = 0 ∧
    orient left.start left.finish right.finish = 0

def GridSegment.PositiveOverlap (left right : GridSegment) : Prop :=
  left.Collinear right ∧
    if left.start.x ≠ left.finish.x then
      openIntervalsOverlap
        left.start.x left.finish.x right.start.x right.finish.x
    else
      openIntervalsOverlap
        left.start.y left.finish.y right.start.y right.finish.y

def GridSegment.ProperCross (left right : GridSegment) : Prop :=
  orient left.start left.finish right.start *
      orient left.start left.finish right.finish < 0 ∧
    orient right.start right.finish left.start *
      orient right.start right.finish left.finish < 0

def StrictStraightDegreeTwo (center left right : GridPoint) : Prop :=
  orient center left right = 0 ∧ directionDot center left right < 0

structure GrammarSignals where
  zeroLengthMember : Bool := false
  duplicateAtomCoordinate : Bool := false
  passesNonincidentAtom : Bool := false
  positiveCollinearOverlap : Bool := false
  implicitStraightDegreeTwo : Bool := false
  nonsemanticTouch : Bool := false
  multiwayInternalCrossing : Bool := false
  crossingMixedWithSemanticObject : Bool := false
  ambiguousMultiplicity : Bool := false
  noCarrierAfterAmbiguityMask : Bool := false
  properCross : Bool := false
deriving DecidableEq, Repr, BEq

def GrammarSignals.hasInvalidGeometry (signals : GrammarSignals) : Bool :=
  signals.zeroLengthMember ||
    signals.duplicateAtomCoordinate ||
    signals.passesNonincidentAtom ||
    signals.positiveCollinearOverlap ||
    signals.implicitStraightDegreeTwo ||
    signals.nonsemanticTouch ||
    signals.multiwayInternalCrossing ||
    signals.crossingMixedWithSemanticObject ||
    signals.ambiguousMultiplicity ||
    signals.noCarrierAfterAmbiguityMask

inductive GrammarLayer where
  | simpleLineAngleValid
  | crossingAllowed
  | adversarialInvalid
deriving DecidableEq, Repr, BEq

/-!
互斥层只描述单个场景的分类；可辨识性必须在累积允许域上陈述。
`crossingDomain` 因而同时包含 simple 与 crossing 两层。
-/
def GrammarLayer.InSimpleDomain : GrammarLayer → Prop
  | .simpleLineAngleValid => True
  | _ => False

def GrammarLayer.InCrossingDomain : GrammarLayer → Prop
  | .simpleLineAngleValid => True
  | .crossingAllowed => True
  | .adversarialInvalid => False

def GrammarLayer.InAllDomain (_ : GrammarLayer) : Prop := True

theorem simple_domain_subset_crossing_domain
    {layer : GrammarLayer}
    (member : layer.InSimpleDomain) :
    layer.InCrossingDomain := by
  cases layer <;> simp_all [GrammarLayer.InSimpleDomain, GrammarLayer.InCrossingDomain]

theorem crossing_domain_subset_all_domain
    {layer : GrammarLayer}
    (_member : layer.InCrossingDomain) :
    layer.InAllDomain := by
  trivial

theorem grammar_layers_pairwise_distinct :
    GrammarLayer.simpleLineAngleValid ≠ GrammarLayer.crossingAllowed ∧
      GrammarLayer.simpleLineAngleValid ≠ GrammarLayer.adversarialInvalid ∧
      GrammarLayer.crossingAllowed ≠ GrammarLayer.adversarialInvalid := by
  decide

def classifyGrammar (signals : GrammarSignals) : GrammarLayer :=
  if signals.hasInvalidGeometry then
    .adversarialInvalid
  else if signals.properCross then
    .crossingAllowed
  else
    .simpleLineAngleValid

theorem classifyGrammar_invalid_priority
    (signals : GrammarSignals)
    (invalid : signals.hasInvalidGeometry = true) :
    classifyGrammar signals = .adversarialInvalid := by
  simp [classifyGrammar, invalid]

theorem classifyGrammar_crossing
    (signals : GrammarSignals)
    (regular : signals.hasInvalidGeometry = false)
    (crossing : signals.properCross = true) :
    classifyGrammar signals = .crossingAllowed := by
  simp [classifyGrammar, regular, crossing]

theorem classifyGrammar_simple
    (signals : GrammarSignals)
    (regular : signals.hasInvalidGeometry = false)
    (noCrossing : signals.properCross = false) :
    classifyGrammar signals = .simpleLineAngleValid := by
  simp [classifyGrammar, regular, noCrossing]

theorem classifyGrammar_exhaustive (signals : GrammarSignals) :
    classifyGrammar signals = .simpleLineAngleValid ∨
      classifyGrammar signals = .crossingAllowed ∨
      classifyGrammar signals = .adversarialInvalid := by
  cases invalid : signals.hasInvalidGeometry <;>
    cases crossing : signals.properCross <;>
    simp [classifyGrammar, invalid, crossing]

theorem classifyGrammar_unique
    (signals : GrammarSignals)
    {left right : GrammarLayer}
    (leftClassification : classifyGrammar signals = left)
    (rightClassification : classifyGrammar signals = right) :
    left = right := by
  rw [← leftClassification, ← rightClassification]

def north : GridPoint := { x := 0, y := -1 }
def east : GridPoint := { x := 1, y := 0 }
def south : GridPoint := { x := 0, y := 1 }
def west : GridPoint := { x := -1, y := 0 }
def center : GridPoint := { x := 0, y := 0 }

def verticalCrossingSegment : GridSegment := { start := north, finish := south }
def horizontalCrossingSegment : GridSegment := { start := west, finish := east }
def northHalfSegment : GridSegment := { start := north, finish := center }
def southHalfSegment : GridSegment := { start := center, finish := south }

theorem crossing_fixture_is_proper :
    verticalCrossingSegment.ProperCross horizontalCrossingSegment := by
  unfold GridSegment.ProperCross verticalCrossingSegment horizontalCrossingSegment
  unfold north south west east orient
  decide

theorem crossing_center_is_strict_interior :
    verticalCrossingSegment.StrictInterior center ∧
      horizontalCrossingSegment.StrictInterior center := by
  unfold GridSegment.StrictInterior verticalCrossingSegment horizontalCrossingSegment
  unfold north south west east center orient
  decide

theorem straight_degree_two_fixture : StrictStraightDegreeTwo center north south := by
  unfold StrictStraightDegreeTwo center north south orient directionDot
  decide

theorem opposite_halves_do_not_positive_overlap :
    Not (northHalfSegment.PositiveOverlap southHalfSegment) := by
  unfold GridSegment.PositiveOverlap GridSegment.Collinear
  unfold northHalfSegment southHalfSegment north south center orient
  unfold openIntervalsOverlap
  decide

def longVerticalSegment : GridSegment := {
  start := { x := 0, y := -2 }
  finish := { x := 0, y := 2 }
}

def overlappingVerticalSegment : GridSegment := {
  start := { x := 0, y := -1 }
  finish := { x := 0, y := 2 }
}

theorem overlap_fixture_has_positive_length :
    longVerticalSegment.PositiveOverlap overlappingVerticalSegment := by
  unfold GridSegment.PositiveOverlap GridSegment.Collinear
  unfold longVerticalSegment overlappingVerticalSegment orient openIntervalsOverlap
  decide

end ChemGraph
