import ChemGraph.Identifiability
import ChemGraph.Scene
import Lean.Elab.Tactic.Omega

namespace ChemGraph

/-!
用自然数距离表达有限精度观测。这里不要求连续空间，只需要对称性和
三角不等式；production tensor 可以在后续用量化后的 L1/Hamming 距离实例化。
-/
structure NatDistance (Observation : Type) where
  dist : Observation -> Observation -> Nat
  refl : forall observation, dist observation observation = 0
  symm : forall left right, dist left right = dist right left
  triangle : forall left middle right,
    dist left right <= dist left middle + dist middle right

def RobustlySeparated {Scene Observation Graph : Type}
    (metric : NatDistance Observation)
    (valid : Scene -> Prop) (equivalent : Graph -> Graph -> Prop)
    (render : Scene -> Observation) (meaning : Scene -> Graph)
    (noiseRadius : Nat) : Prop :=
  forall s t, valid s -> valid t -> Not (equivalent (meaning s) (meaning t)) ->
    noiseRadius + noiseRadius < metric.dist (render s) (render t)

/-!
编码论式唯一解码定理：若所有异图渲染之间的距离严格大于两倍噪声半径，
则一个观测的噪声球中不可能同时出现两个不等价化学图。
-/
theorem robust_ball_has_unique_graph
    {Scene Observation Graph : Type}
    (metric : NatDistance Observation)
    (valid : Scene -> Prop) (equivalent : Graph -> Graph -> Prop)
    (render : Scene -> Observation) (meaning : Scene -> Graph)
    (noiseRadius : Nat)
    (separated : RobustlySeparated metric valid equivalent render meaning noiseRadius)
    (observation : Observation) (s t : Scene)
    (validS : valid s) (validT : valid t)
    (nearS : metric.dist observation (render s) <= noiseRadius)
    (nearT : metric.dist observation (render t) <= noiseRadius) :
    equivalent (meaning s) (meaning t) := by
  apply Classical.byContradiction
  intro notEquivalent
  have far := separated s t validS validT notEquivalent
  have reverseNearS : metric.dist (render s) observation <= noiseRadius := by
    rw [metric.symm]
    exact nearS
  have pathBound := metric.triangle (render s) observation (render t)
  have radiusBound :
      metric.dist (render s) observation + metric.dist observation (render t) <=
        noiseRadius + noiseRadius :=
    Nat.add_le_add reverseNearS nearT
  have tooClose : metric.dist (render s) (render t) <= noiseRadius + noiseRadius :=
    Nat.le_trans pathBound radiusBound
  exact (Nat.not_lt_of_ge tooClose) far

def bitDistance (left right : Bool) : Nat :=
  if left == right then 0 else 1

theorem bitDistance_refl (value : Bool) : bitDistance value value = 0 := by
  cases value <;> decide

theorem bitDistance_symm (left right : Bool) :
    bitDistance left right = bitDistance right left := by
  cases left <;> cases right <;> decide

theorem bitDistance_triangle (left middle right : Bool) :
    bitDistance left right <= bitDistance left middle + bitDistance middle right := by
  cases left <;> cases middle <;> cases right <;> decide

def Ink.hamming (left right : Ink) : Nat :=
  bitDistance left.north right.north +
    bitDistance left.east right.east +
    bitDistance left.south right.south +
    bitDistance left.west right.west

theorem Ink.hamming_refl (ink : Ink) : ink.hamming ink = 0 := by
  simp [Ink.hamming, bitDistance_refl]

theorem Ink.hamming_symm (left right : Ink) :
    left.hamming right = right.hamming left := by
  simp only [Ink.hamming]
  rw [bitDistance_symm left.north right.north]
  rw [bitDistance_symm left.east right.east]
  rw [bitDistance_symm left.south right.south]
  rw [bitDistance_symm left.west right.west]

theorem Ink.hamming_triangle (left middle right : Ink) :
    left.hamming right <= left.hamming middle + middle.hamming right := by
  have north := bitDistance_triangle left.north middle.north right.north
  have east := bitDistance_triangle left.east middle.east right.east
  have south := bitDistance_triangle left.south middle.south right.south
  have west := bitDistance_triangle left.west middle.west right.west
  simp only [Ink.hamming]
  omega

def inkHammingDistance : NatDistance Ink := {
  dist := Ink.hamming
  refl := Ink.hamming_refl
  symm := Ink.hamming_symm
  triangle := Ink.hamming_triangle
}

end ChemGraph
