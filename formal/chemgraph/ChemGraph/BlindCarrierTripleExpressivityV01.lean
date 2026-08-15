import Std

namespace ChemGraph

namespace BlindCarrierTripleExpressivityV01

/-!
Blind Carrier v0.1 的三线 expressivity 合同反例。

若三条平行 lane 的相邻中心线间距都至少为 `2c + A`，而 terminal 到每条
中心线的允许半径为 `c + A`，则在 `c > 0` 时不存在一个 terminal 能同时
贴近三条 lane。这里 `c = ceil(width / 2)`，证明只说明两个几何合同不相容。
-/

/-- 自然数上的对称绝对差。 -/
def natAbsDiff (left right : Nat) : Nat :=
  (left - right) + (right - left)

theorem natAbsDiff_comm (left right : Nat) :
    natAbsDiff left right = natAbsDiff right left := by
  unfold natAbsDiff
  exact Nat.add_comm _ _

theorem right_le_left_add_of_natAbsDiff_le
    {left right radius : Nat}
    (bounded : natAbsDiff left right <= radius) :
    right <= left + radius := by
  unfold natAbsDiff at bounded
  have rightSubLeftBounded : right - left <= radius :=
    Nat.le_trans (Nat.le_add_left (right - left) (left - right)) bounded
  by_cases leftLeRight : left <= right
  · have added := Nat.add_le_add_right rightSubLeftBounded left
    rw [Nat.sub_add_cancel leftLeRight] at added
    rw [Nat.add_comm] at added
    exact added
  · have rightLeLeft : right <= left :=
      Nat.le_of_lt (Nat.lt_of_not_ge leftLeRight)
    exact Nat.le_trans rightLeLeft (Nat.le_add_right left radius)

/-- 自然数宽度的一半向上取整。 -/
def ceilHalf (width : Nat) : Nat :=
  (width + 1) / 2

/-- Blind Carrier v0.1 固定的一条 lane 的 terminal 接近半径。 -/
def terminalRadius (width antiAliasMargin : Nat) : Nat :=
  ceilHalf width + antiAliasMargin

/-- Blind Carrier v0.1 固定的平行 lane 最小中心线间距。 -/
def parallelMinimumSeparation (width antiAliasMargin : Nat) : Nat :=
  2 * ceilHalf width + antiAliasMargin

theorem no_shared_terminal_for_three_separated_parallel_lanes
    (width antiAliasMargin x0 x1 x2 : Nat)
    (halfWidthPositive : 0 < ceilHalf width)
    (ordered01 : x0 < x1)
    (ordered12 : x1 < x2)
    (separated01 :
      parallelMinimumSeparation width antiAliasMargin <= x1 - x0)
    (separated12 :
      parallelMinimumSeparation width antiAliasMargin <= x2 - x1) :
    Not (Exists fun terminal =>
      natAbsDiff terminal x0 <= terminalRadius width antiAliasMargin /\
      natAbsDiff terminal x1 <= terminalRadius width antiAliasMargin /\
      natAbsDiff terminal x2 <= terminalRadius width antiAliasMargin) := by
  rintro ⟨terminal, near0, _near1, near2⟩
  let c := ceilHalf width
  let radius := c + antiAliasMargin
  let minimumSeparation := 2 * c + antiAliasMargin
  have separation01 : x0 + minimumSeparation <= x1 := by
    have converted :=
      (Nat.le_sub_iff_add_le (Nat.le_of_lt ordered01)).mp separated01
    rw [Nat.add_comm] at converted
    exact converted
  have separation12 : x1 + minimumSeparation <= x2 := by
    have converted :=
      (Nat.le_sub_iff_add_le (Nat.le_of_lt ordered12)).mp separated12
    rw [Nat.add_comm] at converted
    exact converted
  have outerLower : x0 + minimumSeparation + minimumSeparation <= x2 :=
    Nat.le_trans
      (Nat.add_le_add_right separation01 minimumSeparation)
      separation12
  have terminalUpper : terminal <= x0 + radius := by
    apply right_le_left_add_of_natAbsDiff_le
    rw [natAbsDiff_comm]
    exact near0
  have outerToTerminal : x2 <= terminal + radius :=
    right_le_left_add_of_natAbsDiff_le near2
  have outerUpper : x2 <= x0 + radius + radius :=
    Nat.le_trans outerToTerminal (Nat.add_le_add_right terminalUpper radius)
  have combined :
      x0 + (minimumSeparation + minimumSeparation) <=
        x0 + (radius + radius) := by
    exact Nat.le_trans
      (Nat.le_of_eq (Nat.add_assoc x0 minimumSeparation minimumSeparation).symm)
      (Nat.le_trans outerLower
        (Nat.le_trans outerUpper
          (Nat.le_of_eq (Nat.add_assoc x0 radius radius))))
  have separationBound :
      minimumSeparation + minimumSeparation <= radius + radius :=
    Nat.le_of_add_le_add_left combined
  have halfBelowDouble : c < 2 * c := by
    rw [Nat.two_mul]
    exact Nat.lt_add_of_pos_right halfWidthPositive
  have radiusBelowSeparation : radius < minimumSeparation :=
    Nat.add_lt_add_right halfBelowDouble antiAliasMargin
  have doubledStrict :
      radius + radius < minimumSeparation + minimumSeparation :=
    Nat.add_lt_add radiusBelowSeparation radiusBelowSeparation
  exact Nat.not_le_of_gt doubledStrict separationBound

#print axioms natAbsDiff_comm
#print axioms right_le_left_add_of_natAbsDiff_le
#print axioms no_shared_terminal_for_three_separated_parallel_lanes

end BlindCarrierTripleExpressivityV01

end ChemGraph
