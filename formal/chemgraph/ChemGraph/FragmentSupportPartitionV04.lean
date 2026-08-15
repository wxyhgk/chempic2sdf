import Std

namespace ChemGraph

namespace FragmentSupportPartitionV04

/-!
Fragment Support Partition v0.4 的有限集合内核。

`parentPixels` 是已经给定的有限、无重复 Pixel 列表；`bucket` 是已经给定的
total function，把每个 Pixel 映射到一个有限 Child index。Child fiber 只通过
filter 构造。本模块不定义或验证 projection、PNG、坐标、support extractor，
也不证明 bucket 对真实像素具有任何几何或语义正确性。
-/

/-- Canonical rational cut `numerator / denominator`.  Reduction and the
strictly internal `[0, 1]` range are checked by the executable boundary; this
kernel records the positive denominator needed by exact comparison. -/
structure ExactCut where
  numerator : Int
  denominator : Nat
  denominatorPositive : 0 < denominator

/-- `projectionNumerator / normSquared` is at or to the right of `cut`.
Positive denominators make exact cross multiplication order preserving. -/
def atOrRightOfCut
    (projectionNumerator : Int) (normSquared : Nat) (cut : ExactCut) : Bool :=
  decide (
    cut.numerator * (normSquared : Int) <=
      projectionNumerator * (cut.denominator : Int))

/-- Right-continuous half-open bucket.  Its value is the number of cuts at or
to the left of the projected pixel, so an exact cut tie enters the right child. -/
def exactRightBucket
    (cuts : List ExactCut) (projectionNumerator : Int) (normSquared : Nat) :
    Fin (cuts.length + 1) :=
  let selected := cuts.filter (atOrRightOfCut projectionNumerator normSquared)
  ⟨selected.length, Nat.lt_succ_of_le (List.length_filter_le _ _)⟩

theorem exactRightBucket_no_cut
    (projectionNumerator : Int) (normSquared : Nat) :
    (exactRightBucket [] projectionNumerator normSquared).val = 0 := by
  rfl

/-- Exact equality is assigned to the right child. -/
theorem exactRightBucket_single_cut_tie
    (cut : ExactCut) (projectionNumerator : Int) (normSquared : Nat)
    (tie : cut.numerator * (normSquared : Int) =
      projectionNumerator * (cut.denominator : Int)) :
    (exactRightBucket [cut] projectionNumerator normSquared).val = 1 := by
  have atOrRight : atOrRightOfCut projectionNumerator normSquared cut = true := by
    simp [atOrRightOfCut, tie]
  simp [exactRightBucket, atOrRight]

/-- A projection left of every cut belongs to the first child.  For internal
cuts this includes begin-cap pixels whose projection parameter is below zero. -/
theorem exactRightBucket_before_all
    (cuts : List ExactCut) (projectionNumerator : Int) (normSquared : Nat)
    (before : forall cut, cut ∈ cuts ->
      atOrRightOfCut projectionNumerator normSquared cut = false) :
    (exactRightBucket cuts projectionNumerator normSquared).val = 0 := by
  change (cuts.filter
    (atOrRightOfCut projectionNumerator normSquared)).length = 0
  rw [List.length_eq_zero_iff]
  apply List.filter_eq_nil_iff.mpr
  intro cut member
  simp [before cut member]

/-- A projection at or right of every cut belongs to the final child.  For
internal cuts this includes end-cap pixels with parameter above one. -/
theorem exactRightBucket_after_all
    (cuts : List ExactCut) (projectionNumerator : Int) (normSquared : Nat)
    (after : forall cut, cut ∈ cuts ->
      atOrRightOfCut projectionNumerator normSquared cut = true) :
    (exactRightBucket cuts projectionNumerator normSquared).val = cuts.length := by
  change (cuts.filter
    (atOrRightOfCut projectionNumerator normSquared)).length = cuts.length
  rw [List.filter_eq_self.mpr after]

structure FragmentSupportPartition (Pixel : Type) (childCount : Nat) where
  parentPixels : List Pixel
  parentNodup : parentPixels.Nodup
  bucket : Pixel -> Fin childCount

/-- Production-shaped specialization.  `projectionNumerator pixel` is the
integer dot product `dot(pixelCenter - A, B - A)` and `normSquared` is
`dot(B - A, B - A)`. -/
def exactProjectionPartition
    {Pixel : Type}
    (parentPixels : List Pixel)
    (parentNodup : parentPixels.Nodup)
    (cuts : List ExactCut)
    (projectionNumerator : Pixel -> Int)
    (normSquared : Nat) :
    FragmentSupportPartition Pixel (cuts.length + 1) where
  parentPixels := parentPixels
  parentNodup := parentNodup
  bucket := fun pixel =>
    exactRightBucket cuts (projectionNumerator pixel) normSquared

def childFiber
    {Pixel : Type} {childCount : Nat}
    (partition : FragmentSupportPartition Pixel childCount)
    (child : Fin childCount) : List Pixel :=
  partition.parentPixels.filter fun pixel =>
    decide (partition.bucket pixel = child)

/-- filter fiber 的精确 membership。 -/
theorem mem_childFiber_iff
    {Pixel : Type} {childCount : Nat}
    (partition : FragmentSupportPartition Pixel childCount)
    (child : Fin childCount) (pixel : Pixel) :
    pixel ∈ childFiber partition child <->
      pixel ∈ partition.parentPixels /\ partition.bucket pixel = child := by
  simp [childFiber]

theorem childFiber_nodup
    {Pixel : Type} {childCount : Nat}
    (partition : FragmentSupportPartition Pixel childCount)
    (child : Fin childCount) :
    (childFiber partition child).Nodup :=
  List.filter_sublist.nodup partition.parentNodup

/-- 避免依赖额外库的 `ExistsUnique`；witness 后的字段表达唯一性。 -/
def HasUniqueChild
    {Pixel : Type} {childCount : Nat}
    (partition : FragmentSupportPartition Pixel childCount)
    (pixel : Pixel) : Prop :=
  Exists fun child : Fin childCount =>
    pixel ∈ childFiber partition child /\
      forall other : Fin childCount,
        pixel ∈ childFiber partition other -> other = child

/-- 每个 parent Pixel 恰好属于 bucket 指定的一个 Child fiber。 -/
theorem parent_pixel_has_unique_child
    {Pixel : Type} {childCount : Nat}
    (partition : FragmentSupportPartition Pixel childCount)
    {pixel : Pixel}
    (parentMember : pixel ∈ partition.parentPixels) :
    HasUniqueChild partition pixel := by
  refine ⟨partition.bucket pixel, ?_, ?_⟩
  · exact (mem_childFiber_iff partition (partition.bucket pixel) pixel).mpr
      ⟨parentMember, rfl⟩
  · intro child childMember
    exact (mem_childFiber_iff partition child pixel).mp childMember |>.2.symm

/-- 两个不同 Child 的 filter fiber 没有共同 Pixel。 -/
theorem distinct_child_fibers_disjoint
    {Pixel : Type} {childCount : Nat}
    (partition : FragmentSupportPartition Pixel childCount)
    {left right : Fin childCount}
    (different : left ≠ right) (pixel : Pixel)
    (leftMember : pixel ∈ childFiber partition left)
    (rightMember : pixel ∈ childFiber partition right) : False := by
  apply different
  exact ((mem_childFiber_iff partition left pixel).mp leftMember).2.symm.trans
    ((mem_childFiber_iff partition right pixel).mp rightMember).2

/-- 按有限 Child index 顺序拼接全部 filter fiber；这里只把它当集合并集。 -/
def allChildPixels
    {Pixel : Type} {childCount : Nat}
    (partition : FragmentSupportPartition Pixel childCount) : List Pixel :=
  (List.finRange childCount).flatMap (childFiber partition)

/-- 全部 Child fiber 的集合并集与 parent support 精确相等。 -/
theorem mem_allChildPixels_iff
    {Pixel : Type} {childCount : Nat}
    (partition : FragmentSupportPartition Pixel childCount)
    (pixel : Pixel) :
    pixel ∈ allChildPixels partition <->
      pixel ∈ partition.parentPixels := by
  constructor
  · intro member
    rcases List.mem_flatMap.mp member with ⟨child, _, fiberMember⟩
    exact ((mem_childFiber_iff partition child pixel).mp fiberMember).1
  · intro parentMember
    apply List.mem_flatMap.mpr
    exact ⟨partition.bucket pixel, List.mem_finRange _,
      (mem_childFiber_iff partition (partition.bucket pixel) pixel).mpr
        ⟨parentMember, rfl⟩⟩

theorem all_child_fibers_cover_parent
    {Pixel : Type} {childCount : Nat}
    (partition : FragmentSupportPartition Pixel childCount)
    {pixel : Pixel}
    (parentMember : pixel ∈ partition.parentPixels) :
    pixel ∈ allChildPixels partition :=
  (mem_allChildPixels_iff partition pixel).mpr parentMember

theorem all_child_fibers_contained_in_parent
    {Pixel : Type} {childCount : Nat}
    (partition : FragmentSupportPartition Pixel childCount)
    {pixel : Pixel}
    (unionMember : pixel ∈ allChildPixels partition) :
    pixel ∈ partition.parentPixels :=
  (mem_allChildPixels_iff partition pixel).mp unionMember

/-- The exact integer-projection specialization inherits unique ownership. -/
theorem exact_projection_parent_pixel_has_unique_child
    {Pixel : Type}
    (parentPixels : List Pixel)
    (parentNodup : parentPixels.Nodup)
    (cuts : List ExactCut)
    (projectionNumerator : Pixel -> Int)
    (normSquared : Nat)
    {pixel : Pixel}
    (parentMember : pixel ∈ parentPixels) :
    HasUniqueChild
      (exactProjectionPartition parentPixels parentNodup cuts
        projectionNumerator normSquared)
      pixel :=
  parent_pixel_has_unique_child _ parentMember

/-- Exact integer-projection child fibers cover precisely the parent support.
Cross-carrier pixel sharing is intentionally outside this per-parent theorem. -/
theorem exact_projection_cover_iff
    {Pixel : Type}
    (parentPixels : List Pixel)
    (parentNodup : parentPixels.Nodup)
    (cuts : List ExactCut)
    (projectionNumerator : Pixel -> Int)
    (normSquared : Nat)
    (pixel : Pixel) :
    pixel ∈ allChildPixels
      (exactProjectionPartition parentPixels parentNodup cuts
        projectionNumerator normSquared) <->
      pixel ∈ parentPixels :=
  mem_allChildPixels_iff _ pixel

#print axioms mem_childFiber_iff
#print axioms childFiber_nodup
#print axioms parent_pixel_has_unique_child
#print axioms distinct_child_fibers_disjoint
#print axioms mem_allChildPixels_iff
#print axioms all_child_fibers_cover_parent
#print axioms all_child_fibers_contained_in_parent
#print axioms exactRightBucket_no_cut
#print axioms exactRightBucket_single_cut_tie
#print axioms exactRightBucket_before_all
#print axioms exactRightBucket_after_all
#print axioms exact_projection_parent_pixel_has_unique_child
#print axioms exact_projection_cover_iff

end FragmentSupportPartitionV04

end ChemGraph
