import Std

namespace ChemGraph

namespace FragmentRootConstructorV04

/-!
Fragment Root Constructor v0.4 的最小有限列表内核。

本模块的输入是 root 中已经物化并按规范顺序排列的 Cut 列表。构造器只在每个
Carrier 的 start/cut/finish delimiter 列表上取相邻 pair，形成 FragmentCell。
它不定义 PNG、像素坐标、intersection extractor、support、primitive kind、
ownership 或化学语义，也不声明 root Cut 列表对真实图像完备。
-/

inductive Delimiter (Cut : Type) where
  | start
  | cut (value : Cut)
  | finish

/-- FragmentCell 只表达一个 Carrier 上两个相邻 delimiter 之间的有限区间。 -/
structure FragmentCell (Carrier Cut : Type) where
  carrier : Carrier
  left : Delimiter Cut
  right : Delimiter Cut

/-- `orderedCutsFor` 的列表顺序就是冻结 canonical order；本模块不重算该顺序。 -/
structure FragmentRoot (Carrier Cut : Type) where
  carrierUniverse : List Carrier
  carrierNodup : carrierUniverse.Nodup
  orderedCutsFor : Carrier -> List Cut
  orderedCutsNodup : forall carrier, (orderedCutsFor carrier).Nodup

def orderedDelimiters {Cut : Type} (cuts : List Cut) : List (Delimiter Cut) :=
  .start :: cuts.map Delimiter.cut ++ [.finish]

def adjacentPairs {Alpha : Type} : List Alpha -> List (Alpha × Alpha)
  | [] => []
  | [_] => []
  | left :: right :: rest =>
      (left, right) :: adjacentPairs (right :: rest)
termination_by values => values.length

def fragmentCells
    {Carrier Cut : Type}
    (carrier : Carrier) (cuts : List Cut) :
    List (FragmentCell Carrier Cut) :=
  (adjacentPairs (orderedDelimiters cuts)).map fun pair =>
    { carrier := carrier, left := pair.1, right := pair.2 }

def rootFragmentCells
    {Carrier Cut : Type}
    (root : FragmentRoot Carrier Cut) (carrier : Carrier) :
    List (FragmentCell Carrier Cut) :=
  fragmentCells carrier (root.orderedCutsFor carrier)

theorem adjacentPairs_cons_length
    {Alpha : Type} (head : Alpha) (tail : List Alpha) :
    (adjacentPairs (head :: tail)).length = tail.length := by
  induction tail generalizing head with
  | nil => simp [adjacentPairs]
  | cons next rest ih => simp [adjacentPairs, ih]

/-- `n` 个 ordered Cut 恰好产生 `n + 1` 个 FragmentCell。 -/
theorem fragmentCells_length
    {Carrier Cut : Type} (carrier : Carrier) (cuts : List Cut) :
    (fragmentCells carrier cuts).length = cuts.length + 1 := by
  simp [fragmentCells, orderedDelimiters, adjacentPairs_cons_length]

theorem rootFragmentCells_length
    {Carrier Cut : Type}
    (root : FragmentRoot Carrier Cut)
    {carrier : Carrier}
    (_carrierMember : carrier ∈ root.carrierUniverse) :
    (rootFragmentCells root carrier).length =
      (root.orderedCutsFor carrier).length + 1 :=
  fragmentCells_length carrier (root.orderedCutsFor carrier)

/-- 独立的结构关系：`left` 与 `right` 在 delimiter 列表中直接相邻。 -/
inductive ListAdjacent
    {Alpha : Type} (left right : Alpha) : List Alpha -> Prop where
  | here (rest : List Alpha) :
      ListAdjacent left right (left :: right :: rest)
  | there (head : Alpha) {tail : List Alpha} :
      ListAdjacent left right tail ->
        ListAdjacent left right (head :: tail)

theorem adjacentPairs_sound
    {Alpha : Type} {values : List Alpha} {pair : Alpha × Alpha}
    (member : pair ∈ adjacentPairs values) :
    ListAdjacent pair.1 pair.2 values := by
  induction values generalizing pair with
  | nil => simp [adjacentPairs] at member
  | cons head tail ih =>
      cases tail with
      | nil => simp [adjacentPairs] at member
      | cons next rest =>
          simp only [adjacentPairs, List.mem_cons] at member
          rcases member with equal | member
          · subst pair
            exact ListAdjacent.here rest
          · exact ListAdjacent.there head (ih member)

/-- 独立 `ListAdjacent` 关系中的每个合法 pair 都由构造器枚举。 -/
theorem adjacentPairs_complete
    {Alpha : Type} {left right : Alpha} {values : List Alpha}
    (adjacent : ListAdjacent left right values) :
    (left, right) ∈ adjacentPairs values := by
  induction adjacent with
  | here rest =>
      simp [adjacentPairs]
  | there head adjacent ih =>
      cases adjacent with
      | here rest =>
          simp [adjacentPairs]
      | there next inner =>
          simp only [adjacentPairs, List.mem_cons]
          exact Or.inr ih

theorem adjacentPairs_exact
    {Alpha : Type} {left right : Alpha} {values : List Alpha} :
    (left, right) ∈ adjacentPairs values <->
      ListAdjacent left right values :=
  ⟨adjacentPairs_sound, adjacentPairs_complete⟩

/-- 构造出的每个 Cell 都来自输入 delimiter 列表中的相邻 pair。 -/
theorem fragmentCell_adjacent
    {Carrier Cut : Type} (carrier : Carrier) (cuts : List Cut)
    {cell : FragmentCell Carrier Cut}
    (member : cell ∈ fragmentCells carrier cuts) :
    cell.carrier = carrier /\
      ListAdjacent cell.left cell.right (orderedDelimiters cuts) := by
  simp only [fragmentCells, List.mem_map] at member
  rcases member with ⟨pair, pairMember, pairEq⟩
  rw [← pairEq]
  exact ⟨rfl, adjacentPairs_sound pairMember⟩

/-- 任意合法相邻 delimiter pair 都产生对应的 FragmentCell。 -/
theorem fragmentCell_complete
    {Carrier Cut : Type} (carrier : Carrier) (cuts : List Cut)
    {left right : Delimiter Cut}
    (adjacent : ListAdjacent left right (orderedDelimiters cuts)) :
    ({ carrier := carrier, left := left, right := right } :
      FragmentCell Carrier Cut) ∈ fragmentCells carrier cuts := by
  simp only [fragmentCells, List.mem_map]
  exact ⟨(left, right), adjacentPairs_complete adjacent, rfl⟩

theorem rootFragmentCell_adjacent
    {Carrier Cut : Type}
    (root : FragmentRoot Carrier Cut)
    {carrier : Carrier}
    (_carrierMember : carrier ∈ root.carrierUniverse)
    {cell : FragmentCell Carrier Cut}
    (member : cell ∈ rootFragmentCells root carrier) :
    cell.carrier = carrier /\
      ListAdjacent cell.left cell.right
        (orderedDelimiters (root.orderedCutsFor carrier)) :=
  fragmentCell_adjacent carrier (root.orderedCutsFor carrier) member

theorem rootFragmentCell_complete
    {Carrier Cut : Type}
    (root : FragmentRoot Carrier Cut)
    {carrier : Carrier}
    (_carrierMember : carrier ∈ root.carrierUniverse)
    {left right : Delimiter Cut}
    (adjacent : ListAdjacent left right
      (orderedDelimiters (root.orderedCutsFor carrier))) :
    ({ carrier := carrier, left := left, right := right } :
      FragmentCell Carrier Cut) ∈ rootFragmentCells root carrier :=
  fragmentCell_complete carrier (root.orderedCutsFor carrier) adjacent

theorem fragmentCells_nonempty
    {Carrier Cut : Type} (carrier : Carrier) (cuts : List Cut) :
    Not (fragmentCells carrier cuts = []) := by
  intro empty
  have lengths := congrArg List.length empty
  rw [fragmentCells_length] at lengths
  simp at lengths

/-- 每个 root Carrier 的 FragmentCell fiber 非空。 -/
theorem rootFragmentFiber_nonempty
    {Carrier Cut : Type}
    (root : FragmentRoot Carrier Cut)
    {carrier : Carrier}
    (_carrierMember : carrier ∈ root.carrierUniverse) :
    Exists fun cell => cell ∈ rootFragmentCells root carrier :=
  List.exists_mem_of_ne_nil _
    (fragmentCells_nonempty carrier (root.orderedCutsFor carrier))

/-- 零 Cut 时唯一 Cell 是 start 到 finish。 -/
theorem fragmentCells_zero_cut
    {Carrier Cut : Type} (carrier : Carrier) :
    fragmentCells carrier ([] : List Cut) =
      ([{ carrier := carrier,
          left := (Delimiter.start : Delimiter Cut),
          right := (Delimiter.finish : Delimiter Cut) }] :
        List (FragmentCell Carrier Cut)) := by
  simp [fragmentCells, orderedDelimiters, adjacentPairs]

theorem rootFragmentCells_zero_cut
    {Carrier Cut : Type}
    (root : FragmentRoot Carrier Cut)
    {carrier : Carrier}
    (_carrierMember : carrier ∈ root.carrierUniverse)
    (noCuts : root.orderedCutsFor carrier = []) :
    rootFragmentCells root carrier =
      ([{ carrier := carrier,
          left := (Delimiter.start : Delimiter Cut),
          right := (Delimiter.finish : Delimiter Cut) }] :
        List (FragmentCell Carrier Cut)) := by
  simp [rootFragmentCells, noCuts, fragmentCells_zero_cut]

#print axioms adjacentPairs_cons_length
#print axioms fragmentCells_length
#print axioms rootFragmentCells_length
#print axioms adjacentPairs_sound
#print axioms adjacentPairs_complete
#print axioms adjacentPairs_exact
#print axioms fragmentCell_adjacent
#print axioms fragmentCell_complete
#print axioms rootFragmentCell_adjacent
#print axioms rootFragmentCell_complete
#print axioms fragmentCells_nonempty
#print axioms rootFragmentFiber_nonempty
#print axioms fragmentCells_zero_cut
#print axioms rootFragmentCells_zero_cut

end FragmentRootConstructorV04

end ChemGraph
