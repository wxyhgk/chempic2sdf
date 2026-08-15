import ChemGraph.BlindCarrierTerminalRegionV02

namespace ChemGraph

namespace BlindCarrierTerminalComponentsV02

/-!
Blind Carrier v0.2 adapter 的 terminal component 合同。

若每条 endpoint-disk 相交边都保留 terminal 身份，且同一 terminal 内任意两个
端点都可达，则图可达关系恰好等于“属于同一 terminal”。因此不同 terminal
之间不可达，每个 terminal fiber 恰好是一个连通分量；此处只证明分区合同，
不涉及图像或训练结论。
-/

open BlindCarrierTerminalRegionV02

def EdgePreservesTerminal
    {Vertex Terminal : Type}
    (edge : Vertex -> Vertex -> Prop)
    (terminalOf : Vertex -> Terminal) : Prop :=
  forall {left right}, edge left right -> terminalOf left = terminalOf right

def TerminalInternallyConnected
    {Vertex Terminal : Type}
    (edge : Vertex -> Vertex -> Prop)
    (terminalOf : Vertex -> Terminal) : Prop :=
  forall {left right}, terminalOf left = terminalOf right -> Reachable edge left right

theorem reachable_preserves_terminal
    {Vertex Terminal : Type}
    {edge : Vertex -> Vertex -> Prop}
    {terminalOf : Vertex -> Terminal}
    (edgePreserves : EdgePreservesTerminal edge terminalOf)
    {left right : Vertex}
    (path : Reachable edge left right) :
    terminalOf left = terminalOf right := by
  induction path with
  | refl vertex => rfl
  | step edgeHead _pathTail inductionHypothesis =>
      exact Eq.trans (edgePreserves edgeHead) inductionHypothesis

theorem different_terminals_not_reachable
    {Vertex Terminal : Type}
    {edge : Vertex -> Vertex -> Prop}
    {terminalOf : Vertex -> Terminal}
    (edgePreserves : EdgePreservesTerminal edge terminalOf)
    {left right : Vertex}
    (different : Not (terminalOf left = terminalOf right)) :
    Not (Reachable edge left right) := by
  intro path
  exact different (reachable_preserves_terminal
    (edge := edge) (terminalOf := terminalOf) edgePreserves path)

theorem same_terminal_reachable
    {Vertex Terminal : Type}
    {edge : Vertex -> Vertex -> Prop}
    {terminalOf : Vertex -> Terminal}
    (internalConnected : TerminalInternallyConnected edge terminalOf)
    {left right : Vertex}
    (same : terminalOf left = terminalOf right) :
    Reachable edge left right :=
  internalConnected same

/-- Terminal fiber 与 endpoint-disk intersection graph 的连通分量完全一致。 -/
theorem reachable_iff_same_terminal
    {Vertex Terminal : Type}
    {edge : Vertex -> Vertex -> Prop}
    {terminalOf : Vertex -> Terminal}
    (edgePreserves : EdgePreservesTerminal edge terminalOf)
    (internalConnected : TerminalInternallyConnected edge terminalOf)
    (left right : Vertex) :
    Reachable edge left right <-> terminalOf left = terminalOf right := by
  constructor
  · exact reachable_preserves_terminal
      (edge := edge) (terminalOf := terminalOf) edgePreserves
  · exact internalConnected

/-- 任意另一份精确 component 标号都诱导相同分区，因此 terminal partition 唯一。 -/
theorem terminal_partition_unique_up_to_class_equivalence
    {Vertex Terminal OtherClass : Type}
    {edge : Vertex -> Vertex -> Prop}
    {terminalOf : Vertex -> Terminal}
    (edgePreserves : EdgePreservesTerminal edge terminalOf)
    (internalConnected : TerminalInternallyConnected edge terminalOf)
    (otherClassOf : Vertex -> OtherClass)
    (otherExact : forall left right,
      Reachable edge left right <-> otherClassOf left = otherClassOf right)
    (left right : Vertex) :
    terminalOf left = terminalOf right <->
      otherClassOf left = otherClassOf right := by
  constructor
  · intro sameTerminal
    have path : Reachable edge left right := internalConnected sameTerminal
    exact (otherExact left right).mp path
  · intro sameOtherClass
    have path : Reachable edge left right :=
      (otherExact left right).mpr sameOtherClass
    exact reachable_preserves_terminal
      (edge := edge) (terminalOf := terminalOf) edgePreserves path

#print axioms reachable_preserves_terminal
#print axioms different_terminals_not_reachable
#print axioms same_terminal_reachable
#print axioms reachable_iff_same_terminal
#print axioms terminal_partition_unique_up_to_class_equivalence

end BlindCarrierTerminalComponentsV02

end ChemGraph
