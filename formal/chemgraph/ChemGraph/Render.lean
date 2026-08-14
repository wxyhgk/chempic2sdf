import ChemGraph.Scene

namespace ChemGraph

/-!
`renderOpaque` 只保留墨迹并用布尔并集叠加。它故意忘记每一片墨迹
来自哪一个语义 Bond，用来刻画普通栅格图像的核心信息损失。
-/
def renderOpaque (scene : Scene) : Ink :=
  scene.bonds.foldl (fun observation bond => Ink.merge observation bond.trace) Ink.empty

/-!
`BondGroupingOracle` 不是最终模型输入，而是一个很弱的 Oracle 规格：
查询某个完整 trace 是否作为一个语义 Bond group 出现。这个定义不依赖
Bond 在列表里的编号或顺序；它不含键级、端口坐标或 Atom 身份。
-/
abbrev BondGroupingOracle := Ink -> Bool

def renderBondGroupingOracle (scene : Scene) : BondGroupingOracle :=
  fun trace => scene.bonds.any (fun bond => bond.trace == trace)

end ChemGraph
