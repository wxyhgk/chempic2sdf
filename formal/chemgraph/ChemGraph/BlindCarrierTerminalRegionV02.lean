import ChemGraph.BlindCarrierTripleExpressivityV01

namespace ChemGraph

namespace BlindCarrierTerminalRegionV02

/-!
Blind Carrier v0.2 的 terminal-region 非空性合同。

三条等宽平行 lane 不再要求共享一个 terminal 点。每条 lane endpoint 拥有半径
`R = c + A` 的闭圆盘；相邻中心线间距为 `S = 2c + A`。由于 `S <= 2R`，
相邻 endpoint disk 两两相交，因此交图包含路径 `0-1-2` 并连通。这里不声称
三个圆盘存在共同交点。
-/

open BlindCarrierTripleExpressivityV01

/-- 等宽 lane 的 endpoint disk 半径 `R = c + A`。 -/
def endpointDiskRadius (halfWidth antiAliasMargin : Nat) : Nat :=
  halfWidth + antiAliasMargin

/-- 相邻等宽平行 lane 的中心线间距 `S = 2c + A`。 -/
def adjacentLaneSeparation (halfWidth antiAliasMargin : Nat) : Nat :=
  2 * halfWidth + antiAliasMargin

/-- 一维闭 endpoint disk；它是二维平行切片上的闭区间。 -/
def InEndpointDisk (center radius point : Nat) : Prop :=
  natAbsDiff point center <= radius

/-- 两个 endpoint disk 至少有一个共同点。 -/
def EndpointDisksOverlap (leftCenter rightCenter radius : Nat) : Prop :=
  Exists fun point =>
    InEndpointDisk leftCenter radius point /\
    InEndpointDisk rightCenter radius point

theorem separation_le_twice_radius
    (halfWidth antiAliasMargin : Nat) :
    adjacentLaneSeparation halfWidth antiAliasMargin <=
      2 * endpointDiskRadius halfWidth antiAliasMargin := by
  unfold adjacentLaneSeparation endpointDiskRadius
  omega

/-- 相邻两条 lane 的圆盘以 `leftCenter + c` 为显式交点见证。 -/
theorem adjacent_endpoint_disks_overlap
    (leftCenter halfWidth antiAliasMargin : Nat) :
    EndpointDisksOverlap
      leftCenter
      (leftCenter + adjacentLaneSeparation halfWidth antiAliasMargin)
      (endpointDiskRadius halfWidth antiAliasMargin) := by
  refine ⟨leftCenter + halfWidth, ?_, ?_⟩
  · unfold InEndpointDisk natAbsDiff endpointDiskRadius
    omega
  · unfold InEndpointDisk natAbsDiff adjacentLaneSeparation endpointDiskRadius
    omega

theorem endpointDisksOverlap_symm
    {leftCenter rightCenter radius : Nat}
    (overlap : EndpointDisksOverlap leftCenter rightCenter radius) :
    EndpointDisksOverlap rightCenter leftCenter radius := by
  rcases overlap with ⟨point, inLeft, inRight⟩
  exact ⟨point, inRight, inLeft⟩

inductive LaneIndex where
  | lane0
  | lane1
  | lane2
deriving DecidableEq, Repr

def laneCenter
    (baseCenter halfWidth antiAliasMargin : Nat) : LaneIndex -> Nat
  | .lane0 => baseCenter
  | .lane1 =>
      baseCenter + adjacentLaneSeparation halfWidth antiAliasMargin
  | .lane2 =>
      baseCenter + adjacentLaneSeparation halfWidth antiAliasMargin +
        adjacentLaneSeparation halfWidth antiAliasMargin

/-- Endpoint-disk intersection graph 的无向边。 -/
def DiskIntersectionEdge
    (baseCenter halfWidth antiAliasMargin : Nat)
    (left right : LaneIndex) : Prop :=
  Not (left = right) /\
    EndpointDisksOverlap
      (laneCenter baseCenter halfWidth antiAliasMargin left)
      (laneCenter baseCenter halfWidth antiAliasMargin right)
      (endpointDiskRadius halfWidth antiAliasMargin)

inductive Reachable {Vertex : Type} (edge : Vertex -> Vertex -> Prop) :
    Vertex -> Vertex -> Prop where
  | refl (vertex : Vertex) : Reachable edge vertex vertex
  | step {beginNode middleNode endNode : Vertex} :
      edge beginNode middleNode ->
      Reachable edge middleNode endNode ->
      Reachable edge beginNode endNode

def Connected {Vertex : Type} (edge : Vertex -> Vertex -> Prop) : Prop :=
  forall left right, Reachable edge left right

theorem disk_edge_lane0_lane1
    (baseCenter halfWidth antiAliasMargin : Nat) :
    DiskIntersectionEdge baseCenter halfWidth antiAliasMargin
      .lane0 .lane1 := by
  constructor
  · decide
  · exact adjacent_endpoint_disks_overlap
      baseCenter halfWidth antiAliasMargin

theorem disk_edge_lane1_lane2
    (baseCenter halfWidth antiAliasMargin : Nat) :
    DiskIntersectionEdge baseCenter halfWidth antiAliasMargin
      .lane1 .lane2 := by
  constructor
  · decide
  · exact adjacent_endpoint_disks_overlap
      (baseCenter + adjacentLaneSeparation halfWidth antiAliasMargin)
      halfWidth antiAliasMargin

theorem diskIntersectionEdge_symm
    {baseCenter halfWidth antiAliasMargin : Nat}
    {left right : LaneIndex}
    (edge : DiskIntersectionEdge baseCenter halfWidth antiAliasMargin left right) :
    DiskIntersectionEdge baseCenter halfWidth antiAliasMargin right left := by
  exact ⟨Ne.symm edge.1, endpointDisksOverlap_symm edge.2⟩

theorem three_lane_endpoint_disk_intersection_graph_connected
    (baseCenter halfWidth antiAliasMargin : Nat) :
    Connected (DiskIntersectionEdge baseCenter halfWidth antiAliasMargin) := by
  intro left right
  have edge01 := disk_edge_lane0_lane1
    baseCenter halfWidth antiAliasMargin
  have edge12 := disk_edge_lane1_lane2
    baseCenter halfWidth antiAliasMargin
  have edge10 := diskIntersectionEdge_symm edge01
  have edge21 := diskIntersectionEdge_symm edge12
  cases left <;> cases right
  · exact Reachable.refl LaneIndex.lane0
  · exact Reachable.step edge01 (Reachable.refl LaneIndex.lane1)
  · exact Reachable.step edge01
      (Reachable.step edge12 (Reachable.refl LaneIndex.lane2))
  · exact Reachable.step edge10 (Reachable.refl LaneIndex.lane0)
  · exact Reachable.refl LaneIndex.lane1
  · exact Reachable.step edge12 (Reachable.refl LaneIndex.lane2)
  · exact Reachable.step edge21
      (Reachable.step edge10 (Reachable.refl LaneIndex.lane0))
  · exact Reachable.step edge21 (Reachable.refl LaneIndex.lane1)
  · exact Reachable.refl LaneIndex.lane2

#print axioms separation_le_twice_radius
#print axioms adjacent_endpoint_disks_overlap
#print axioms three_lane_endpoint_disk_intersection_graph_connected

end BlindCarrierTerminalRegionV02

end ChemGraph
