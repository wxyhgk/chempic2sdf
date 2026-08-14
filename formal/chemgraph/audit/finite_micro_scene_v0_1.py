"""Finite 3x3 micro-scene grammar and graph canonicalization.

The scene AST is the only source of graph truth. Raster inference never reads
this structure, an SDF, or a Render JSON document.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations, permutations, product
from math import comb
from typing import Iterable, Iterator


SCHEMA_VERSION = "finite-micro-scene-0.1"
CANVAS_SIZE = 64
GRID_COORDINATES = (12, 32, 52)
MAXIMUM_CORE_EDGES = 4


@dataclass(frozen=True, slots=True)
class Node:
    atom_id: int
    x: int
    y: int
    color: str = "IMPLICIT"


@dataclass(frozen=True, slots=True)
class PhysicalMember:
    begin: tuple[int, int]
    end: tuple[int, int]


@dataclass(frozen=True, slots=True)
class Bond:
    begin_atom: int
    end_atom: int
    order: int = 1
    display_type: str = "SINGLE"
    members: tuple[PhysicalMember, ...] = ()

    @property
    def endpoints(self) -> tuple[int, int]:
        return tuple(sorted((self.begin_atom, self.end_atom)))


@dataclass(frozen=True, slots=True)
class Scene:
    scene_id: str
    family: str
    nodes: tuple[Node, ...]
    bonds: tuple[Bond, ...]


def grid_nodes() -> tuple[Node, ...]:
    return tuple(
        Node(atom_id=row * 3 + column, x=x, y=y)
        for row, y in enumerate(GRID_COORDINATES)
        for column, x in enumerate(GRID_COORDINATES)
    )


GRID_NODES = grid_nodes()
GRID_NODE_BY_ID = {node.atom_id: node for node in GRID_NODES}
GRID_EDGES = tuple(combinations(range(len(GRID_NODES)), 2))
EXPECTED_CORE_SCENES = sum(
    comb(len(GRID_EDGES), count) for count in range(MAXIMUM_CORE_EDGES + 1)
)


def validate_scene(scene: Scene) -> None:
    node_ids = [node.atom_id for node in scene.nodes]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError(f"Scene {scene.scene_id} contains duplicate atom IDs")
    known = set(node_ids)
    endpoint_pairs: set[tuple[int, int]] = set()
    for bond in scene.bonds:
        if bond.begin_atom == bond.end_atom:
            raise ValueError(f"Scene {scene.scene_id} contains a self bond")
        if bond.begin_atom not in known or bond.end_atom not in known:
            raise ValueError(f"Scene {scene.scene_id} references an unknown atom")
        if bond.order not in {1, 2, 3}:
            raise ValueError(f"Scene {scene.scene_id} has an invalid bond order")
        if bond.endpoints in endpoint_pairs:
            raise ValueError(f"Scene {scene.scene_id} repeats a semantic endpoint pair")
        endpoint_pairs.add(bond.endpoints)
        if not bond.members:
            raise ValueError(f"Scene {scene.scene_id} has a bond with no physical member")


def _core_scene(edge_indices: tuple[int, ...], ordinal: int) -> Scene:
    selected = tuple(GRID_EDGES[index] for index in edge_indices)
    active_ids = sorted({atom_id for edge in selected for atom_id in edge})
    nodes = tuple(GRID_NODE_BY_ID[atom_id] for atom_id in active_ids)
    bonds = []
    for begin_atom, end_atom in selected:
        begin = GRID_NODE_BY_ID[begin_atom]
        end = GRID_NODE_BY_ID[end_atom]
        bonds.append(
            Bond(
                begin_atom=begin_atom,
                end_atom=end_atom,
                members=(PhysicalMember((begin.x, begin.y), (end.x, end.y)),),
            )
        )
    scene = Scene(
        scene_id=f"core-{ordinal:05d}",
        family=f"core_edges_{len(selected)}",
        nodes=nodes,
        bonds=tuple(bonds),
    )
    validate_scene(scene)
    return scene


def iter_core_scenes(limit: int | None = None) -> Iterator[Scene]:
    if limit is not None and limit < 0:
        raise ValueError("Core scene limit must be nonnegative")
    ordinal = 0
    for edge_count in range(MAXIMUM_CORE_EDGES + 1):
        for edge_indices in combinations(range(len(GRID_EDGES)), edge_count):
            if limit is not None and ordinal >= limit:
                return
            yield _core_scene(edge_indices, ordinal)
            ordinal += 1


def _node(atom_id: int, x: int, y: int) -> Node:
    return Node(atom_id=atom_id, x=x, y=y)


def crossing_junction_fixtures() -> tuple[Scene, Scene]:
    north = _node(0, 32, 8)
    east = _node(1, 56, 32)
    south = _node(2, 32, 56)
    west = _node(3, 8, 32)
    center = _node(4, 32, 32)
    crossing = Scene(
        scene_id="fixture-crossing",
        family="fixture_crossing_junction",
        nodes=(north, east, south, west),
        bonds=(
            Bond(0, 2, members=(PhysicalMember((32, 8), (32, 56)),)),
            Bond(1, 3, members=(PhysicalMember((56, 32), (8, 32)),)),
        ),
    )
    junction = Scene(
        scene_id="fixture-junction",
        family="fixture_crossing_junction",
        nodes=(north, east, south, west, center),
        bonds=(
            Bond(0, 4, members=(PhysicalMember((32, 8), (32, 32)),)),
            Bond(1, 4, members=(PhysicalMember((56, 32), (32, 32)),)),
            Bond(2, 4, members=(PhysicalMember((32, 56), (32, 32)),)),
            Bond(3, 4, members=(PhysicalMember((8, 32), (32, 32)),)),
        ),
    )
    validate_scene(crossing)
    validate_scene(junction)
    return crossing, junction


def double_parallel_fixtures() -> tuple[Scene, Scene]:
    upper = PhysicalMember((8, 28), (56, 28))
    lower = PhysicalMember((8, 36), (56, 36))
    double = Scene(
        scene_id="fixture-double-bond",
        family="fixture_double_parallel",
        nodes=(_node(0, 8, 32), _node(1, 56, 32)),
        bonds=(Bond(0, 1, order=2, display_type="DOUBLE", members=(upper, lower)),),
    )
    parallel = Scene(
        scene_id="fixture-parallel-singles",
        family="fixture_double_parallel",
        nodes=(
            _node(0, 8, 28),
            _node(1, 56, 28),
            _node(2, 8, 36),
            _node(3, 56, 36),
        ),
        bonds=(
            Bond(0, 1, members=(upper,)),
            Bond(2, 3, members=(lower,)),
        ),
    )
    validate_scene(double)
    validate_scene(parallel)
    return double, parallel


def fixture_scenes() -> tuple[Scene, ...]:
    return (*crossing_junction_fixtures(), *double_parallel_fixtures())


def _components(scene: Scene) -> list[tuple[int, ...]]:
    adjacency = {node.atom_id: set() for node in scene.nodes}
    for bond in scene.bonds:
        adjacency[bond.begin_atom].add(bond.end_atom)
        adjacency[bond.end_atom].add(bond.begin_atom)
    components = []
    unseen = set(adjacency)
    while unseen:
        start = min(unseen)
        stack = [start]
        current = set()
        while stack:
            atom_id = stack.pop()
            if atom_id in current:
                continue
            current.add(atom_id)
            stack.extend(adjacency[atom_id] - current)
        unseen -= current
        components.append(tuple(sorted(current)))
    return components


def _canonical_component(scene: Scene, atom_ids: tuple[int, ...], *, erase_text: bool) -> str:
    node_by_id = {node.atom_id: node for node in scene.nodes}
    edge_labels = {
        bond.endpoints: f"{bond.order}:{bond.display_type}"
        for bond in scene.bonds
        if bond.begin_atom in atom_ids and bond.end_atom in atom_ids
    }

    def color(atom_id: int) -> str:
        value = node_by_id[atom_id].color
        if erase_text and value.startswith("TEXT:"):
            return "TEXT"
        return value

    color_groups: dict[str, list[int]] = {}
    for atom_id in atom_ids:
        color_groups.setdefault(color(atom_id), []).append(atom_id)
    ordered_groups = sorted(color_groups.items())
    candidates = []
    for group_permutations in product(
        *(permutations(group) for _, group in ordered_groups)
    ):
        order = tuple(atom_id for group in group_permutations for atom_id in group)
        node_part = ",".join(color(atom_id) for atom_id in order)
        edge_part = []
        for left_index, left in enumerate(order):
            for right in order[left_index + 1 :]:
                edge_part.append(edge_labels.get(tuple(sorted((left, right))), "."))
        candidates.append(f"N[{node_part}]E[{','.join(edge_part)}]")
    return min(candidates)


def canonical_graph_signature(scene: Scene, *, erase_text: bool = False) -> str:
    validate_scene(scene)
    component_signatures = sorted(
        _canonical_component(scene, component, erase_text=erase_text)
        for component in _components(scene)
    )
    return "G{" + "|".join(component_signatures) + "}"


def relabel_scene(
    scene: Scene,
    atom_mapping: dict[int, int],
    *,
    reverse_bonds: bool = False,
) -> Scene:
    if set(atom_mapping) != {node.atom_id for node in scene.nodes}:
        raise ValueError("Atom relabeling must cover the complete node set")
    nodes = tuple(replace(node, atom_id=atom_mapping[node.atom_id]) for node in scene.nodes)
    bonds = tuple(
        replace(
            bond,
            begin_atom=atom_mapping[bond.begin_atom],
            end_atom=atom_mapping[bond.end_atom],
        )
        for bond in scene.bonds
    )
    if reverse_bonds:
        bonds = tuple(reversed(bonds))
    output = replace(scene, scene_id=scene.scene_id + "-relabel", nodes=nodes, bonds=bonds)
    validate_scene(output)
    return output


def scene_record(scene: Scene) -> dict:
    """Return a canonical JSON-compatible record sufficient to rebuild a scene."""
    return {
        "scene_id": scene.scene_id,
        "family": scene.family,
        "nodes": [
            {
                "atom_id": node.atom_id,
                "color": node.color,
                "x": node.x,
                "y": node.y,
            }
            for node in sorted(scene.nodes, key=lambda item: item.atom_id)
        ],
        "bonds": [
            {
                "begin_atom": bond.begin_atom,
                "display_type": bond.display_type,
                "end_atom": bond.end_atom,
                "members": [
                    {"begin": list(member.begin), "end": list(member.end)}
                    for member in bond.members
                ],
                "order": bond.order,
            }
            for bond in scene.bonds
        ],
    }


def core_scene_count() -> int:
    return sum(1 for _ in iter_core_scenes())


__all__ = [
    "Bond",
    "CANVAS_SIZE",
    "EXPECTED_CORE_SCENES",
    "GRID_EDGES",
    "GRID_NODES",
    "Node",
    "PhysicalMember",
    "SCHEMA_VERSION",
    "Scene",
    "canonical_graph_signature",
    "core_scene_count",
    "crossing_junction_fixtures",
    "double_parallel_fixtures",
    "fixture_scenes",
    "iter_core_scenes",
    "relabel_scene",
    "scene_record",
    "validate_scene",
]
