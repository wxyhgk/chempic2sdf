"""Joint AtomObject/BondObject/incidence decoding with a sparse binary program."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix
import torch


@dataclass(frozen=True, slots=True)
class JointGraphMilpConfig:
    endpoint_topk: int = 2
    atom_presence_weight: float = 1.0
    atom_bias: float = 0.0
    bond_presence_weight: float = 1.0
    bond_bias: float = 0.0
    incidence_weight: float = 0.5
    time_limit_seconds: float = 2.0
    require_incident_atoms: bool = True

    def __post_init__(self) -> None:
        if self.endpoint_topk < 1:
            raise ValueError("endpoint_topk must be positive")
        if self.atom_presence_weight < 0.0:
            raise ValueError("atom_presence_weight must be non-negative")
        if self.bond_presence_weight < 0.0:
            raise ValueError("bond_presence_weight must be non-negative")
        if self.incidence_weight < 0.0:
            raise ValueError("incidence_weight must be non-negative")
        if self.time_limit_seconds <= 0.0:
            raise ValueError("time_limit_seconds must be positive")


def solve_joint_graph_milp(
    *,
    atom_presence_logits: torch.Tensor,
    atom_mask: torch.Tensor,
    bond_presence_logits: torch.Tensor,
    bond_mask: torch.Tensor,
    incidence_logits: torch.Tensor,
    atom_threshold: float,
    bond_threshold: float,
    config: JointGraphMilpConfig = JointGraphMilpConfig(),
) -> tuple[list[int], dict[int, tuple[int, int]], dict]:
    """Select a graph and endpoint assignments without geometry or chemistry rules."""
    started = time.perf_counter()
    atom_indices = torch.where(atom_mask)[0].tolist()
    bond_indices = torch.where(bond_mask)[0].tolist()
    if not atom_indices:
        raise ValueError("Joint graph MILP received no AtomObject candidates")
    atom_variable = {atom: index for index, atom in enumerate(atom_indices)}
    atom_set = set(atom_indices)
    log_probabilities = incidence_logits.float().log_softmax(-1)
    pair_candidates: list[tuple[int, int, int, float]] = []
    per_bond_variables: dict[int, list[int]] = {}
    per_atom_variables: dict[int, list[int]] = {atom: [] for atom in atom_indices}
    per_pair_variables: dict[tuple[int, int], list[int]] = {}
    topk = min(config.endpoint_topk, len(atom_indices))
    for bond in bond_indices:
        endpoint_options = []
        for endpoint in range(2):
            values = log_probabilities[bond, endpoint].clone()
            values[~atom_mask] = -1e9
            endpoint_options.append(values.topk(topk).indices.tolist())
        best_log_probability = sum(
            float(log_probabilities[bond, endpoint, endpoint_options[endpoint][0]])
            for endpoint in range(2)
        )
        for begin in endpoint_options[0]:
            for end in endpoint_options[1]:
                if begin == end or begin not in atom_set or end not in atom_set:
                    continue
                pair = tuple(sorted((begin, end)))
                relative_incidence = (
                    float(log_probabilities[bond, 0, begin])
                    + float(log_probabilities[bond, 1, end])
                    - best_log_probability
                )
                variable = len(atom_indices) + len(pair_candidates)
                pair_candidates.append((bond, begin, end, relative_incidence))
                per_bond_variables.setdefault(bond, []).append(variable)
                per_atom_variables[begin].append(variable)
                per_atom_variables[end].append(variable)
                per_pair_variables.setdefault(pair, []).append(variable)
    variable_count = len(atom_indices) + len(pair_candidates)
    if not pair_candidates:
        return [], {}, {
            "status": "no-valid-bond-pairs",
            "solver_success": True,
            "selected_atoms": 0,
            "selected_bonds": 0,
            "variables": variable_count,
            "elapsed_seconds": time.perf_counter() - started,
        }

    objective = np.zeros(variable_count, dtype=np.float64)
    for atom, variable in atom_variable.items():
        score = (
            config.atom_presence_weight
            * (float(atom_presence_logits[atom]) - atom_threshold)
            + config.atom_bias
        )
        objective[variable] = -score + 1e-9 * variable
    for offset, (bond, _, _, relative_incidence) in enumerate(pair_candidates):
        variable = len(atom_indices) + offset
        score = (
            config.bond_presence_weight
            * (float(bond_presence_logits[bond]) - bond_threshold)
            + config.bond_bias
            + config.incidence_weight * relative_incidence
        )
        objective[variable] = -score + 1e-9 * variable

    rows: list[dict[int, float]] = []
    lower: list[float] = []
    upper: list[float] = []

    def add_row(values: dict[int, float], low: float, high: float) -> None:
        rows.append(values)
        lower.append(low)
        upper.append(high)

    for variables in per_bond_variables.values():
        add_row({variable: 1.0 for variable in variables}, 0.0, 1.0)
    for offset, (_, begin, end, _) in enumerate(pair_candidates):
        variable = len(atom_indices) + offset
        add_row({variable: 1.0, atom_variable[begin]: -1.0}, -np.inf, 0.0)
        add_row({variable: 1.0, atom_variable[end]: -1.0}, -np.inf, 0.0)
    for variables in per_pair_variables.values():
        add_row({variable: 1.0 for variable in variables}, 0.0, 1.0)
    if config.require_incident_atoms:
        for atom, variable in atom_variable.items():
            values = {variable: 1.0}
            for bond_pair_variable in per_atom_variables[atom]:
                values[bond_pair_variable] = -1.0
            add_row(values, -np.inf, 0.0)

    matrix = lil_matrix((len(rows), variable_count), dtype=np.float64)
    for row_index, values in enumerate(rows):
        for variable, value in values.items():
            matrix[row_index, variable] = value
    result = milp(
        objective,
        integrality=np.ones(variable_count, dtype=np.int8),
        bounds=Bounds(np.zeros(variable_count), np.ones(variable_count)),
        constraints=LinearConstraint(matrix.tocsr(), np.asarray(lower), np.asarray(upper)),
        options={"time_limit": config.time_limit_seconds, "presolve": True},
    )
    if result.x is None:
        raise RuntimeError(f"Joint graph MILP failed: {result.message}")
    selected_atoms = [
        atom for atom, variable in atom_variable.items() if result.x[variable] >= 0.5
    ]
    selected_bonds: dict[int, tuple[int, int]] = {}
    for offset, (bond, begin, end, _) in enumerate(pair_candidates):
        variable = len(atom_indices) + offset
        if result.x[variable] >= 0.5:
            selected_bonds[bond] = (begin, end)
    diagnostics = {
        "status": str(result.message),
        "solver_success": bool(result.success),
        "solver_status": int(result.status),
        "objective": float(result.fun),
        "selected_atoms": len(selected_atoms),
        "selected_bonds": len(selected_bonds),
        "atom_candidates": len(atom_indices),
        "bond_candidates": len(bond_indices),
        "pair_candidates": len(pair_candidates),
        "variables": variable_count,
        "constraints": len(rows),
        "elapsed_seconds": time.perf_counter() - started,
    }
    return sorted(selected_atoms), selected_bonds, diagnostics


def apply_joint_graph_milp(
    *,
    atom_presence_logits: torch.Tensor,
    atom_mask: torch.Tensor,
    bond_presence_logits: torch.Tensor,
    bond_mask: torch.Tensor,
    incidence_logits: torch.Tensor,
    atom_threshold: float,
    bond_threshold: float,
    config: JointGraphMilpConfig = JointGraphMilpConfig(),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    atoms, bonds, diagnostics = solve_joint_graph_milp(
        atom_presence_logits=atom_presence_logits,
        atom_mask=atom_mask,
        bond_presence_logits=bond_presence_logits,
        bond_mask=bond_mask,
        incidence_logits=incidence_logits,
        atom_threshold=atom_threshold,
        bond_threshold=bond_threshold,
        config=config,
    )
    atom_output = torch.full_like(atom_presence_logits, -10.0)
    bond_output = torch.full_like(bond_presence_logits, -10.0)
    incidence_output = incidence_logits.clone()
    if atoms:
        atom_output[torch.tensor(atoms, dtype=torch.long)] = 10.0
    for bond, (begin, end) in bonds.items():
        bond_output[bond] = 10.0
        incidence_output[bond] = -10.0
        incidence_output[bond, 0, begin] = 10.0
        incidence_output[bond, 1, end] = 10.0
    return atom_output, bond_output, incidence_output, diagnostics


__all__ = [
    "JointGraphMilpConfig",
    "apply_joint_graph_milp",
    "solve_joint_graph_milp",
]
