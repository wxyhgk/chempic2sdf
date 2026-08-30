"""Assemble the frozen joint Atom/Bond inference components."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from chemvector.benchmark_joint_mr_tadf import (
    _load_atom_pair_group_head,
    _load_set_head,
)
from chemvector.atom_pair_graph_context import load_atom_pair_graph_context_head
from chemvector.atom_pair_deletion_veto import AtomPairDeletionVetoConfig
from chemvector.atom_pair_one_move import load_atom_pair_one_move_head
from chemvector.atom_pair_chemical_retype import load_atom_pair_chemical_retype_head
from chemvector.atom_pair_chemical_retype_verifier import (
    load_atom_pair_chemical_retype_verifier,
)
from chemvector.joint_atom_pair_refinement import JointAtomPairExperts
from chemvector.line_atom_pair_head import load_line_atom_pair_head
from chemvector.train_spatial_incidence import load_frozen_components


@dataclass(frozen=True, slots=True)
class JointInferenceComponents:
    vision: Any
    atom_head: Any
    bond_head: Any
    primary: Any
    fallback: Any
    experts: JointAtomPairExperts


def load_joint_components(
    *,
    grounded_graph_checkpoint: Path,
    atom_checkpoint: Path,
    bond_checkpoint: Path,
    primary_checkpoint: str,
    fallback_checkpoint: str,
    atom_pair_group_checkpoint: str | None,
    atom_pair_graph_context_checkpoint: str | None,
    atom_pair_one_move_checkpoint: str | None,
    atom_pair_chemical_retype_verifier_checkpoint: str | None,
    line_atom_pair_checkpoint: str | None,
    device: torch.device,
) -> JointInferenceComponents:
    """Load frozen base heads plus the validated AtomPair expert cascade."""

    vision, atom_head, bond_head, _ = load_frozen_components(
        grounded_graph_checkpoint,
        atom_checkpoint,
        bond_checkpoint,
        device,
    )
    primary, _ = _load_set_head(primary_checkpoint, device)
    fallback, _ = _load_set_head(fallback_checkpoint, device)
    group_head = None
    group_payload = None
    if atom_pair_group_checkpoint:
        group_head, group_payload = _load_atom_pair_group_head(
            atom_pair_group_checkpoint, device
        )
    line_atom_pair_head = None
    line_atom_pair_payload = None
    if line_atom_pair_checkpoint:
        line_atom_pair_head, line_atom_pair_payload = load_line_atom_pair_head(
            line_atom_pair_checkpoint, device
        )
        if group_head is None:
            raise ValueError("Line AtomPair inference requires the AtomPair group head")
    graph_context_head = None
    graph_context_payload = None
    if atom_pair_graph_context_checkpoint:
        if group_head is None:
            raise ValueError("AtomPair graph context requires the AtomPair group head")
        graph_context_head, graph_context_payload = (
            load_atom_pair_graph_context_head(
                atom_pair_graph_context_checkpoint, device
            )
        )
        if Path(graph_context_payload["group_checkpoint"]) != Path(
            atom_pair_group_checkpoint
        ):
            raise ValueError("AtomPair graph context was trained for another group head")
    one_move_head = None
    one_move_payload = None
    one_move_relation_config = None
    if atom_pair_one_move_checkpoint:
        if graph_context_head is None:
            raise ValueError("AtomPair one-move requires AtomPair graph context")
        one_move_head, one_move_payload = load_atom_pair_one_move_head(
            atom_pair_one_move_checkpoint, device
        )
        if Path(one_move_payload["base_context_checkpoint"]) != Path(
            atom_pair_graph_context_checkpoint
        ):
            raise ValueError("AtomPair one-move was trained for another graph context")
        if Path(one_move_payload["group_checkpoint"]) != Path(
            atom_pair_group_checkpoint
        ):
            raise ValueError("AtomPair one-move was trained for another group head")
        one_move_relation_config = AtomPairDeletionVetoConfig(
            **one_move_payload["relation_config"]
        )
    retype_head = None
    retype_payload = None
    retype_verifier_head = None
    retype_verifier_payload = None
    if atom_pair_chemical_retype_verifier_checkpoint:
        if one_move_head is None:
            raise ValueError("Chemical RETYPE verifier requires AtomPair one-move")
        (
            retype_verifier_head,
            retype_verifier_payload,
        ) = load_atom_pair_chemical_retype_verifier(
            atom_pair_chemical_retype_verifier_checkpoint, device
        )
        (
            retype_head,
            retype_payload,
        ) = load_atom_pair_chemical_retype_head(
            retype_verifier_payload["retype_init_checkpoint"],
            device,
        )
        if Path(retype_payload["first_move_checkpoint"]) != Path(
            atom_pair_one_move_checkpoint
        ):
            raise ValueError("Chemical RETYPE verifier uses another one-move model")
        if retype_verifier_payload["relation_config"] != (
            one_move_payload["relation_config"]
        ):
            raise ValueError("Chemical RETYPE verifier uses another relation protocol")
    return JointInferenceComponents(
        vision=vision,
        atom_head=atom_head,
        bond_head=bond_head,
        primary=primary,
        fallback=fallback,
        experts=JointAtomPairExperts(
            group_head=group_head,
            group_payload=group_payload,
            line_atom_pair_head=line_atom_pair_head,
            line_atom_pair_payload=line_atom_pair_payload,
            graph_context_head=graph_context_head,
            graph_context_payload=graph_context_payload,
            one_move_head=one_move_head,
            one_move_payload=one_move_payload,
            one_move_relation_config=one_move_relation_config,
            retype_verifier_head=retype_verifier_head,
            retype_verifier_payload=retype_verifier_payload,
            retype_head=retype_head,
            retype_payload=retype_payload,
        ),
    )