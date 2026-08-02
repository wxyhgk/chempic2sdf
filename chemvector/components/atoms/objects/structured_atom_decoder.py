"""Failure-driven atom-state decoding on a fixed ChemScene topology."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import heapq
import math
import re
from typing import Sequence

import torch
from rdkit import rdBase

from Data.sdf.chemscene_v0_1 import (
    mol_from_chemscene_v0_1,
    parse_chemscene_v0_1,
)
from chemvector.ctc_ocr import (
    ctc_text_log_score,
    decode_ctc_prefix_beam,
)
from chemvector.ocr import atom_display_text, parse_atom_display_text


@dataclass(frozen=True, slots=True)
class AtomStateCandidate:
    text: str
    atom: dict | None
    raw_log_score: float
    normalized_score: float
    source: str
    rank: int = 0


def atom_state_from_text(text: str) -> dict | None:
    parsed = parse_atom_display_text(text)
    if parsed is None:
        return None
    explicit_h_count = int(parsed["explicit_h_count"])
    return {
        "element": str(parsed["element"]),
        "formal_charge": int(parsed["formal_charge"]),
        "isotope": int(parsed["isotope"]),
        "aromatic": False,
        "explicit_h_count": explicit_h_count,
        "no_implicit": bool(explicit_h_count),
        "radical_electrons": int(parsed["radical_electrons"]),
        "canonical_chiral_tag": "CHI_UNSPECIFIED",
    }


def atom_state_candidate(
    text: str,
    normalized_score: float,
    *,
    raw_log_score: float | None = None,
    source: str = "test",
    rank: int = 0,
) -> AtomStateCandidate:
    return AtomStateCandidate(
        text=text,
        atom=atom_state_from_text(text),
        raw_log_score=(normalized_score if raw_log_score is None else raw_log_score),
        normalized_score=normalized_score,
        source=source,
        rank=rank,
    )


def _atom_state_key(atom: dict | None) -> tuple | None:
    if atom is None:
        return None
    return (
        atom["element"],
        atom["formal_charge"],
        atom["isotope"],
        atom["explicit_h_count"],
        atom["radical_electrons"],
    )


def merge_ctc_atom_candidates(
    logits_by_source: Sequence[tuple[str, torch.Tensor]],
    *,
    current_text: str,
    beam_size: int = 12,
    top_paths_per_source: int = 8,
    maximum_candidates: int = 8,
    maximum_bytes: int = 24,
    token_beam: int | None = 24,
) -> list[AtomStateCandidate]:
    """Merge shifted and anchor crop beams using comparable normalized scores."""

    if not logits_by_source:
        return [atom_state_candidate(current_text, -math.inf, source="current")]
    candidates: dict[str, AtomStateCandidate] = {}
    for source, logits in logits_by_source:
        beam = decode_ctc_prefix_beam(
            logits,
            beam_size=beam_size,
            top_paths=top_paths_per_source,
            maximum_bytes=maximum_bytes,
            token_beam=token_beam,
        )
        texts = {item.text for item in beam if item.text}
        if current_text:
            texts.add(current_text)
        raw_scores = {item.text: item.log_score for item in beam}
        for text in texts:
            try:
                normalized = ctc_text_log_score(
                    logits, text, normalize_by_bytes=True
                )
            except ValueError:
                continue
            candidate = AtomStateCandidate(
                text=text,
                atom=atom_state_from_text(text),
                raw_log_score=raw_scores.get(
                    text, ctc_text_log_score(logits, text)
                ),
                normalized_score=normalized,
                source=source,
            )
            previous = candidates.get(text)
            if previous is None or candidate.normalized_score > previous.normalized_score:
                candidates[text] = candidate

    ranked = sorted(
        candidates.values(), key=lambda item: item.normalized_score, reverse=True
    )
    deduplicated = []
    seen_states = set()
    for candidate in ranked:
        key = _atom_state_key(candidate.atom)
        if key is not None and key in seen_states:
            continue
        if key is not None:
            seen_states.add(key)
        deduplicated.append(replace(candidate, rank=len(deduplicated)))
        if len(deduplicated) >= maximum_candidates:
            break
    if current_text and all(item.text != current_text for item in deduplicated):
        current = candidates.get(current_text)
        if current is not None:
            if len(deduplicated) >= maximum_candidates:
                deduplicated[-1] = replace(current, rank=len(deduplicated) - 1)
            else:
                deduplicated.append(replace(current, rank=len(deduplicated)))
    return deduplicated


def scene_topology_signature(scene: dict) -> tuple:
    """Return every ChemScene field that an atom-state reranker must preserve."""

    return (
        tuple(
            (tuple(item["coordinate"]), item["role"])
            for item in scene["junctions"]
        ),
        tuple(
            (
                tuple(map(tuple, item["endpoints"])),
                item["display_type"],
                item["chemical_type"],
                item["direction"],
                item["stereo"],
                tuple(map(tuple, item.get("stereo_references", []))),
            )
            for item in scene["bond_objects"]
        ),
    )


def repair_impossible_explicit_h(scene: dict) -> tuple[dict, list[dict]]:
    """Drop OCR H only when it alone exceeds a conservative ordinary valence."""

    repaired = copy.deepcopy(scene)
    maximum_valence = {
        "B": 3.0,
        "C": 4.0,
        "N": 3.0,
        "O": 2.0,
        "F": 1.0,
        "Si": 4.0,
        "P": 5.0,
        "S": 6.0,
        "Cl": 1.0,
        "Se": 6.0,
        "Br": 1.0,
        "I": 1.0,
    }
    bond_orders = {"SINGLE": 1.0, "DOUBLE": 2.0, "TRIPLE": 3.0}
    coordinate_to_index = {
        tuple(item["coordinate"]): index
        for index, item in enumerate(repaired["junctions"])
    }
    incident_orders = [0.0] * len(repaired["junctions"])
    has_aromatic = [False] * len(repaired["junctions"])
    for bond in repaired["bond_objects"]:
        chemical_type = str(bond["chemical_type"])
        for endpoint in bond["endpoints"]:
            index = coordinate_to_index.get(tuple(endpoint))
            if index is None:
                continue
            if chemical_type == "AROMATIC":
                has_aromatic[index] = True
            else:
                incident_orders[index] += bond_orders.get(chemical_type, 0.0)
    repairs = []
    for index, junction in enumerate(repaired["junctions"]):
        atom = junction["atom"]
        explicit_h = int(atom.get("explicit_h_count", 0))
        element = str(atom.get("element", ""))
        if not explicit_h or has_aromatic[index] or element not in maximum_valence:
            continue
        allowed = maximum_valence[element]
        charge = int(atom.get("formal_charge", 0))
        if element == "N" and charge > 0:
            allowed = 4.0
        elif element == "O" and charge > 0:
            allowed = 3.0
        bonded = incident_orders[index]
        if bonded > allowed or bonded + explicit_h <= allowed:
            continue
        keep_h = max(0, int(allowed - bonded))
        if keep_h >= explicit_h:
            continue
        atom["explicit_h_count"] = keep_h
        atom["no_implicit"] = bool(keep_h)
        coordinate = junction["coordinate"]
        for text_object in repaired["text_objects"]:
            if text_object["anchor"] == coordinate:
                text_object["text"] = atom_display_text(atom) or element
        repairs.append(
            {
                "junction_index": index,
                "element": element,
                "bond_order_sum": bonded,
                "explicit_h_before": explicit_h,
                "explicit_h_after": keep_h,
            }
        )
    parse_chemscene_v0_1(repaired)
    return repaired, repairs


def _text_by_junction(scene: dict) -> list[str]:
    text_by_anchor = {
        tuple(item["anchor"]): str(item["text"])
        for item in scene["text_objects"]
    }
    return [
        text_by_anchor.get(tuple(item["coordinate"]), "")
        for item in scene["junctions"]
    ]


def _apply_overrides(
    scene: dict,
    overrides: dict[int, AtomStateCandidate],
) -> dict:
    updated = copy.deepcopy(scene)
    text_by_anchor = {
        tuple(item["anchor"]): item for item in updated["text_objects"]
    }
    for index, candidate in overrides.items():
        if candidate.atom is None:
            raise ValueError("Unparsed OCR text cannot become a chemical atom state")
        junction = updated["junctions"][index]
        atom = dict(candidate.atom)
        atom["aromatic"] = bool(junction["atom"].get("aromatic", False))
        junction["atom"] = atom
        text_object = text_by_anchor.get(tuple(junction["coordinate"]))
        if text_object is None:
            raise ValueError("Atom-state override targets a Junction without text")
        text_object["text"] = candidate.text
    parse_chemscene_v0_1(updated)
    return updated


_ATOM_FAILURE = re.compile(r"atom #\s*(\d+)", re.IGNORECASE)
_AROMATIC_FAILURES = re.compile(
    r"(?:unkekulized atoms|non-ring atom)\s*:?\s*([\d\s]+)", re.IGNORECASE
)


def failure_atom_indices(error: str, atom_count: int) -> tuple[int, ...]:
    indices = {int(value) for value in _ATOM_FAILURE.findall(error)}
    for group in _AROMATIC_FAILURES.findall(error):
        indices.update(int(value) for value in group.split())
    return tuple(sorted(index for index in indices if 0 <= index < atom_count))


def _sanitize_error(scene: dict) -> str | None:
    repaired, _ = repair_impossible_explicit_h(scene)
    try:
        with rdBase.BlockLogs():
            mol_from_chemscene_v0_1(repaired)
    except Exception as error:
        return str(error)
    return None


def rerank_invalid_scene_atom_states(
    scene: dict,
    candidates_by_junction: Sequence[Sequence[AtomStateCandidate]],
    *,
    max_states: int = 128,
    max_changes: int = 4,
    minimum_score_delta: float = -math.log(100.0),
) -> tuple[dict, dict]:
    """Repair only sanitize-invalid atom states while preserving the full topology."""

    if len(candidates_by_junction) != len(scene["junctions"]):
        raise ValueError("Atom candidates and ChemScene Junctions differ")
    if max_states < 1 or max_changes < 1:
        raise ValueError("Structured atom search limits must be positive")
    original_signature = scene_topology_signature(scene)
    original_text = _text_by_junction(scene)
    base_error = _sanitize_error(scene)
    _, base_h_repairs = repair_impossible_explicit_h(scene)
    diagnostics = {
        "atom_decode_mode": "structured_invalid_only",
        "base_sanitize_valid": base_error is None,
        "final_sanitize_valid": base_error is None,
        "attempted_states": 1,
        "candidate_junctions": sum(bool(items) for items in candidates_by_junction),
        "parsed_candidate_junctions": sum(
            any(item.atom is not None for item in items)
            for items in candidates_by_junction
        ),
        "changes": [],
        "base_error": base_error,
        "base_explicit_h_repairs": base_h_repairs,
    }
    if base_error is None:
        return scene, diagnostics

    current_scores = []
    for text, candidates in zip(original_text, candidates_by_junction):
        match = next((item for item in candidates if item.text == text), None)
        current_scores.append(
            match.normalized_score
            if match is not None
            else max((item.normalized_score for item in candidates), default=0.0)
        )

    queue = []
    serial = 0
    heapq.heappush(queue, (-0.0, 0, serial, {}, base_error, scene))
    visited = {()}
    attempted = 1
    winner = None
    while queue and attempted < max_states:
        negative_delta, changes, _, overrides, error, current_scene = heapq.heappop(
            queue
        )
        if error is None:
            winner = (current_scene, overrides, -negative_delta)
            break
        targets = failure_atom_indices(error, len(scene["junctions"]))
        if not targets:
            continue
        for index in targets:
            if scene["junctions"][index]["role"] != "TEXT":
                continue
            effective_text = overrides.get(index)
            effective_text = (
                effective_text.text if effective_text is not None else original_text[index]
            )
            for candidate in candidates_by_junction[index]:
                if candidate.atom is None or candidate.text == effective_text:
                    continue
                next_overrides = dict(overrides)
                if candidate.text == original_text[index]:
                    next_overrides.pop(index, None)
                else:
                    next_overrides[index] = candidate
                next_changes = len(next_overrides)
                if next_changes > max_changes:
                    continue
                key = tuple(sorted((item, value.text) for item, value in next_overrides.items()))
                if key in visited:
                    continue
                visited.add(key)
                score_delta = sum(
                    value.normalized_score - current_scores[item]
                    for item, value in next_overrides.items()
                )
                if score_delta < minimum_score_delta:
                    continue
                trial = _apply_overrides(scene, next_overrides)
                trial_error = _sanitize_error(trial)
                attempted += 1
                serial += 1
                heapq.heappush(
                    queue,
                    (
                        -score_delta,
                        next_changes,
                        serial,
                        next_overrides,
                        trial_error,
                        trial,
                    ),
                )
                if attempted >= max_states:
                    break
            if attempted >= max_states:
                break

    diagnostics["attempted_states"] = attempted
    if winner is None:
        diagnostics["final_error"] = base_error
        return scene, diagnostics
    repaired, overrides, score_delta = winner
    if scene_topology_signature(repaired) != original_signature:
        raise AssertionError("Structured atom decoding changed ChemScene topology")
    diagnostics["final_sanitize_valid"] = True
    diagnostics["score_delta"] = score_delta
    diagnostics["changes"] = [
        {
            "junction_index": index,
            "from": original_text[index],
            "to": candidate.text,
            "source": candidate.source,
            "rank": candidate.rank,
            "normalized_score": candidate.normalized_score,
        }
        for index, candidate in sorted(overrides.items())
    ]
    return repaired, diagnostics
