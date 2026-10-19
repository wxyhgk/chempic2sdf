"""Small, metadata-only manifest flattener shared by r17 admission tools.

Historical manifests are not schema-uniform: current releases use a top-level
``rows`` list while older pilot manifests use ``split -> molecules[] ->
rows[]``.  Admission must inspect both forms, without opening image/graph
payloads.  This helper deliberately returns only row dictionaries and carries
an inherited split label when the nested row omitted it.
"""
from __future__ import annotations

from typing import Any


_SPLITS = ("train", "holdout", "dev", "validation", "test", "fresh_holdout")


def rows_from_payload(payload: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def visit(node: Any, inherited_split: str | None = None) -> None:
        if isinstance(node, list):
            for item in node:
                visit(item, inherited_split)
            return
        if not isinstance(node, dict):
            return

        # A row is identified by the sample/molecule metadata used by the
        # admission contract.  Do not recurse into it after collecting it.
        if any(key in node for key in ("sample_key", "profile_id", "image_sha256", "png_sha256")):
            row = dict(node)
            if inherited_split and not row.get("split"):
                row["split"] = inherited_split
            rows.append(row)
            return

        # Dataset molecule catalogs are often a flat list with only identity
        # fields (no sample_key/profile).  Admit those rows for history scans;
        # container objects such as ``{"molecule_id": ..., "rows": [...]}``
        # are intentionally not collected so their child sample rows remain
        # reachable.
        if any(key in node for key in ("molecule_id", "canonical_smiles", "connection_identity", "legacy_scaffold")) \
                and not any(key in node for key in ("rows", "molecules", "items", "samples", "records")):
            row = dict(node)
            if inherited_split and not row.get("split"):
                row["split"] = inherited_split
            rows.append(row)
            return

        # Preserve the split while descending through split -> molecules.
        for split in _SPLITS:
            if split in node:
                visit(node[split], split)

        # Current manifests have a direct rows list; old manifests may nest
        # rows below molecule groups.  Generic recursion also tolerates
        # additional metadata wrappers without silently dropping them.
        for key, value in node.items():
            if key in _SPLITS:
                continue
            if key in ("rows", "molecules", "items", "samples", "records"):
                visit(value, inherited_split)

    visit(payload)
    return rows


def rows_from(path) -> list[dict[str, Any]]:
    import json
    from pathlib import Path

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = rows_from_payload(payload)
    if not rows:
        raise ValueError(f"{path} is not a row manifest")
    return rows
