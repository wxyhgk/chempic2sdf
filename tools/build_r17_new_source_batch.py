#!/usr/bin/env python3
"""Build a genuinely new, metadata-bound r17 source batch.

The molecules are generated from a new namespace/version/seed and rendered
with the existing deterministic text-native renderer.  This is a data
admission artifact only: it never reads sealed data, queries the model, or
writes a checkpoint.  The resulting manifest and append-only ledger are fed
to ``audit_r17_new_source_prior_use_ledger.py`` before any training.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Iterable
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tools.r17_manifest_rows import rows_from_payload

from Data.sdf.case_diversity_v1.profiles import profiles
from Data.sdf.case_diversity_v1.rendering import render

OUT = ROOT / os.environ.get("R17_NEW_SOURCE_OUT", "outputs/datasets/chemvector-r17-new-source-20261008-v6")
SOURCE_ID = "chemvector-r17-synthetic-source"
# v5 is a strict-visible-inventory successor to the earlier v2 source.  Keep
# the namespace stable for provenance, but bump version/seed so the append-only
# ledger cannot be confused with the superseded 16-canonical-overlap batch.
SOURCE_VERSION = os.environ.get("R17_NEW_SOURCE_VERSION", "20261008.v5")
SOURCE_SEED = os.environ.get("R17_NEW_SOURCE_SEED", "r17-new-source-20261008-seed-e")
PROFILE_IDS = tuple(os.environ.get("R17_NEW_SOURCE_PROFILES", "oa13,oa33").split(","))
TRAIN_COUNT = int(os.environ.get("R17_NEW_SOURCE_TRAIN", "40"))
HOLDOUT_COUNT = int(os.environ.get("R17_NEW_SOURCE_HOLDOUT", "24"))


def digest_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def digest_file(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def canonical(smiles: str) -> tuple[str, Chem.Mol] | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, canonical=True), mol
    except Exception:
        return None


def make_smiles() -> Iterable[str]:
    # Deliberately varied visible O/N, aromatic, double, and triple-bond
    # contexts.  The combinations are new source material, not copied rows.
    r = (
        "CCO", "CCN", "CCCl", "CCBr", "C#CC", "C=CC", "CCC", "CC(C)O",
        "CC(C)N", "CCOC", "CCNC", "c1ccccc1", "c1ccncc1", "c1ccoc1",
        "CC(C)C", "CCOC", "CCSC", "C=CCl", "C#CN", "CC(=O)N",
    )
    templates = (
        *(f"CC(=O)N{x}" for x in r),
        *(f"CS(=O)(=O)N{x}" for x in r),
        *(f"O=C(O){x}" for x in r),
        *(f"O=P(O)(O){x}" for x in r),
        *(f"B(O)(O){x}" for x in r),
        *(f"c1cc({x})ccc1" for x in r),
        *(f"c1ncc({x})cc1" for x in r),
        *(f"c1cc({x})ncc1" for x in r),
    )
    yield from templates
    # A second, independently authored root family expands heterocycle and
    # saturated-ring contexts that were absent from the first smoke list.
    roots = (
        "C1CCNC1", "C1CCOC1", "C1CCSC1", "C1CCNCC1", "C1CCOCC1",
        "C1CCCNC1", "C1CCCOC1", "c1ncncn1", "c1nccnc1", "c1ccncc1",
        "c1ccoc1", "c1ccsc1", "c1cnccn1", "c1cncnc1",
    )
    yield from (f"N{x}" for x in roots)
    yield from (f"O{x}" for x in roots)
    yield from (f"C#C{x}" for x in roots)
    yield from (f"C=C{x}" for x in roots)
    yield from (f"CC(=O)N{x}" for x in roots)
    yield from (f"CS(=O)(=O)N{x}" for x in roots)
    yield from (f"O=C(O){x}" for x in roots)
    yield from (f"B(O)(O){x}" for x in roots)
    yield from (f"c1cc({x})ccc1" for x in roots)
    roots2 = (
        "C1CCSCC1", "C1CCCNCC1", "C1CCCOCC1", "C1CCCSC1", "C1CCCNC1",
        "C1CCN(CC1)C", "C1CCO(CC1)C", "C1CCS(CC1)C", "C1CCN(C)C1",
        "c1cc[nH]c1", "c1cn[nH]c1", "c1scnc1", "c1nscnc1", "c1ccnc2ccccc12",
        "c1ncc2ccccc2n1", "c1ccsc1", "c1csnc1", "c1cc[nH]c2ccccc12",
        "C1CNCCO1", "C1CNCCS1", "C1COCCN1", "C1CSCCN1",
        "C1CC(=O)NC1", "C1CC(=O)OC1", "C1CC(=O)SC1",
    )
    yield from (f"N{x}" for x in roots2)
    yield from (f"O{x}" for x in roots2)
    yield from (f"C#C{x}" for x in roots2)
    yield from (f"C=C{x}" for x in roots2)
    yield from (f"CC(=O)N{x}" for x in roots2)
    yield from (f"CC(=O)O{x}" for x in roots2)
    yield from (f"CS(=O)(=O)N{x}" for x in roots2)
    yield from (f"O=P(O)(O){x}" for x in roots2)
    yield from (f"c1cc({x})ccc1" for x in roots2)
    yield from (f"c1ncc({x})cc1" for x in roots2)
    roots3 = (
        "C1CCCNCC1", "C1CCCOCC1", "C1CCCSC1", "C1CCNCCC1", "C1CCOCCC1",
        "C1CCSCCC1", "C1CCN(C)CC1", "C1CCO(C)CC1", "C1CCS(C)CC1",
        "c1nncc1", "c1nccn1", "c1nccs1", "c1sccn1", "c1occn1", "c1ncc[nH]1",
        "c1ccoc2ccccc12", "c1ccsc2ccccc12", "c1ccnc2ccccc12", "c1ncc2ccccc2n1",
        "c1cc2nccnc2cc1", "c1cc2occc2cc1", "c1cc2sccc2cc1",
        "C1COCCO1", "C1CSCCS1", "C1CNCCN1", "C1CC(=O)NCC1",
        "C1CC(=O)OCC1", "C1CC(=O)SCC1", "C1CCN(CCO)CC1",
    )
    yield from (f"N{x}" for x in roots3)
    yield from (f"O{x}" for x in roots3)
    yield from (f"C#C{x}" for x in roots3)
    yield from (f"C=C{x}" for x in roots3)
    yield from (f"CC(=O)N{x}" for x in roots3)
    yield from (f"CC(=O)O{x}" for x in roots3)
    yield from (f"CS(=O)(=O)N{x}" for x in roots3)
    yield from (f"O=P(O)(O){x}" for x in roots3)
    yield from (f"c1cc({x})ccc1" for x in roots3)
    yield from (f"c1ncc({x})cc1" for x in roots3)


def historical_identity_values() -> dict[str, set[str]]:
    fields = ("molecule_id", "canonical_smiles", "connection_identity", "legacy_scaffold")
    seen = {field: set() for field in fields}
    paths = list((ROOT / "outputs/audits").glob("**/manifest.json"))
    # Dataset-side molecule catalogs are also prior-use evidence.  A batch
    # that was rendered but never placed in an audit manifest is still not a
    # genuinely new source identity.
    paths += list((ROOT / "outputs/datasets").glob("**/molecules.json"))
    paths += list((ROOT / "outputs/datasets").glob("**/manifest.json"))
    for path in sorted(set(paths)):
        if OUT in path.parents:
            continue
        # Metadata-only history checks must not traverse sealed/internal test
        # trees; those identities are intentionally outside this admission
        # scope and their payloads must remain unread.
        if any(part in str(path).lower() for part in ("sealed", "internal-test", "test996")):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = rows_from_payload(payload)
            for row in rows:
                if not isinstance(row, dict):
                    continue
                for field in fields:
                    aliases = {
                        "connection_identity": ("connection_identity", "connection_identity_without_isotope_stereo"),
                        "legacy_scaffold": ("legacy_scaffold", "scaffold", "scaffold_identity"),
                    }.get(field, (field,))
                    value = next((row.get(alias) for alias in aliases if row.get(alias)), None)
                    if value is not None and not isinstance(value, (dict, list)):
                        text = str(value).strip()
                        if text:
                            seen[field].add(text)
        except Exception:
            continue
    return seen


def entry_rows() -> list[dict]:
    history = historical_identity_values()
    candidates: dict[str, dict] = {}
    for smiles in make_smiles():
        item = canonical(smiles)
        if item is None:
            continue
        can, mol = item
        molecule_id = "newr17_" + sha256((SOURCE_ID + "|" + SOURCE_VERSION + "|" + can).encode()).hexdigest()[:16]
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or "ACYCLIC:" + can
        connection = Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(mol)), canonical=True)
        values = {
            "molecule_id": molecule_id,
            "canonical_smiles": can,
            "connection_identity": connection,
            "legacy_scaffold": scaffold,
        }
        if any(values[field] in history[field] for field in values):
            continue
        candidates[molecule_id] = values
    ordered = sorted(candidates.values(), key=lambda row: sha256((SOURCE_SEED + row["molecule_id"]).encode()).hexdigest())
    if len(ordered) < TRAIN_COUNT + HOLDOUT_COUNT:
        raise RuntimeError(f"only {len(ordered)} new candidates; need {TRAIN_COUNT + HOLDOUT_COUNT}")
    # Keep a deterministic reserve because the strict renderer may reject a
    # crowded glyph/bond layout.  Final split assignment happens only after a
    # molecule passes both requested profiles.
    selected = ordered[: min(len(ordered), (TRAIN_COUNT + HOLDOUT_COUNT) * 3)]
    rows = []
    for index, molecule in enumerate(selected):
        split = "candidate"
        for profile_id in PROFILE_IDS:
            rows.append({
                **molecule,
                "sample_key": f"{molecule['molecule_id']}__{profile_id}",
                "profile_id": profile_id,
                "split": split,
                "source_id": SOURCE_ID,
                "source_version": SOURCE_VERSION,
                "source_seed": SOURCE_SEED,
            })
    return rows


def run() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    rows = entry_rows()
    OUT.mkdir(parents=True)
    source_dir = OUT / "samples"
    source_dir.mkdir()
    selected_profiles = {p.profile_id: p for p in profiles() if p.profile_id in PROFILE_IDS}
    rendered = []
    ledger = []
    by_molecule = {}
    for row in rows:
        by_molecule.setdefault(row["molecule_id"], []).append(row)
    successful = []
    for molecule_id, molecule_rows in sorted(by_molecule.items()):
        staged = []
        try:
            for row in sorted(molecule_rows, key=lambda item: item["profile_id"]):
                entry = {
                    **{key: value for key, value in row.items() if key not in {"sample_key", "profile_id", "split"}},
                    "split": "quarantine", "training_authorized": False,
                    "external_overlap_status": "unchecked", "source_record_ids": [SOURCE_ID],
                    "stages": ["r17_new_source"],
                }
                files, meta = render(entry, selected_profiles[row["profile_id"]])
                staged.append((row, files))
        except (ValueError, RuntimeError):
            continue
        successful.append((molecule_id, staged))
        if len(successful) >= TRAIN_COUNT + HOLDOUT_COUNT:
            break
    if len(successful) < TRAIN_COUNT + HOLDOUT_COUNT:
        raise RuntimeError(f"only {len(successful)} renderable molecules; need {TRAIN_COUNT + HOLDOUT_COUNT}")
    for index, (molecule_id, staged) in enumerate(successful[: TRAIN_COUNT + HOLDOUT_COUNT]):
        split = "train" if index < TRAIN_COUNT else "holdout"
        for row, files in staged:
            key = row["sample_key"]
            for suffix, data in files.items():
                (source_dir / f"{key}{suffix}").write_bytes(data)
            png = files[".png"]
            with Image.open(source_dir / f"{key}.png") as image:
                rgb_sha = digest_bytes(image.convert("RGB").tobytes())
            public = {**row, "split": split, "image_sha256": digest_bytes(png),
                      "png_sha256": digest_bytes(png), "rgb_sha256": rgb_sha}
            rendered.append(public)
            ledger.append({**public, "event": "reserved_before_model_use", "source_namespace": SOURCE_ID})
    manifest = {
        "schema_version": "chemvector-r17-new-source-manifest-0.1",
        "source_id": SOURCE_ID,
        "source_version": SOURCE_VERSION,
        "source_seed": SOURCE_SEED,
        "metadata_only": True,
        "training_authorized": False,
        "sealed_test_read": False,
        "rows": rendered,
    }
    manifest_path = OUT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ledger_path = OUT / "prior-use-ledger.jsonl"
    with ledger_path.open("w", encoding="utf-8") as stream:
        for row in ledger:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "schema_version": "chemvector-r17-new-source-build-0.1",
        "complete": True,
        "read_only": True,
        "training_performed": False,
        "gpu_used": False,
        "sealed_test_read": False,
        "source_id": SOURCE_ID,
        "source_version": SOURCE_VERSION,
        "source_seed": SOURCE_SEED,
        "rows": len(rendered),
        "molecules": len({row["molecule_id"] for row in rendered}),
        "profile_counts": dict(Counter(row["profile_id"] for row in rendered)),
        "split_molecules": {split: len({row["molecule_id"] for row in rendered if row["split"] == split}) for split in ("train", "holdout")},
        "manifest_sha256": digest_file(manifest_path),
        "ledger_sha256": digest_file(ledger_path),
        "training_admissible": False,
        "gpu_training_go": False,
        "production_go": False,
        "next": "run metadata-only prior-use ledger admission before any model training",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
