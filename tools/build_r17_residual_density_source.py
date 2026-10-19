#!/usr/bin/env python3
"""Build a small identity-safe r17 residual-density source (quarantine only).

This is intentionally a separate, bounded source from t2.  It selects whole
molecules with endpoint-conditioned O/N bonds (degree and bond type) instead
of only counting aggregate O/N strata.  The source is metadata-only until its
ledger and strata receipts pass.  No model, checkpoint, or sealed payload is
read by this script.

The resulting batch is deliberately small (24 train + 8 holdout molecules,
eight profiles = 256 images) so that frozen CPU replay can answer whether the
new endpoint distribution actually raises residual-positive density before any
training is considered.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import sys

from PIL import Image
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from Data.sdf.case_diversity_v1.profiles import profiles
from Data.sdf.case_diversity_v1.rendering import render
from tools.build_r17_targeted_source_batch import historical_identity_values
from tools.r17_manifest_rows import rows_from_payload

RDLogger.DisableLog("rdApp.*")

OUT = ROOT / os.environ.get(
    "R17_RESIDUAL_SOURCE_OUT",
    "outputs/datasets/chemvector-r17-residual-density-source-20261019-t3",
)
SOURCE_ID = os.environ.get("R17_RESIDUAL_SOURCE_ID", "chemvector-r17-residual-density-visible-source-t3")
SOURCE_VERSION = os.environ.get("R17_RESIDUAL_SOURCE_VERSION", "20261019.t3")
SOURCE_SEED = os.environ.get("R17_RESIDUAL_SOURCE_SEED", "r17-residual-density-20261019-seed-a")
PROFILE_IDS = tuple(x for x in os.environ.get(
    "R17_RESIDUAL_SOURCE_PROFILES", "oa00,oa02,oa11,oa13,oa20,oa22,oa31,oa33"
).split(",") if x)
TRAIN_COUNT = int(os.environ.get("R17_RESIDUAL_SOURCE_TRAIN", "12"))
HOLDOUT_COUNT = int(os.environ.get("R17_RESIDUAL_SOURCE_HOLDOUT", "8"))

# Quotas are independent-molecule counts, not image or edge counts.  They are
# intentionally modest for this first CPU preflight batch.
_DEFAULT_HOLDOUT_TAG_QUOTAS = {
    "EP:O:DOUBLE:D1": 2,
    "EP:O:SINGLE:D2": 2,
    "EP:N:DOUBLE:D1": 2,
    "EP:N:TRIPLE:D1": 2,
    "GRAPH:TRIPLE": 3,
    # The historical scaffold quarantine can legitimately leave only a few
    # novel aromatic/ring identities; require presence without fabricating a
    # quota that would force reuse.
    "GRAPH:AROMATIC": 1,
    "RING": 1,
}

# Keep the historical defaults for t3d/t3e, but allow a new quarantine build
# to demand *several independent* aromatic/ring molecules on both sides of
# the split.  This is intentionally an environment-only override: no caller
# can silently change an existing manifest, and the resolved quotas are
# written into summary.json for auditability.
HOLDOUT_TAG_QUOTAS = {
    key: int(os.environ.get("R17_RESIDUAL_HOLDOUT_QUOTA_" + key.replace(":", "_"), str(value)))
    for key, value in _DEFAULT_HOLDOUT_TAG_QUOTAS.items()
}
TRAIN_TAG_QUOTAS = {key: int(os.environ.get("R17_RESIDUAL_TRAIN_QUOTA_" + key.replace(":", "_"), "1"))
                    for key in HOLDOUT_TAG_QUOTAS}


def digest_bytes(data: bytes) -> str:
    return sha256(data).hexdigest()


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


def candidate_smiles() -> tuple[str, ...]:
    """Fresh endpoint families; identity admission remains the final authority."""
    base = (
        # O double/single and branching endpoint families.
        "O=C=O", "O=C=O", "O=C(O)O", "O=C(O)N", "O=C(N)N",
        "O=C(OC)C", "O=C(O)C", "O=C(OC)CC", "O=C(O)CC",
        "O=C(N)CC", "O=C(N)C#N", "O=C(O)C#N", "O=C(OC)C#N",
        "COC(=O)OC", "CCOC(=O)OCC", "CC(C)(C)OC(=O)N",
        "O=S(=O)(O)C", "O=S(=O)(N)C", "O=P(O)(O)OC",
        # N double/triple endpoint families, including degree-2 junctions.
        "N=C=O", "N=C=S", "N=C(N)N", "N=C(N)O", "N=C(N)S",
        "CC=N", "CCC=N", "CCN=C=O", "CCN=C=S", "COC=N",
        "N#C", "N#CC", "N#CCC", "N#CCO", "N#CCN", "N#CC(=O)O",
        "N#CC(=O)N", "N#CC#N", "N#CC#CC#N", "N#CC#CCO",
        "N#CC(F)(F)F", "N#CC(Cl)(Cl)Cl", "CC(C)C#N", "CCNCC#N",
        # Heteroaromatic endpoint families and ring closures.
        "c1ccncc1", "c1ncc[nH]1", "c1nccnc1", "c1ccoc1",
        "c1occc1", "c1ccsc1", "c1cc[nH]c1", "c1ncncc1",
        "c1cc2ccccc2n1", "c1cc2ccccc2o1", "c1ccc2ncccc2c1",
        "c1ccc2occc2c1", "c1cc2nccnc2c1", "c1cc2ocnc2c1",
        # Substituted heteroaromatics are distinct connection identities from
        # the historical parent rings while preserving aromatic/ring context.
        "Cc1ccncc1", "Fc1ccncc1", "Clc1ccncc1", "COc1ccncc1",
        "Cc1nccnc1", "Fc1nccnc1", "Cc1ccoc1", "Fc1ccoc1",
        "Cc1cc[nH]c1", "Cc1ccc2ncccc2c1", "Cc1ccc2occc2c1",
        # Larger fused systems provide genuinely new Murcko scaffolds rather
        # than reusing substituted pyridine/furan parents.
        "c1ccc2cc3occc3cc2c1", "c1ccc2cc3ocnc3cc2c1",
        "c1ccc2cc3ncccc3cc2c1", "c1ccc2cc3cc4ccccc4n3cc2c1",
        "c1ccc2nc3cc4ccccc4nc3cc2c1", "c1ccc2cc3cc4ccccc4c3cc2c1",
        "c1ccc2cc3ncc4ccccc4c3cc2c1", "c1ccc2cc3oc4ccccc4c3cc2c1",
        # Additional fused heteroaromatic scaffolds reserved for t3f/t4.
        # These are deliberately not substitutions of the small pyridine/
        # furan pool: their Murcko scaffolds remain distinct after the
        # historical identity quarantine.
        "c1ccc2cc3[nH]ccc3cc2c1", "c1ccc2cc3sccc3cc2c1",
        "c1ccc2cc3cc4[nH]ccc4cc3cc2c1",
        "c1ccc2cc3ccccc3cc2c1", "c1ccc2cc3cc4occc4cc3cc2c1",
        "c1ccc2cc3cc4sccc4cc3cc2c1",
        "c1ccc2cc3cc4ncccc4cc3cc2c1",
        "c1ccc2cc3cc4ccccc4cc3cc2c1",
        "c1cc2ncon2c1", "c1cc2ocnn2c1", "c1cc2ncnc2c1",
        "c1cc2nccs2c1", "c1cc2ocnc2c1", "c1cc2nncc2c1",
        # t4 scaffold reserve: distinct tricyclic heteroaromatics.  These
        # are intentionally separate Murcko scaffolds so train/holdout can
        # be stratified without reusing the t3 family.
        "c1ccc2c(c1)ccc1ncccc12", "c1ccc2c(c1)ccc1occc12",
        "c1ccc2c(c1)ccc1sccc12", "c1ccc2c(c1)ccc1ocnc12",
        "c1ccc2c(c1)ccc1scnc12", "c1ccc2c(c1)ccc1nccn12",
        "c1ccc2c(c1)ccc1ccoc12", "c1ccc2c(c1)ccc1ccsc12",
        "c1ccc2c(c1)sc1ccncc12", "c1ccc2c(c1)ncc1ccncc12",
        "c1ccc2c(c1)nnc1ccncc12", "c1ccc2c(c1)cnc1ccncc12",
        # Fused/ring systems with explicit carbonyl/nitrile endpoints.
        "O=C1NCCOCC1", "O=C1NCCSCC1", "O=C1CCOCCN1",
        "O=C1CCNCCN1", "O=C1OC2=CC=CC=C2C1=O",
        "O=C1N(C)CCOCC1", "O=C1CCOC(=O)N1", "O=C1NCC(=O)CC1",
        "c1cc2occc2c1C#N", "c1cc2ncc(C#N)cc2c1",
        "c1cc2occc2c1C(=O)O", "c1cc2ncc(C(=O)O)cc2c1",
        # Explicitly substituted heteroaromatics.  These avoid relying on the
        # small historical unsubstituted ring pool while retaining aromatic
        # bond and ring endpoint conditions.
        "Cc1ccncc1", "Fc1ccncc1", "Clc1ccncc1", "COc1ccncc1",
        "Cc1nccnc1", "Fc1nccnc1", "Cc1ccoc1", "Cc1occc1",
        "Cc1cc[nH]c1", "Cc1ccsc1", "Cc1ccc2ncccc2c1",
        "Cc1ccc2occc2c1", "Fc1ccc2ncccc2c1", "COc1ccc2occc2c1",
        # Multiple endpoint bonds in one compact graph increase positive rows.
        "N#CC(=O)N", "N#CC(=O)O", "N#CC(=O)OC", "N#CC(=N)N",
        "N#CC(=N)O", "N=C(N)C(=O)O", "N=C(N)C(=O)N",
        "CC(=O)N=C=O", "CC(=O)N=C=S", "CC(=O)N#CC#N",
        "O=C(O)C(=O)O", "O=C(O)C(=O)N", "O=C(N)C(=O)N",
    )
    # Generate a larger *acyclic* endpoint pool rather than repeating fused
    # scaffolds.  Each root changes full canonical/connection identity, while
    # the endpoint tag remains controlled.  This is still metadata-only; the
    # historical identity scan below is authoritative and may discard rows.
    roots = ("C", "CC", "CCC", "CCCC", "CO", "CCO", "CCCO", "CN", "CCN",
             "CNC", "CCNC", "CS", "CCS", "CCl", "CBr", "CF", "C(F)(F)F",
             "C#N", "C#CC", "C1CC1", "C1CCC1", "C1CCO1", "C1CCN1")
    dynamic = []
    for root in roots:
        dynamic.extend((f"O=C(O){root}", f"O=C(N){root}", f"O=C(OC){root}",
                        f"O=C(O{root})C", f"O=C(N{root})C"))
        dynamic.extend((f"N=C({root})O", f"N=C({root})N", f"N=C({root})S",
                        f"N=C(O){root}", f"N=C(N){root}"))
        dynamic.extend((f"N#C{root}", f"N#CC{root}", f"N#CC(=O){root}",
                        f"N#CC(=N){root}"))
    # Compact multi-endpoint chains increase positive-pair density without
    # relying on one repeated ring scaffold.
    for left in ("C", "CC", "CCC", "CO", "CN", "CCl", "CBr", "C#N"):
        for right in ("C", "CC", "CO", "CN", "C#N", "C(=O)O", "C(=O)N"):
            dynamic.extend((f"O=C({left}){right}", f"N=C({left}){right}",
                            f"N#C{left}{right}"))
    return tuple(base) + tuple(dynamic)


def endpoint_tags(mol: Chem.Mol) -> set[str]:
    tags: set[str] = set()
    for bond in mol.GetBonds():
        kind = str(bond.GetBondType()).upper()
        if bond.GetIsAromatic() or kind == "AROMATIC":
            kind = "AROMATIC"
        if kind not in {"SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"}:
            continue
        tags.add("GRAPH:" + kind)
        for atom in (bond.GetBeginAtom(), bond.GetEndAtom()):
            symbol = atom.GetSymbol()
            if symbol not in {"O", "N"}:
                continue
            degree = atom.GetDegree()
            tags.add(f"EP:{symbol}:{kind}:D{degree}")
            if degree == 1:
                tags.add(f"EP:{symbol}:{kind}:TERMINAL")
            if degree >= 3:
                tags.add(f"EP:{symbol}:{kind}:JUNCTION")
    if any(bond.IsInRing() for bond in mol.GetBonds()):
        tags.add("RING")
    return tags


def build_candidates() -> list[dict]:
    history = historical_identity_values(OUT)
    result: dict[str, dict] = {}
    for smiles in candidate_smiles():
        item = canonical(smiles)
        if item is None:
            continue
        can, mol = item
        molecule_id = "newr17t3_" + sha256((SOURCE_ID + "|" + SOURCE_VERSION + "|" + can).encode()).hexdigest()[:16]
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or "ACYCLIC:" + can
        connection = Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(mol)), canonical=True)
        values = {"molecule_id": molecule_id, "canonical_smiles": can,
                  "connection_identity": connection, "legacy_scaffold": scaffold}
        if any(values[field] in history[field] for field in values):
            continue
        result[molecule_id] = {**values, "tags": sorted(endpoint_tags(mol))}
    ordered = sorted(result.values(), key=lambda row: sha256((SOURCE_SEED + row["molecule_id"]).encode()).hexdigest())
    # Rendering every chemically valid variant is unnecessarily expensive for
    # this first CPU preflight.  Reserve a deterministic, tag-diverse prefix;
    # the full candidate pool remains reproducible by raising the limit.
    limit = int(os.environ.get("R17_RESIDUAL_CANDIDATE_LIMIT", "96"))
    if len(ordered) <= limit:
        return ordered
    selected: list[dict] = []
    remaining = list(ordered)
    covered = Counter()
    while len(selected) < limit and any(covered[k] < v for k, v in HOLDOUT_TAG_QUOTAS.items()):
        choices = [(sum(covered[k] < v and k in item["tags"] for k, v in HOLDOUT_TAG_QUOTAS.items()), item)
                   for item in remaining]
        choices = [pair for pair in choices if pair[0] > 0]
        if not choices:
            break
        _, chosen = max(choices, key=lambda pair: (pair[0], -remaining.index(pair[1])))
        selected.append(chosen); remaining.remove(chosen)
        for key in chosen["tags"]:
            if key in HOLDOUT_TAG_QUOTAS:
                covered[key] += 1
    selected.extend(remaining[: max(0, limit - len(selected))])
    return selected


def render_pool(candidates: list[dict], selected_profiles: dict[str, object]) -> list[dict]:
    def one(molecule: dict) -> dict | None:
        staged = []
        try:
            for profile_id in PROFILE_IDS:
                entry = {**molecule, "split": "quarantine", "source_id": SOURCE_ID,
                         "source_version": SOURCE_VERSION, "source_seed": SOURCE_SEED,
                         "training_authorized": False, "external_overlap_status": "unchecked",
                         "source_record_ids": [SOURCE_ID], "stages": ["r17_residual_density_source"]}
                files, _meta = render(entry, selected_profiles[profile_id])
                # The renderer itself is the source of truth for graph validity;
                # only metadata is inspected here, never model output.
                staged.append(({**entry, "sample_key": molecule["molecule_id"] + "__" + profile_id,
                                "profile_id": profile_id}, files))
        except (ValueError, RuntimeError, KeyError):
            return None
        return {"molecule": molecule, "tags": set(molecule["tags"]), "files": staged}
    # Cairo rasterization dominates this bounded metadata build.  Parallel
    # workers do not touch shared model/checkpoint state; preserve candidate
    # order after collection for deterministic split assignment.
    workers = max(1, int(os.environ.get("R17_RESIDUAL_RENDER_WORKERS", "4")))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        values = list(executor.map(one, candidates))
    return [value for value in values if value is not None]


def choose_split(pool: list[dict]) -> tuple[list[dict], list[dict]]:
    if len(pool) < TRAIN_COUNT + HOLDOUT_COUNT:
        raise RuntimeError(f"only {len(pool)} complete molecules; need {TRAIN_COUNT + HOLDOUT_COUNT}")
    remaining = list(pool)
    holdout: list[dict] = []
    covered = Counter()
    while len(holdout) < HOLDOUT_COUNT and any(covered[k] < v for k, v in HOLDOUT_TAG_QUOTAS.items()):
        choices = [(sum(covered[k] < v and k in item["tags"] for k, v in HOLDOUT_TAG_QUOTAS.items()), item)
                   for item in remaining]
        choices = [pair for pair in choices if pair[0] > 0]
        if not choices:
            missing = {k: v - covered[k] for k, v in HOLDOUT_TAG_QUOTAS.items() if covered[k] < v}
            raise RuntimeError("cannot satisfy holdout endpoint quotas: " + repr(missing))
        _, chosen = max(choices, key=lambda pair: (pair[0], -remaining.index(pair[1])))
        holdout.append(chosen); remaining.remove(chosen)
        for key in chosen["tags"]:
            if key in HOLDOUT_TAG_QUOTAS:
                covered[key] += 1
    if any(covered[k] < v for k, v in HOLDOUT_TAG_QUOTAS.items()):
        raise RuntimeError("holdout endpoint quota failure")
    holdout.extend(remaining[:HOLDOUT_COUNT - len(holdout)])
    hold_ids = {x["molecule"]["molecule_id"] for x in holdout}
    train_pool = [x for x in remaining if x["molecule"]["molecule_id"] not in hold_ids]
    train: list[dict] = []; covered = Counter()
    while len(train) < TRAIN_COUNT and any(covered[k] < v for k, v in TRAIN_TAG_QUOTAS.items()):
        choices = [(sum(covered[k] < v and k in item["tags"] for k, v in TRAIN_TAG_QUOTAS.items()), item)
                   for item in train_pool if item not in train]
        choices = [pair for pair in choices if pair[0] > 0]
        if not choices:
            missing = {k: v - covered[k] for k, v in TRAIN_TAG_QUOTAS.items() if covered[k] < v}
            raise RuntimeError("cannot satisfy train endpoint quotas: " + repr(missing))
        _, chosen = max(choices, key=lambda pair: (pair[0], -train_pool.index(pair[1])))
        train.append(chosen)
        for key in chosen["tags"]:
            if key in TRAIN_TAG_QUOTAS:
                covered[key] += 1
    train.extend(x for x in train_pool if x not in train)
    train = train[:TRAIN_COUNT]
    if len(train) != TRAIN_COUNT:
        raise RuntimeError("not enough train molecules after holdout reservation")
    return train, holdout


def run() -> dict:
    if OUT.exists():
        raise FileExistsError(OUT)
    selected_profiles = {p.profile_id: p for p in profiles() if p.profile_id in PROFILE_IDS}
    if set(selected_profiles) != set(PROFILE_IDS):
        raise ValueError("unknown or duplicate profile ids")
    candidates = build_candidates()
    pool = render_pool(candidates, selected_profiles)
    train, holdout = choose_split(pool)
    OUT.mkdir(parents=True); source_dir = OUT / "samples"; source_dir.mkdir()
    rows = []; ledger = []
    for split, items in (("train", train), ("holdout", holdout)):
        for item in items:
            for entry, files in item["files"]:
                key = entry["sample_key"]
                for suffix, data in files.items():
                    (source_dir / f"{key}{suffix}").write_bytes(data)
                png = files[".png"]
                with Image.open(io.BytesIO(png)) as image:
                    rgb_sha = digest_bytes(image.convert("RGB").tobytes())
                public = {**{k: v for k, v in entry.items() if k not in {"training_authorized", "external_overlap_status", "source_record_ids", "stages", "tags"}},
                          "split": split, "training_authorized": False,
                          "external_overlap_status": "unchecked", "image_sha256": digest_bytes(png),
                          "png_sha256": digest_bytes(png), "rgb_sha256": rgb_sha,
                          "endpoint_tags": item["molecule"]["tags"]}
                rows.append(public); ledger.append({**public, "event": "reserved_before_model_use", "source_namespace": SOURCE_ID})
    manifest = {"schema_version": "chemvector-r17-residual-density-source-manifest-0.1", "source_id": SOURCE_ID,
                "source_version": SOURCE_VERSION, "source_seed": SOURCE_SEED, "metadata_only": True,
                "training_authorized": False, "sealed_test_read": False, "rows": rows}
    manifest_path = OUT / "manifest.json"; manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    ledger_path = OUT / "prior-use-ledger.jsonl"
    ledger_path.write_text("\n".join(json.dumps(x, sort_keys=True) for x in ledger) + "\n")
    split_items = {s: [x for x in rows if x["split"] == s] for s in ("train", "holdout")}
    tag_counts = {s: {k: len({x["molecule_id"] for x in rs if k in x["endpoint_tags"]}) for k in sorted(HOLDOUT_TAG_QUOTAS)} for s, rs in split_items.items()}
    summary = {"schema_version": "chemvector-r17-residual-density-source-build-0.1", "complete": True,
               "read_only": True, "training_performed": False, "gpu_used": False, "sealed_test_read": False,
               "source_id": SOURCE_ID, "source_version": SOURCE_VERSION, "source_seed": SOURCE_SEED,
               "rows": len(rows), "molecules": len({x["molecule_id"] for x in rows}),
               "profile_counts": dict(Counter(x["profile_id"] for x in rows)),
               "split_molecules": {s: len({x["molecule_id"] for x in rs}) for s, rs in split_items.items()},
               "candidate_molecules": len(candidates), "render_complete_molecules": len(pool),
               "endpoint_tag_molecule_counts": tag_counts, "holdout_quotas": HOLDOUT_TAG_QUOTAS,
               "train_quotas": TRAIN_TAG_QUOTAS, "manifest_sha256": digest_file(manifest_path),
               "ledger_sha256": digest_file(ledger_path), "training_admissible": False,
               "gpu_training_go": False, "production_go": False}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2)); return summary


if __name__ == "__main__":
    run()
