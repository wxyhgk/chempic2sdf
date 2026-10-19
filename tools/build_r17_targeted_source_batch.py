#!/usr/bin/env python3
"""Build a targeted, metadata-only r17 source batch.

This is deliberately separate from ``build_r17_new_source_batch.py``.  The
older builder selects candidates by a hash order and therefore cannot promise
coverage of rare visible O/N bond strata.  This builder renders a deterministic
candidate pool first, measures the resulting graph/layers, and greedily
reserves molecule-disjoint holdout rows for explicit strata quotas.  It never
reads model outputs, sealed evaluation payloads, or writes a checkpoint.

The generated source remains quarantine-only.  Run the prior-use ledger audit
before treating it as admissible for any training experiment.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import sys
from typing import Iterable

from PIL import Image
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.r17_manifest_rows import rows_from_payload

from Data.sdf.case_diversity_v1.profiles import profiles
from Data.sdf.case_diversity_v1.rendering import render

RDLogger.DisableLog("rdApp.*")


OUT = ROOT / os.environ.get(
    "R17_TARGETED_SOURCE_OUT",
    "outputs/datasets/chemvector-r17-targeted-source-20261008-t1",
)
SOURCE_ID = os.environ.get("R17_TARGETED_SOURCE_ID", "chemvector-r17-targeted-visible-source")
SOURCE_VERSION = os.environ.get("R17_TARGETED_SOURCE_VERSION", "20261008.t1")
SOURCE_SEED = os.environ.get("R17_TARGETED_SOURCE_SEED", "r17-targeted-visible-20261008-seed-a")
PROFILE_IDS = tuple(
    p.strip()
    for p in os.environ.get(
        "R17_TARGETED_SOURCE_PROFILES",
        "oa00,oa02,oa11,oa13,oa20,oa22,oa31,oa33",
    ).split(",")
    if p.strip()
)
TRAIN_COUNT = int(os.environ.get("R17_TARGETED_SOURCE_TRAIN", "48"))
HOLDOUT_COUNT = int(os.environ.get("R17_TARGETED_SOURCE_HOLDOUT", "24"))

# Quotas are molecule counts, not bond-incidence counts.  This prevents a
# single molecule with several oxygen atoms from making the holdout appear
# broad when it is not.  Values can be lowered for a bounded smoke build, but
# a production-candidate run should retain the defaults.
DEFAULT_QUOTAS = {
    "O:AROMATIC": 8,
    "N:TRIPLE": 8,
    "N:AROMATIC": 8,
    "N:DOUBLE": 8,
    "O:DOUBLE": 8,
    "O:SINGLE": 8,
    "GRAPH:TRIPLE": 8,
}


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


def targeted_smiles() -> Iterable[str]:
    """Yield intentionally varied candidates for rare visible O/N strata.

    The list includes acyclic nitrile/amide/carboxyl examples (which are
    generally new by full identity) and larger fused O/N aromatic systems.
    Candidates are still subjected to the full historical identity scan below;
    this list is not an identity exemption.
    """
    # Nitrile families: N:TRIPLE and GRAPH:TRIPLE, with O/N context variants.
    nitrile_roots = (
        "C", "CC", "CCC", "CCCC", "CO", "CCO", "CCCO", "CN", "CCN",
        "CNC", "CCNC", "C(=O)O", "C(=O)N", "C(=O)OC", "C(=O)NC",
        "CS", "CCS", "CCl", "CBr", "C1CC1", "C1CCC1", "C1CCO1",
        "C1CCN1", "C1CCOC1", "C1CCNC1",
    )
    for root in nitrile_roots:
        yield f"N#C{root}"
    yield from ("N#CN", "N#CC#N", "N#CC(=N)N", "N#CC(=O)N", "N#CC(=O)O")

    # Carbonyl/sulfur/phosphorus families: visible O double/single and N single.
    carbonyl_roots = (
        "C", "CC", "CCC", "CCCC", "CO", "CCO", "CCCO", "CN", "CCN",
        "CNC", "CCNC", "CS", "CCS", "CCl", "CBr", "C1CC1", "C1CCC1",
        "C1CCO1", "C1CCN1", "C1CCOC1", "C1CCNC1", "C1CCS1",
    )
    for root in carbonyl_roots:
        yield f"CC(=O){root}"
        yield f"O=C(O){root}"
        yield f"O=P(O)(O){root}"
    yield from (
        "CS(=O)(=O)C", "CS(=O)(=O)N", "CS(=O)(=O)O", "O=C1CCOCCN1",
        "O=C1CCSCCN1", "O=C1CCOCCO1", "O=C1CCNCCN1", "O=C1NCCOCC1",
        "O=C1NCCSCC1", "O=C1OC2=CC=CC=C2C1", "O=C1Oc2ccccc2C1=O",
    )

    # Aromatic O/N: use fused systems absent from the old small heterocycle
    # pool.  Substitution changes full identity while preserving the visible
    # heteroatom bond context; the historical scaffold gate still applies.
    aromatic = (
        "c1ccc2c(c1)oc1ccccc12",       # dibenzofuran
        "c1cc2cc3ccccc3oc-2c1",        # larger fused O aromatic system
        "c1cc2oc3ccccc3cc2c1",
        "c1cc2cc3occc3cc2c1",
        "c1ccc2c(c1)[nH]c1ccccc12",    # carbazole
        "c1ccc2c(c1)cnc1ccccc12",      # acridine-like N aromatic
        "c1cc2[nH]cc3ccccc3c-2c1",
        "c1cc2ncc3ccccc3c2c1",
        "c1cc2nccn3ccccc23",
        "c1cc2ocnc3ccccc23",
        # Larger N-fused systems used by the optional train-side quota.  They
        # are intentionally distinct scaffolds from the historical inventory.
        "c1ccc2nc3ccccc3nc2c1",
        "c1ccc2cc3ncccc3cc2c1",
        "c1ccc2cc3nccnc3cc2c1",
        "c1ccc2cc3ncc(C)cc3cc2c1",
        "c1ccc2cc3ncc(C#N)cc3cc2c1",
        "c1ccc2cc3ncc(C(=O)O)cc3cc2c1",
    )
    substituents = ("", "C", "CC", "Cl", "F", "C#N", "C(=O)O", "C(=O)N")
    for base in aromatic:
        yield base
        # Add a small deterministic set of explicit substituent variants.
        # Some bases cannot be safely substituted by this textual operation;
        # they are simply discarded by canonical()/render() below.
        for suffix in substituents[1:]:
            yield f"{base}{suffix}"

    # Distinct O-aromatic/N-aromatic heterocycles with nitrile or carbonyl
    # substituents.  These intentionally overlap quotas to keep holdout small.
    yield from (
        "c1cc2occc2c1C#N", "c1cc2ncc(C#N)cc2c1", "c1cc2occc2c1C(=O)O",
        "c1cc2ncc(C(=O)O)cc2c1", "c1ccc2c(c1)oc1ccccc12C#N",
        "c1ccc2c(c1)cnc1ccccc12C#N", "O=C(O)c1cc2occc2c1",
        "O=C(O)c1cc2nccc2c1", "N#Cc1cc2occc2c1", "N#Cc1cc2nccc2c1",
    )

    # Additional, valid polycyclic heteroaromatics.  These are deliberately
    # separate scaffolds rather than more copies of the old benzofuran/
    # quinoline families; they increase independent-molecule support for the
    # train-side O/N-aromatic quotas.  Substitution is not appended blindly
    # because fused-ring SMILES need an explicit attachment point.
    yield from (
        "c1ccc2cc3c(cc2c1)oc1ccccc13",
        "c1ccc2cc3c(cc2c1)oc1ccc2ccccc2c13",
        "c1ccc2cc3c(cc2c1)oc1cc2ccccc2cc13",
        "c1ccc2cc3ccnc4ccccc4c3c2c1",
        "c1ccc2cc3ncc4ccccc4c3cc2c1",
        "c1ccc2cc3c(cc2c1)n1cccc1-3",
    )

    # Fresh O-aromatic scaffolds retained after the historical identity scan.
    # Keep these as standalone canonical candidates; the renderer decides
    # whether every style profile can represent them.
    yield from (
        "C1=CC=c2nnc(o2)=C1", "c1cc2cc3cccnc3oc-2c1",
        "c1cc2cc3ccncc3oc-2c1", "c1cc2cc3ccnnc3oc-2c1",
        "c1cc2cc3cnccc3oc-2c1", "c1cc2cc3ncccc3oc-2c1",
        "c1cc2cc3nccnc3oc-2c1", "c1cc2cc3occc3oc-2c1",
        "c1cc2ccoc2cn1", "c1cc2occc2cn1", "c1cc2occc2o1",
        "c1ccc2c(c1)oc1cccnc12", "c1ccc2c(c1)oc1ccncc12",
        "c1ccc2c(c1)oc1cnccc12", "c1ccc2c(c1)oc1ncccc12",
        "c1cnc2occc2c1", "c1cnc2occc2n1", "c1nnco1",
    )

    # Acyclic nitrile buffer: these add independent N-triple and graph-triple
    # molecules without relying on one repeated nitrile scaffold.
    yield from (
        "N#CC(F)(F)F", "N#CC(Cl)(Cl)Cl", "N#CC(Br)(Br)Br",
        "CC(C)(O)C#N", "CC(C)(N)C#N", "CCOCCC#N", "CN(C)CC#N",
        "CCNCC#N", "CCSCC#N", "CCNC(=O)C#N", "CN(C)C(=O)C#N",
        "N#CC(N)=S", "N#CC#CC#N", "N#CC#CC(=O)O", "N#CC#CC(N)=O",
        "N#CC#CCO",
    )

    # Additional simple O-single examples for train-side support.
    yield from (
        "FC(F)(F)CO", "CCOCC#N", "CCOCC(=O)N", "CCOCC(=O)O",
        "CCOCCS", "CCOC(=O)CBr", "CCOC(=O)CCl", "CCOC(=O)C#N",
        "CCOC(=O)C(F)F", "CC(C)OC#N", "CC(C)OCC#N",
    )

    # N-double buffer.  t1 had almost no train support and no holdout support
    # for this stratum, so it must be explicit in both split quotas.
    yield from (
        "N=C=O", "N=C=S", "CC=N", "CCC=N", "N=CCO", "N=CCN",
        "N=CCCl", "N=CCBr", "N=C(N)N", "N=C(N)O", "N=C(N)S",
        "N=C(Cl)Cl", "N=C(Br)Br", "N#CC(=N)O", "N=C(N)C(=O)O",
        "N=C(N)C(N)=O", "CCOC=N", "CCN=C=O", "CCN=C=S", "CC=NCC",
        "COC=N", "COC(=N)N", "N=c1cccccn1", "N=c1nccccn1",
        "C1=CC=NNC=C1", "N=c1nccco1", "N=c1nccc[nH]1",
        "N=c1nccncn1", "N=C1C=NC2=CC=CC2=C1",
    )


def historical_identity_values(candidate_out: Path) -> dict[str, set[str]]:
    fields = ("molecule_id", "canonical_smiles", "connection_identity", "legacy_scaffold")
    seen = {field: set() for field in fields}
    paths = list((ROOT / "outputs/audits").glob("**/manifest.json"))
    paths += list((ROOT / "outputs/datasets").glob("**/molecules.json"))
    paths += list((ROOT / "outputs/datasets").glob("**/manifest.json"))
    for path in sorted(set(paths)):
        if candidate_out in path.parents:
            continue
        if any(part in str(path).lower() for part in ("sealed", "internal-test", "test996")):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = rows_from_payload(payload)
        except Exception:
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            aliases = {
                "connection_identity": ("connection_identity", "connection_identity_without_isotope_stereo"),
                "legacy_scaffold": ("legacy_scaffold", "scaffold", "scaffold_identity"),
            }
            for field in fields:
                names = aliases.get(field, (field,))
                value = next((row.get(name) for name in names if row.get(name)), None)
                if value is not None and not isinstance(value, (dict, list)) and str(value).strip():
                    seen[field].add(str(value).strip())
    return seen


def quotas() -> dict[str, int]:
    result = dict(DEFAULT_QUOTAS)
    for key in result:
        env_key = "R17_TARGETED_QUOTA_" + key.replace(":", "_")
        if env_key in os.environ:
            result[key] = int(os.environ[env_key])
    return result


def train_min_quotas() -> dict[str, int]:
    """Optional training-side molecule quotas (default zero for t1).

    A later source can reserve learning examples by setting, for example,
    ``R17_TARGETED_TRAIN_MIN_N_AROMATIC=4``.  Keeping this opt-in avoids
    silently changing the already-audited t1 split.
    """
    result = {key: 0 for key in DEFAULT_QUOTAS}
    for key in result:
        env_key = "R17_TARGETED_TRAIN_MIN_" + key.replace(":", "_")
        if env_key in os.environ:
            result[key] = int(os.environ[env_key])
    return result


def molecule_strata(layers: dict, graph: dict) -> set[str]:
    atoms = {int(a["atom_id"]): a for a in layers.get("atoms", [])}
    visible = {
        i for i, atom in atoms.items()
        if atom.get("visible_text") and atom.get("element") in {"O", "N"}
    }
    out: set[str] = set()
    for bond in graph.get("bonds", []):
        begin, end = int(bond["begin_atom"]), int(bond["end_atom"])
        typ = str(bond.get("chemical_type", "SINGLE")).upper()
        if typ == "TRIPLE":
            out.add("GRAPH:TRIPLE")
        for atom_id in (begin, end):
            if atom_id not in visible:
                continue
            element = str(atoms[atom_id].get("element"))
            if element in {"O", "N"}:
                out.add(f"{element}:{typ}")
    return out


def build_candidates() -> list[dict]:
    history = historical_identity_values(OUT)
    candidates: dict[str, dict] = {}
    for smiles in targeted_smiles():
        item = canonical(smiles)
        if item is None:
            continue
        can, mol = item
        molecule_id = "newr17t_" + sha256((SOURCE_ID + "|" + SOURCE_VERSION + "|" + can).encode()).hexdigest()[:16]
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
    # Hash order is stable but independent of input ordering.  The actual
    # holdout choice below is quota-greedy over rendered strata.
    return sorted(candidates.values(), key=lambda row: sha256((SOURCE_SEED + row["molecule_id"]).encode()).hexdigest())


def render_pool(candidates: list[dict], selected_profiles: dict[str, object]) -> list[dict]:
    """Render all profiles; retain only molecules with a complete profile tuple."""
    rendered: list[dict] = []
    for molecule in candidates:
        staged = []
        strata: set[str] | None = None
        try:
            for profile_id in PROFILE_IDS:
                entry = {
                    **molecule,
                    "split": "quarantine",
                    "source_id": SOURCE_ID,
                    "source_version": SOURCE_VERSION,
                    "source_seed": SOURCE_SEED,
                    "training_authorized": False,
                    "external_overlap_status": "unchecked",
                    "source_record_ids": [SOURCE_ID],
                    "stages": ["r17_targeted_source"],
                }
                files, _meta = render(entry, selected_profiles[profile_id])
                layers = json.loads(files[".layers.json"])
                graph = json.loads(files[".graph.json"])
                current = molecule_strata(layers, graph)
                strata = current if strata is None else strata
                if current != strata:
                    raise ValueError("profile chemistry strata changed")
                # ``render`` derives sample_key/profile_id in its metadata;
                # keep the explicit key only in this staging wrapper so it
                # is not passed twice through ``dict(..., sample_key=...)``.
                staged.append(({**entry, "sample_key": f"{molecule['molecule_id']}__{profile_id}", "profile_id": profile_id}, files))
        except (ValueError, RuntimeError, KeyError):
            continue
        # Store files in memory until deterministic split assignment is done.
        rendered.append({"molecule": molecule, "strata": sorted(strata or ()), "files": staged})
    return rendered


def choose_split(
    pool: list[dict], target_train: int, target_holdout: int,
    target_quotas: dict[str, int], train_quotas: dict[str, int] | None = None,
) -> tuple[list[dict], list[dict]]:
    if len(pool) < target_train + target_holdout:
        raise RuntimeError(f"only {len(pool)} complete molecules; need {target_train + target_holdout}")
    by_id = {item["molecule"]["molecule_id"]: item for item in pool}
    remaining = list(pool)
    holdout: list[dict] = []
    covered = Counter()
    # Greedy max-unmet-quota selection with deterministic hash tie-break order.
    while len(holdout) < target_holdout and any(covered[k] < v for k, v in target_quotas.items()):
        scored = []
        for item in remaining:
            gain = sum(1 for key, goal in target_quotas.items() if covered[key] < goal and key in item["strata"])
            if gain:
                scored.append((gain, item))
        if not scored:
            break
        _, chosen = max(scored, key=lambda pair: (pair[0], -remaining.index(pair[1])))
        holdout.append(chosen)
        remaining.remove(chosen)
        for key in chosen["strata"]:
            covered[key] += 1
    if any(covered[k] < v for k, v in target_quotas.items()):
        missing = {k: v - covered[k] for k, v in target_quotas.items() if covered[k] < v}
        raise RuntimeError(f"rendered pool cannot satisfy holdout strata quotas: {missing}")
    # Fill holdout then train from the same deterministic pool.  The source
    # identity is molecule-disjoint by construction.
    holdout.extend(remaining[: target_holdout - len(holdout)])
    holdout_ids = {item["molecule"]["molecule_id"] for item in holdout}
    train_pool = [item for item in remaining if item["molecule"]["molecule_id"] not in holdout_ids]
    train_quotas = train_quotas or {}
    train: list[dict] = []
    train_covered = Counter()
    # Optional training-side reservation.  It fails loudly if holdout
    # allocation consumed all candidates for a requested training stratum.
    while len(train) < target_train and any(train_covered[k] < v for k, v in train_quotas.items()):
        scored = []
        for item in train_pool:
            if item in train:
                continue
            gain = sum(1 for key, goal in train_quotas.items() if train_covered[key] < goal and key in item["strata"])
            if gain:
                scored.append((gain, item))
        if not scored:
            missing = {k: v - train_covered[k] for k, v in train_quotas.items() if train_covered[k] < v}
            raise RuntimeError(f"rendered pool cannot satisfy train strata quotas after holdout allocation: {missing}")
        _, chosen = max(scored, key=lambda pair: (pair[0], -train_pool.index(pair[1])))
        train.append(chosen)
        for key in chosen["strata"]:
            train_covered[key] += 1
    train.extend(item for item in train_pool if item not in train)
    train = train[:target_train]
    if len(train) != target_train:
        raise RuntimeError("not enough non-holdout molecules for train split")
    return train, holdout


def run() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    if not PROFILE_IDS or len(set(PROFILE_IDS)) != len(PROFILE_IDS):
        raise ValueError("profiles must be non-empty and unique")
    selected_profiles = {p.profile_id: p for p in profiles() if p.profile_id in PROFILE_IDS}
    if set(selected_profiles) != set(PROFILE_IDS):
        raise ValueError(f"unknown profile(s): {sorted(set(PROFILE_IDS) - set(selected_profiles))}")
    target_quotas = quotas()
    if HOLDOUT_COUNT < max(target_quotas.values(), default=0):
        raise ValueError("holdout count must be at least the largest molecule quota")
    target_train_quotas = train_min_quotas()
    candidates = build_candidates()
    pool = render_pool(candidates, selected_profiles)
    train, holdout = choose_split(pool, TRAIN_COUNT, HOLDOUT_COUNT, target_quotas, target_train_quotas)

    OUT.mkdir(parents=True)
    source_dir = OUT / "samples"
    source_dir.mkdir()
    rows: list[dict] = []
    ledger: list[dict] = []
    split_items = [("train", train), ("holdout", holdout)]
    for split, items in split_items:
        for item in items:
            for entry, files in item["files"]:
                key = entry["sample_key"]
                for suffix, data in files.items():
                    (source_dir / f"{key}{suffix}").write_bytes(data)
                png = files[".png"]
                with Image.open(io.BytesIO(png)) as image:
                    rgb_sha = digest_bytes(image.convert("RGB").tobytes())
                public = {
                    **{k: v for k, v in entry.items() if k not in {"training_authorized", "external_overlap_status", "source_record_ids", "stages"}},
                    "split": split,
                    "training_authorized": False,
                    "external_overlap_status": "unchecked",
                    "image_sha256": digest_bytes(png),
                    "png_sha256": digest_bytes(png),
                    "rgb_sha256": rgb_sha,
                }
                rows.append(public)
                ledger.append({**public, "event": "reserved_before_model_use", "source_namespace": SOURCE_ID})

    manifest = {
        "schema_version": "chemvector-r17-targeted-source-manifest-0.1",
        "source_id": SOURCE_ID,
        "source_version": SOURCE_VERSION,
        "source_seed": SOURCE_SEED,
        "metadata_only": True,
        "training_authorized": False,
        "sealed_test_read": False,
        "targeted_holdout_quotas": target_quotas,
        "rows": rows,
    }
    manifest_path = OUT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ledger_path = OUT / "prior-use-ledger.jsonl"
    with ledger_path.open("w", encoding="utf-8") as stream:
        for row in ledger:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    holdout_strata = Counter()
    train_strata = Counter()
    for item in holdout:
        holdout_strata.update(item["strata"])
    for item in train:
        train_strata.update(item["strata"])
    summary = {
        "schema_version": "chemvector-r17-targeted-source-build-0.1",
        "complete": True,
        "read_only": True,
        "training_performed": False,
        "gpu_used": False,
        "sealed_test_read": False,
        "source_id": SOURCE_ID,
        "source_version": SOURCE_VERSION,
        "source_seed": SOURCE_SEED,
        "rows": len(rows),
        "molecules": len({row["molecule_id"] for row in rows}),
        "profile_counts": dict(Counter(row["profile_id"] for row in rows)),
        "split_molecules": {split: len({row["molecule_id"] for row in rows if row["split"] == split}) for split in ("train", "holdout")},
        "candidate_molecules": len(candidates),
        "render_complete_molecules": len(pool),
        "holdout_strata_molecules": dict(sorted(holdout_strata.items())),
        "train_strata_molecules": dict(sorted(train_strata.items())),
        "targeted_holdout_quotas": target_quotas,
        "targeted_train_min_quotas": target_train_quotas,
        "holdout_quota_gate": all(holdout_strata[key] >= goal for key, goal in target_quotas.items()),
        "train_min_quota_gate": all(train_strata[key] >= goal for key, goal in target_train_quotas.items()),
        "manifest_sha256": digest_file(manifest_path),
        "ledger_sha256": digest_file(ledger_path),
        "training_admissible": False,
        "gpu_training_go": False,
        "production_go": False,
        "next": "run metadata-only prior-use ledger admission and strata audit before any model training",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
