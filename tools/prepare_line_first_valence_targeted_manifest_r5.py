"""Prepare a train-only manifest for the remaining line-first failures.

This is deliberately a metadata/graph inventory.  It does not render samples,
run a model, read the sealed test, or write a checkpoint.  The existing r5
train identities are partitioned into chemically meaningful strata so a future
render can oversample the failure modes seen in the full holdout replay:

* heteroatom valence environments (N/O/S/P/B with non-trivial bond order),
* N/O incident double or triple bonds,
* non-aromatic bridges between separate aromatic components, and
* the weak cov13 raster profiles.

The graph labels are used only to make a data plan; they are not exposed to a
model by this script.  A future GPU shadow must consume a newly rendered,
identity-disjoint train panel and pass the CPU gate first.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path
import sys

from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/datasets/chemvector-identity-aligned-visible-text-20261017-r5-render"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-valence-targeted-manifest-r5"
FAILURE_REPORT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-targeted-failure-strata-r5/report.json"
OWNERSHIP_REPORT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-ownership-roundtrip-r5/report.json"
GUARD_REPORT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-valence-topology-guard-r5/report.json"

sys.path.insert(0, str(ROOT))
from Data.sdf.chemvector_curriculum_v0_2.catalog import entries as catalog_entries

TARGET_PROFILES = ("cov13_base48", "cov13_cap52", "cov13_stroke16_48")
HETERO = {"N", "O", "S", "P", "B", "Si"}
ORDER = {"SINGLE": 1.0, "DOUBLE": 2.0, "TRIPLE": 3.0, "AROMATIC": 1.5,
         "DATIVE": 1.0, "OTHER": 1.0}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_failure_evidence() -> dict:
    """Capture only public train/holdout aggregate evidence, never sealed data."""
    evidence = {}
    for path, name in ((FAILURE_REPORT, "targeted_train_roundtrip"),
                       (OWNERSHIP_REPORT, "full_ownership_roundtrip"),
                       (GUARD_REPORT, "full_valence_topology_guard")):
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        aggregate = report.get("splits", {}).get("holdout", report.get("by_stratum", {}).get("all", {}))
        # The target-free valence/topology probe stores a confidence sweep
        # rather than a single ``splits`` field.  Capture the train-selected
        # reference point so the manifest remains auditable without exposing
        # any sealed content.
        if name == "full_valence_topology_guard" and not aggregate:
            aggregate = report.get("sweeps", {}).get("0.4", {}).get("by_split", {}).get("holdout", {})
        evidence[name] = {
            "path": str(path),
            "sha256": digest(path),
            "sealed_test_read": bool(report.get("sealed_test_read", False)),
            "training_performed": bool(report.get("training_performed", False)),
            "gpu_used": bool(report.get("gpu_used", False)),
            "aggregate": aggregate,
        }
    return evidence


def guard_recommendation() -> dict:
    if not GUARD_REPORT.exists():
        return {"available": False}
    report = json.loads(GUARD_REPORT.read_text())
    sweeps = report.get("sweeps", {})
    # Preserve the pre-guard recall floor while taking the largest sanitizer
    # gain among thresholds whose recall does not drop below .85.
    eligible = []
    for tau, value in sweeps.items():
        hold = value.get("by_split", {}).get("holdout", {})
        if float(hold.get("edge_recall", 0.0)) >= 0.85:
            eligible.append((float(hold.get("oracle_sanitize_rate", 0.0)), float(tau), hold))
    if not eligible:
        return {"available": True, "recommended_tau": None, "reason": "no threshold preserves recall floor"}
    # For equal sanitize rates use the lower confidence threshold; it avoids
    # discarding recoverable edges and is deterministic across Python versions.
    sanitize, tau, hold = max(eligible, key=lambda item: (round(item[0], 6), -item[1]))
    return {
        "available": True,
        "recommended_tau": tau,
        "criterion": "max oracle sanitize subject to full-holdout edge recall >= 0.85",
        "oracle_sanitize_rate": sanitize,
        "edge_recall": float(hold.get("edge_recall", 0.0)),
        "edge_precision": float(hold.get("edge_precision", 0.0)),
        "bond_type_accuracy": float(hold.get("bond_type_accuracy", 0.0)),
    }


def components(nodes: set[int], adjacency: dict[int, list[int]]) -> list[set[int]]:
    left = set(nodes)
    result = []
    while left:
        root = left.pop()
        comp = {root}
        todo = [root]
        while todo:
            x = todo.pop()
            for y in adjacency.get(x, []):
                if y in left:
                    left.remove(y)
                    comp.add(y)
                    todo.append(y)
        result.append(comp)
    return result


def graph_strata(graph: dict) -> tuple[list[str], dict]:
    atoms = graph.get("atoms", [])
    bonds = graph.get("bonds", [])
    atom_by_id = {int(a["atom_id"]): a for a in atoms}
    adjacency: dict[int, list[tuple[int, dict]]] = defaultdict(list)
    aromatic_adj: dict[int, list[int]] = defaultdict(list)
    aromatic_atoms: set[int] = set()
    for atom in atoms:
        if bool(atom.get("aromatic", False)):
            aromatic_atoms.add(int(atom["atom_id"]))
    for bond in bonds:
        a, b = int(bond["begin_atom"]), int(bond["end_atom"])
        typ = str(bond.get("chemical_type", "OTHER")).upper()
        adjacency[a].append((b, bond))
        adjacency[b].append((a, bond))
        if typ == "AROMATIC":
            aromatic_adj[a].append(b)
            aromatic_adj[b].append(a)

    aromatic_components = components(aromatic_atoms, {k: v for k, v in aromatic_adj.items()})
    component_of = {node: i for i, comp in enumerate(aromatic_components) for node in comp}
    # Non-aromatic paths of length two or more joining two aromatic components
    # are the bridge cases which can be confused with an aromatic edge.  Direct
    # aryl-to-aryl single bonds are included as length-one bridges.
    bridge_edges = []
    bridge_paths = []
    for bond in bonds:
        a, b = int(bond["begin_atom"]), int(bond["end_atom"])
        typ = str(bond.get("chemical_type", "OTHER")).upper()
        if typ != "AROMATIC" and a in aromatic_atoms and b in aromatic_atoms:
            bridge_edges.append([a, b, typ])
    # A small BFS through non-aromatic edges catches aryl-O-aryl and similar
    # bridges without treating an entire aromatic ring as a bridge.
    nonaromatic = defaultdict(list)
    for bond in bonds:
        typ = str(bond.get("chemical_type", "OTHER")).upper()
        if typ == "AROMATIC":
            continue
        a, b = int(bond["begin_atom"]), int(bond["end_atom"])
        nonaromatic[a].append(b)
        nonaromatic[b].append(a)
    for start in sorted(aromatic_atoms):
        queue = deque([(start, 0)])
        seen = {start}
        while queue:
            node, depth = queue.popleft()
            if depth >= 1 and node in aromatic_atoms and component_of.get(node) != component_of.get(start):
                bridge_paths.append([start, node, depth])
                continue
            if depth >= 3:
                continue
            for nxt in nonaromatic.get(node, []):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, depth + 1))

    hetero_valence = []
    no_multiple = []
    high_degree = []
    for atom_id, atom in atom_by_id.items():
        element = str(atom.get("element", "OTHER"))
        incident = adjacency.get(atom_id, [])
        weighted = sum(ORDER.get(str(b.get("chemical_type", "OTHER")).upper(), 1.0)
                       for _, b in incident)
        if element in HETERO and (weighted > 2.0 or len(incident) >= 3 or atom.get("formal_charge", 0)):
            hetero_valence.append([atom_id, element, weighted, int(atom.get("formal_charge", 0))])
        if element in {"N", "O"} and any(
            ORDER.get(str(b.get("chemical_type", "OTHER")).upper(), 1.0) >= 2.0
            for _, b in incident
        ):
            no_multiple.append(atom_id)
        if len(incident) >= 3:
            high_degree.append(atom_id)

    tags = []
    if hetero_valence:
        tags.append("heteroatom_valence")
    if no_multiple:
        tags.append("NO_multiple_bond")
    if bridge_edges or bridge_paths:
        tags.append("nonring_aromatic_bridge")
    if high_degree:
        tags.append("junction_degree3plus")
    if any(str(b.get("chemical_type", "OTHER")).upper() in {"DOUBLE", "TRIPLE"} for b in bonds):
        tags.append("multiple_bond")
    return tags, {
        "heteroatom_valence_atoms": hetero_valence,
        "NO_multiple_bond_atoms": no_multiple,
        "aromatic_bridge_edges": bridge_edges,
        "aromatic_bridge_paths": bridge_paths,
        "junction_atoms": high_degree,
        "aromatic_components": [sorted(c) for c in aromatic_components],
        "bond_count": len(bonds),
        "atom_count": len(atoms),
    }


def connection_identity(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid catalog SMILES: {smiles}")
    Chem.RemoveStereochemistry(mol)
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    return "conn-v1-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


def smiles_graph(smiles: str) -> dict:
    """Build the tiny graph view needed by graph_strata, never a render target."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid catalog SMILES: {smiles}")
    type_name = {
        Chem.BondType.SINGLE: "SINGLE",
        Chem.BondType.DOUBLE: "DOUBLE",
        Chem.BondType.TRIPLE: "TRIPLE",
        Chem.BondType.AROMATIC: "AROMATIC",
    }
    return {
        "atoms": [
            {
                "atom_id": a.GetIdx(),
                "element": a.GetSymbol(),
                "aromatic": bool(a.GetIsAromatic()),
                "formal_charge": int(a.GetFormalCharge()),
                "explicit_h_count": int(a.GetNumExplicitHs()),
            }
            for a in mol.GetAtoms()
        ],
        "bonds": [
            {
                "begin_atom": b.GetBeginAtomIdx(),
                "end_atom": b.GetEndAtomIdx(),
                "chemical_type": type_name.get(b.GetBondType(), "OTHER"),
            }
            for b in mol.GetBonds()
        ],
    }


def main() -> dict:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    rows = [json.loads(line) for line in (SOURCE / "samples.jsonl").read_text().splitlines() if line.strip()]
    train = [r for r in rows if r.get("split") == "train" and r.get("profile_id") in TARGET_PROFILES]
    holdout_ids = {r.get("molecule_id") for r in rows if r.get("split") == "holdout"}
    train_ids = {r.get("molecule_id") for r in train}
    if train_ids & holdout_ids:
        raise ValueError("train/holdout molecule overlap")
    # One graph is shared by all profiles; inventory each train molecule once.
    by_molecule = {}
    for row in sorted(train, key=lambda item: item["sample_key"]):
        by_molecule.setdefault(row["molecule_id"], row)
    selected = []
    counts = Counter()
    profile_counts = Counter(r["profile_id"] for r in train)
    for molecule_id, row in sorted(by_molecule.items()):
        graph_path = SOURCE / "samples" / row["sample_key"]
        graph = json.loads(graph_path.with_suffix(".graph.json").read_text())
        tags, detail = graph_strata(graph)
        # Every target molecule is rendered in all three cov13 profiles later;
        # profile is kept as a separate planned axis rather than duplicating IDs.
        item = {
            "molecule_id": molecule_id,
            "canonical_smiles": row.get("canonical_smiles"),
            "connection_identity": row.get("connection_identity"),
            "source_profile": row.get("profile_id"),
            "target_profiles": list(TARGET_PROFILES),
            "tags": tags,
            "detail": detail,
        }
        selected.append(item)
        for tag in tags:
            counts[tag] += 1
    target = {"heteroatom_valence", "NO_multiple_bond", "nonring_aromatic_bridge"}
    coverage = {tag: sum(tag in x["tags"] for x in selected) for tag in sorted(target)}
    planned_images = len(selected) * len(TARGET_PROFILES)

    # The existing r5 train panel contains only three independent aromatic
    # bridges.  Propose a deterministic, identity-disjoint catalog extension;
    # these are metadata rows only and are not admitted to training here.
    source_connection_ids = {
        str(r.get("connection_identity")) for r in rows if r.get("connection_identity")
    }
    candidate_by_id = {}
    for entry in catalog_entries():
        ci = connection_identity(entry["canonical_smiles"])
        if ci in source_connection_ids or ci in candidate_by_id:
            continue
        tags, detail = graph_strata(smiles_graph(entry["canonical_smiles"]))
        required = set(tags) & target
        if not required:
            continue
        candidate_by_id[ci] = {
            "molecule_id": entry["molecule_id"],
            "canonical_smiles": entry["canonical_smiles"],
            "connection_identity": ci,
            "tags": tags,
            "detail": detail,
            "provenance": "deterministic curriculum catalog; not rendered or admitted",
        }
    # Quotas are intentionally bounded so the next CPU gate remains cheap.
    # Bridge gets priority because it is the rarest failure stratum.
    quota = {
        "nonring_aromatic_bridge": 24,
        "NO_multiple_bond": 24,
        "heteroatom_valence": 24,
    }
    extension = []
    used = set()
    for tag in ("nonring_aromatic_bridge", "NO_multiple_bond", "heteroatom_valence"):
        for ci, item in sorted(candidate_by_id.items(), key=lambda pair: pair[1]["molecule_id"]):
            if ci in used or tag not in item["tags"]:
                continue
            extension.append({**item, "target_profiles": list(TARGET_PROFILES), "selection_stratum": tag})
            used.add(ci)
            if sum(x["selection_stratum"] == tag for x in extension) >= quota[tag]:
                break
    extension_counts = Counter(x["selection_stratum"] for x in extension)
    report = {
        "schema_version": "chemvector-identity-aligned-line-first-valence-targeted-manifest-r5-0.1",
        "complete": True,
        "metadata_only": True,
        "status": "quarantine_train_only",
        "training_performed": False,
        "images_rendered": False,
        "gpu_used": False,
        "sealed_test_read": False,
        "production_checkpoint_written": False,
        "source": str(SOURCE),
        "source_summary_sha256": digest(SOURCE / "summary.json"),
        "source_samples_sha256": digest(SOURCE / "samples.jsonl"),
        "failure_evidence": read_failure_evidence(),
        "guard_recommendation": guard_recommendation(),
        "train_images_in_profiles": len(train),
        "train_molecules": len(train_ids),
        "holdout_molecules": len(holdout_ids),
        "molecule_overlap": len(train_ids & holdout_ids),
        "profile_counts": dict(sorted(profile_counts.items())),
        "target_molecule_count": len(selected),
        "planned_images": planned_images,
        "target_strata_counts": dict(sorted(counts.items())),
        "required_failure_strata_coverage": coverage,
        "items": selected,
        "catalog_extension": {
            "identity_disjoint_from_fixed_source": not bool(used & source_connection_ids),
            "candidate_pool": len(candidate_by_id),
            "selected_molecules": len(extension),
            "planned_images": len(extension) * len(TARGET_PROFILES),
            "selection_quota": quota,
            "selection_counts": dict(sorted(extension_counts.items())),
            "items": extension,
            "admitted_to_training": False,
            "rendered": False,
        },
        "future_render_factors": {
            "profiles": list(TARGET_PROFILES),
            "anti_alias_variants": ["baseline", "thin", "high_contrast"],
            "font_family": ["DejaVu Serif"],
            "rotation_degrees": [180.0],
        },
        "next_gpu_shadow_protocol": {
            "must_render_targeted_panel_first": True,
            "max_optimizer_steps": 800,
            "max_wall_minutes": 30,
            "single_gpu": True,
            "frozen": ["production_checkpoint", "text_role_head", "OCR", "bond_head", "production_graph"],
            "trainable": ["line_first_sidecar_only"],
            "early_stop": "fixed full holdout CPU gate no improvement after 200 steps",
            "requires_full_holdout": True,
        },
        "gate": {
            "identity_disjoint": not bool(train_ids & holdout_ids),
            "metadata_only": True,
            "sealed_test_read": False,
            "training_authorized": False,
            "target_strata_nonempty": all(v > 0 for v in coverage.values()),
            "gpu_training_admissible": False,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    # Keep a compact machine-readable index for later renderer adapters.
    (OUT / "items.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in selected))
    (OUT / "catalog_extension.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in extension))
    print(json.dumps({k: report[k] for k in ("target_molecule_count", "planned_images", "target_strata_counts", "required_failure_strata_coverage", "gate")}, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
