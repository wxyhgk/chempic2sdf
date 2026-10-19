"""Read-only contract/identity audit for the r6 quarantine render."""
from __future__ import annotations
from collections import Counter
import hashlib, json
from pathlib import Path
import sys
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-targeted-diversity-r6/manifest.json"
DATA = ROOT / "outputs/datasets/chemvector-identity-aligned-line-first-targeted-diversity-r6"
SOURCE = ROOT / "outputs/datasets/chemvector-identity-aligned-visible-text-20261017-r5-render"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-targeted-diversity-r6-contract"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
REQUIRED = (".png", ".svg", ".layers.json", ".text-records.json", ".text.png", ".bond.png", ".auxiliary.png", ".semantic.png", ".graph.json", ".sdf", ".meta.json")

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()

def main() -> dict:
    manifest = json.loads(MANIFEST.read_text()); summary = json.loads((DATA / "summary.json").read_text())
    rows = [json.loads(x) for x in (DATA / "samples.jsonl").read_text().splitlines() if x.strip()]
    source = [json.loads(x) for x in (SOURCE / "samples.jsonl").read_text().splitlines() if x.strip()]
    source_ci = {str(x.get("connection_identity")) for x in source if x.get("connection_identity")}
    required = hashes = graphs = sdfs = 0; failures=[]
    for row in rows:
        base = DATA / "samples" / row["sample_key"]
        missing = [s for s in REQUIRED if not base.with_suffix(s).exists()]
        if missing: failures.append({"sample_key":row["sample_key"],"kind":"missing","fields":missing}); continue
        required += 1
        actual = {s: sha(base.with_suffix(s)) for s in REQUIRED}; expected = {s: row.get("sha256",{}).get(s) for s in REQUIRED}
        if actual == expected: hashes += 1
        else: failures.append({"sample_key":row["sample_key"],"kind":"hash"})
        try:
            graph = json.loads(base.with_suffix(".graph.json").read_text())
            if graph.get("molecule_id") == row["molecule_id"] and graph.get("canonical_smiles") == row["canonical_smiles"]: graphs += 1
            else: failures.append({"sample_key":row["sample_key"],"kind":"graph_identity"})
            mol = Chem.MolFromMolFile(str(base.with_suffix(".sdf")), sanitize=True, removeHs=False, strictParsing=True)
            ref = Chem.MolFromSmiles(row["canonical_smiles"])
            if mol is not None and ref is not None and Chem.MolToSmiles(mol) == Chem.MolToSmiles(ref): sdfs += 1
            else: failures.append({"sample_key":row["sample_key"],"kind":"sdf_identity"})
        except Exception as exc: failures.append({"sample_key":row["sample_key"],"kind":"parse","error":str(exc)})
    split_counts = Counter(x["split"] for x in rows); profile_counts = Counter(x["profile_id"] for x in rows); cat_counts = Counter(x["category"] for x in rows)
    ci = {x.get("connection_identity") for x in rows}; keys = {x.get("sample_key") for x in rows}
    expected_profiles = {x["profile_id"] for x in manifest["styles"]}
    report = {"schema_version":"chemvector-identity-aligned-line-first-targeted-diversity-r6-contract-0.1","complete":True,"read_only":True,"training_performed":False,"gpu_used":False,"sealed_test_read":False,"production_checkpoint_written":False,"initial_sha256":INITIAL_SHA,"manifest_sha256":sha(MANIFEST),"rows":len(rows),"unique_sample_keys":len(keys),"unique_connection_identities":len(ci),"split_counts":dict(split_counts),"profile_counts":dict(profile_counts),"category_counts":dict(cat_counts),"required_files_ok":required,"hash_rows_ok":hashes,"graph_identity_ok":graphs,"sdf_identity_ok":sdfs,"source_connection_overlap":len(ci & source_ci),"failure_count":len(failures),"failures":failures[:100],"gate":{"identity_disjoint":not bool(ci & source_ci),"sample_keys_unique":len(keys)==len(rows),"connection_identities_split_disjoint":len({x["connection_identity"] for x in rows if x["split"]=="train"}&{x["connection_identity"] for x in rows if x["split"]!="train"})==0,"style_complete":set(profile_counts)==expected_profiles,"required_files_complete":required==len(rows),"hashes_complete":hashes==len(rows),"graph_identity_complete":graphs==len(rows),"sdf_identity_complete":sdfs==len(rows),"sealed_test_read":False,"training_authorized":False,"endpoint_ceiling_admissible":False}}
    OUT.mkdir(parents=True,exist_ok=True); (OUT/"report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n"); print(json.dumps({"output":str(OUT/"report.json"),"rows":len(rows),"failures":len(failures),"gate":report["gate"]},indent=2,ensure_ascii=False)); return report

if __name__ == "__main__": main()
