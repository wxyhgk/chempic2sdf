"""Render the r6 quarantine manifest; no model inference or training."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from Data.sdf.chemvector_curriculum_v0_1.rendering import Profile, render
from Data.sdf.chemvector_curriculum_v0_2.catalog import entries

MANIFEST = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-targeted-diversity-r6/manifest.json"
OUT = ROOT / "outputs/datasets/chemvector-identity-aligned-line-first-targeted-diversity-r6"


def job(payload):
    row, entry, out_root = payload
    profile = Profile(profile_id=row["profile_id"], font_family=row["font_family"], font_size=int(row["font_size"]),
                      bond_length_cap=float(row["bond_length_cap"]), bond_line_width=float(row["bond_line_width"]),
                      rotation_degrees=float(row["rotation_degrees"]))
    sample_key = row["sample_key"]
    result = {**row, "render_status": "accepted", "sha256": {}}
    try:
        files, meta = render({"molecule_id": entry["molecule_id"], "canonical_smiles": entry["canonical_smiles"]}, profile, typography="stereo-aware-v1")
    except ValueError as exc:
        result["render_status"] = "quality_rejected"; result["reason"] = str(exc); return result
    result["measured"] = meta["measured"]
    sample_dir = Path(out_root) / "samples"; sample_dir.mkdir(parents=True, exist_ok=True)
    for suffix, data in files.items():
        path = sample_dir / (sample_key + suffix); path.write_bytes(data); result["sha256"][suffix] = hashlib.sha256(data).hexdigest()
    return result


def run(workers: int = 16):
    if OUT.exists(): raise FileExistsError(OUT)
    manifest = json.loads(MANIFEST.read_text())
    if manifest.get("status") != "quarantine_plan_only" or not manifest["gate"]["quota_complete"]:
        raise ValueError("r6 manifest is not admissible for rendering")
    catalog = {x["molecule_id"]: x for x in entries()}
    rows = manifest["rows"]
    payloads = []
    for row in rows:
        entry = catalog.get(row["molecule_id"])
        if entry is None or entry["canonical_smiles"] != row["canonical_smiles"]: raise ValueError("catalog mismatch")
        payloads.append((row, entry, str(OUT)))
    OUT.mkdir(parents=True); accepted=[]; rejected=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(job,p) for p in payloads]
        for i,f in enumerate(as_completed(futures),1):
            row=f.result(); (accepted if row["render_status"] == "accepted" else rejected).append(row)
            if i % 160 == 0: print(json.dumps({"completed":i,"planned":len(payloads),"accepted":len(accepted),"rejected":len(rejected)}),flush=True)
    accepted.sort(key=lambda x:x["sample_key"]); rejected.sort(key=lambda x:x["sample_key"])
    hashes={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (OUT/"samples").iterdir() if p.is_file()}
    summary={"schema_version":"chemvector-identity-aligned-line-first-targeted-diversity-r6-render-0.1","complete":True,"status":"quarantine_train_dev_fresh_holdout","training_authorized":False,"training_performed":False,"gpu_used":False,"model_forwards":0,"sealed_test_read":False,"production_checkpoint_written":False,"manifest_sha256":hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),"planned_images":len(payloads),"accepted_images":len(accepted),"rejected_images":len(rejected),"accepted_by_split":dict(Counter(x["split"] for x in accepted)),"accepted_by_profile":dict(Counter(x["profile_id"] for x in accepted)),"accepted_by_category":dict(Counter(x["category"] for x in accepted)),"rejection_reasons":dict(Counter(x.get("reason") for x in rejected)),"verified_sample_file_hashes":len(hashes),"next":"run contract/graph-SDF identity and endpoint ceiling audits; no training admission"}
    (OUT/"samples.jsonl").write_text("".join(json.dumps(x,sort_keys=True,ensure_ascii=False)+"\n" for x in accepted)); (OUT/"rejected.jsonl").write_text("".join(json.dumps(x,sort_keys=True,ensure_ascii=False)+"\n" for x in rejected)); (OUT/"sample-sha256.json").write_text(json.dumps(hashes,indent=2,sort_keys=True)+"\n"); (OUT/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(summary,indent=2,ensure_ascii=False)); return summary


if __name__ == "__main__": run()
