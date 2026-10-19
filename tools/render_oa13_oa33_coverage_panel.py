"""Render the fixed oa13/oa33 coverage panel into quarantine.

This is a quality-only stage.  It consumes the metadata panel plan, renders
the eight declared profiles with the existing font-capable renderer, and writes
an isolated dataset.  It never loads the model, reads sealed evaluation data,
updates a selector, or authorizes training.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import io, json, os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from PIL import Image

from Data.sdf.case_diversity_v1.rendering import render
from Data.sdf.chemvector_curriculum_v0_1.rendering import Profile
from chemvector.training.curriculum.data import validate_records
from Data.sdf.case_diversity_v1.coverage import visible_features
from Data.sdf.chemvector_curriculum_v0_2.coverage import visible_coverage

ROOT = Path(__file__).resolve().parents[1]
PLAN_DIR = Path(os.environ.get("CHEMVECTOR_PANEL_PLAN_DIR", str(ROOT / "outputs/audits/chemvector-oa13-oa33-coverage-panel-plan-20261017-r3")))
OUT = Path(os.environ.get("CHEMVECTOR_OA13_OA33_RENDER_OUTPUT", str(ROOT / "outputs/datasets/chemvector-oa13-oa33-coverage-panel-20261017-r1")))
PAYLOADS = (".png", ".svg", ".sdf", ".graph.json", ".layers.json", ".text-records.json", ".meta.json", ".semantic.png", ".auxiliary.png", ".text.png", ".bond.png")

def digest_bytes(data: bytes) -> str: return sha256(data).hexdigest()
def digest(path: Path) -> str: return sha256(path.read_bytes()).hexdigest()

def render_one(job):
    row, profile_row = job
    profile = Profile(profile_id=profile_row["profile_id"], width=profile_row.get("width",768), height=profile_row.get("height",768),
                      font_size=profile_row["font_size"], bond_length_cap=profile_row["bond_length_cap"],
                      bond_line_width=profile_row["bond_line_width"], rotation_degrees=profile_row["rotation_degrees"],
                      font_family=profile_row["font_family"])
    entry = {"molecule_id":row["molecule_id"], "canonical_smiles":row["canonical_smiles"], "split":"quarantine", "training_authorized":False, "external_overlap_status":"unchecked"}
    try:
        files, meta = render(entry, profile)
        graph, layers, records = (json.loads(files[s]) for s in (".graph.json",".layers.json",".text-records.json"))
        validate_records(graph, layers, records)
        return row, files, meta, visible_features(layers, records), visible_coverage(layers), None
    except ValueError as exc:
        return row, None, None, None, None, str(exc)

def run() -> dict:
    if OUT.exists(): raise FileExistsError(OUT)
    plan = json.loads((PLAN_DIR / "manifest.json").read_text())
    profile_rows = json.loads((PLAN_DIR / "profiles.json").read_text())
    expected_rows = int(os.environ.get("CHEMVECTOR_PANEL_EXPECTED_ROWS", "2560"))
    if len(plan["rows"]) != expected_rows or not profile_rows: raise ValueError("fixed panel size/profile gate failed")
    profiles = [Profile(profile_id=row["profile_id"], width=row.get("width",768), height=row.get("height",768),
                        font_size=row["font_size"], bond_length_cap=row["bond_length_cap"],
                        bond_line_width=row["bond_line_width"], rotation_degrees=row["rotation_degrees"],
                        font_family=row["font_family"]) for row in profile_rows]
    by_id = {p.profile_id:p for p in profiles}; rows = sorted(plan["rows"], key=lambda r:r["sample_key"])
    max_rows = int(os.environ.get("CHEMVECTOR_PANEL_MAX_ROWS", "0"))
    if max_rows:
        if max_rows < 1 or max_rows > len(rows): raise ValueError("invalid max rows")
        rows = rows[:max_rows]
    OUT.mkdir(parents=True); (OUT / "samples").mkdir()
    accepted=[]; rejected=[]; hashes={}; coverage=Counter()
    workers = int(os.environ.get("CHEMVECTOR_PANEL_WORKERS", "8"))
    if not 1 <= workers <= 16: raise ValueError("invalid workers")
    jobs = [(row, next(p for p in profile_rows if p["profile_id"] == row["profile_id"])) for row in rows]
    with ProcessPoolExecutor(max_workers=workers) as pool:
      for i,(row, files, meta, visible, geometry, error) in enumerate(pool.map(render_one, jobs, chunksize=1),1):
        try:
          if error is not None: raise ValueError(error)
          for suffix,data in files.items():
              path = OUT / "samples" / (row["sample_key"] + suffix)
              path.write_bytes(data); hashes[str(path.relative_to(OUT))] = digest_bytes(data)
          sample = {"sample_key":row["sample_key"], "molecule_id":row["molecule_id"], "profile_id":row["profile_id"], "split":row["split"], "parent_profile_id":row["parent_profile_id"], "changed_factor":row["changed_factor"], "measured":meta["measured"], "png_sha256":meta["sha256"][".png"], "visible":visible, "geometry":geometry}
          accepted.append(sample); coverage[row["profile_id"]] += 1
        except ValueError as exc: rejected.append({**row,"reason":str(exc)})
        if i % 100 == 0 or i == len(rows): print(json.dumps({"processed":i,"accepted":len(accepted),"rejected":len(rejected)}), flush=True)
    for name,expected in hashes.items():
        if digest(OUT/name) != expected: raise RuntimeError("payload hash changed")
    summary={"schema_version":"chemvector-oa13-oa33-coverage-panel-render-summary-0.1","complete":True,"status":"quarantine_cases_only","planned_images":len(rows),"full_panel_planned_images":expected_rows,"smoke_subset":bool(max_rows),"accepted_images":len(accepted),"rejected_images":len(rejected),"accepted_by_profile":dict(sorted(coverage.items())),"rejection_reasons":dict(Counter(r["reason"] for r in rejected)),"distinct_png":len({r["png_sha256"] for r in accepted}),"verified_payload_files":len(hashes),"training_authorized":False,"gpu_used":False,"model_forwards":0,"sealed_test_read":False,"identity_plan_sha256":digest(PLAN_DIR/"manifest.json"),"profiles_sha256":digest(PLAN_DIR/"profiles.json"),"next":"quality audit then one frozen initial replay; no training"}
    (OUT/"samples.json").write_text(json.dumps(accepted,indent=2,ensure_ascii=False)+"\n")
    (OUT/"rejections.json").write_text(json.dumps(rejected,indent=2,ensure_ascii=False)+"\n")
    (OUT/"sample-sha256.json").write_text(json.dumps(hashes,indent=2,ensure_ascii=False)+"\n")
    (OUT/"profiles.json").write_text(json.dumps(profile_rows,indent=2,ensure_ascii=False)+"\n")
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(summary,indent=2,ensure_ascii=False)); return summary

if __name__ == "__main__": run()
