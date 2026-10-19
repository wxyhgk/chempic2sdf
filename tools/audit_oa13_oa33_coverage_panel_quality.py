"""Apply the no-silent-drop quality gate to the rendered coverage panel."""
from __future__ import annotations
import hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs/datasets/chemvector-oa13-oa33-coverage-panel-20261017-r2"
OUT = ROOT / "outputs/audits/chemvector-oa13-oa33-coverage-panel-quality-20261017-r1/report.json"
EXPECTED = {"cov13_base":320,"cov13_size24":320,"cov13_rot0":320,"cov13_sans32":320,
            "cov33_base":320,"cov33_size24":320,"cov33_rot180":320,"cov33_serif32":320}

def digest(p: Path) -> str: return hashlib.sha256(p.read_bytes()).hexdigest()

def run():
    if OUT.exists(): raise FileExistsError(OUT)
    summary=json.loads((DATA/"summary.json").read_text()); rej=json.loads((DATA/"rejections.json").read_text())
    accepted={k:int(v) for k,v in summary["accepted_by_profile"].items()}
    complete = summary.get("planned_images") == 2560 and not rej and accepted == EXPECTED
    report={"schema_version":"chemvector-oa13-oa33-coverage-panel-quality-0.1","complete":True,"read_only":True,
            "training_performed":False,"gpu_used":False,"sealed_test_read":False,
            "dataset_summary_sha256":digest(DATA/"summary.json"),"rejections_sha256":digest(DATA/"rejections.json"),
            "planned_images":summary.get("planned_images"),"accepted_images":summary.get("accepted_images"),
            "rejected_images":len(rej),"accepted_by_profile":accepted,"expected_by_profile":EXPECTED,
            "gate":{"complete_profile_denominator":complete,"zero_quality_rejections":not bool(rej),
                    "training_admissible":False,"decision":"PASS" if complete else "NO-GO",
                    "reason":"all cells complete" if complete else "profile denominator/rejection gate failed"}}
    OUT.parent.mkdir(parents=True,exist_ok=False); OUT.write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2)); return report
if __name__ == "__main__": run()
