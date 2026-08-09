#!/usr/bin/env python3
"""Resumable state machine for the gated P6 residual long cycle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from chemvector.evaluation.audits.atoms import build_g3_p6e8_residual_cycle_data as p6e8
from chemvector.evaluation.audits.atoms import split_g3_p6e8_residual_caches as split
from chemvector.training.atoms import train_g3_p6e9_structural_residual_arms as p6e9
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import _atomic_json
from chemvector.g3_structural_data import file_sha256


OUTPUT=Path("outputs/checkpoints/chemvector-g3-p6-residual-long-cycle")
STATE=OUTPUT/"cycle_state.json"
STAGES=(
    ("p6e8_data","chemvector.evaluation.audits.atoms.build_g3_p6e8_residual_cycle_data",p6e8.OUTPUT/"audit.json"),
    ("p6e8_split","chemvector.evaluation.audits.atoms.split_g3_p6e8_residual_caches",split.AUDIT),
    ("p6e9_arms","chemvector.training.atoms.train_g3_p6e9_structural_residual_arms",p6e9.OUTPUT/"audit.json"),
)


def cycle_decision(p6e8_audit:dict|None,p6e9_audit:dict|None)->str:
    if p6e8_audit is None: return "run_p6e8_data"
    if p6e8_audit.get("decision")!="residual_cycle_data_go": return "stopped_p6e8_data_no_go"
    if p6e9_audit is None: return "run_p6e9_arms"
    if p6e9_audit.get("decision")!="residual_calibration_winner_selected": return "stopped_p6e9_no_safe_winner"
    return "ready_for_p6e10_holdout"


def _load(path:Path)->dict|None:
    return json.loads(path.read_text()) if path.is_file() else None


def _write_state(status:str,stages:list[dict])->None:
    OUTPUT.mkdir(parents=True,exist_ok=True); _atomic_json(STATE,{"schema_version":"g3-p6-residual-long-cycle-state-0.1","status":status,"updated_unix":time.time(),"stages":stages,"holdout_read":False,"dev_confirm_external_sealed_read":False})


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--audit-existing",action="store_true"); args=parser.parse_args()
    stage_rows=[]
    for name,module,audit_path in STAGES:
        before=cycle_decision(_load(p6e8.OUTPUT/"audit.json"),_load(p6e9.OUTPUT/"audit.json"))
        if name!="p6e8_data" and before=="stopped_p6e8_data_no_go": break
        if not audit_path.is_file() and not args.audit_existing:
            subprocess.run([sys.executable,"-m",module],check=True)
        payload=_load(audit_path)
        stage_rows.append({"name":name,"audit":str(audit_path.resolve()),"exists":payload is not None,"sha256":file_sha256(audit_path) if payload else None,"decision":payload.get("decision",payload.get("status")) if payload else None})
        status=cycle_decision(_load(p6e8.OUTPUT/"audit.json"),_load(p6e9.OUTPUT/"audit.json"))
        _write_state(status,stage_rows)
    _write_state(cycle_decision(_load(p6e8.OUTPUT/"audit.json"),_load(p6e9.OUTPUT/"audit.json")),stage_rows)
    print(json.dumps(json.loads(STATE.read_text()),indent=2))


if __name__=="__main__": main()
