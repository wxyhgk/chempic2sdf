#!/usr/bin/env python3
"""Read-only contract audit for the text/glyph + line-endpoint seam."""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib, json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"outputs/datasets/chemvector-identity-aligned-visible-text-20261017-r5-render"
OUT=ROOT/"outputs/audits/chemvector-identity-aligned-visible-text-seam-contract-r5/report.json"
SHA="09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def one(row):
 key=row["sample_key"];base=DATA/"samples"/key
 layers=json.loads(base.with_suffix(".layers.json").read_text()); rec=json.loads(base.with_suffix(".text-records.json").read_text()); graph=json.loads(base.with_suffix(".graph.json").read_text())
 mask=np.asarray(Image.open(base.with_suffix(".text.png")).convert("L"))>0; h,w=mask.shape; union=np.zeros_like(mask); boxes=[]; valid=True; overlap=0
 for g in rec.get("glyphs",[]):
  b=g.get("bbox"); valid &= isinstance(b,list) and len(b)==4 and all(isinstance(v,int) for v in b) and 0<=b[0]<b[2]<=w and 0<=b[1]<b[3]<=h
  if valid:
   for old in boxes:
    overlap += int(min(b[2],old[2])>max(b[0],old[0]) and min(b[3],old[3])>max(b[1],old[1]))
   boxes.append(b); union[b[1]:b[3],b[0]:b[2]]=True
 atoms={int(a["atom_id"]) for a in layers["atoms"]}; visible={int(a["atom_id"]) for a in layers["atoms"] if a.get("visible_text")}; glyph_atoms={int(g["atom_id"]) for g in rec.get("glyphs",[])}; endpoints={int(x) for b in layers.get("bonds",[]) for x in (b["begin_atom"],b["end_atom"])}; graph_bonds={(int(b["begin_atom"]),int(b["end_atom"])) for b in graph.get("bonds",[])}; layer_bonds={(int(b["begin_atom"]),int(b["end_atom"])) for b in layers.get("bonds",[])}
 return {"sample_key":key,"split":row["split"],"profile_id":row["profile_id"],"atoms":len(atoms),"visible_atoms":len(visible),"glyph_atoms":len(glyph_atoms),"endpoint_atoms":len(endpoints),"union_atoms":len(glyph_atoms|endpoints),"bonds":len(graph_bonds),"line_endpoint_records":len(layer_bonds),"glyph_bbox_valid":bool(valid),"glyph_bbox_overlap_count":overlap,"text_pixels":int(mask.sum()),"text_pixels_outside_glyph_boxes":int((mask&~union).sum()),"visible_glyph_ceiling":len(glyph_atoms&visible)/max(len(visible),1),"line_endpoint_ceiling":len(endpoints&atoms)/max(len(atoms),1),"union_atom_ceiling":len((glyph_atoms|endpoints)&atoms)/max(len(atoms),1),"bond_geometry_ceiling":len(graph_bonds&layer_bonds)/max(len(graph_bonds),1)}

def summary(vals):
 n=len(vals)
 return {"images":n,"atoms":sum(v["atoms"] for v in vals),"visible_atoms":sum(v["visible_atoms"] for v in vals),"glyph_atoms":sum(v["glyph_atoms"] for v in vals),"endpoint_atoms":sum(v["endpoint_atoms"] for v in vals),"union_atoms":sum(v["union_atoms"] for v in vals),"bonds":sum(v["bonds"] for v in vals),"glyph_bbox_valid_all":all(v["glyph_bbox_valid"] for v in vals),"glyph_bbox_overlap_total":sum(v["glyph_bbox_overlap_count"] for v in vals),"text_pixels_outside_glyph_boxes":sum(v["text_pixels_outside_glyph_boxes"] for v in vals),"visible_glyph_ceiling":sum(v["glyph_atoms"] for v in vals)/max(sum(v["visible_atoms"] for v in vals),1),"line_endpoint_ceiling":sum(v["endpoint_atoms"] for v in vals)/max(sum(v["atoms"] for v in vals),1),"union_atom_ceiling":sum(v["union_atoms"] for v in vals)/max(sum(v["atoms"] for v in vals),1),"bond_geometry_ceiling":sum(v["bonds"] for v in vals)/max(sum(v["bonds"] for v in vals),1)}

def main():
 rows=[json.loads(x) for x in (DATA/"samples.jsonl").read_text().splitlines() if x.strip() and json.loads(x).get("render_status","accepted")=="accepted"]; vals=[one(r) for r in rows]; split={s:summary([v for v in vals if v["split"]==s]) for s in ("train","holdout")}; prof={p:summary([v for v in vals if v["profile_id"]==p]) for p in sorted({v["profile_id"] for v in vals})}
 report={"schema_version":"chemvector-identity-aligned-visible-text-seam-contract-r5-0.1","complete":True,"read_only":True,"training_performed":False,"gpu_used":False,"sealed_test_read":False,"production_checkpoint_written":False,"data_dir":str(DATA),"samples":len(rows),"split":split,"by_profile":prof,"ceiling_definition":"oracle union of same-render glyph masks/records and exact layer bond endpoints; labels are used only to measure an upper bound","gate":{"accepted_artifacts_complete":all((DATA/"samples"/(r["sample_key"]+s)).exists() for r in rows for s in (".png",".layers.json",".text-records.json",".text.png",".bond.png",".graph.json")),"glyph_masks_present":all(v["text_pixels"]>0 for v in vals if v["visible_atoms"]>0),"glyph_boxes_valid":all(v["glyph_bbox_valid"] for v in vals),"glyph_boxes_disjoint":sum(v["glyph_bbox_overlap_count"] for v in vals)==0,"glyph_mask_attribution_exact":sum(v["text_pixels_outside_glyph_boxes"] for v in vals)==0,"line_endpoints_present":all(v["line_endpoint_records"]==v["bonds"] for v in vals),"union_ceiling_one":all(v["union_atom_ceiling"]==1.0 for v in vals),"bond_geometry_ceiling_one":all(v["bond_geometry_ceiling"]==1.0 for v in vals),"training_admissible":False,"production_go":False},"sha256":{"samples_jsonl":digest(DATA/"samples.jsonl")}}
 OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n"); print(json.dumps({"output":str(OUT),"split":split,"gate":report["gate"]},indent=2,ensure_ascii=False));return report
if __name__=="__main__":main()
