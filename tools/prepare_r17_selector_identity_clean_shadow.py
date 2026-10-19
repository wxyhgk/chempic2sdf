#!/usr/bin/env python3
"""Prepare a known-clean (not historically complete) selector shadow cohort.

This is metadata-only.  It excludes every molecule and every known
connection/scaffold identity present in the r17 plan or materialized r17
caches, then chooses a deterministic release train/holdout slice.  The
release itself is quarantine and its historical prior-use ledger is incomplete;
the resulting manifest is therefore only eligible for an isolated CPU shadow,
never production training.
"""
from __future__ import annotations
from collections import defaultdict
import hashlib, json
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
RELEASE=ROOT/'outputs/audits/chemvector-structure-expansion-10k-release-plan-20261016-r1/manifest.json'
MOLECULES=ROOT/'outputs/datasets/structure-expansion-batch-12272-20260923/molecules.json'
PLAN=ROOT/'outputs/audits/chemvector-aromatic-specialist-plan-r17-v2/rows.jsonl'
OUT=ROOT/'outputs/audits/chemvector-r17-selector-known-clean-shadow-cohort-v1'
TRAIN_N=128; HOLDOUT_N=64

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def ident(row):
    return (str(row.get('canonical_smiles') or ''),
            str(row.get('connection_identity') or row.get('connection_identity_without_isotope_stereo') or ''),
            str(row.get('legacy_scaffold') or row.get('scaffold') or ''))
def rank(mid): return hashlib.sha256(('r17-known-clean-selector-v1:'+mid).encode()).hexdigest()

def run():
    if OUT.exists(): raise FileExistsError(OUT)
    release=json.loads(RELEASE.read_text())['rows']
    molecules={str(r['molecule_id']):r for r in json.loads(MOLECULES.read_text())}
    known_ids=set(); known_fields=[set(),set(),set()]
    for line in PLAN.read_text().splitlines():
        if not line.strip(): continue
        r=json.loads(line); known_ids.add(str(r['molecule_id']))
        values=ident(r)
        for i,v in enumerate(values):
            if v: known_fields[i].add(v)
    cache_files=sorted(ROOT.glob('outputs/audits/chemvector-graph-context-aromatic-cpu-r17-*/feature-cache.pt'))
    cache_ids=set()
    for path in cache_files:
        data=torch.load(path,map_location='cpu',weights_only=False)
        for records in data.get('records',{}).values():
            for r in records:
                if r.get('sample_key'): cache_ids.add(str(r['sample_key']).rsplit('__',1)[0])
    known_ids |= cache_ids
    by=defaultdict(list)
    for r in release: by[str(r['molecule_id'])].append(r)
    clean=[]
    for mid,rows in by.items():
        if mid in known_ids or len(rows)!=8: continue
        meta=molecules.get(mid,{})
        values=(str(meta.get('canonical_smiles') or rows[0].get('canonical_smiles') or ''),
                str(meta.get('connection_identity_without_isotope_stereo') or rows[0].get('connection_identity') or ''),
                str(meta.get('scaffold') or rows[0].get('legacy_scaffold') or ''))
        if any(v and v in known_fields[i] for i,v in enumerate(values)): continue
        clean.append((mid,rows,values))
    train=[x for x in sorted(clean,key=lambda x:rank(x[0])) if x[1][0]['split']=='train'][:TRAIN_N]
    hold=[x for x in sorted(clean,key=lambda x:rank('holdout:'+x[0])) if x[1][0]['split']=='holdout'][:HOLDOUT_N]
    if len(train)!=TRAIN_N or len(hold)!=HOLDOUT_N: raise RuntimeError(f'clean cohort too small: {len(train)}/{len(hold)}')
    def vals(items,i): return {x[2][i] for x in items if x[2][i]}
    overlap={field:len(vals(train,i)&vals(hold,i)) for i,field in enumerate(('canonical_smiles','connection_identity','scaffold'))}
    if any(overlap.values()): raise RuntimeError(f'train/holdout identity overlap: {overlap}')
    rows=[]
    for split,items in (('train',train),('holdout',hold)):
        for mid,group,_ in items:
            for r in sorted(group,key=lambda x:x['sample_key']): rows.append({**r,'split':split,'shadow_identity_clean':True})
    report={'schema_version':'chemvector-r17-selector-known-clean-shadow-cohort-0.1','complete':True,'read_only':True,'payloads_read':False,'training_performed':False,'training_authorized':False,'sealed_test_read':False,'production_checkpoint_written':False,'release_historical_prior_use_ledger_complete':False,'known_excluded_molecules':len(known_ids),'clean_candidates':len(clean),'selection':{'train_molecules':len(train),'holdout_molecules':len(hold),'images':len(rows),'profiles':sorted({r['profile_id'] for r in rows})},'identity_overlap':overlap,'inputs':{'release_sha256':digest(RELEASE),'molecules_sha256':digest(MOLECULES),'plan_sha256':digest(PLAN),'cache_files':[str(p.relative_to(ROOT)) for p in cache_files]},'gate':{'known_plan_and_cache_excluded':True,'train_holdout_identity_disjoint':not any(overlap.values()),'historical_ledger_complete':False,'cpu_shadow_only':True,'gpu_training_go':False}}
    OUT.mkdir(parents=False); (OUT/'manifest.json').write_text(json.dumps({'schema_version':report['schema_version'],'rows':rows},indent=2,ensure_ascii=False)+'\n'); (OUT/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n'); print(json.dumps({'output':str(OUT),'selection':report['selection'],'overlap':overlap,'gate':report['gate']},indent=2,ensure_ascii=False))
if __name__=='__main__': run()
