"""Prepare a metadata-only targeted line-first data plan.

This does not render images or expose sealed identities. It inventories the
existing visible r5 train split and selects strata for a future cov13-focused
render, keeping molecule/scaffold identities out of the holdout.
"""
from __future__ import annotations
import hashlib, json
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'outputs/datasets/chemvector-identity-aligned-visible-text-20261017-r5-render'
OUT=ROOT/'outputs/audits/chemvector-identity-aligned-line-first-targeted-data-manifest-r5/report.json'

def main():
 rows=[json.loads(x) for x in (DATA/'samples.jsonl').read_text().splitlines() if x and json.loads(x).get('render_status')=='accepted']
 train=[r for r in rows if r['split']=='train']; hold=[r for r in rows if r['split']=='holdout']; hold_ids={r.get('molecule_id') for r in hold};
 profiles=Counter(r['profile_id'] for r in train); strata=Counter(); molecule_strata=defaultdict(set)
 for r in train:
  base=DATA/'samples'/r['sample_key']; graph=json.loads(base.with_suffix('.graph.json').read_text()); layers=json.loads(base.with_suffix('.layers.json').read_text()); atoms=graph.get('atoms',[]); bonds=graph.get('bonds',[])
  kinds={str(b.get('chemical_type','OTHER')).upper() for b in bonds}; elements={str(a.get('element','OTHER')).upper() for a in atoms}; degree=Counter()
  for b in bonds: degree[int(b['begin_atom'])]+=1; degree[int(b['end_atom'])]+=1
  tags=[]
  if r['profile_id'].startswith('cov13'): tags.append('cov13_weak_profile')
  if r['profile_id']=='cov13_stroke16_48': tags.append('thin_stroke_1p6')
  if r['profile_id'] in {'cov13_base48','cov13_cap52'}: tags.append('short_cap_180deg')
  if 'AROMATIC' in kinds: tags.append('aromatic_ring')
  if 'DOUBLE' in kinds or 'TRIPLE' in kinds: tags.append('multiple_bond')
  if any(v>=3 for v in degree.values()): tags.append('junction_degree3plus')
  if 'O' in elements or 'N' in elements: tags.append('visible_heteroatom_candidate')
  for tag in tags: strata[tag]+=1; molecule_strata[tag].add(r.get('molecule_id'))
 report={'schema_version':'chemvector-identity-aligned-line-first-targeted-data-manifest-r5-0.1','complete':True,'metadata_only':True,'images_rendered':False,'training_performed':False,'sealed_test_read':False,'production_checkpoint_written':False,'data_sha256':hashlib.sha256((DATA/'summary.json').read_bytes()).hexdigest(),'fixed_source':{'train_images':len(train),'holdout_images':len(hold),'train_molecule_ids':len({r.get('molecule_id') for r in train}),'holdout_molecule_ids':len(hold_ids),'molecule_overlap':len({r.get('molecule_id') for r in train}&hold_ids)},'existing_profile_counts':dict(sorted(profiles.items())),'target_strata_counts':dict(sorted(strata.items())),'target_strata_molecules':{k:len(v) for k,v in sorted(molecule_strata.items())},'targeted_profiles':['cov13_base48','cov13_cap52','cov13_stroke16_48'],'future_render_factors':{'rotation_degrees':[180.0],'font_family':['DejaVu Serif'],'bond_length_cap':[32.0,52.0],'bond_line_width':[1.6,3.2],'anti_alias_variants':['baseline','thin','high_contrast']},'future_required_strata':['aromatic_ring','junction_degree3plus','multiple_bond','visible_heteroatom_candidate'],'gate':{'identity_disjoint':len({r.get('molecule_id') for r in train}&hold_ids)==0,'metadata_only':True,'sealed_test_read':False,'training_authorized':False,'next':'render targeted train-only panel, then CPU coverage/roundtrip gate before GPU'}}
 OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n'); print(json.dumps(report,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
