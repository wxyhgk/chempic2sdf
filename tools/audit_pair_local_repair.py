"""Fixed58 saved-scene candidate deletions and a non-deployed conservative selector."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

from rdkit import rdBase

from chemvector.training.curriculum import runtime
from chemvector.training.curriculum.validation_truth import canonical_molblock
from tools import pair_local_repair as repair
from tools.preview_pair_false_edges import read, ROOT, TRIAL

REVIEW = ROOT/'outputs/audits/chemvector-pair-local-evidence58-20260921-r1'
OUTPUT = ROOT/'outputs/audits/chemvector-pair-local-repair58-20260921'


def write(path,obj):
    encoded=json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)
    with Path(path).open('x',encoding='utf-8') as stream:stream.write(encoded+'\n')


def score(compilation, expected):
    return dict(valid=compilation['valid'],exact=compilation['valid'] and compilation['canonical']==expected,
                fallback=compilation['fallback'])


def worker():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Saved-scene CPU-only audit')
    if not OUTPUT.is_dir():raise ValueError('Fresh budgeted output required')
    hashes={}
    def data(path,digest=None):return json.loads(read(path,hashes,digest))
    manifest=data(ROOT/'outputs/datasets/chemvector-selector-retention-fit-20260919/manifest.json',
        'f5e3a7163eef5c49118ad78dcf05778f4ad889dc9b6389097ef20896b79e2111')
    previous=data(REVIEW/'report.json','6bcc160efc5ce7c918c3f2edeef419829ec18ff17f1d63294117512a8c9ab7a9')
    post=data(TRIAL/'postcheck.json','b09b82c98fe5280de0d79a6cca858a2854398fa66dca1d29205997f4cece1f54')
    trial_report=data(TRIAL/'report.json','d12d93e5e012f7566f4cc3123abf411b67f49c44ecb898404034105b3c4603eb')
    authorization=data(ROOT/'docs/model/training/releases/20260921-pair-only-training-authorization.json',
                       trial_report['authorization_sha256'])
    # Compiler/scorer implementations are reused, never modified for this repair audit.
    dependencies=('Data/sdf/chemscene_v0_2.py','Data/sdf/chemscene_v0_1.py','Data/sdf/chemmol_sequence.py',
                  'chemvector/placeholder_scene.py','chemvector/training/curriculum/validation_truth.py')
    for relative in dependencies:
        read(ROOT/relative,hashes,authorization['implementation_sha256'].get(relative))
    for path in (Path(__file__),Path(repair.__file__),Path(runtime.__file__),
                 ROOT/'tools/preview_pair_false_edges.py',ROOT/'tests/chemvector/test_pair_local_repair.py'):
        read(path,hashes)
    rows=manifest['training_samples']
    if (len(rows)!=58 or len({r['sample_key'] for r in rows})!=58 or any(r['split']!='train' for r in rows)
            or Counter(r['fit_domain'] for r in rows)!={'witness':26,'old_full_molecule':32}):
        raise ValueError('Fixed58 scope changed')
    write(OUTPUT/'plan.json',dict(schema_version='pair-local-repair-plan-0.1',
        sample_keys=[r['sample_key'] for r in rows],phases=[0,50],wall_seconds=120,
        maximum_warned_edges_per_scene=2,maximum_nonempty_subsets_per_scene=3,
        edits='Only subsets of existing combined-warning edges; ordinary non-stereo SINGLE deletion',
        baseline='Recompile all 116 unmodified saved scenes and require historical identity/validity/fallback parity',
        selection='Keep valid originals; otherwise unique fewest-deletion valid, no-fallback, nonfragmenting candidate',
        target_labels_used_for_selection=False,new_training=False,model_forwards=0,checkpoint_loads=0,gpu_used=False,
        production_changed=False,independent_accuracy_claim=False,warning_rule_changed=False,input_sha256=dict(hashes)))
    counts={str(step):Counter() for step in (0,50)}
    focused,artifacts=[],{}
    for row in rows:
        key=row['sample_key']; case=data(REVIEW/(key+'.json'),previous['case_sha256'][key+'.json'])
        if case['sample_key']!=key:raise ValueError('Wrong saved evidence sample')
        phases={}
        for step in (0,50):
            name=f'check-{step:06d}--{key}.json'
            record=data(TRIAL/name,post['artifact_sha256'][name])
            if record['metrics']['key']!=row['source']+':'+key:raise ValueError('Wrong saved prediction')
            evidence_rows=case['phases'][str(step)]
            present={tuple(r['pair']):r for r in evidence_rows if r['raw_present']}
            scene=record['trace']['scene_before_fallback']; lineage=record['trace']['compiled_candidate_indices']
            if set(repair.edge_map(scene,lineage))!=set(present):raise ValueError('Saved evidence topology mismatch')
            warnings=sorted(pair for pair,e in present.items() if e['flags']['combined_review'])
            # Pass only predicted scenes and warning pairs. Scoring happens AFTER selection.
            result=repair.explore(scene,lineage,warnings)
            baseline=result['baseline']; old=record['prediction']; metrics=record['metrics']
            if (baseline['canonical']!=canonical_molblock(old['molblock'])
                    or baseline['valid']!=metrics['sdf_valid'] or baseline['fallback']!=old['fallback']):
                raise ValueError('Saved-scene baseline compiler parity failed: '+key)
            expected=metrics['expected_smiles']
            baseline_score=score(baseline,expected)
            if baseline_score['exact']!=metrics['sdf_exact']:raise ValueError('Historical identity score mismatch')
            stat=counts[str(step)];stat['images']+=1;stat['unchanged_scene_recompilations']+=1
            stat['warned_edges']+=len(warnings);stat['warned_images']+=bool(warnings)
            stat['candidate_subsets']+=len(result['candidates'])
            for candidate in result['candidates']:
                removed=[tuple(p) for p in candidate['removed']]
                candidate['removed_scoring_labels']=[present[p]['scoring_label'] for p in removed]
                candidate['scoring']=(None if not candidate['edit_allowed'] else score(candidate['compilation'],expected))
                stat['candidate_compilations']+=candidate['edit_allowed']
                stat['refused_edit_candidates']+=not candidate['edit_allowed']
            chosen=result['decision']['candidate_index']
            selected=baseline if chosen is None else result['candidates'][chosen]['compilation']
            removed=set() if chosen is None else {tuple(p) for p in result['candidates'][chosen]['removed']}
            selected_score=score(selected,expected)
            stat['selected_edited_images']+=chosen is not None
            stat['selected_removed_edges']+=len(removed)
            stat['exact_rescued']+=not baseline_score['exact'] and selected_score['exact']
            stat['exact_broken']+=baseline_score['exact'] and not selected_score['exact']
            stat['valid_rescued']+=not baseline_score['valid'] and selected_score['valid']
            stat['valid_broken']+=baseline_score['valid'] and not selected_score['valid']
            for metric in ('valid','exact','fallback'):
                stat['before_'+metric]+=baseline_score[metric];stat['selected_'+metric]+=selected_score[metric]
            for pair,e in present.items():
                label=e['scoring_label'];stat['before_'+label]+=1
                stat[('removed_' if pair in removed else 'kept_')+label]+=1
            result.update(warnings=[list(p) for p in warnings],baseline_scoring=baseline_score,
                selected_scoring=selected_score,selected_removed=[list(p) for p in sorted(removed)])
            phases[str(step)]=result
            if warnings:
                focused.append(dict(sample_key=key,step=step,warnings=result['warnings'],
                    decision=result['decision'],baseline_scoring=baseline_score,selected_scoring=selected_score,
                    candidates=[{k:v for k,v in c.items() if k!='scene'} for c in result['candidates']]))
        path=OUTPUT/(key+'.json');write(path,dict(sample_key=key,group=row['fit_domain'],phases=phases))
        artifacts[path.name]=sha256(path.read_bytes()).hexdigest()
    if (counts['0']['images']!=58 or counts['50']['images']!=58
            or counts['0']['warned_edges']!=0 or counts['50']['warned_edges']!=3
            or counts['50']['candidate_subsets']!=4):raise ValueError('Incomplete fixed audit scope')
    for path,digest in list(hashes.items()):read(path,hashes,digest)
    write(OUTPUT/'report.json',dict(schema_version='pair-local-repair58-0.1',totals=counts,
        focused_cases=focused,input_sha256=hashes,case_sha256=artifacts,rdkit_version=rdBase.rdkitVersion,
        candidate_selection_used_truth=False,production_wired=False,original_predictions_modified=False,
        default_changed=False,original_acceptance_changed=False,training_performed=False,
        model_forwards=0,checkpoint_loads=0,optimizer_steps=0,gpu_used=False,
        original_images_read=False,validation_sample_artifacts_read=False,sealed_test_read=False,new_identity_queries=0,
        limitations=['Post-hoc fixed training/replay cases, not independent generalization',
            'Offline candidate selection does not deploy a repair policy',
            'Sanitizable does not mean the complete molecule is correct',
            'Candidate deletion keeps saved upstream/text/bond semantics; it cannot restore missing atoms or repair OCR',
            'Sampled labels are only for scoring; unclassified edges are not automatically nonedges',
            'Initial and 50-step results are not pooled as independent molecules']))
    print(json.dumps(dict(totals=counts),indent=2))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',action='store_true')
    args=parser.parse_args(argv)
    if args.worker:worker();return 0
    receipt=runtime.run_budgeted([sys.executable,'-m','tools.audit_pair_local_repair','--worker'],
        output=OUTPUT,budget=runtime.ResourceBudget(120))
    print(json.dumps(receipt,indent=2))
    return 0 if receipt['status']=='completed' else 1


if __name__=='__main__':raise SystemExit(main())
