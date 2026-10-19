"""Bounded fixed58 coordinate-contract audit; no training or label release."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import html
import json
import os
from pathlib import Path
import sys

from PIL import Image, ImageDraw
import torch

from chemvector.training.curriculum import runtime
from tools import selector_coordinate_contract as coordinates
from tools import selector_node_groups as groups
from tools import selector_node_semantics as semantics

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT / 'outputs/audits/chemvector-selector-node-groups-20260920'
PREVIOUS_SHA = '7d72ca17759ac5c2d17742c0e5efe0c86b468116c971f74513c926999326d4be'
FAILED_ATTEMPT = ROOT / 'outputs/audits/chemvector-selector-coordinate-contract-20260920'
OUTPUT = ROOT / 'outputs/audits/chemvector-selector-coordinate-contract-20260920-r1'
MANIFEST = ROOT / 'outputs/datasets/chemvector-selector-retention-fit-20260919/manifest.json'
CONTROL = ROOT / 'outputs/experiments/chemvector-selector-full-loss-control-20260920'
RETENTION = ROOT / 'outputs/experiments/chemvector-selector-retention-trial-20260920'
SOURCE_FILES = (
    'chemvector/training/curriculum/data.py', 'chemvector/training/curriculum/pixels.py',
    'chemvector/training/curriculum/continuation.py',
    'chemvector/components/atoms/objects/spatial_atom_object.py',
    'chemvector/components/graph/joint/predicted_incidence.py',
    'chemvector/training/graph/relations/train_direct_atom_pair_bond.py',
    'Data/sdf/text_source_renderer/layers.py', 'Data/sdf/text_source_renderer/v2/layers.py',
    'Data/sdf/text_source_layout/drawing.py',
)


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')


class Inputs:
    def __init__(self):
        self.hashes = {}

    def read(self, path, expected=None):
        path = Path(path).absolute()
        if path.resolve()!=path or 'sealed_test' in path.parts:
            raise ValueError('Explicit nonsealed nonsymlink inputs only')
        value = path.read_bytes()
        digest = sha256(value).hexdigest()
        if ((expected is not None and expected!=digest)
                or str(path) in self.hashes and self.hashes[str(path)]!=digest):
            raise ValueError('Input changed: '+str(path))
        self.hashes[str(path)] = digest
        return value

    def json(self, path, expected=None):
        return json.loads(self.read(path,expected))


def preflight(inputs):
    previous = inputs.json(PREVIOUS/'report.json',PREVIOUS_SHA)
    bindings = previous['input_sha256']
    manifest = inputs.json(MANIFEST,bindings[str(MANIFEST)])
    rows = groups.training_rows(manifest)
    if Counter(r['fit_domain'] for r in rows)!={'witness':26,'old_full_molecule':32}:
        raise ValueError('Fixed58 scope changed')
    return manifest,rows,bindings


def proposal_equal(previous, reconstructed):
    # SVG parser points are tuples in memory but JSON arrays on disk. Compare
    # the actual serialized contract; do not relax values or float precision.
    return all(previous.get(k) == json.loads(json.dumps(v, allow_nan=False))
               for k,v in reconstructed.items())


def overlay(png, key, target):
    with Image.open(png) as image:
        canvas = image.convert('RGB')
    points = [target[k] for k in ('source_anchor','quantized_source_anchor','nearest_source_xy')]
    box = target['bbox']
    if box:
        points += [box[:2],box[2:]]
    crop = (max(0,int(min(p[0] for p in points))-18), max(0,int(min(p[1] for p in points))-18),
            min(768,int(max(p[0] for p in points))+19), min(768,int(max(p[1] for p in points))+19))
    canvas = canvas.crop(crop).resize(((crop[2]-crop[0])*5,(crop[3]-crop[1])*5))
    draw = ImageDraw.Draw(canvas)
    def xy(point):
        return ((point[0]-crop[0])*5,(point[1]-crop[1])*5)
    if box:
        draw.rectangle((*xy(box[:2]),*xy(box[2:])),outline='#888888',width=1)
    for name,color,shape in (('source_anchor','#008040','plus'),
                             ('quantized_source_anchor','#1565c0','square'),
                             ('nearest_source_xy','#d01030','circle')):
        x,y=xy(target[name])
        if shape=='plus':
            draw.line((x-7,y,x+7,y),fill=color,width=2)
            draw.line((x,y-7,x,y+7),fill=color,width=2)
        elif shape=='square':
            draw.rectangle((x-4,y-4,x+4,y+4),outline=color,width=2)
        else:
            draw.ellipse((x-5,y-5,x+5,y+5),outline=color,width=2)
    name=f'{key}--atom-{target["target_id"]}.png'
    canvas.save(OUTPUT/name)
    return name


def gallery(cards, sensitive):
    sections=[]
    for key,target,name in cards:
        sections.append(f'<h2>{html.escape(key)} / atom {target["target_id"]} / {html.escape(target["text"])}</h2>'
            f'<p>原匹配距离 {target["legacy_float32_nearest_px"]:.4f}px；去量化诊断距离 '
            f'{target["unquantized_model_nearest_px"]:.4f}px。诊断口径，不改变验收。</p><img src="{name}">')
    rows=''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in (
        r['sample_key'],r['fragment_slot'],r['owner_target'],r['review_reasons']))+'</tr>' for r in sensitive)
    page='<!doctype html><meta charset="utf-8"><title>坐标契约与文字锚点</title>' \
        '<style>body{font:16px sans-serif;margin:24px}td,th{border:1px solid #aaa;padding:6px}table{border-collapse:collapse}img{max-width:100%}</style>' \
        '<h1>坐标契约审计：未修改模型、标签或匹配半径</h1>' \
        '<p>所有点按归一化坐标乘767叠回原图。绿十字=原始化学节点锚点；蓝框=q1023量化锚点；红圈=最近候选；灰框=文字墨迹框。</p>' \
        '<p>文字框中心不等于化学节点锚点；这些图不是新的模型预测。所有边界配对暂缓，不作为训练标签。</p>' \
        ''.join(sections)+'<h2>上一轮配对的坐标敏感项</h2><table><tr><th>样本</th><th>片段槽</th><th>归属节点</th><th>复核原因</th></tr>'+rows+'</table>'
    with (OUTPUT/'index.html').open('x',encoding='utf-8') as stream:
        stream.write(page)


def worker():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_initialized():
        raise ValueError('CPU-only audit required')
    if not OUTPUT.is_dir() or (OUTPUT/'plan.json').exists():
        raise ValueError('Fresh budgeted output required')
    torch.set_num_threads(2)
    inputs=Inputs()
    manifest,rows,bindings=preflight(inputs)
    failed=inputs.json(FAILED_ATTEMPT/'runtime.json')
    if failed['status']!='worker_failed' or failed['gpu_devices'] or failed['training']:
        raise ValueError('Expected preserved CPU-only failed audit attempt')
    inputs.read(FAILED_ATTEMPT/'worker.log')
    code=[Path(__file__),Path(coordinates.__file__),Path(groups.__file__),Path(semantics.__file__),Path(runtime.__file__)]
    for path in code+[ROOT/p for p in SOURCE_FILES]+[ROOT/'frontend/inference_profiles.json']:
        inputs.read(path,bindings.get(str(path)))
    write(OUTPUT/'plan.json',dict(schema_version='selector-coordinate-plan-0.1',
        sample_keys=[r['sample_key'] for r in rows],gpu_count=0,model_forwards=0,optimizer_steps=0,
        training_authorized=False,wall_seconds=120,
        diagnostics=['all saved targets/candidate cell consistency','original float32 vs direct distance',
            'inverse source projection via767','q1023 sensitivity without label change',
            'fixed-cell geometric reachability bound','all prior525 pair evidence recheck'],
        unchanged=['original768 matching metric','6px radius','-.25 threshold','all historical predictions and labels'],
        supervision_experiment='deferred until coordinate-sensitive proposal evidence is reviewed'))
    totals=defaultdict(Counter)
    phase_counts=defaultdict(Counter)
    sensitive=[]; missing=[]; cards=[]
    maximum_numeric_error=0.
    for row in rows:
        key,group=row['sample_key'],row['fit_domain']
        source=Path(manifest['sources'][row['source']])/'samples'
        meta=inputs.json(source/(key+'.meta.json'),row['meta_sha256'])
        layers=inputs.json(source/(key+'.layers.json'),meta['sha256']['.layers.json'])
        glyphs=inputs.json(source/(key+'.text-records.json'),meta['sha256']['.text-records.json'])['glyphs']
        svg=inputs.read(source/(key+'.svg'),meta['sha256']['.svg'])
        if layers['sample_key']!=key or layers['molecule_id']!=row['molecule_id']:
            raise ValueError('Fixed source identity mismatch')
        path=CONTROL/('geometry--'+key+'.json')
        geometry=inputs.json(path,bindings[str(path)])
        semantics.check_geometry(geometry,layers)
        paths,unknown=semantics.strokes(svg,layers)
        prior=inputs.json(PREVIOUS/(key+'.json'))
        rebuilt=groups.propose(geometry,layers,glyphs,paths,unknown)
        if not proposal_equal(prior,rebuilt):
            raise ValueError('Prior proposals do not reproduce from bound source geometry')
        result=coordinates.target_diagnostics(geometry,layers)
        pairs=coordinates.pair_robustness(geometry,layers,glyphs,paths,unknown,prior)
        count=totals[group]
        count['images']+=1; count['targets']+=len(result['targets']); count['raw_candidates']+=len(geometry['candidates'])
        count['float32_direct_membership_flips']+=result['float32_direct_membership_flips']
        maximum_numeric_error=max(maximum_numeric_error,result['maximum_float32_direct_distance_difference_px'])
        count['prior_pairs']+=len(pairs)
        count['coordinate_stable_pairs']+=sum(p['coordinate_stable'] for p in pairs)
        held={p['fragment_slot'] for p in pairs if not p['coordinate_stable']}
        count['coordinate_sensitive_pairs']+=len(held)
        for p in pairs:
            if not p['coordinate_stable']:
                count.update(p['review_reasons'])
                sensitive.append(dict(sample_key=key,group=group,**p))
        for target in result['targets']:
            count['target_coverage_gain_without_q1023']+=target['legacy_no_candidate'] and target['unquantized_model_nearest_px']<=6
            count['target_coverage_loss_without_q1023']+=not target['legacy_no_candidate'] and target['unquantized_model_nearest_px']>6
            if target['legacy_no_candidate']:
                missing.append(dict(sample_key=key,group=group,**target))
                count['legacy_no_candidate_targets']+=1
                count['missing_reachable_with_fixed_cell_offsets']+=target['best_existing_cell_lower_bound_px']<6
                count['missing_target_cell_present']+=target['target_cell_in_raw']
                png=source/(key+'.png')
                inputs.read(png,meta['sha256']['.png'])
                cards.append((key,target,overlay(png,key,target)))
        for phase,parent,prefix in (('initial',CONTROL,'baseline'),('retention-200',RETENTION,'check-000200'),
                                   ('control-200',CONTROL,'check-000200'),('retention-400',RETENTION,'check-000400')):
            path=parent/(prefix+'--'+key+'.json')
            record=inputs.json(path,bindings[str(path)])
            if record['key']!=geometry['key'] or record['group']!=group:
                raise ValueError('Prediction/source identity mismatch')
            phase_counts[phase][group]+=sum(record['metrics']['logits'][str(slot)]>=-.25 for slot in held)
        write(OUTPUT/(key+'.json'),dict(sample_key=key,group=group,**result,prior_pair_review=pairs,
            training_authorized=False,requires_review=True,threshold_or_labels_changed=False))
    if len(missing)!=10:
        raise ValueError('Original ten missing-target cases changed')
    gallery(cards,sensitive)
    for path,digest in list(inputs.hashes.items()):
        inputs.read(path,digest)
    if torch.cuda.is_initialized():
        raise ValueError('Unexpected CUDA initialization')
    write(OUTPUT/'report.json',dict(schema_version='selector-coordinate-audit-0.1',input_sha256=inputs.hashes,
        totals=totals,missing_targets=missing,coordinate_sensitive_pairs=sensitive,
        selected_sensitive_pairs_by_phase=phase_counts,
        maximum_float32_direct_distance_difference_px=maximum_numeric_error,
        training_authorized=False,training_performed=False,model_forwards=0,optimizer_steps=0,
        gpu_used=False,validation_sample_artifacts_read=False,new_identity_queries=0,
        threshold_or_labels_changed=False,default_changed=False,
        prior_failed_attempt=str(FAILED_ATTEMPT),
        prior_failure='JSON arrays versus in-memory tuples in strict proposal comparison; same numeric evidence; preserved and regression-tested',
        conclusion='Matching/training normalized coordinates agree; previous source-overlay locations used768 instead of inverse767. Hold sensitive proposals, do not alter acceptance.',
        group_gradient_experiment='not started: coordinate-sensitive proposals require review first',
        limitations=['Unquantized distances are sensitivity diagnostics, not replacement targets',
            'Fixed-cell reachability is a geometric lower bound, not a trained-model result',
            'Stable location evidence does not certify a training label',
            'No new recognition accuracy or test-set generalization measured']))
    print(json.dumps(dict(totals=totals,missing_targets=len(missing),sensitive_pairs=len(sensitive)),ensure_ascii=False))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args.worker:
        worker(); return 0
    preflight(Inputs())
    receipt=runtime.run_budgeted([sys.executable,'-m','tools.audit_selector_coordinate_contract','--worker'],
        output=OUTPUT,budget=runtime.ResourceBudget(120))
    print(json.dumps(receipt,indent=2))
    return 0 if receipt['status']=='completed' else 1


if __name__=='__main__':
    raise SystemExit(main())
