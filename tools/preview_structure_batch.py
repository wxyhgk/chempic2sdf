"""Small human-review contact sheets from audited original images, no rerender."""

import argparse
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw

from Data.sdf.chemvector_curriculum_v0_1.gallery import _font, _foreground_thumbnail
from Data.sdf.structure_expansion_v1.batch import SOURCE, REPORT_SHA, digest, verify_hashes
from chemvector.training.curriculum.interface_diagnostic import write_json
from chemvector.training.curriculum.runtime import run_budgeted, ResourceBudget


def choose_rows(selection, rows, retained):
    by_id = {}
    for row in sorted(rows,key=lambda r:(r['profile_id']!='oa22',r['sample_key'])):
        if row['sample_key'] in retained:
            by_id.setdefault(row['molecule_id'],row)
    return [(e,by_id.get(e['molecule_id'])) for e in selection]


def write_pages(root, output, selected):
    names=[]
    heading, label, small = _font(18), _font(13), _font(11)
    for start in range(0,len(selected),16):
        canvas=Image.new('RGB',(1200,1090),'white'); draw=ImageDraw.Draw(canvas)
        draw.text((12,12),'QUARANTINE / NOT TRAINED / fixed structural representatives',fill='black',font=heading)
        draw.text((12,40),'Foreground-cropped thumbnails for HUMAN REVIEW ONLY; original inputs remain 768 x 768.',fill='#923411',font=label)
        for i,(entry,row) in enumerate(selected[start:start+16]):
            x,y=i%4*300,80+i//4*250
            draw.rectangle((x,y,x+299,y+249),outline='#cbd5e1')
            draw.text((x+8,y+6),entry['molecule_id'],fill='black',font=label)
            draw.text((x+8,y+27),', '.join(entry['concepts'])[:44],fill='#475569',font=small)
            if row is None:
                draw.text((x+25,y+110),'NO RETAINED IMAGE',fill='red',font=label)
                continue
            thumb=_foreground_thumbnail(root/'samples'/(row['sample_key']+'.png'),(278,164))
            canvas.paste(thumb,(x+(300-thumb.width)//2,y+49+(164-thumb.height)//2))
            draw.text((x+8,y+219),row['profile_id']+' '+entry['canonical_smiles'][:37],fill='#475569',font=small)
        name=f'structures-{start//16+1:02d}.png'; canvas.save(output/name); names.append(name)
    return names


def worker(dataset, audit_path, output):
    root, output = Path(dataset),Path(output)
    report=json.loads(Path(audit_path).read_bytes())
    if not report['integrity_passed'] or report['training_authorized']:
        raise ValueError('Audited quarantine batch required')
    inputs={str(root/name):value for name,value in report['manifest_sha256'].items()}
    inputs[str(audit_path)]=digest(audit_path)
    inputs[str(SOURCE/'report.json')]=REPORT_SHA
    verify_hashes(inputs)
    parent=json.loads((SOURCE/'report.json').read_bytes())
    inputs[str(SOURCE/'preview-selection.json')]=parent['preview_selection_sha256']
    verify_hashes(inputs)
    selection=json.loads((SOURCE/'preview-selection.json').read_bytes())
    rows=json.loads((root/'samples.json').read_bytes())
    retained=set(json.loads((root/'quality-partition.json').read_bytes())['retained'])
    hashes=json.loads((root/'sample-sha256.json').read_bytes())
    chosen=choose_rows(selection,rows,retained)
    for _,row in chosen:
        if row:
            name='samples/'+row['sample_key']+'.png'
            inputs[str(root/name)]=hashes[name]
    inputs[str(Path(__file__))]=digest(Path(__file__))
    inputs['Data/sdf/chemvector_curriculum_v0_1/gallery.py']=digest('Data/sdf/chemvector_curriculum_v0_1/gallery.py')
    verify_hashes(inputs)
    names=write_pages(root,output,chosen)
    verify_hashes(inputs)
    write_json(output/'report.json',dict(status='human_review_only_not_model_inputs',
        selected_structures=len(chosen),shown_images=sum(r is not None for _,r in chosen),
        missing_structures=[e['molecule_id'] for e,r in chosen if r is None],
        sample_keys=[r['sample_key'] for _,r in chosen if r is not None],input_sha256=inputs,
        pages={name:digest(output/name) for name in names},training_authorized=False,
        original_images_modified=False,model_forwards=0,gpu_used=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True); parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True); parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    if args.worker:
        worker(args.dataset,args.audit,args.output); return 0
    command=[sys.executable,'-m','tools.preview_structure_batch','--worker','--dataset',str(args.dataset),
             '--audit',str(args.audit),'--output',str(args.output)]
    result=run_budgeted(command,output=args.output,budget=ResourceBudget(60))
    print(json.dumps(result,indent=2)); return 0 if result['status']=='completed' else 1


if __name__ == '__main__': raise SystemExit(main())
