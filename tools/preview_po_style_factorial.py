"""Human-only style/probability contact sheets from completed saved evidence."""

import json
from pathlib import Path

from PIL import Image, ImageDraw

from Data.sdf.chemvector_curriculum_v0_1.gallery import _font, _foreground_thumbnail
from tools import diagnose_po_style_factorial as experiment
from tools import review_structure_expansion_identity as fixed

OUTPUT = fixed.REPO/"outputs/reviews/chemvector-po-style-factorial-20260923"


def run():
    fixed.require(not OUTPUT.exists(),"Fresh review directory required")
    report=json.loads(fixed.read(experiment.OUTPUT/"report.json"))
    audit=json.loads(fixed.read(experiment.AUDIT/"report.json"))
    plan=json.loads(fixed.read(experiment.OUTPUT/"plan.json"))
    fixed.require(report["complete"] is True and audit["status"]=="passed_saved_factorial_postcheck"
        and audit["source_report_sha256"]==fixed.digest(experiment.OUTPUT/"report.json"),"Completed audited factorial required")
    hashes=json.loads(fixed.read(experiment.DATA/"sample-sha256.json"))
    # Predeclared representative rule: smallest structure, then stable identity.
    entry=min(plan["entries"],key=lambda row:(row["heavy_atoms"],row["molecule_id"]))
    rows=[]; bindings={}
    for cell in plan["profiles"]:
        key=entry["molecule_id"]+"__"+cell["profile"]["profile_id"]
        path=experiment.OUTPUT/"cases"/(key+".json")
        bindings[str(path)]=report["case_sha256"][key]
        image_path=experiment.DATA/"samples"/(key+".png")
        bindings[str(image_path)]=hashes["samples/"+key+".png"]
        rows.append((cell,json.loads(fixed.read(path)),image_path))
    for path in (experiment.OUTPUT/"report.json",experiment.AUDIT/"report.json",experiment.OUTPUT/"plan.json",
                 Path(__file__).absolute(),fixed.REPO/"Data/sdf/chemvector_curriculum_v0_1/gallery.py"):
        bindings[str(path)]=fixed.digest(path)
    fixed.verify_bindings(bindings); OUTPUT.mkdir(parents=True,exist_ok=False)
    pages=[]
    for start in (0,8):
        canvas=Image.new("RGB",(1120,1320),"white"); draw=ImageDraw.Draw(canvas)
        draw.text((12,10),"P-O factorial / frozen initial / HUMAN REVIEW ONLY",font=_font(19),fill="black")
        draw.text((12,38),"Original inputs stay 768px; foreground thumbnails. P-O probabilities use predicted endpoints.",font=_font(12),fill="#475569")
        for i,(cell,case,image_path) in enumerate(rows[start:start+8]):
            x,y=i%2*560,70+i//2*310
            profile=cell["profile"]
            draw.rectangle((x,y,x+559,y+309),outline="#cbd5e1")
            draw.text((x+10,y+8),f"{profile['profile_id']} / {profile['font_family']}",font=_font(16),fill="black")
            draw.text((x+10,y+30),f"font {profile['font_size']} / length cap {profile['bond_length_cap']:g} / stroke {profile['bond_line_width']:g}",font=_font(13),fill="#475569")
            thumb=_foreground_thumbnail(image_path,(536,190))
            canvas.paste(thumb,(x+(560-thumb.width)//2,y+52+(190-thumb.height)//2))
            singles=sorted([r for r in case["po_bonds"] if r["expected_type"]=="SINGLE"],key=lambda r:r["bond_id"])
            values=["unreachable" if r["presence_probability"] is None else f"{r['presence_probability']:.4f}" for r in singles]
            correct=sum(r["stage"]=="reachable_correct" for r in singles)
            draw.text((x+10,y+253),"P-O p: "+" / ".join(values)+"    threshold 0.85",font=_font(15),fill="#166534" if correct==2 else "#b91c1c")
            draw.text((x+10,y+279),f"Local P-O correct: {correct}/2. Not whole-molecule accuracy.",font=_font(12),fill="#475569")
        name=f"styles-{start//8+1:02d}.png"; canvas.save(OUTPUT/name); pages.append(name)
    fixed.verify_bindings(bindings)
    result=dict(status="human_review_only",molecule_id=entry["molecule_id"],images=16,structures=1,
        input_sha256=bindings,pages={name:fixed.digest(OUTPUT/name) for name in pages},
        selection="smallest of four structures then molecule ID",model_forwards=0,training_authorized=False,original_images_modified=False)
    fixed.write_new(OUTPUT/"report.json",result)
    return result


if __name__=="__main__":
    print(json.dumps({k:v for k,v in run().items() if k in ("status","molecule_id","images","pages")}))
