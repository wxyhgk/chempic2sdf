"""Narrow post-hoc probe: chemical CH2 truth versus renderer-visible carbon labels."""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import chemvector.external_ocr_r0_contracts as contract
from chemvector.text_entry_protocol import load_protocol, source_files


ATTRIBUTION = Path("outputs/evaluations/text-failure-attribution-20260909")
COMPARISON = Path("outputs/evaluations/text-entry-comparison-20260909")


def visible_svg_labels(mol, parameters):
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D

    prepared = rdMolDraw2D.PrepareMolForDrawing(
        Chem.Mol(mol), kekulize=parameters.get("kekulize", True),
        addChiralHs=False, wedgeBonds=True,
    )
    drawer = rdMolDraw2D.MolDraw2DSVG(768, 768, -1, -1, True)
    options = drawer.drawOptions()
    options.rotate = parameters.get("rotation_degrees", 0)
    options.bondLineWidth = parameters.get("bond_line_width", 2)
    options.fixedFontSize = parameters.get("font_size", 16)
    options.padding = .08
    options.multipleBondOffset = parameters.get("multiple_bond_offset", .16)
    options.addStereoAnnotation = False
    options.includeMetadata = False
    options.flagCloseContactsDist = -1
    drawer.DrawMolecule(prepared)
    drawer.FinishDrawing()
    labels = {}
    for element in ET.fromstring(drawer.GetDrawingText()).iter():
        identity = element.attrib.get("class", "")
        if element.tag.rsplit("}", 1)[-1] == "text" and identity.startswith("atom-"):
            labels.setdefault(int(identity[5:]), []).append("".join(element.itertext()).strip())
    return labels


def run(output_dir):
    from Data.sdf.build_chemvision_layers import explicit_atom_text, restore_source_geometry
    from chemvector.text_entry_predictions import selected_images

    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Label probe requires empty output")
    original = load_protocol(COMPARISON)
    report = json.loads((ATTRIBUTION / "report.json").read_text())
    for name, sha in report["artifacts"].items():
        contract.validate_frozen_file(ATTRIBUTION / name, sha, "attribution artifact")
    failures = contract.read_jsonl(ATTRIBUTION / "failures.jsonl")
    selected = {row["gt_key"]: row for row in failures if row["truth"] == "CH2"}
    wanted = {row["sample_key"] for row in selected.values()}
    protocol = {"schema_version": "text-failure-label-probe-v1",
                "selection": "all truth=CH2 objects among the frozen 211 failures; post-hoc",
                "attribution_report_sha256": contract.sha256_file(ATTRIBUTION / "report.json"),
                "sources": source_files(), "gt_keys": sorted(selected),
                "method": "same chemical graph and layout; parse noFreetype SVG text nodes",
                "limitation": "text-backend diagnostic, not pixel-identical raster replay or dataset-wide census",
                "training_authorized": False, "promotion_authorized": False}
    contract.write_new_json(output / "protocol.json", protocol)
    subset = {**original, "sample_metadata": {key: original["sample_metadata"][key] for key in sorted(wanted)}}
    records = []
    oracle = {r["gt_key"]: r for r in contract.read_jsonl(ATTRIBUTION / "ocr_counterfactual.jsonl")}
    for sample in selected_images(subset):
        mol = restore_source_geometry(sample["chemmol"], sample["render"])
        params = {**sample["layers"]["render_parameters"], **sample["render"]["render_parameters"]}
        labels = visible_svg_labels(mol, params)
        chemical = {a["id"]: a for a in sample["chemmol"]["atoms"]}
        for index, atom in enumerate(sorted(sample["render"]["atoms"], key=lambda a: a["atom_id"])):
            key = f"{sample['key']}:t{index:03d}"
            if key not in selected:
                continue
            atom_id = atom["atom_id"]
            row = selected[key]
            records.append({"gt_key": key, "sample_key": sample["key"],
                            "molecule_id": row["molecule_id"], "source_atom_id": atom_id,
                            "frozen_truth": row["truth"], "renderer_tokens": labels.get(atom_id, []),
                            "chemical_fallback_text": explicit_atom_text(chemical[atom_id]),
                            "plain_C_vs_CH2": labels.get(atom_id) == ["C"],
                            "baseline_text": row["baseline"]["predicted_text"],
                            "oracle_text": oracle[key]["oracle_text"] if key in oracle else None,
                            "source_crop": oracle[key]["oracle_path"] if key in oracle else None})
    if {r["gt_key"] for r in records} != set(selected):
        raise ValueError("Label probe population incomplete")
    records.sort(key=lambda r: r["gt_key"])
    contract.write_new_jsonl(output / "records.jsonl", records)
    summary = {"population": len(records), "images": len(wanted),
               "molecules": len({r["molecule_id"] for r in records}),
               "plain_C_vs_CH2": sum(r["plain_C_vs_CH2"] for r in records),
               "baseline_reads_C": sum(r["baseline_text"] == "C" for r in records),
               "oracle_reads_C": sum(r["oracle_text"] == "C" for r in records),
               "oracle_reads_chemical_CH2": sum(r["oracle_text"] == "CH2" for r in records),
               "training_authorized": False, "promotion_authorized": False,
               "historical_scores_modified": False,
               "protocol_sha256": contract.sha256_file(output / "protocol.json"),
               "records_sha256": contract.sha256_file(output / "records.jsonl")}
    contract.write_new_json(output / "report.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    print(json.dumps(run(parser.parse_args().output_dir), sort_keys=True))
