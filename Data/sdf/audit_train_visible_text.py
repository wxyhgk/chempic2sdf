"""Freeze a train-only sample and export visible-carbon evidence; never revise data."""

import argparse
from datetime import datetime, timezone
import io
import json
from pathlib import Path

from Data.sdf.rendered_text_evidence import no_freetype_text_diagnostic
from Data.sdf.train_visible_text_inputs import APPROVED_ROOT, freeze_selection, iter_selected_samples
from Data.sdf.train_visible_text_report import (
    candidate_record, digest, neutral_visible_carbons, save_previews,
    summarize, write_gallery, write_json,
)


def source_hashes():
    names = ("audit_train_visible_text.py", "train_visible_text_inputs.py",
             "train_visible_text_report.py", "rendered_text_evidence.py",
             "visible_text_revision.py", "build_chemvision_layers.py", "build_webdataset_sample.py")
    return {str(Path(__file__).parent / name): digest((Path(__file__).parent / name).read_bytes())
            for name in names}


def run(output_dir, *, molecule_count=250):
    from PIL import Image
    from rdkit import rdBase
    from Data.sdf.build_chemvision_layers import restore_source_geometry

    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Train review requires an empty output directory")
    print(f"freezing explicit train manifest: {molecule_count} molecules", flush=True)
    selection = freeze_selection(APPROVED_ROOT, count=molecule_count)
    output.mkdir(parents=True, exist_ok=True)
    protocol = {"schema_version": "train-visible-carbon-review-0.1",
                "frozen_utc": datetime.now(timezone.utc).isoformat(),
                "selection": selection, "sources": source_hashes(),
                "object_scope": "all visible neutral nonisotopic nonradical carbon, including C controls",
                "flag": "alternate-renderer tokens exactly [C] and layer text is not C",
                "review_policy": "diagnostic results cannot authorize source-raster corrections",
                "crop_context_px": 4, "context_view_margin_px": 28,
                "training_authorized": False, "revision_authorized": False,
                "source_data_modified": False, "rdkit_version": rdBase.rdkitVersion}
    write_json(output / "protocol.json", protocol)
    rows, sample_counts = [], []
    for sample in iter_selected_samples(APPROVED_ROOT, selection):
        key, metadata = sample["key"], sample["metadata"]
        layers, chemical = sample["layers"], sample["chemmol"]
        with Image.open(io.BytesIO(sample["payloads"]["png"])) as opened:
            image = opened.convert("RGB")
        if image.size != (768, 768) or (layers["width"], layers["height"]) != image.size:
            raise ValueError("Source image/layer coordinate frame differs")
        candidates = neutral_visible_carbons(layers, chemical)
        sample_rows = []
        if candidates:
            try:
                molecule = restore_source_geometry(chemical, sample["render"])
                parameters = {**layers["render_parameters"], **sample["render"]["render_parameters"]}
                diagnostic = no_freetype_text_diagnostic(molecule, parameters)
            except Exception as error:
                # Failures remain in the population and are explicitly unresolved.
                diagnostic = {"labels": {}, "evidence_kind": "nofreetype_diagnostic",
                              "revision_authorized": False,
                              "error": f"{type(error).__name__}: {error}"}
            hashes = {kind: digest(payload) for kind, payload in sample["payloads"].items()}
            sample_rows = [candidate_record(key, metadata["molecule_id"], metadata["variant"],
                                           layer, atom, diagnostic, hashes)
                           for layer, atom in candidates]
            save_previews(output, image, sample["payloads"]["png"], sample_rows)
            rows.extend(sample_rows)
        sample_counts.append({"sample_key": key, "molecule_id": metadata["molecule_id"],
                              "variant": metadata["variant"],
                              "visible_text_objects": sum(a["visible_text"] for a in layers["atoms"]),
                              "visible_neutral_carbons": len(sample_rows)})
        if len(sample_counts) % 100 == 0:
            print(f"inspected={len(sample_counts)}/{len(selection['selection'])} "
                  f"visible_carbon_objects={len(rows)}", flush=True)
    if (len(sample_counts) != len(selection["selection"])
            or {r["sample_key"] for r in sample_counts} != set(selection["selection"])):
        raise ValueError("Train inspection population incomplete")
    rows.sort(key=lambda row: row["object_key"])
    sample_counts.sort(key=lambda row: row["sample_key"])
    for name, values in (("records.jsonl", rows), ("samples.jsonl", sample_counts)):
        with (output / name).open("x", encoding="utf-8") as handle:
            for value in values:
                handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
    write_gallery(output, rows)
    for path, sha in protocol["sources"].items():
        if digest(Path(path).read_bytes()) != sha:
            raise ValueError("Source code changed during frozen inspection")
    report = summarize(rows, sample_counts)
    report["artifacts"] = {str(path.relative_to(output)): digest(path.read_bytes())
                           for path in sorted(output.rglob("*")) if path.is_file()}
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "report.json", report)
    return {"report": str(output / "report.json"),
            **{name: report[name] for name in ("images", "molecules", "visible_neutral_carbons",
                                              "diagnostic_plain_c_disagreements", "unresolved_diagnostic_objects")}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--molecules", type=int, default=250)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.output_dir, molecule_count=arguments.molecules), sort_keys=True))
