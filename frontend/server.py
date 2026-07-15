#!/usr/bin/env python3
"""Serve the local ChemVector benchmark browser and its read-only API."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
from urllib.parse import unquote, urlparse

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator, rdFMCS
from rdkit.Chem.Draw import rdMolDraw2D


FRONTEND_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = FRONTEND_ROOT.parent


def sample_status(record: dict) -> str:
    if record.get("chemical_match"):
        return "exact"
    if not record.get("sdf_valid"):
        return "invalid"
    if record.get("atom_count_match") and record.get("bond_count_match"):
        return "similar"
    return "mismatch"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    resolved = path.resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"Path is outside the project: {value}")
    return resolved


def load_sdf(path: Path, *, sanitize: bool = True) -> Chem.Mol:
    mol = Chem.MolFromMolFile(
        str(path),
        sanitize=sanitize,
        removeHs=False,
        strictParsing=False,
    )
    if mol is None:
        raise ValueError(f"Cannot read SDF: {path}")
    if not sanitize:
        mol.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(mol)
    return mol


def fingerprint_similarity(expected: Chem.Mol, predicted: Chem.Mol) -> float:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return float(
        DataStructs.TanimotoSimilarity(
            generator.GetFingerprint(expected),
            generator.GetFingerprint(predicted),
        )
    )


def sdf_comparison(expected_path: Path, predicted_path: Path) -> dict:
    expected = load_sdf(expected_path)
    predicted = load_sdf(predicted_path)
    result = rdFMCS.FindMCS(
        [expected, predicted],
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareOrder,
        ringMatchesRingOnly=True,
        completeRingsOnly=False,
        timeout=2,
    )
    return {
        "fingerprint_similarity": fingerprint_similarity(expected, predicted),
        "mcs_atoms": result.numAtoms,
        "mcs_bonds": result.numBonds,
        "mcs_timed_out": bool(result.canceled),
        "expected_atom_coverage": result.numAtoms / max(expected.GetNumAtoms(), 1),
        "predicted_atom_coverage": result.numAtoms / max(predicted.GetNumAtoms(), 1),
        "expected_bond_coverage": result.numBonds / max(expected.GetNumBonds(), 1),
        "predicted_bond_coverage": result.numBonds / max(predicted.GetNumBonds(), 1),
    }


def render_sdf(sdf_path: Path, output_path: Path) -> None:
    try:
        mol = load_sdf(sdf_path)
        sanitized = True
    except ValueError:
        mol = load_sdf(sdf_path, sanitize=False)
        sanitized = False
    drawer = rdMolDraw2D.MolDraw2DCairo(768, 768)
    options = drawer.drawOptions()
    options.prepareMolsBeforeDrawing = sanitized
    options.bondLineWidth = 2.0
    options.fixedFontSize = 16
    options.padding = 0.08
    options.addStereoAnnotation = False
    options.setBackgroundColour((1.0, 1.0, 1.0, 1.0))
    options.useBWAtomPalette()
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_bytes(drawer.GetDrawingText())
    temporary.replace(output_path)


class BenchmarkRepository:
    def __init__(self, evaluations_root: Path) -> None:
        self.evaluations_root = evaluations_root.resolve()

    def run_directories(self) -> list[Path]:
        if not self.evaluations_root.is_dir():
            return []
        runs = []
        for path in self.evaluations_root.iterdir():
            if not path.is_dir():
                continue
            required = ("summary.json", "benchmark_config.json", "records.jsonl")
            if not all((path / name).is_file() for name in required):
                continue
            try:
                config = load_json(path / "benchmark_config.json")
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if not isinstance(config.get("benchmark_manifest"), str):
                continue
            runs.append(path)
        return sorted(
            runs,
            key=lambda path: (path / "records.jsonl").stat().st_mtime,
            reverse=True,
        )

    def run_directory(self, run_id: str) -> Path:
        if not run_id or Path(run_id).name != run_id:
            raise FileNotFoundError(run_id)
        path = (self.evaluations_root / run_id).resolve()
        if not path.is_relative_to(self.evaluations_root) or not path.is_dir():
            raise FileNotFoundError(run_id)
        return path

    def load_manifest(self, config: dict) -> tuple[Path, dict[str, dict]]:
        manifest_path = resolve_project_path(config["benchmark_manifest"])
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        return manifest_path.parent, {row["molecule_id"]: row for row in rows}

    def load_run(self, run_id: str) -> dict:
        run_dir = self.run_directory(run_id)
        config = load_json(run_dir / "benchmark_config.json")
        summary = load_json(run_dir / "summary.json")
        records = load_jsonl(run_dir / "records.jsonl")
        benchmark_root, manifest = self.load_manifest(config)
        return {
            "id": run_id,
            "directory": run_dir,
            "config": config,
            "summary": summary,
            "records": records,
            "record_index": {row["molecule_id"]: row for row in records},
            "benchmark_root": benchmark_root,
            "manifest": manifest,
            "updated_at": datetime.fromtimestamp(
                (run_dir / "records.jsonl").stat().st_mtime
            ).astimezone().isoformat(timespec="seconds"),
        }

    def list_runs(self) -> list[dict]:
        payload = []
        for path in self.run_directories():
            run = self.load_run(path.name)
            statuses = {"exact": 0, "similar": 0, "mismatch": 0, "invalid": 0}
            for record in run["records"]:
                statuses[sample_status(record)] += 1
            payload.append(
                {
                    "id": run["id"],
                    "updated_at": run["updated_at"],
                    "samples": len(run["records"]),
                    "statuses": statuses,
                    "summary": run["summary"],
                    "checkpoint": run["config"].get("checkpoint"),
                }
            )
        return payload

    def run_payload(self, run_id: str) -> dict:
        run = self.load_run(run_id)
        samples = []
        for record in run["records"]:
            manifest = run["manifest"].get(record["molecule_id"], {})
            samples.append(
                {
                    "molecule_id": record["molecule_id"],
                    "status": sample_status(record),
                    "difficulty": record.get("difficulty"),
                    "sdf_valid": bool(record.get("sdf_valid")),
                    "chemical_match": bool(record.get("chemical_match")),
                    "fingerprint_similarity": record.get("fingerprint_similarity"),
                    "expected_formula": manifest.get("formula"),
                    "actual_formula": record.get("actual_formula"),
                    "expected_atoms": record.get("expected_atoms"),
                    "predicted_atoms": record.get("predicted_atoms"),
                    "expected_bonds": record.get("expected_bonds"),
                    "predicted_bonds": record.get("predicted_bonds"),
                    "failure_category": record.get("failure_category"),
                    "primary_error": record.get("primary_error"),
                    "error_flags": record.get("error_flags", []),
                    "diagnostics": record.get("diagnostics"),
                    "error": record.get("error"),
                    "seconds": record.get("seconds"),
                }
            )
        return {
            "id": run["id"],
            "updated_at": run["updated_at"],
            "config": run["config"],
            "summary": run["summary"],
            "samples": samples,
        }

    def prediction_sdf(self, run: dict, molecule_id: str) -> Path | None:
        record = run["record_index"].get(molecule_id, {})
        relative = record.get("prediction_sdf_file")
        if relative:
            path = (run["directory"] / relative).resolve()
            if path.is_relative_to(run["directory"]) and path.is_file():
                return path
        legacy = run["directory"] / "sdf" / f"{molecule_id}.sdf"
        return legacy if legacy.is_file() else None

    def detail_payload(self, run_id: str, molecule_id: str) -> dict:
        run = self.load_run(run_id)
        record = run["record_index"].get(molecule_id)
        manifest = run["manifest"].get(molecule_id)
        if record is None or manifest is None:
            raise FileNotFoundError(molecule_id)
        predicted_sdf = self.prediction_sdf(run, molecule_id)
        comparison = None
        if predicted_sdf is not None and record.get("sdf_valid"):
            comparison = sdf_comparison(
                run["benchmark_root"] / manifest["sdf_file"],
                predicted_sdf,
            )
        base = f"/api/artifacts/{run_id}/{molecule_id}"
        public_manifest = {
            key: value
            for key, value in manifest.items()
            if key != "canonical_isomeric_smiles"
        }
        return {
            "record": record,
            "manifest": public_manifest,
            "status": sample_status(record),
            "comparison": comparison,
            "prediction_kind": (
                predicted_sdf.name.split("_", 1)[0]
                if predicted_sdf is not None and "_" in predicted_sdf.name
                else None
            ),
            "artifacts": {
                "ground_truth_image": f"{base}/ground-truth.png",
                "ground_truth_sdf": f"{base}/ground-truth.sdf",
                "prediction_image": (
                    f"{base}/prediction.png" if predicted_sdf is not None else None
                ),
                "prediction_sdf": (
                    f"{base}/prediction.sdf" if predicted_sdf is not None else None
                ),
                "raw_chemsvg": (
                    f"{base}/raw.chemsvg.txt"
                    if (
                        run["directory"]
                        / "raw_chemsvg"
                        / f"{molecule_id}.chemsvg.txt"
                    ).is_file()
                    else None
                ),
                "repaired_chemsvg": (
                    f"{base}/repaired.chemsvg.txt"
                    if (
                        run["directory"]
                        / "repaired_chemsvg"
                        / f"{molecule_id}.chemsvg.txt"
                    ).is_file()
                    else None
                ),
            },
        }

    def artifact_path(self, run_id: str, molecule_id: str, artifact: str) -> Path:
        run = self.load_run(run_id)
        manifest = run["manifest"].get(molecule_id)
        if manifest is None or molecule_id not in run["record_index"]:
            raise FileNotFoundError(molecule_id)
        if artifact == "ground-truth.png":
            path = run["benchmark_root"] / manifest["image_file"]
        elif artifact == "ground-truth.sdf":
            path = run["benchmark_root"] / manifest["sdf_file"]
        elif artifact == "prediction.sdf":
            path = self.prediction_sdf(run, molecule_id)
        elif artifact == "prediction.png":
            sdf_path = self.prediction_sdf(run, molecule_id)
            if sdf_path is None:
                path = None
            else:
                path = (
                    run["directory"]
                    / "visualizations"
                    / "predicted_images"
                    / f"{molecule_id}.png"
                )
                if not path.is_file():
                    render_sdf(sdf_path, path)
        elif artifact == "raw.chemsvg.txt":
            path = (
                run["directory"]
                / "raw_chemsvg"
                / f"{molecule_id}.chemsvg.txt"
            )
        elif artifact == "repaired.chemsvg.txt":
            path = (
                run["directory"]
                / "repaired_chemsvg"
                / f"{molecule_id}.chemsvg.txt"
            )
        else:
            path = None
        if path is None or not path.is_file():
            raise FileNotFoundError(artifact)
        return path


class BenchmarkHandler(SimpleHTTPRequestHandler):
    repository: BenchmarkRepository

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        parts = [unquote(part) for part in parsed.path.strip("/").split("/") if part]
        try:
            if parts == ["api", "runs"]:
                self.send_json({"runs": self.repository.list_runs()})
                return
            if len(parts) == 3 and parts[:2] == ["api", "runs"]:
                self.send_json(self.repository.run_payload(parts[2]))
                return
            if (
                len(parts) == 5
                and parts[:2] == ["api", "runs"]
                and parts[3] == "samples"
            ):
                self.send_json(self.repository.detail_payload(parts[2], parts[4]))
                return
            if len(parts) == 5 and parts[:2] == ["api", "artifacts"]:
                self.send_file(
                    self.repository.artifact_path(parts[2], parts[3], parts[4])
                )
                return
            self.serve_frontend(parsed.path)
        except FileNotFoundError:
            self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as error:
            self.send_json({"error": str(error)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def serve_frontend(self, request_path: str) -> None:
        relative = request_path.lstrip("/") or "index.html"
        path = (FRONTEND_ROOT / relative).resolve()
        if not path.is_relative_to(FRONTEND_ROOT) or not path.is_file():
            path = FRONTEND_ROOT / "index.html"
        self.send_file(path)

    def send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path) -> None:
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if path.suffix in {".sdf", ".txt"}:
            content_type = "text/plain; charset=utf-8"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--evaluations-root", default="outputs/evaluations")
    args = parser.parse_args()
    RDLogger.DisableLog("rdApp.*")
    BenchmarkHandler.repository = BenchmarkRepository(
        resolve_project_path(args.evaluations_root)
    )
    server = ThreadingHTTPServer((args.host, args.port), BenchmarkHandler)
    print(f"ChemVector benchmark browser: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
