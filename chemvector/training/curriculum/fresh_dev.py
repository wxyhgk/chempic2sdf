"""Read-only, validation-only access to committed legacy fresh-dev PNG/SDF.

Plan fixes the commitment and deterministic molecule selection BEFORE reading
samples.csv or any pixels. Only the two allowlisted fresh-dev path patterns
are followed. No legacy builder, private identity worker or model is imported.
The plan is provenance, not training admission or sealed-test authorization.
"""

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import io
import json
from pathlib import Path
import re
import tarfile

from PIL import Image

from .admission import identities
from .validation_truth import expected_identity

FORMAT = "fresh-dev-original-image-sdf"
PLAN_SCHEMA = "chemvector-fresh-dev-access-plan-0.1"
VARIANTS = {"original", "rdkit", "coordgen", "stress"}
KEY = r"fresh_dev_[0-9]+_(original|rdkit|coordgen|stress)"
SHARD = r"shards/fresh_dev-[0-9]{6}\.tar"
HASH_FIELDS = ("png_sha256", "sdf_sha256", "meta_sha256")
ROW_FIELDS = ("sample_key", "molecule_id", "inchikey", "variant", "output_shard", *HASH_FIELDS)


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _safe_path(path):
    path = Path(path)
    if (not path.is_absolute() or ".." in path.parts
            or {"sealed_test", "work"} & set(path.parts)
            or any(p.is_symlink() for p in (path, *path.parents))):
        raise ValueError("Unsafe, sealed or symlinked validation path")
    return path


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)


def _signature(stream):
    import os
    s = os.fstat(stream.fileno())
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def _metadata(root):
    raw = {name: _safe_path(root / name).read_bytes() for name in ("commitment.json", "validation.json")}
    commitment, validation = (json.loads(raw[name]) for name in ("commitment.json", "validation.json"))
    unsigned = {k: v for k, v in commitment.items() if k != "commitment_sha256"}
    if (commitment.get("schema_version") != "chemvector-joint-commitment-1.0"
            or commitment.get("dataset") != "fresh_eval_12k"
            or commitment.get("sealed_access") != "locked_until_checkpoint_gate"
            or sha256(_json_bytes(unsigned)).hexdigest() != commitment.get("commitment_sha256")
            or set(commitment.get("split_counts", {})) != {"fresh_dev", "sealed_test"}
            or validation.get("dataset") != "fresh_eval_12k"
            or validation.get("status") != "passed" or validation.get("errors")):
        raise ValueError("Expected passed, hash-consistent fresh evaluation commitment")
    counts = commitment["split_counts"]
    if (any(type(n) is not int or n <= 0 or n % 4 for n in counts.values())
            or commitment.get("samples") != sum(counts.values())
            or commitment.get("molecules") != sum(counts.values()) // 4):
        raise ValueError("Invalid committed evaluation counts")
    for field in ("molecule_id_set_sha256", "inchikey_set_sha256", "scaffold_set_sha256",
                  "font_family_groups_sha256"):
        if not _digest(commitment.get(field)):
            raise ValueError("Missing committed identity/font set hash")
    if not _digest(commitment.get("renderer_style_schema", {}).get("sha256")):
        raise ValueError("Missing committed renderer/style hash")
    for kind in ("samples_csv", "tars"):
        records = commitment.get(kind, [])
        if not records or len({r["path"] for r in records}) != len(records):
            raise ValueError("Missing/duplicate commitment file records")
        for record in records:
            if not _digest(record.get("sha256")) or type(record.get("bytes")) is not int or record["bytes"] <= 0:
                raise ValueError("Invalid commitment file hash/size")
    csv_paths = {r["path"] for r in commitment["samples_csv"]}
    if csv_paths != {"samples.csv", "sealed_test/samples.csv"}:
        raise ValueError("Unexpected evaluation manifest paths")
    if not any(re.fullmatch(SHARD, r["path"]) for r in commitment["tars"]):
        raise ValueError("No committed fresh-dev shards")
    return commitment, {name: sha256(data).hexdigest() for name, data in raw.items()}


def prepare_plan(root, *, molecules, seed):
    """Aggregate-only. Caller must save and hash this plan before using it."""
    root = _safe_path(root)
    commitment, hashes = _metadata(root)
    if (type(molecules) is not int or not 1 <= molecules <= commitment["split_counts"]["fresh_dev"] // 4
            or type(seed) is not int):
        raise ValueError("Explicit bounded molecule count and integer seed required")
    return {"schema_version": PLAN_SCHEMA, "root": str(root), "commitment": commitment,
            "metadata_sha256": hashes, "selection": {"molecules": molecules, "seed": seed,
                "policy": "sha256(seed:molecule_id); all four variants; no image/model-dependent selection"},
            "use": "development_validation_only", "training_authorized": False,
            "sealed_access": False,
            "limitations": ["Aggregate identity hashes are pinned, not reverified by opening sealed rows.",
                "Legacy grouping is not proof of stereo/isotope-independent fresh/sealed separation.",
                "This plan is neither data admission nor evidence of model accuracy."]}


class FreshDevSource:
    def __init__(self, root, binding):
        if binding.get("format") != FORMAT or not _digest(binding.get("access_plan_sha256")):
            raise ValueError("Explicit validation format and access-plan hash required")
        self.root = _safe_path(root)
        payload = _safe_path(binding["access_plan"]).read_bytes()
        if sha256(payload).hexdigest() != binding["access_plan_sha256"]:
            raise ValueError("Validation access-plan hash mismatch")
        plan = json.loads(payload)
        selection = plan.get("selection", {})
        expected = prepare_plan(self.root, molecules=selection.get("molecules"), seed=selection.get("seed"))
        if plan != expected:
            raise ValueError("Validation access plan changed or belongs to a different source")
        self.plan = plan
        committed = plan["commitment"]
        self.shards = {r["path"]: r for r in committed["tars"] if re.fullmatch(SHARD, r["path"])}
        record = next(r for r in committed["samples_csv"] if r["path"] == "samples.csv")
        payload = _safe_path(self.root / "samples.csv").read_bytes()
        if len(payload) != record["bytes"] or sha256(payload).hexdigest() != record["sha256"]:
            raise ValueError("Committed fresh-dev sample manifest hash/size mismatch")
        rows = list(csv.DictReader(io.StringIO(payload.decode())))
        if (len(rows) != committed["split_counts"]["fresh_dev"]
                or len({r["sample_key"] for r in rows}) != len(rows)):
            raise ValueError("Fresh-dev sample count or duplicate key mismatch")
        by_molecule = defaultdict(list)
        for row in rows:
            if (row["split"] != "fresh_dev" or not re.fullmatch(KEY, row["sample_key"])
                    or row["variant"] not in VARIANTS or not row["sample_key"].endswith("_" + row["variant"])
                    or row["output_shard"] not in self.shards or not row["molecule_id"]
                    or not re.fullmatch(r"[A-Z]{14}-[A-Z]{10}-[A-Z]", row["inchikey"])
                    or any(not _digest(row.get(k)) for k in HASH_FIELDS)):
                raise ValueError("Unsafe/non-fresh-dev or invalid manifest row")
            by_molecule[row["molecule_id"]].append(row)
        if len(by_molecule) * 4 != len(rows):
            raise ValueError("Fresh-dev molecule count mismatch")
        for group in by_molecule.values():
            if (len(group) != 4 or {r["variant"] for r in group} != VARIANTS
                    or len({(r["inchikey"], r["scaffold"], r["sdf_sha256"]) for r in group}) != 1):
                raise ValueError("Fresh-dev molecule variants/identity inconsistent")
        selected = sorted(by_molecule, key=lambda mid: (
            sha256(f"{selection['seed']}:{mid}".encode()).hexdigest(), mid))[:selection["molecules"]]
        self.rows = {r["sample_key"]: r for mid in selected for r in sorted(by_molecule[mid], key=lambda r: r["sample_key"])}
        self._indexes = {}

    def _read_payloads(self, row):
        name = row["output_shard"]
        path = _safe_path(self.root / name)
        with path.open("rb") as stream:
            signature = _signature(stream)
            if name not in self._indexes:
                committed = self.shards[name]
                digest = sha256()
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
                if signature[2] != committed["bytes"] or digest.hexdigest() != committed["sha256"]:
                    raise ValueError("Committed fresh-dev shard hash/size mismatch")
                stream.seek(0)
                members = {}
                with tarfile.open(fileobj=stream, mode="r:") as archive:
                    for member in archive:
                        if (not re.fullmatch(KEY + r"\.(png|text\.png|bond\.png|semantic\.png|chemmol\.json|render\.json|layers\.json|sdf|meta\.json)", member.name)
                                or not member.isfile() or member.name in members
                                or not 0 < member.size <= 16 * 1024 * 1024):
                            raise ValueError("Unsafe, duplicate or non-file fresh-dev tar member")
                        members[member.name] = (member.offset_data, member.size)
                if _signature(stream) != signature:
                    raise ValueError("Validation shard changed while indexing")
                self._indexes[name] = (signature, members)
            saved_signature, members = self._indexes[name]
            if signature != saved_signature:
                raise ValueError("Validation shard changed after verification")
            payloads = {}
            for suffix, field in ((".png", "png_sha256"), (".sdf", "sdf_sha256")):
                member = row["sample_key"] + suffix
                if member not in members:
                    raise ValueError("Missing validation image/SDF member")
                offset, size = members[member]
                stream.seek(offset)
                payload = stream.read(size)
                if len(payload) != size or sha256(payload).hexdigest() != row[field]:
                    raise ValueError("Validation member hash mismatch")
                payloads[suffix] = payload
            if _signature(stream) != signature:
                raise ValueError("Validation shard changed while reading")
        return payloads

    def check_row(self, row):
        if row.get("split") != "validation" or row.get("domain") != "full_molecule":
            raise ValueError("Original fresh-dev data is validation-only, never training")
        original = self.rows.get(row.get("sample_key"))
        if original is None or any(row.get(k) != original[k] for k in ROW_FIELDS):
            raise ValueError("Validation row differs from fixed selected manifest")

    def load(self, row):
        self.check_row(row)  # Reject training before reading any tar.
        original = self.rows[row["sample_key"]]
        payloads = self._read_payloads(original)
        sample = {"target_molblock": payloads[".sdf"].decode(), "purpose": "diagnostic"}
        canonical = expected_identity(sample)
        ids = identities(canonical)
        if (ids["inchikey"] != original["inchikey"]
                or row.get("canonical_smiles", canonical) != canonical
                or row.get("group_key", ids["group_key"]) != ids["group_key"]):
            raise ValueError("Original SDF and validation molecular identity disagree")
        with Image.open(io.BytesIO(payloads[".png"])) as image:
            if (image.format != "PNG" or image.mode != "RGB" or image.size != (768, 768)
                    or getattr(image, "n_frames", 1) != 1):
                raise ValueError("Original validation image shape/mode must be 768px RGB PNG")
            image.load()
            sample["image"] = image.copy()  # Never resize, convert, render or annotate.
        return sample

    def candidate_rows(self, source):
        """Engineering check/manifest fragment only, without training admission."""
        result = []
        for original in self.rows.values():
            row = {k: original[k] for k in ROW_FIELDS}
            row.update(source=source, split="validation", domain="full_molecule")
            sample = self.load(row)
            canonical = expected_identity(sample)
            row.update(canonical_smiles=canonical, group_key=identities(canonical)["group_key"])
            result.append(row)
        return result


def _write(path, value):
    with Path(path).open("xb") as stream:
        stream.write(_json_bytes(value))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="Pin aggregates and selection; no per-sample reads")
    plan.add_argument("--root", type=Path, required=True)
    plan.add_argument("--molecules", type=int, required=True)
    plan.add_argument("--seed", type=int, required=True)
    plan.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("check", help="CPU original-image/SDF verification, never training")
    check.add_argument("--plan", type=Path, required=True)
    check.add_argument("--plan-sha256", required=True)
    check.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "plan":
        value = prepare_plan(args.root, molecules=args.molecules, seed=args.seed)
        _write(args.output, value)
        print(json.dumps({"access_plan": str(args.output), "sha256": sha256(_json_bytes(value)).hexdigest(),
                          "sample_payloads_read": False}))
        return 0
    binding = {"format": FORMAT, "access_plan": str(_safe_path(args.plan)), "access_plan_sha256": args.plan_sha256}
    payload = args.plan.read_bytes()
    if sha256(payload).hexdigest() != args.plan_sha256:
        raise ValueError("Validation access-plan hash mismatch")
    root = json.loads(payload)["root"]
    source = FreshDevSource(root, binding)
    args.output.mkdir(parents=True, exist_ok=False)
    rows = source.candidate_rows("fresh_dev")
    fragment = {"schema_version": "chemvector-validation-candidates-0.1", "training_authorized": False,
                "sources": {"fresh_dev": root}, "validation_sources": {"fresh_dev": binding}, "samples": rows}
    _write(args.output / "validation-candidates.json", fragment)
    report = {"images": len(rows), "molecules": len({r["molecule_id"] for r in rows}),
              "by_variant": dict(Counter(r["variant"] for r in rows)),
              "original_pixels": True, "training_authorized": False, "gpu_used": False,
              "model_loaded": False, "sealed_payloads_read": False,
              "access_plan_sha256": args.plan_sha256,
              "validation_candidates_sha256": sha256(_json_bytes(fragment)).hexdigest(),
              "verified_shards": [source.shards[name] for name in sorted(source._indexes)],
              "limitations": source.plan["limitations"]}
    _write(args.output / "report.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
