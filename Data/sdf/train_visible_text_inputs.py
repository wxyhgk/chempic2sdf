"""Read-only, allowlisted train inputs for the visible-carbon review.

No renderer, OCR, dataset discovery, or revision writes belong here. Freeze
hashes the selected tar files (not unselected training tar files). Iteration
revalidates the freeze and binds every returned payload to the manifest.
"""

from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import tarfile


APPROVED_ROOT = Path(__file__).resolve().parent / "chemvector_joint_v1/style_train_100k"
VARIANTS = ("original", "rdkit", "coordgen", "stress")
SELECTION_SALT = "visible-carbon-review-v1:"
SCHEMA = "train-visible-text-selection-0.1"
SUFFIXES = {"png": ".png", "layers": ".layers.json",
            "chemmol": ".chemmol.json", "render": ".render.json"}
SHARD_PATTERN = re.compile(r"shards/train-[0-9]{6}\.tar\Z")
KEY_PATTERN = re.compile(r"train_[0-9]{6}_(original|rdkit|coordgen|stress)\Z")
HASH_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


def _canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True,
                       separators=(",", ":")) + "\n").encode()


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _reject_symlinks(path: Path) -> None:
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError(f"Symlink paths are forbidden: {path}")


def _root(root: str | Path) -> Path:
    root = Path(root).absolute()
    if ".." in root.parts:
        raise ValueError("Traversal is forbidden")
    _reject_symlinks(root)
    if root != APPROVED_ROOT.absolute():
        raise ValueError("Only the approved style_train_100k root is allowed")
    if not root.is_dir():
        raise ValueError("Approved train root is missing")
    return root


def _path(root: Path, relative: str, *, shard: bool = False) -> Path:
    if not isinstance(relative, str):
        raise ValueError("Input path must be a string")
    parts = PurePosixPath(relative)
    if (parts.is_absolute() or ".." in parts.parts or "\\" in relative
            or parts.as_posix() != relative):
        raise ValueError("Noncanonical or traversal input path")
    if shard and not SHARD_PATTERN.fullmatch(relative):
        raise ValueError("Only explicit train shards are allowed")
    path = root / relative
    _reject_symlinks(path)
    if not path.is_file():
        raise ValueError(f"Train input is not a regular file: {relative}")
    return path


def _file_record(path: Path, relative: str) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": relative, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _check_record(record: dict) -> None:
    if (not isinstance(record.get("sha256"), str)
            or not HASH_PATTERN.fullmatch(record["sha256"])
            or type(record.get("bytes")) is not int or record["bytes"] < 0):
        raise ValueError("Malformed committed file record")


def _read_manifest(root: Path) -> tuple[dict, dict, dict]:
    commitment_path = _path(root, "commitment.json")
    commitment_bytes = commitment_path.read_bytes()
    commitment = json.loads(commitment_bytes)
    claimed = commitment.pop("commitment_sha256", None)
    if claimed != _digest(_canonical(commitment)):
        raise ValueError("Commitment self-hash mismatch")
    if (commitment.get("dataset") != "style_train_100k"
            or commitment.get("schema_version") != "chemvector-joint-commitment-1.0"
            or commitment.get("split_counts") != {"train": commitment.get("samples")}):
        raise ValueError("Commitment must describe a train-only style dataset")
    manifest_records = commitment.get("samples_csv", [])
    if len(manifest_records) != 1 or manifest_records[0].get("path") != "samples.csv":
        raise ValueError("Only the explicit train samples.csv manifest is allowed")
    manifest_record = manifest_records[0]
    _check_record(manifest_record)
    manifest_path = _path(root, "samples.csv")
    manifest_bytes = manifest_path.read_bytes()
    if (len(manifest_bytes) != manifest_record["bytes"]
            or _digest(manifest_bytes) != manifest_record["sha256"]):
        raise ValueError("Manifest hash/size mismatch")
    tar_records = {}
    for record in commitment.get("tars", []):
        _check_record(record)
        relative = record["path"]
        _path(root, relative, shard=True)
        if relative in tar_records:
            raise ValueError("Duplicate committed shard")
        tar_records[relative] = record
    rows = {}
    grouped = defaultdict(set)
    for row in csv.DictReader(io.StringIO(manifest_bytes.decode("utf-8"))):
        key = row.get("sample_key", "")
        if (row.get("split") != "train" or not KEY_PATTERN.fullmatch(key)
                or row.get("variant") not in VARIANTS
                or not key.endswith("_" + row["variant"])):
            raise ValueError("Manifest sample must have a valid train identity/variant")
        if key in rows or not row.get("molecule_id"):
            raise ValueError("Duplicate sample or missing molecule identity")
        relative = row.get("output_shard", "")
        if not SHARD_PATTERN.fullmatch(relative) or relative not in tar_records:
            raise ValueError("Manifest points outside committed train shards")
        for kind in SUFFIXES:
            if not HASH_PATTERN.fullmatch(row.get(kind + "_sha256", "")):
                raise ValueError("Manifest payload hash missing or malformed")
        if row["variant"] in grouped[row["molecule_id"]]:
            raise ValueError("Duplicate molecule variant")
        grouped[row["molecule_id"]].add(row["variant"])
        rows[key] = row
    if (len(rows) != commitment["samples"]
            or len(grouped) != commitment.get("molecules")
            or any(value != set(VARIANTS) for value in grouped.values())):
        raise ValueError("Manifest counts or four-variant molecule groups differ")
    sources = {"commitment": {"path": "commitment.json", "bytes": len(commitment_bytes),
                               "sha256": _digest(commitment_bytes)},
               "manifest": manifest_record, "commitment_sha256": claimed}
    return rows, tar_records, sources


def freeze_selection(root: str | Path, count: int = 50) -> dict:
    """Return a JSON-serializable selection; this function writes nothing."""
    root = _root(root)
    if type(count) is not int or count < 1:
        raise ValueError("Molecule count must be a positive integer")
    rows, tar_records, sources = _read_manifest(root)
    molecules = sorted({row["molecule_id"] for row in rows.values()},
                       key=lambda value: (_digest((SELECTION_SALT + value).encode()), value))
    if count > len(molecules):
        raise ValueError("Requested more molecules than available")
    chosen = set(molecules[:count])
    selected = {key: row for key, row in sorted(rows.items()) if row["molecule_id"] in chosen}
    shards = {}
    for relative in sorted({row["output_shard"] for row in selected.values()}):
        record = _file_record(_path(root, relative, shard=True), relative)
        if record != tar_records[relative]:
            raise ValueError(f"Committed shard hash/size mismatch: {relative}")
        shards[relative] = record
    return {"schema_version": SCHEMA, "root": str(root), "selection_salt": SELECTION_SALT,
            "molecule_count": count, "molecule_ids": molecules[:count],
            "selection": list(selected), "sample_metadata": selected,
            "selected_shards": shards, "sources": sources}


def iter_selected_samples(root: str | Path, selection: dict):
    """Yield identity-checked JSON objects and original bytes, never extract tar paths.

    Item keys: key, metadata, payloads (png/layers/chemmol/render bytes),
    layers, chemmol, render. PNG decoding is the presentation layer's job.
    """
    root = _root(root)
    if selection != freeze_selection(root, selection.get("molecule_count")):
        raise ValueError("Frozen selection differs from verified train inputs")
    wanted = selection["sample_metadata"]
    for relative in selection["selected_shards"]:
        shard_keys = {key for key, row in wanted.items() if row["output_shard"] == relative}
        payloads = defaultdict(dict)
        seen = set()
        with tarfile.open(_path(root, relative, shard=True), "r:") as archive:
            for member in archive:
                name = member.name
                if (not member.isfile() or PurePosixPath(name).name != name
                        or "\\" in name or not name.startswith("train_")):
                    raise ValueError("Unsafe/nontrain tar member")
                key, separator, suffix = name.partition(".")
                if not separator or not KEY_PATTERN.fullmatch(key):
                    raise ValueError("Invalid train tar member identity")
                if name in seen:
                    raise ValueError("Duplicate tar member")
                seen.add(name)
                if key not in shard_keys:
                    continue
                kinds = [kind for kind, ending in SUFFIXES.items() if "." + suffix == ending]
                if not kinds:
                    continue
                kind = kinds[0]
                handle = archive.extractfile(member)
                if handle is None:
                    raise ValueError("Missing regular tar payload")
                with handle:
                    payload = handle.read()
                if _digest(payload) != wanted[key][kind + "_sha256"]:
                    raise ValueError(f"Payload hash mismatch: {name}")
                payloads[key][kind] = payload
        for key in sorted(shard_keys):
            raw = payloads[key]
            if set(raw) != set(SUFFIXES):
                raise ValueError(f"Missing selected payloads: {key}")
            documents = {kind: json.loads(raw[kind]) for kind in ("layers", "chemmol", "render")}
            row = wanted[key]
            for kind, doc in documents.items():
                if doc.get("molecule_id") != row["molecule_id"]:
                    raise ValueError(f"Payload molecule identity differs: {key}/{kind}")
                if kind in {"layers", "render"} and doc.get("variant") != row["variant"]:
                    raise ValueError("Payload variant differs")
            layers = documents["layers"]
            if layers.get("sample_key") != key or layers.get("split") != "train":
                raise ValueError("Layers are not the selected train sample")
            yield {"key": key, "metadata": row, "payloads": raw, **documents}
