"""Frozen protocol constants and artifact I/O for the external OCR R0 audit."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable


SCHEMA_VERSION = "external-ocr-candidate-audit-0.1"
SELECTION_SCHEMA_VERSION = "external-ocr-selection-0.1"
CROP_SCHEMA_VERSION = "external-ocr-crops-0.1"
ARM_SCHEMA_VERSION = "external-ocr-arm-0.1"
TRUTH_SCHEMA_VERSION = "external-ocr-truth-0.1"

FRESH_COMMITMENT_SHA256 = "13a456afb2c475a73fc4dca82df5fe62d4a31277d6b584637d9cd92736024512"
FRESH_COMMITMENT_FILE_SHA256 = "2e437737e8938ffd82ff09b230aa89878c7c646c42e37ced14255414ab165a76"
FRESH_SAMPLES_SHA256 = "b46571e188620e8e117e294648ae1a5a936cc8e0958c88da6ad830af0e9d0619"
FRESH_SAMPLES_PATH = Path("Data/sdf/chemvector_joint_v1/fresh_eval_12k/samples.csv")
FRESH_COMMITMENT_PATH = Path("Data/sdf/chemvector_joint_v1/fresh_eval_12k/commitment.json")

ROLE_CHECKPOINT = Path("outputs/checkpoints/chemvector-v0105-direct-atom-role-style10k/selected.pt")
BBOX_CHECKPOINT = Path("outputs/checkpoints/chemvector-v0108-direct-atom-text-bbox-style10k/selected.pt")
CTC_CHECKPOINT = Path("outputs/checkpoints/chemvector-v0109-correct-label-ocr-style10k/selected.pt")
SOURCE_CHECKPOINT = Path("outputs/checkpoints/chemvector-v0101-set-first-style10k/selected.pt")
GROUNDED_GRAPH_CHECKPOINT = Path(
    "outputs/checkpoints/chemvector-v0729-whole-set-stable-distance-overfit100/last.pt"
)
ROLE_SHA256 = "f7f8fe2e7df510ee9dfea36712cb90bdc4a386dd5d3db6973bc5079cd23c5f1b"
BBOX_SHA256 = "2370263a2a59b26d31939108ea9ebf0744fb73bc27f21dc0638719ad6033aece"
V0109_CTC_SHA256 = "dff869139c8e63451794ce8083932c06344d407774f652dc3baa18729abc8540"
SOURCE_SHA256 = "e982057d57dd4c03864eae890df12e1d8710fce60dfc452b94511d86a5f967ac"
GROUNDED_GRAPH_SHA256 = "1eede2e814efc0c440c26eba88bc8e9ed86e0911d7727b770a992c1ea2671c6a"

SELECTION_MOLECULES = 500
GROUP_MINIMUM = 200
GROUP_NAMES = ("simple_element", "modifier", "open_string")
REQUIRED_NATURAL_GROUPS = ("simple_element", "modifier")
HARD_GROUPS = ("modifier", "open_string")
CTC_BEAM_SIZE = 12
CTC_TOP_PATHS = 8
CTC_TOKEN_BEAM = 24
OVERALL_DELTA_PP = 2.0
HARD_GROUP_DELTA_PP = 5.0
INCREMENTAL_COVERAGE = 0.01
ARM_NAMES = (
    "ctc_v0109_greedy",
    "ctc_v0109_top8",
    "ppocrv5_mobile",
    "ppocrv5_server",
)
PADDLE_MODELS = {
    "ppocrv5_mobile": "en_PP-OCRv5_mobile_rec",
    "ppocrv5_server": "PP-OCRv5_server_rec",
}
PADDLE_MODEL_HASHES = {
    "ppocrv5_mobile": {
        "inference.yml": "27e91d0582f40168aa218303c76e184bc78fa7a5d105aad0cfbad8458b441067",
        "inference.json": "fd1b6ec722ea841a72d3ba43e527df1d1066d5d7808e0503ee3eec7265188753",
        "inference.pdiparams": "3ec8a97ed6cefe8568d3e2ee90bb193299b566a7661aa4fd52d224b96b59f66b",
    },
    "ppocrv5_server": {
        "inference.yml": "2c719dba044c4e2228aef8ff92f5f575394d75d24c16de096a33b7cfd902f66d",
        "inference.json": "8e6e12e5d42531840310977fffb58165bf889fc5061408c5a8afdb6985f47fcb",
        "inference.pdiparams": "63853f062a5f4089befc16f565a68277618e0da5cb45468b49d11079de0ada77",
    },
}

# These are filled only after their preceding audit phase has completed.
FROZEN_CROP_MANIFEST_SHA256: str | None = (
    "1bfe3dcb36e8f034cf933e940b24c1dffaf74c639bc7de3af1d3272ea34a0a29"
)
FROZEN_ARM_BUNDLE_SHA256: dict[str, str] = {
    "ctc_v0109_greedy": (
        "8ff71908574323348d3739f28aecb77dccac44a77c0073f44dd01d58f8b382cc"
    ),
    "ctc_v0109_top8": (
        "b086351badd139762e65cd2317ff9e5581c4257f12e1de35cf199cf2356631e3"
    ),
    "ppocrv5_mobile": (
        "e5c98b191a765d5d518a7fe097f1cef7f82500b1c7c73b39a1de2511b382cfd8"
    ),
    "ppocrv5_server": (
        "6fc8d8f5f655aaf25fcba29e6cb89fd5b6ca72b4c0eb4e62bb5abc829913798e"
    ),
}
FROZEN_TRUTH_MANIFEST_SHA256: str | None = (
    "d063dd9c4bcf339c377799aab3b4647ee3fc9cd89e67d04e684bd183d7df87c7"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def arm_bundle_sha256(record_path: str | Path) -> str:
    records = Path(record_path)
    metadata = records.with_suffix(".metadata.json")
    return sha256_bytes(
        canonical_json_bytes(
            {
                "metadata_sha256": sha256_file(metadata),
                "records_sha256": sha256_file(records),
            }
        )
    )


def write_new_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite audit artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(value))


def write_new_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    output = Path(path)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite audit artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line:
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row {number} is not an object: {path}")
        rows.append(value)
    return rows


def label_group(text: str) -> str:
    from chemvector.ocr import parse_atom_display_text

    parsed = parse_atom_display_text(text)
    if parsed is None:
        return "open_string"
    return "simple_element" if text == parsed["element"] else "modifier"


def has_modifier(text: str) -> bool:
    from chemvector.ocr import parse_atom_display_text

    parsed = parse_atom_display_text(text)
    if parsed is not None:
        return text != parsed["element"]
    return bool(re.search(r"H|[+-]|\d|\.", text))


def git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def validate_frozen_file(path: Path, expected_sha256: str, name: str) -> None:
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"Frozen {name} artifact SHA256 differs: {path}")
