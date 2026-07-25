#!/usr/bin/env python3
"""Build the committed 2,000-molecule/8,000-image fresh-dev package.

This is a data-only builder.  It consumes the already frozen fresh-dev
selection and the passed 16/64 preflight, writes through a staging directory,
and never imports or runs a model.  The first 64 generated samples are checked
byte-for-byte against the preflight tar before any final artifact is installed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from concurrent.futures import Future, ProcessPoolExecutor
import csv
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import stat
import tarfile
from typing import Any, Mapping, Sequence

import fastjsonschema
from rdkit import rdBase

try:
    from . import build_chemvision_layers as layer_builder
    from . import build_visual_svg_layers_c_fresh_dev_preflight as preflight
    from .build_visual_svg_capacity_100 import (
        GROUP_STRUCTURAL_PROTOCOL,
        json_bytes,
    )
    from .visual_svg_layers_c import HASHED_SUFFIXES, MEMBER_SUFFIXES, assemble_c_sample
    from .visual_svg_tokenizer_v0_1 import VisualSVGTokenizerV01
except ImportError:
    import build_chemvision_layers as layer_builder
    import build_visual_svg_layers_c_fresh_dev_preflight as preflight
    from build_visual_svg_capacity_100 import GROUP_STRUCTURAL_PROTOCOL, json_bytes
    from visual_svg_layers_c import HASHED_SUFFIXES, MEMBER_SUFFIXES, assemble_c_sample
    from visual_svg_tokenizer_v0_1 import VisualSVGTokenizerV01


SCHEMA_VERSION = "visual-svg-layers-c-fresh-dev-build-0.1"
PARTIAL_SCHEMA_VERSION = "visual-svg-layers-c-fresh-dev-partial-build-0.1"
MOLECULE_COUNT = 2_000
SAMPLE_COUNT = 8_000
MOLECULES_PER_SHARD = 250
DEFAULT_WORKERS = 10
PREFLIGHT_MOLECULE_COUNT = 16
PREFLIGHT_SAMPLE_COUNT = 64
FULL_RASTER_AUDIT_STRIDE = 100
PREFLIGHT_VALIDATION_SCHEMA = (
    "visual-svg-layers-c-fresh-dev-preflight-validation-0.1"
)
PREFLIGHT_STATUS = "PREFLIGHT-PASS"
MOLECULE_MANIFEST = "molecule_manifest.dev.csv"
SAMPLE_MANIFEST = "sample_manifest.dev.csv"
PAIRED_MANIFEST = "paired_manifest.dev.csv"
BUILD_MANIFEST = "build.dev.json"
FONT_INVENTORY = [
    {
        "family": "STIX",
        "sha256": "73b6d579f94b4870856e4ae8ddc19cd31b353078a02db57edf93ee8ca5049f45",
    },
    {
        "family": "P052",
        "sha256": "f054a7389d8d0e9f4c5f7b4c21feb7e2b14bec8004a1e92778a78d149524eea9",
    },
]
MOLECULE_FIELDS = (
    *preflight.SELECTION_FIELDS,
    "inchikey",
    "role",
    "split",
    "variants",
    "sdf_sha256",
    "sdf_bytes",
    "atom_count",
    "bond_count",
)
SAMPLE_FIELDS = (
    "sample_key",
    "selection_index",
    "molecule_id",
    "role",
    "split",
    "variant",
    "scaffold_id",
    "source_record_id",
    "source_structure",
    "source_sdf_sha256",
    "style_id",
    "font_sha",
    "render_profile",
    "group_diagnostics_mode",
    "image_sha256",
    "sdf_sha256",
    "shard",
    "sdf_inchikey",
    "sequence_tokens",
    "primitive_count",
    "visual_svg_v02_group_count",
)


class FreshDevBuildError(ValueError):
    """Raised when a frozen full-build precondition is violated."""


_WORKER_SOURCE: Path | None = None
_WORKER_SAMPLE_ROWS: Mapping[tuple[str, str], Mapping[str, str]] | None = None
_WORKER_SOURCE_ASSETS: Mapping[
    tuple[str, str],
    tuple[bytes, Mapping[str, Any]],
] | None = None
_WORKER_FONTS: Sequence[Mapping[str, str]] | None = None
_WORKER_MOLECULES_PER_SHARD: int | None = None
_WORKER_PRIMITIVE_VALIDATOR = None
_WORKER_TOKENIZER: VisualSVGTokenizerV01 | None = None


def _regular_file(path: Path) -> Path:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError as error:
            raise FreshDevBuildError(f"missing_file:{absolute}") from error
        if stat.S_ISLNK(mode):
            raise FreshDevBuildError(f"symlink_rejected:{current}")
    if not stat.S_ISREG(absolute.stat().st_mode):
        raise FreshDevBuildError(f"not_regular_file:{absolute}")
    return absolute


def _read_json(path: Path) -> dict[str, Any]:
    path = _regular_file(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FreshDevBuildError(f"invalid_json:{path}") from error
    if not isinstance(value, dict):
        raise FreshDevBuildError(f"json_root_not_object:{path}")
    return value


def _safe_relative_file(root: Path, value: object, *, label: str) -> Path:
    text = str(value)
    preflight._assert_allowed_text(text, context=label)
    relative = Path(text)
    if relative.is_absolute() or ".." in relative.parts:
        raise FreshDevBuildError(f"unsafe_relative_path:{label}:{text}")
    candidate = _regular_file(root / relative)
    if os.path.commonpath((str(root), str(candidate))) != str(root):
        raise FreshDevBuildError(f"path_escapes_root:{label}:{text}")
    return candidate


def _load_preflight_members(
    preflight_root: Path,
    *,
    selection_commitment_sha256: str,
) -> tuple[dict[str, dict[str, bytes]], dict[str, Any], str]:
    """Load only the already passed canonical 16/64 preflight members."""

    report_path = _regular_file(
        preflight_root / "preflight_validation.json"
    )
    report = _read_json(report_path)
    expected = {
        "schema_version": PREFLIGHT_VALIDATION_SCHEMA,
        "status": PREFLIGHT_STATUS,
        "role": preflight.ROLE,
        "split": preflight.SPLIT,
        "training_steps": 0,
        "parameter_updates": 0,
        "model_run": False,
        "threshold_or_checkpoint_selection": False,
        "sealed_rows_read": False,
        "molecules": PREFLIGHT_MOLECULE_COUNT,
        "samples": PREFLIGHT_SAMPLE_COUNT,
        "checked_samples": PREFLIGHT_SAMPLE_COUNT,
        "errors": [],
        "selection_commitment_sha256": selection_commitment_sha256,
    }
    for name, expected_value in expected.items():
        if report.get(name) != expected_value:
            raise FreshDevBuildError(f"preflight_validation_mismatch:{name}")
    input_commitment = _read_json(preflight_root / "preflight_input_commitment.json")
    shard_ref = input_commitment.get("shard")
    if not isinstance(shard_ref, Mapping):
        raise FreshDevBuildError("preflight_shard_reference_missing")
    shard = _safe_relative_file(
        preflight_root,
        shard_ref.get("path", ""),
        label="preflight_shard",
    )
    shard_sha256 = preflight.file_sha256(shard)
    if (
        shard_ref.get("sha256") != shard_sha256
        or report.get("shard_sha256") != shard_sha256
        or int(shard_ref.get("samples", -1)) != PREFLIGHT_SAMPLE_COUNT
        or int(shard_ref.get("members", -1))
        != PREFLIGHT_SAMPLE_COUNT * len(MEMBER_SUFFIXES)
    ):
        raise FreshDevBuildError("preflight_shard_commitment_mismatch")

    expected_keys = {
        f"fresh_dev_{index:06d}_{variant}"
        for index in range(PREFLIGHT_MOLECULE_COUNT)
        for variant in preflight.VARIANTS
    }
    canonical: dict[str, dict[str, bytes]] = defaultdict(dict)
    with tarfile.open(shard, "r") as archive:
        members = archive.getmembers()
        if len(members) != PREFLIGHT_SAMPLE_COUNT * len(MEMBER_SUFFIXES):
            raise FreshDevBuildError("preflight_tar_member_count")
        for member in members:
            if not member.isfile() or "/" in member.name or "\\" in member.name:
                raise FreshDevBuildError(f"preflight_tar_member:{member.name}")
            suffix = next(
                (
                    item
                    for item in sorted(MEMBER_SUFFIXES, key=len, reverse=True)
                    if member.name.endswith(item)
                ),
                None,
            )
            if suffix is None:
                raise FreshDevBuildError(f"preflight_unknown_suffix:{member.name}")
            key = member.name[: -len(suffix)]
            if key not in expected_keys or suffix in canonical[key]:
                raise FreshDevBuildError(f"preflight_member_identity:{member.name}")
            extracted = archive.extractfile(member)
            if extracted is None:
                raise FreshDevBuildError(f"preflight_member_unreadable:{member.name}")
            canonical[key][suffix] = extracted.read()
    if set(canonical) != expected_keys or any(
        set(members) != set(MEMBER_SUFFIXES) for members in canonical.values()
    ):
        raise FreshDevBuildError("preflight_member_closure")
    return dict(canonical), report, preflight.file_sha256(report_path)


def _write_csv(
    path: Path,
    fields: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fields),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def _generated_paths(output: Path) -> tuple[Path, ...]:
    return (
        output / "shards",
        output / MOLECULE_MANIFEST,
        output / SAMPLE_MANIFEST,
        output / PAIRED_MANIFEST,
        output / BUILD_MANIFEST,
        output / "candidate_rejected.dev.csv",
    )


def _ensure_output_available(output: Path, *, overwrite: bool) -> None:
    existing = [path for path in _generated_paths(output) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Fresh-dev output exists; pass --overwrite: {existing[0]}"
        )


def _install_staged(stage: Path, output: Path, *, overwrite: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    _ensure_output_available(output, overwrite=overwrite)
    if overwrite:
        for path in _generated_paths(output):
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
    for name in (
        "shards",
        MOLECULE_MANIFEST,
        SAMPLE_MANIFEST,
        PAIRED_MANIFEST,
        BUILD_MANIFEST,
    ):
        os.replace(stage / name, output / name)


def _copy_exact_if_needed(source: Path, destination: Path) -> None:
    source = _regular_file(source)
    if destination.exists():
        destination = _regular_file(destination)
        if preflight.file_sha256(source) != preflight.file_sha256(destination):
            raise FreshDevBuildError(f"supporting_artifact_conflict:{destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _install_supporting_artifacts(
    *,
    selection_commitment_path: Path,
    freeze_path: Path,
    preflight_root: Path,
    output: Path,
) -> None:
    """Make alternate/partial output roots self-contained using exact copies."""

    source_fresh_root = selection_commitment_path.parent
    if source_fresh_root != output:
        commitment = _read_json(selection_commitment_path)
        references = [
            selection_commitment_path.name,
            str(commitment["selection_manifest"]["path"]),
            str(commitment["fresh_replacement_manifest"]["path"]),
            str(commitment["identity_isolation_report"]["path"]),
            str(commitment["selection_provenance"]["path"]),
        ]
        for relative in references:
            _copy_exact_if_needed(
                source_fresh_root / relative,
                output / relative,
            )
        _copy_exact_if_needed(freeze_path, output.parent / "phase_a_freeze.json")

    input_commitment = _read_json(preflight_root / "preflight_input_commitment.json")
    shard_ref = input_commitment.get("shard")
    if not isinstance(shard_ref, Mapping):
        raise FreshDevBuildError("preflight_shard_reference_missing")
    preflight_files = (
        "preflight_validation.json",
        "preflight_input_commitment.json",
        "molecules.csv",
        "samples.csv",
        "paired.csv",
        "summary.json",
        str(shard_ref["path"]),
    )
    if preflight_root != output / "preflight_64":
        for relative in preflight_files:
            _copy_exact_if_needed(
                preflight_root / relative,
                output / "preflight_64" / relative,
            )


def _shard_relative(selection_index: int, molecules_per_shard: int) -> str:
    return f"shards/dev-{selection_index // molecules_per_shard:06d}.tar"


def _group_diagnostics_mode(selection_index: int) -> str:
    if (
        selection_index < PREFLIGHT_MOLECULE_COUNT
        or selection_index % FULL_RASTER_AUDIT_STRIDE == 0
    ):
        return "full_raster"
    return "structural"


def _compare_preflight_members(
    key: str,
    members: Mapping[str, bytes],
    canonical: Mapping[str, Mapping[str, bytes]],
) -> None:
    expected = canonical.get(key)
    if expected is None:
        return
    for suffix in MEMBER_SUFFIXES:
        if members[suffix] != expected[suffix]:
            raise FreshDevBuildError(
                f"preflight_byte_mismatch:{key}{suffix}"
            )


def _init_build_worker(
    source: str,
    source_sample_rows: Mapping[tuple[str, str], Mapping[str, str]],
    source_assets: Mapping[
        tuple[str, str],
        tuple[bytes, Mapping[str, Any]],
    ],
    fonts: Sequence[Mapping[str, str]],
    molecules_per_shard: int,
    primitive_schema_path: str,
) -> None:
    """Install immutable build inputs in one rendering process."""

    global _WORKER_SOURCE
    global _WORKER_SAMPLE_ROWS
    global _WORKER_SOURCE_ASSETS
    global _WORKER_FONTS
    global _WORKER_MOLECULES_PER_SHARD
    global _WORKER_PRIMITIVE_VALIDATOR
    global _WORKER_TOKENIZER
    _WORKER_SOURCE = Path(source)
    _WORKER_SAMPLE_ROWS = source_sample_rows
    _WORKER_SOURCE_ASSETS = source_assets
    _WORKER_FONTS = fonts
    _WORKER_MOLECULES_PER_SHARD = molecules_per_shard
    schema = json.loads(Path(primitive_schema_path).read_text(encoding="utf-8"))
    _WORKER_PRIMITIVE_VALIDATOR = fastjsonschema.compile(schema)
    _WORKER_TOKENIZER = VisualSVGTokenizerV01()


def _build_molecule(identity: Mapping[str, str]) -> dict[str, Any]:
    """Render all four variants for one committed molecule without file writes."""

    if (
        _WORKER_SOURCE is None
        or _WORKER_SAMPLE_ROWS is None
        or _WORKER_SOURCE_ASSETS is None
        or _WORKER_FONTS is None
        or _WORKER_MOLECULES_PER_SHARD is None
        or _WORKER_PRIMITIVE_VALIDATOR is None
        or _WORKER_TOKENIZER is None
    ):
        raise FreshDevBuildError("build_worker_not_initialized")

    selection_index = int(identity["selection_index"])
    sdf_bytes, sdf_sha = preflight._load_source_sdf(_WORKER_SOURCE, identity)
    validated_chemmol_sha: set[str] = set()
    molecule_samples = []
    for variant in preflight.VARIANTS:
        source_row = _WORKER_SAMPLE_ROWS[(identity["molecule_id"], variant)]
        chemmol_bytes, source_render = _WORKER_SOURCE_ASSETS[
            (identity["molecule_id"], variant)
        ]
        chemmol_sha = preflight.sha256_bytes(chemmol_bytes)
        if chemmol_sha != identity["source_chemmol_sha256"]:
            raise FreshDevBuildError(
                f"committed_chemmol_sha256:{identity['molecule_id']}:{variant}"
            )
        preflight._validate_chemmol_against_sdf(
            chemmol_bytes,
            sdf_bytes,
            identity=identity,
        )
        validated_chemmol_sha.add(chemmol_sha)
        font = _WORKER_FONTS[
            preflight._stable_int(
                preflight.FREEZE_ID,
                "font",
                identity["molecule_id"],
                variant,
            )
            % len(_WORKER_FONTS)
        ]
        task, style = preflight._render_task(
            identity,
            source_row,
            variant=variant,
            font=font,
            selection_index=selection_index,
        )
        shard_relative = _shard_relative(
            selection_index,
            _WORKER_MOLECULES_PER_SHARD,
        )
        task.output_shard = shard_relative
        members, diagnostics = assemble_c_sample(
            task,
            chemmol_bytes,
            sdf_bytes,
            source_render,
            primitive_validator=_WORKER_PRIMITIVE_VALIDATOR,
            tokenizer=_WORKER_TOKENIZER,
            output_split=preflight.SPLIT,
            source_dataset="webdataset_full_v0/train",
            structure_ref=identity["source_structure"],
            fresh_audit_eligible=True,
            max_tokens=preflight.MAX_TOKENS,
            group_diagnostics_mode=_group_diagnostics_mode(selection_index),
        )
        preflight._enrich_members(
            members,
            identity=identity,
            style=style,
            source_sdf_sha256=sdf_sha,
        )
        diagnostics.update(
            {
                "selection_index": selection_index,
                "role": preflight.ROLE,
                "scaffold_id": identity["scaffold_id"],
                "source_record_id": identity["source_record_id"],
                "source_structure": identity["source_structure"],
                "source_sdf_sha256": sdf_sha,
                "style_id": style["style_id"],
                "font_sha": style["descriptor"]["font_sha256"],
                "render_profile": preflight.RENDER_PROFILE,
                "shard": shard_relative,
            }
        )
        molecule_samples.append((task.sample_key, members, diagnostics))

    if validated_chemmol_sha != {identity["source_chemmol_sha256"]}:
        raise FreshDevBuildError(
            f"source_chemmol_variant_bytes:{identity['molecule_id']}"
        )
    sdf_mol = preflight._parse_single_sdf(
        sdf_bytes,
        molecule_id=identity["molecule_id"],
    )
    return {
        "selection_index": selection_index,
        "sdf_sha256": sdf_sha,
        "sdf_bytes": len(sdf_bytes),
        "atom_count": sdf_mol.GetNumAtoms(),
        "bond_count": sdf_mol.GetNumBonds(),
        "samples": molecule_samples,
    }


def _ordered_molecule_futures(
    selected: Sequence[dict[str, str]],
    *,
    workers: int,
    source: Path,
    source_sample_rows: Mapping[tuple[str, str], Mapping[str, str]],
    source_assets: Mapping[
        tuple[str, str],
        tuple[bytes, Mapping[str, Any]],
    ],
    fonts: Sequence[Mapping[str, str]],
    molecules_per_shard: int,
    primitive_schema_path: Path,
):
    """Yield futures in selection order with bounded parallel buffering."""

    initargs = (
        str(source),
        source_sample_rows,
        source_assets,
        fonts,
        molecules_per_shard,
        str(primitive_schema_path),
    )
    if workers == 1:
        _init_build_worker(*initargs)
        for identity in selected:
            future: Future = Future()
            try:
                future.set_result(_build_molecule(identity))
            except BaseException as error:
                future.set_exception(error)
            yield identity, future
        return

    effective_workers = min(workers, len(selected))
    context = mp.get_context("fork")
    with ProcessPoolExecutor(
        max_workers=effective_workers,
        mp_context=context,
        initializer=_init_build_worker,
        initargs=initargs,
    ) as executor:
        identities = iter(selected)
        pending = deque()
        for _ in range(min(len(selected), effective_workers * 2)):
            identity = next(identities)
            pending.append((identity, executor.submit(_build_molecule, identity)))
        while pending:
            identity, future = pending.popleft()
            try:
                next_identity = next(identities)
            except StopIteration:
                pass
            else:
                pending.append(
                    (next_identity, executor.submit(_build_molecule, next_identity))
                )
            yield identity, future


def build(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.source).expanduser().resolve()
    freeze_path = Path(args.freeze).expanduser().resolve()
    selection_commitment_path = Path(args.selection_commitment).expanduser().resolve()
    preflight_root = Path(args.preflight_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve(strict=False)
    for value, label in (
        (source, "source_root"),
        (selection_commitment_path, "selection_commitment"),
        (preflight_root, "preflight_root"),
        (output, "output_root"),
    ):
        preflight._assert_allowed_text(value, context=label)
    molecule_limit = int(args.molecule_limit)
    molecules_per_shard = int(args.molecules_per_shard)
    workers = int(getattr(args, "workers", DEFAULT_WORKERS))
    if not 1 <= molecule_limit <= MOLECULE_COUNT:
        raise FreshDevBuildError("molecule_limit_out_of_range")
    if molecules_per_shard < 1:
        raise FreshDevBuildError("molecules_per_shard_must_be_positive")
    if workers < 1:
        raise FreshDevBuildError("workers_must_be_positive")
    full_build = molecule_limit == MOLECULE_COUNT

    preflight._validate_frozen_source(source)
    selection, selection_commitment, identity_audit = (
        preflight.load_committed_selection(selection_commitment_path, freeze_path)
    )
    selection_sha256 = preflight.file_sha256(selection_commitment_path)
    canonical_preflight, preflight_report, preflight_validation_sha256 = (
        _load_preflight_members(
            preflight_root,
            selection_commitment_sha256=selection_sha256,
        )
    )
    freeze = _read_json(freeze_path)
    fonts = preflight._font_specs_checked(freeze)
    selected = selection[:molecule_limit]
    if [int(row["selection_index"]) for row in selected] != list(
        range(molecule_limit)
    ):
        raise FreshDevBuildError("selection_index_order_mismatch")
    preflight._verify_source_molecule_rows(source, selected)
    source_sample_rows = preflight._load_source_sample_rows(source, selected)
    source_assets = preflight._read_tar_members(source, source_sample_rows)
    _ensure_output_available(output, overwrite=args.overwrite)

    schema_path = (
        Path(__file__).resolve().parent
        / "visual_svg_v0_1/schemas/visual-primitives-0.1.schema.json"
    )
    stage = output / f".fresh-dev-build-{os.getpid()}.tmp"
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "shards").mkdir(parents=True)

    molecule_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    shard_buffers: dict[
        str,
        list[tuple[str, Mapping[str, bytes], Mapping[str, Any]]],
    ] = defaultdict(list)
    rejected: list[dict[str, str]] = []
    preflight_member_matches = 0
    max_tokens = 0
    group_diagnostic_sample_counts = {"full_raster": 0, "structural": 0}
    try:
        molecule_futures = _ordered_molecule_futures(
            selected,
            workers=workers,
            source=source,
            source_sample_rows=source_sample_rows,
            source_assets=source_assets,
            fonts=fonts,
            molecules_per_shard=molecules_per_shard,
            primitive_schema_path=schema_path,
        )
        for identity, future in molecule_futures:
            selection_index = int(identity["selection_index"])
            try:
                molecule_result = future.result()
                if molecule_result["selection_index"] != selection_index:
                    raise FreshDevBuildError("worker_selection_index_mismatch")
                molecule_samples = molecule_result["samples"]
                for key, members, diagnostics in molecule_samples:
                    expected_mode = _group_diagnostics_mode(selection_index)
                    actual_mode = diagnostics.get("group_diagnostics_mode")
                    if actual_mode != expected_mode:
                        raise FreshDevBuildError(
                            "group_diagnostics_schedule_mismatch:"
                            f"{key}:{actual_mode}:{expected_mode}"
                        )
                    group_diagnostic_sample_counts[actual_mode] += 1
                    _compare_preflight_members(
                        key,
                        members,
                        canonical_preflight,
                    )
                    if key in canonical_preflight:
                        preflight_member_matches += len(MEMBER_SUFFIXES)
                    max_tokens = max(max_tokens, int(diagnostics["sequence_tokens"]))
            except Exception as error:
                rejected.append(
                    {
                        "selection_index": identity["selection_index"],
                        "molecule_id": identity["molecule_id"],
                        "stage": "pre_install_contract",
                        "error_code": type(error).__name__,
                        "error_message": str(error),
                    }
                )
                raise

            molecule_rows.append(
                {
                    **{field: identity[field] for field in preflight.SELECTION_FIELDS},
                    "inchikey": identity["inchikey_full"],
                    "role": preflight.ROLE,
                    "split": preflight.SPLIT,
                    "variants": len(preflight.VARIANTS),
                    "sdf_sha256": molecule_result["sdf_sha256"],
                    "sdf_bytes": molecule_result["sdf_bytes"],
                    "atom_count": molecule_result["atom_count"],
                    "bond_count": molecule_result["bond_count"],
                }
            )
            for key, members, diagnostics in molecule_samples:
                shard_buffers[diagnostics["shard"]].append(
                    (key, members, diagnostics)
                )
                paired_row = {
                    "sample_key": key,
                    "molecule_id": diagnostics["molecule_id"],
                    "inchikey": diagnostics["sdf_inchikey"],
                    "scaffold_id": diagnostics["scaffold_id"],
                    "source_record_id": diagnostics["source_record_id"],
                    "role": preflight.ROLE,
                    "split": preflight.SPLIT,
                    "variant": diagnostics["variant"],
                    "shard": diagnostics["shard"],
                    "style_id": diagnostics["style_id"],
                    "font_sha": diagnostics["font_sha"],
                    "render_profile": preflight.RENDER_PROFILE,
                    "image_sha256": preflight.sha256_bytes(members[".png"]),
                    "sdf_sha256": preflight.sha256_bytes(members[".sdf"]),
                    "primitives_sha256": preflight.sha256_bytes(
                        members[".primitives.json"]
                    ),
                    "visualsvg_v01_sha256": preflight.sha256_bytes(
                        members[".visualsvg.txt"]
                    ),
                    "visualsvg_v02_sha256": preflight.sha256_bytes(
                        members[".visualsvg-v02.json"]
                    ),
                    "layers_sha256": preflight.sha256_bytes(
                        members[".layers.json"]
                    ),
                    "render_sha256": preflight.sha256_bytes(
                        members[".render.json"]
                    ),
                    "chemmol_sha256": preflight.sha256_bytes(
                        members[".chemmol.json"]
                    ),
                    "meta_sha256": preflight.sha256_bytes(
                        members[".meta.json"]
                    ),
                }
                paired_rows.append(paired_row)
                sample_rows.append(
                    {
                        **diagnostics,
                        "image_sha256": paired_row["image_sha256"],
                        "sdf_sha256": paired_row["sdf_sha256"],
                    }
                )

            shard_relative = _shard_relative(selection_index, molecules_per_shard)
            shard_end = (
                selection_index + 1 == molecule_limit
                or (selection_index + 1) % molecules_per_shard == 0
            )
            if shard_end:
                shard_samples = shard_buffers.pop(shard_relative)
                preflight._write_shard(stage / shard_relative, shard_samples)

        expected_samples = molecule_limit * len(preflight.VARIANTS)
        if shard_buffers:
            raise FreshDevBuildError("unflushed_shard_buffers")
        if len(molecule_rows) != molecule_limit or len(sample_rows) != expected_samples:
            raise FreshDevBuildError("build_count_mismatch")
        if preflight_member_matches != min(
            molecule_limit,
            PREFLIGHT_MOLECULE_COUNT,
        ) * len(preflight.VARIANTS) * len(MEMBER_SUFFIXES):
            raise FreshDevBuildError("preflight_byte_match_count")

        sample_rows.sort(key=lambda row: row["sample_key"])
        paired_rows.sort(key=lambda row: row["sample_key"])
        _write_csv(stage / MOLECULE_MANIFEST, MOLECULE_FIELDS, molecule_rows)
        _write_csv(stage / SAMPLE_MANIFEST, SAMPLE_FIELDS, sample_rows)
        _write_csv(stage / PAIRED_MANIFEST, preflight.PAIRED_FIELDS, paired_rows)

        shard_records = []
        for shard in sorted((stage / "shards").glob("dev-*.tar")):
            relative = shard.relative_to(stage).as_posix()
            count = sum(row["shard"] == relative for row in paired_rows)
            shard_records.append(
                {
                    "path": relative,
                    "sha256": preflight.file_sha256(shard),
                    "bytes": shard.stat().st_size,
                    "samples": count,
                    "members": count * len(MEMBER_SUFFIXES),
                }
            )
        if sum(row["samples"] for row in shard_records) != expected_samples:
            raise FreshDevBuildError("shard_sample_count_mismatch")

        selection_artifacts = {
            "selection_commitment_sha256": selection_sha256,
            "selection_manifest_sha256": selection_commitment["selection_manifest"][
                "sha256"
            ],
            "replacement_manifest_sha256": selection_commitment[
                "fresh_replacement_manifest"
            ]["sha256"],
            "identity_isolation_report_sha256": selection_commitment[
                "identity_isolation_report"
            ]["sha256"],
            "selection_provenance_sha256": selection_commitment[
                "selection_provenance"
            ]["sha256"],
        }
        full_raster_molecules = sum(
            _group_diagnostics_mode(index) == "full_raster"
            for index in range(molecule_limit)
        )
        structural_molecules = molecule_limit - full_raster_molecules
        expected_group_diagnostic_sample_counts = {
            "full_raster": full_raster_molecules * len(preflight.VARIANTS),
            "structural": structural_molecules * len(preflight.VARIANTS),
        }
        if group_diagnostic_sample_counts != expected_group_diagnostic_sample_counts:
            raise FreshDevBuildError("group_diagnostics_count_mismatch")
        build_report = {
            "schema_version": SCHEMA_VERSION if full_build else PARTIAL_SCHEMA_VERSION,
            "status": "built_pending_independent_validation"
            if full_build
            else "partial_built_for_verification",
            "finalizable": full_build,
            "freeze_id": preflight.FREEZE_ID,
            "freeze_sha256": preflight.file_sha256(freeze_path),
            "role": preflight.ROLE,
            "split": preflight.SPLIT,
            "training_steps": 0,
            "parameter_updates": 0,
            "model_run": False,
            "threshold_or_checkpoint_selection": False,
            "sealed_rows_read": False,
            "molecules": molecule_limit,
            "samples": expected_samples,
            "members_per_sample": len(MEMBER_SUFFIXES),
            "members": expected_samples * len(MEMBER_SUFFIXES),
            "width": 768,
            "height": 768,
            "max_tokens": preflight.MAX_TOKENS,
            "max_sequence_tokens": max_tokens,
            "sdf_protocol": preflight.SDF_PROTOCOL,
            "render_profile": preflight.RENDER_PROFILE,
            "variants": list(preflight.VARIANTS),
            "molecules_per_shard": molecules_per_shard,
            "rdkit_version": rdBase.rdkitVersion,
            "preflight_validation": {
                "path": "preflight_64/preflight_validation.json",
                "sha256": preflight_validation_sha256,
                "status": preflight_report["status"],
            },
            "preflight_byte_identity": {
                "molecules": min(molecule_limit, PREFLIGHT_MOLECULE_COUNT),
                "samples": min(molecule_limit, PREFLIGHT_MOLECULE_COUNT)
                * len(preflight.VARIANTS),
                "members": preflight_member_matches,
                "all_members_equal": True,
            },
            "group_diagnostics": {
                "full_raster_rule": (
                    f"selection_index<{PREFLIGHT_MOLECULE_COUNT} or "
                    f"selection_index%{FULL_RASTER_AUDIT_STRIDE}==0"
                ),
                "full_raster_molecules": full_raster_molecules,
                "full_raster_samples": group_diagnostic_sample_counts[
                    "full_raster"
                ],
                "structural_protocol": GROUP_STRUCTURAL_PROTOCOL,
                "structural_molecules": structural_molecules,
                "structural_samples": group_diagnostic_sample_counts["structural"],
            },
            "selection_artifact_sha256": selection_artifacts,
            "identity_isolation_status": identity_audit["status"],
            "font_inventory": FONT_INVENTORY,
            "implementation_sha256": {
                "builder_sha256": preflight.file_sha256(Path(__file__).resolve()),
                "style_id_algorithm_sha256": identity_audit[
                    "style_id_algorithm_sha256"
                ],
                "scaffold_id_algorithm_sha256": identity_audit[
                    "scaffold_id_algorithm_sha256"
                ],
                "source_record_normalization_algorithm_sha256": identity_audit[
                    "source_record_normalization_algorithm_sha256"
                ],
            },
            "manifests": {
                "molecule": {
                    "path": MOLECULE_MANIFEST,
                    "sha256": preflight.file_sha256(stage / MOLECULE_MANIFEST),
                    "rows": len(molecule_rows),
                },
                "sample": {
                    "path": SAMPLE_MANIFEST,
                    "sha256": preflight.file_sha256(stage / SAMPLE_MANIFEST),
                    "rows": len(sample_rows),
                },
                "paired": {
                    "path": PAIRED_MANIFEST,
                    "sha256": preflight.file_sha256(stage / PAIRED_MANIFEST),
                    "rows": len(paired_rows),
                },
            },
            "shards": shard_records,
            "errors": [],
        }
        (stage / BUILD_MANIFEST).write_bytes(json_bytes(build_report))
        _install_staged(stage, output, overwrite=args.overwrite)
        _install_supporting_artifacts(
            selection_commitment_path=selection_commitment_path,
            freeze_path=freeze_path,
            preflight_root=preflight_root,
            output=output,
        )
        stage.rmdir()
        return {"output": str(output), **build_report}
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        if rejected:
            output.mkdir(parents=True, exist_ok=True)
            rejected_path = output / "candidate_rejected.dev.csv"
            _write_csv(
                rejected_path,
                (
                    "selection_index",
                    "molecule_id",
                    "stage",
                    "error_code",
                    "error_message",
                ),
                rejected,
            )
        raise


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    version_root = root / "visual_svg_layers_c_v0_1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(root / "webdataset_full_v0"))
    parser.add_argument("--freeze", default=str(version_root / "phase_a_freeze.json"))
    parser.add_argument(
        "--selection-commitment",
        default=str(version_root / "fresh_dev/selection.commitment.json"),
    )
    parser.add_argument(
        "--preflight-root",
        default=str(version_root / "fresh_dev/preflight_64"),
    )
    parser.add_argument("--output", default=str(version_root / "fresh_dev"))
    parser.add_argument("--molecule-limit", type=int, default=MOLECULE_COUNT)
    parser.add_argument(
        "--molecules-per-shard",
        type=int,
        default=MOLECULES_PER_SHARD,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Molecule rendering processes (10 is suitable for a 16-core host).",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    report = build(build_parser().parse_args())
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "BUILD_MANIFEST",
    "DEFAULT_WORKERS",
    "FONT_INVENTORY",
    "MOLECULE_COUNT",
    "MOLECULE_MANIFEST",
    "PAIRED_MANIFEST",
    "SAMPLE_COUNT",
    "SAMPLE_MANIFEST",
    "SCHEMA_VERSION",
    "FreshDevBuildError",
    "build",
    "build_parser",
    "main",
]
