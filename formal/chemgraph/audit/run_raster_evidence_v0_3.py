"""Run the preregistered Raster-neutral Evidence v0.3 finite audit."""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Iterable, Sequence

from .raster_candidate_enumerator_v0_3 import (
    ForwardCandidateIndex,
    build_forward_candidate_index,
    enumerate_candidate_graphs,
    foreground_key_from_image,
    iter_domain_scenes,
    render_public_scene,
)


SCHEMA_VERSION = "raster-evidence-audit-v0.3"
SUPPORTED_DOMAINS = ("simple_domain", "crossing_domain")
ALLOWED_EXTRACTOR_IMPORTS = {"__future__", "json", "struct", "sys", "zlib"}
FORBIDDEN_SOURCE_TOKENS = (
    "canonical_graph",
    "graph_signature",
    "hashlib",
    "manifest",
    "networkx",
    "rdkit",
    "scene_id",
    "socket",
    ".sdf",
)
FORBIDDEN_CALL_NAMES = {
    "breakpoint", "compile", "eval", "exec", "globals", "input", "locals",
    "open", "vars", "__import__",
}


class RasterAuditError(RuntimeError):
    """A finite audit invariant failed."""


@dataclass(frozen=True, slots=True)
class ObservationInput:
    domain: str
    foreground_key: tuple[tuple[int, int, int], ...]
    png_bytes: bytes
    oracle_graphs: tuple[str, ...]
    scene_ids: tuple[str, ...]


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value: Any) -> str:
    data = _json_bytes(value)
    path.write_bytes(data)
    return _sha256(data)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    with path.open("wb") as output:
        for row in rows:
            data = _canonical_json_bytes(row)
            output.write(data)
            digest.update(data)
    return digest.hexdigest()


def _png_bytes(image) -> bytes:
    output = BytesIO()
    image.save(output, format="PNG", optimize=False)
    return output.getvalue()


def static_extractor_audit(path: Path) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    imports: set[str] = set()
    calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add((node.module or "").split(".")[0])
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            calls.add(node.func.id)
    forbidden_tokens = sorted(
        token for token in FORBIDDEN_SOURCE_TOKENS if token in source.lower()
    )
    unexpected_imports = sorted(imports - ALLOWED_EXTRACTOR_IMPORTS)
    forbidden_calls = sorted(calls & FORBIDDEN_CALL_NAMES)
    return {
        "allowed_imports": sorted(ALLOWED_EXTRACTOR_IMPORTS),
        "forbidden_call_names_found": forbidden_calls,
        "forbidden_source_tokens_found": forbidden_tokens,
        "imports_found": sorted(imports),
        "passed": not unexpected_imports and not forbidden_calls and not forbidden_tokens,
        "source_sha256": _sha256(path.read_bytes()),
        "unexpected_imports": unexpected_imports,
    }


def _isolated_command(extractor_path: Path) -> list[str]:
    return ["/usr/bin/python3", "-I", "-S", str(extractor_path.resolve())]


def run_extractor(extractor_path: Path, png_bytes: bytes, workdir: Path) -> tuple[bytes, dict[str, Any]]:
    completed = subprocess.run(
        _isolated_command(extractor_path),
        input=png_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=workdir,
        env={"LC_ALL": "C", "PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"},
        check=False,
        timeout=20,
    )
    if completed.returncode != 0:
        raise RasterAuditError(
            f"extractor_exit_{completed.returncode}:{completed.stderr.decode('utf-8', 'replace')}"
        )
    if completed.stderr:
        raise RasterAuditError("extractor_stderr_not_empty")
    try:
        evidence = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RasterAuditError(f"extractor_invalid_json:{error}") from error
    if _canonical_json_bytes(evidence) != completed.stdout:
        raise RasterAuditError("extractor_output_not_canonical_json")
    return completed.stdout, evidence


def runtime_syscall_audit(extractor_path: Path, png_bytes: bytes) -> dict[str, Any]:
    strace = shutil.which("strace")
    if strace is None:
        return {"passed": False, "reason": "strace_not_available"}
    with tempfile.TemporaryDirectory(prefix="raster-evidence-strace-") as raw_workdir:
        workdir = Path(raw_workdir)
        trace_path = workdir / "trace.txt"
        completed = subprocess.run(
            [
                strace, "-f", "-qq", "-e", "trace=%file,%network", "-o", str(trace_path),
                *_isolated_command(extractor_path),
            ],
            input=png_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=workdir,
            env={"LC_ALL": "C", "PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"},
            check=False,
            timeout=30,
        )
        trace = trace_path.read_text(encoding="utf-8", errors="replace")
    repository_root = Path(__file__).resolve().parents[3]
    extractor = extractor_path.resolve()
    quoted_paths = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', trace)
    forbidden_paths: list[str] = []
    for raw_path in quoted_paths:
        if not raw_path.startswith("/"):
            continue
        path = Path(raw_path)
        try:
            inside_repo = path.resolve().is_relative_to(repository_root)
        except (OSError, RuntimeError):
            inside_repo = False
        if inside_repo and path.resolve() != extractor:
            forbidden_paths.append(raw_path)
    network_calls = sorted(
        set(
            line.split("(", 1)[0].split()[-1]
            for line in trace.splitlines()
            if any(token in line for token in ("socket(", "connect(", "accept(", "sendto(", "recvfrom("))
        )
    )
    normalized_trace = trace.replace(str(extractor), "<EXTRACTOR>")
    normalized_trace = normalized_trace.replace(raw_workdir, "<WORKDIR>")
    normalized_trace = re.sub(r"(?m)^\d+\s+", "<PID> ", normalized_trace)
    normalized_trace = re.sub(r"0x[0-9a-f]+", "<ADDR>", normalized_trace)
    return {
        "extractor_exit_code": completed.returncode,
        "extractor_stderr_empty": not completed.stderr,
        "forbidden_repository_paths": sorted(set(forbidden_paths)),
        "network_calls": network_calls,
        "passed": (
            completed.returncode == 0
            and not completed.stderr
            and not forbidden_paths
            and not network_calls
        ),
        "trace_sha256_normalized": _sha256(normalized_trace.encode("utf-8")),
    }


def _observation_inputs(domain: str, index: ForwardCandidateIndex) -> list[ObservationInput]:
    if set(index.png_by_foreground) != set(index.candidates_by_foreground):
        raise RasterAuditError("forward_index_and_runner_domain_disagree")
    return [
        ObservationInput(
            domain=domain,
            foreground_key=key,
            png_bytes=index.png_by_foreground[key],
            oracle_graphs=tuple(sorted(index.candidates_by_foreground[key])),
            scene_ids=index.scene_ids_by_foreground[key],
        )
        for key in sorted(index.png_by_foreground)
    ]


def _extract_inputs(
    extractor_path: Path,
    inputs: Sequence[ObservationInput],
    workdir: Path,
    workers: int,
) -> list[tuple[bytes, dict[str, Any]]]:
    def execute(item: ObservationInput):
        return run_extractor(extractor_path, item.png_bytes, workdir)

    if workers == 1:
        return [execute(item) for item in inputs]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(execute, inputs))


def _fixture_factorization(extractor_path: Path, workdir: Path) -> dict[str, Any]:
    from .finite_micro_scene_v0_1 import crossing_junction_fixtures, double_parallel_fixtures

    pairs = {
        "crossing_junction": crossing_junction_fixtures(),
        "parallel_multiplicity": double_parallel_fixtures(),
    }
    results: dict[str, Any] = {}
    for kind, pair in pairs.items():
        pngs = tuple(_png_bytes(render_public_scene(scene)) for scene in pair)
        evidences = tuple(run_extractor(extractor_path, png, workdir)[0] for png in pngs)
        results[kind] = {
            "evidence_bytes_equal": evidences[0] == evidences[1],
            "evidence_sha256": _sha256(evidences[0]),
            "png_bytes_equal": pngs[0] == pngs[1],
            "png_sha256": _sha256(pngs[0]),
            "scene_ids": [scene.scene_id for scene in pair],
        }
    return results


def run_audit(
    output_dir: Path,
    *,
    extractor_path: Path,
    smoke_limit: int | None,
    workers: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    extractor_path = extractor_path.resolve()
    static_audit = static_extractor_audit(extractor_path)
    manifest_rows: list[dict[str, Any]] = []
    domain_summary: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="raster-evidence-v0-3-") as raw_workdir:
        workdir = Path(raw_workdir)
        fixture_png = _png_bytes(render_public_scene(next(iter(iter_domain_scenes("crossing_domain")))))
        syscall_audit = runtime_syscall_audit(extractor_path, fixture_png)
        fixture_factorization = _fixture_factorization(extractor_path, workdir)
        for domain in SUPPORTED_DOMAINS:
            index = build_forward_candidate_index(domain)
            inputs = _observation_inputs(domain, index)
            full_observation_count = len(inputs)
            if smoke_limit is not None:
                collision_inputs = [item for item in inputs if len(item.oracle_graphs) > 1]
                selected = inputs[:smoke_limit]
                seen = {item.foreground_key for item in selected}
                selected.extend(item for item in collision_inputs if item.foreground_key not in seen)
                inputs = sorted(selected, key=lambda item: item.foreground_key)
            extracted = _extract_inputs(extractor_path, inputs, workdir, workers)
            counters = {
                "ambiguous": 0,
                "candidate_complete": 0,
                "candidate_sound": 0,
                "graph": 0,
                "reject": 0,
                "wrong_graph": 0,
            }
            for item, (evidence_bytes, evidence) in zip(inputs, extracted, strict=True):
                predicted = tuple(sorted(enumerate_candidate_graphs(evidence, index)))
                oracle = item.oracle_graphs
                sound = set(predicted) <= set(oracle)
                complete = set(oracle) <= set(predicted)
                counters["candidate_sound"] += sound
                counters["candidate_complete"] += complete
                if not predicted:
                    decision = "Reject"
                    counters["reject"] += 1
                    output_graph = None
                elif len(predicted) == 1:
                    decision = "Graph"
                    counters["graph"] += 1
                    output_graph = predicted[0]
                    counters["wrong_graph"] += output_graph not in oracle or len(oracle) != 1
                else:
                    decision = "Ambiguous"
                    counters["ambiguous"] += 1
                    output_graph = None
                manifest_rows.append(
                    {
                        "candidate_complete": complete,
                        "candidate_sound": sound,
                        "decision": decision,
                        "domain": domain,
                        "evidence_sha256": _sha256(evidence_bytes),
                        "oracle_candidate_graphs": list(oracle),
                        "output_graph": output_graph,
                        "png_sha256_audit_identifier_only": _sha256(item.png_bytes),
                        "predicted_candidate_graphs": list(predicted),
                        "scene_ids_audit_only": list(item.scene_ids),
                    }
                )
            domain_summary[domain] = {
                "audited_observation_count": len(inputs),
                "candidate_complete_count": counters["candidate_complete"],
                "candidate_sound_count": counters["candidate_sound"],
                "decision_counts": {
                    "Ambiguous": counters["ambiguous"],
                    "Graph": counters["graph"],
                    "Reject": counters["reject"],
                },
                "forward_enumerator_observation_count": index.observation_count,
                "forward_enumerator_scene_count": index.scene_count,
                "full_observation_count": full_observation_count,
                "wrong_graph_count": counters["wrong_graph"],
            }

    is_full = smoke_limit is None
    finite_candidate_gate = is_full and all(
        summary["candidate_complete_count"] == summary["audited_observation_count"]
        and summary["candidate_sound_count"] == summary["audited_observation_count"]
        and summary["wrong_graph_count"] == 0
        and summary["decision_counts"]["Reject"] == 0
        for summary in domain_summary.values()
    )
    fixture_gate = all(
        record["png_bytes_equal"] and record["evidence_bytes_equal"]
        for record in fixture_factorization.values()
    )
    summary = {
        "candidate_enumerator": {
            "method": "runtime_full_foreground_forward_enumeration_no_digest_lookup",
            "status": "passed_finite_domain" if finite_candidate_gate else "not_passed",
        },
        "domains": domain_summary,
        "extractor": {
            "runtime_syscall_audit": syscall_audit,
            "static_audit": static_audit,
        },
        "fixture_factorization": fixture_factorization,
        "mode": "full" if is_full else "smoke",
        "preregistration_commit": "caa8ad1",
        "schema_version": SCHEMA_VERSION,
        "training_authorized": False,
    }
    manifest_sha = _write_jsonl(
        output_dir / "raster_evidence_manifest.jsonl",
        sorted(manifest_rows, key=lambda row: (row["domain"], row["png_sha256_audit_identifier_only"])),
    )
    summary["artifacts"] = {"raster_evidence_manifest_sha256": manifest_sha}
    summary_sha = _write_json(output_dir / "summary.json", summary)
    validation = {
        "candidate_soundness_and_completeness_100_percent": finite_candidate_gate,
        "exact_collision_factorization": fixture_gate,
        "extractor_runtime_isolation": syscall_audit["passed"],
        "extractor_static_isolation": static_audit["passed"],
        "hidden_challenge_passed": False,
        "independent_verifier_passed": False,
        "overall_status": "no_go_pending_hidden_challenge_stability_and_independent_review",
        "raster_factorization_finite_domain": finite_candidate_gate and fixture_gate,
        "schema_version": SCHEMA_VERSION,
        "stability_gate_passed": False,
        "summary_sha256": summary_sha,
        "training_authorized": False,
    }
    validation_sha = _write_json(output_dir / "validation.json", validation)
    return {
        "summary": summary,
        "summary_sha256": summary_sha,
        "validation": validation,
        "validation_sha256": validation_sha,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--extractor",
        type=Path,
        default=Path(__file__).with_name("raster_neutral_extractor_v0_3.py"),
    )
    parser.add_argument("--smoke-limit", type=int)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_limit is not None and args.smoke_limit < 1:
        raise SystemExit("--smoke-limit must be positive")
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    result = run_audit(
        args.output_dir,
        extractor_path=args.extractor,
        smoke_limit=args.smoke_limit,
        workers=args.workers,
    )
    print(json.dumps(result["validation"], ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
