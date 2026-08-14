from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile

from PIL import Image, ImageDraw

from .finite_micro_scene_v0_1 import (
    crossing_junction_fixtures,
    double_parallel_fixtures,
)
from .raster_candidate_enumerator_v0_3 import (
    CandidateEnumerationError,
    ForwardCandidateIndex,
    enumerate_candidate_graphs,
    foreground_key_from_evidence,
    render_public_scene,
)
from .run_raster_evidence_v0_3 import (
    _png_bytes,
    run_extractor,
    static_extractor_audit,
)


EXTRACTOR = Path(__file__).with_name("raster_neutral_extractor_v0_3.py")


def _extract(image: Image.Image) -> tuple[bytes, dict]:
    with tempfile.TemporaryDirectory() as raw_workdir:
        return run_extractor(EXTRACTOR, _png_bytes(image), Path(raw_workdir))


def test_extractor_static_dependency_closure() -> None:
    result = static_extractor_audit(EXTRACTOR)
    assert result["passed"] is True
    assert result["unexpected_imports"] == []
    assert result["forbidden_call_names_found"] == []
    assert result["forbidden_source_tokens_found"] == []


def test_crossing_and_junction_factorize_to_identical_evidence() -> None:
    crossing, junction = crossing_junction_fixtures()
    crossing_png = _png_bytes(render_public_scene(crossing))
    junction_png = _png_bytes(render_public_scene(junction))
    assert crossing_png == junction_png
    with tempfile.TemporaryDirectory() as raw_workdir:
        workdir = Path(raw_workdir)
        crossing_bytes, crossing_evidence = run_extractor(EXTRACTOR, crossing_png, workdir)
        junction_bytes, junction_evidence = run_extractor(EXTRACTOR, junction_png, workdir)
    assert crossing_bytes == junction_bytes
    assert crossing_evidence == junction_evidence
    assert len(crossing_evidence["skeleton"]["branch_events"]) == 1
    assert len(crossing_evidence["skeleton"]["terminal_events"]) == 4
    assert len(crossing_evidence["lanes"]) == 4
    assert "graph" not in crossing_evidence


def test_double_and_parallel_singles_factorize_to_identical_evidence() -> None:
    double, parallel = double_parallel_fixtures()
    double_png = _png_bytes(render_public_scene(double))
    parallel_png = _png_bytes(render_public_scene(parallel))
    assert double_png == parallel_png
    double_bytes, double_evidence = _extract(render_public_scene(double))
    parallel_bytes, parallel_evidence = _extract(render_public_scene(parallel))
    assert double_bytes == parallel_bytes
    assert double_evidence == parallel_evidence
    assert len(double_evidence["components"]) == 2
    assert len(double_evidence["lanes"]) == 2


def test_color_encodings_have_same_foreground_geometry() -> None:
    images = []
    for mode, white, black in (
        ("L", 255, 0),
        ("RGB", (255, 255, 255), (0, 0, 0)),
        ("RGBA", (255, 255, 255, 255), (0, 0, 0, 255)),
    ):
        image = Image.new(mode, (64, 64), white)
        ImageDraw.Draw(image).line((8, 32, 56, 32), fill=black, width=3)
        images.append(image)
    evidences = [_extract(image)[1] for image in images]
    foregrounds = [evidence["foreground"] for evidence in evidences]
    skeletons = [evidence["skeleton"] for evidence in evidences]
    assert foregrounds[0] == foregrounds[1] == foregrounds[2]
    assert skeletons[0] == skeletons[1] == skeletons[2]
    assert [evidence["image"]["color_type"] for evidence in evidences] == [
        "gray8", "rgb8", "rgba8"
    ]


def test_invalid_png_is_canonical_reject() -> None:
    completed = subprocess.run(
        ["/usr/bin/python3", "-I", "-S", str(EXTRACTOR.resolve())],
        input=b"not a png",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={"LC_ALL": "C", "PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"},
        check=False,
    )
    assert completed.returncode == 0
    assert completed.stderr == b""
    assert completed.stdout.endswith(b"\n")
    assert json.loads(completed.stdout) == {
        "reject_reason": "invalid_png_signature",
        "schema_version": "raster-neutral-evidence-v0.3",
        "status": "reject",
    }


def test_crc_tamper_is_rejected() -> None:
    png = bytearray(_png_bytes(Image.new("L", (8, 8), 255)))
    png[-5] ^= 1
    completed = subprocess.run(
        ["/usr/bin/python3", "-I", "-S", str(EXTRACTOR.resolve())],
        input=bytes(png),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={"LC_ALL": "C", "PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"},
        check=False,
    )
    assert json.loads(completed.stdout)["reject_reason"] == "png_crc_mismatch"


def test_candidate_enumerator_consumes_complete_runs_not_digest() -> None:
    _, evidence = _extract(render_public_scene(crossing_junction_fixtures()[0]))
    key = foreground_key_from_evidence(evidence)
    expected = frozenset({"graph-a", "graph-b"})
    index = ForwardCandidateIndex(
        domain="test",
        scene_count=2,
        observation_count=1,
        candidates_by_foreground={key: expected},
        png_by_foreground={},
        scene_ids_by_foreground={key: ("left", "right")},
    )
    assert enumerate_candidate_graphs(evidence, index) == expected
    tampered = json.loads(json.dumps(evidence))
    tampered["foreground"]["row_runs"][0][1] += 1
    assert enumerate_candidate_graphs(tampered, index) == frozenset()


def test_candidate_enumerator_rejects_noncanonical_runs() -> None:
    evidence = {
        "schema_version": "raster-neutral-evidence-v0.3",
        "status": "ok",
        "foreground": {"row_runs": [[1, 4, 5], [1, 2, 3]]},
    }
    try:
        foreground_key_from_evidence(evidence)
    except CandidateEnumerationError as error:
        assert str(error) == "noncanonical_foreground_row_runs"
    else:
        raise AssertionError("noncanonical runs were accepted")
