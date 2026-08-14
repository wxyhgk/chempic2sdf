import math

import numpy as np

from formal.chemgraph.evidence_core_v0_4.extractor.extract_evidence_core_v0_4 import (
    Primitive,
    RawLine,
    canonicalize_primitives,
    cluster_hough_lines,
    line_source_without_tiny_components,
)


def test_unoriented_angle_seam_clusters_without_rotating_ninety_degrees() -> None:
    lines = [
        RawLine(-49.95, 50.0, 49.95, -50.0),
        RawLine(-50.05, 50.0, 50.05, -50.0),
    ]
    clusters = cluster_hough_lines(lines)
    assert len(clusters) == 1
    assert abs(abs(math.degrees(clusters[0].angle)) - 45.0) < 0.1


def test_tiny_isolated_component_is_not_hough_input() -> None:
    binary = np.zeros((64, 64), dtype=bool)
    binary[1, 1] = True
    binary[20, 10:40] = True
    filtered = line_source_without_tiny_components(binary)
    assert not filtered[1, 1]
    assert np.all(filtered[20, 10:40])


def test_primitive_direction_does_not_change_canonical_order() -> None:
    forward = Primitive((10.0, 20.0), (30.0, 40.0), 2.0, 9000, 0)
    reverse = Primitive((50.0, 60.0), (40.0, 50.0), 2.0, 9000, 0)
    result = canonicalize_primitives([reverse, forward])
    assert [(item.begin, item.end) for item in result] == [
        ((10.0, 20.0), (30.0, 40.0)),
        ((40.0, 50.0), (50.0, 60.0)),
    ]
