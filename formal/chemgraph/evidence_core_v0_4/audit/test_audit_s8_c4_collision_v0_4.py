from formal.chemgraph.evidence_core_v0_4.audit.audit_s8_c4_collision_v0_4 import (
    Q,
    audit_collision,
    collision_segments,
    collision_styles,
    semantic_graph_sets,
)


def test_public_coordinates_have_quarter_pixel_residuals() -> None:
    s8, c4 = collision_segments()
    assert [(item.x0_q16 // (Q // 4), item.x1_q16 // (Q // 4)) for item in s8] == [
        (1350, 1491),
        (1581, 1581),
    ]
    assert [(item.x0_q16 // (Q // 4), item.x1_q16 // (Q // 4)) for item in c4] == [
        (1580, 1580),
        (1490, 1350),
    ]


def test_all_eight_styles_are_byte_exact_and_support_separated() -> None:
    result = audit_collision()
    rows = result["collision"]["style_results"]
    assert len(rows) == len(collision_styles()) == 8
    assert all(item["gray_bytes_equal"] for item in rows)
    assert all(item["png_bytes_equal"] for item in rows)
    assert all(item["later_c4_hits_seen_png"] for item in rows)
    assert all(item["s8_support_gate_passed"] for item in rows)
    assert all(item["c4_support_gate_passed"] for item in rows)


def test_graph_sets_share_lane_graph_but_require_different_decisions() -> None:
    s8_graphs, c4_graphs = semantic_graph_sets()
    assert len(s8_graphs) == 1
    assert len(c4_graphs) == 2
    assert s8_graphs < c4_graphs
    result = audit_collision()
    assert result["conclusion"]["semantic_demand_conflict"] is True
    assert result["semantics"]["s8_decode_requirement"].startswith("Graph")
    assert result["semantics"]["c4_decode_requirement"].startswith("Ambiguous")


def test_audit_reports_exact_collision_with_strict_scope() -> None:
    result = audit_collision()
    assert result["status"] == "no_go_cross_family_exact_png_collision"
    assert result["conclusion"] == {
        "accepted_same_pack_pair": False,
        "evidence_collision_for_deterministic_png_extractor": True,
        "generator_duplicate_observation_gate_would_retry_later_c4": True,
        "is_only_near_neighbor": False,
        "semantic_demand_conflict": True,
    }
    assert result["scope"]["formal_hidden_seed_used"] is False
    assert result["scope"]["lean_proves_png_equality"] is False
    assert result["scope"]["parameter_space_witness_only"] is True
    assert result["scope"]["prng_joint_reachability_verified"] is False
    assert result["scope"]["training_authorized"] is False
