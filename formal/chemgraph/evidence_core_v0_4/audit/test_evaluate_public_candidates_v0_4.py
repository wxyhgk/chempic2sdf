from formal.chemgraph.evidence_core_v0_4.audit.evaluate_public_candidates_v0_4 import graph_signature


def test_graph_signature_ignores_node_ids_and_edge_direction() -> None:
    left = {
        "nodes": [{"id": 0, "role": "implicit"}, {"id": 1, "role": "text_stroke"}],
        "edges": [{"begin": 0, "end": 1, "order": 2}],
    }
    right = {
        "nodes": [{"id": 8, "role": "text_stroke"}, {"id": 4, "role": "implicit"}],
        "edges": [{"begin": 8, "end": 4, "order": 2}],
    }
    assert graph_signature(left) == graph_signature(right)


def test_graph_signature_distinguishes_edge_order() -> None:
    single = {
        "nodes": [{"id": 0, "role": "implicit"}, {"id": 1, "role": "implicit"}],
        "edges": [{"begin": 0, "end": 1, "order": 1}],
    }
    double = {**single, "edges": [{"begin": 0, "end": 1, "order": 2}]}
    assert graph_signature(single) != graph_signature(double)
