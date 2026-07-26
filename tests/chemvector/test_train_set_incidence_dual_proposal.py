from chemvector.train_set_incidence_dual_proposal import selection_key


def _official(chemvision: tuple[int, int], full_v0: tuple[int, int]) -> dict:
    domains = {}
    for name, values in (("chemvision", chemvision), ("full_v0", full_v0)):
        domains[name] = {
            "bond_deployed_exact": values[0],
            "full_object_exact": values[1],
        }
    return {
        "domains": domains,
        "pooled": {
            "full_object_exact": sum(item["full_object_exact"] for item in domains.values())
        },
    }


def _style(full: int, bond: int) -> dict:
    return {
        "domains": {
            "style": {
                "full_object_exact": full,
                "bond_deployed_exact": bond,
            }
        }
    }


def test_selection_rejects_regression_on_either_proposal_path() -> None:
    baseline_candidate = _official((10, 8), (11, 9))
    baseline_original = _official((12, 10), (13, 11))
    unsafe_original = _official((12, 10), (12, 11))

    unsafe = selection_key(
        baseline_candidate,
        unsafe_original,
        _style(20, 21),
        baseline_candidate,
        baseline_original,
    )
    safe = selection_key(
        baseline_candidate,
        baseline_original,
        _style(10, 11),
        baseline_candidate,
        baseline_original,
    )

    assert safe > unsafe


def test_selection_prefers_style_full_exact_after_safety() -> None:
    baseline_candidate = _official((10, 8), (11, 9))
    baseline_original = _official((12, 10), (13, 11))

    lower = selection_key(
        baseline_candidate,
        baseline_original,
        _style(10, 20),
        baseline_candidate,
        baseline_original,
    )
    higher = selection_key(
        baseline_candidate,
        baseline_original,
        _style(11, 19),
        baseline_candidate,
        baseline_original,
    )

    assert higher > lower
