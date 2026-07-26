from chemvector.train_proposal_delta_set_adapter import selection_key


def _metrics(chemvision: tuple[int, int], full_v0: tuple[int, int], style: tuple[int, int]):
    def domain(values):
        return {"bond_deployed_exact": values[0], "full_object_exact": values[1]}

    return {
        "official": {
            "domains": {"chemvision": domain(chemvision), "full_v0": domain(full_v0)},
            "pooled": {"full_object_exact": chemvision[1] + full_v0[1]},
        },
        "style": {"domains": {"style": domain(style)}},
    }


def test_selection_requires_each_official_domain_to_be_safe() -> None:
    baseline = _metrics((10, 8), (11, 9), (7, 6))
    unsafe = _metrics((11, 9), (10, 9), (9, 8))

    assert selection_key(baseline, baseline) > selection_key(unsafe, baseline)


def test_selection_prefers_style_full_exact_after_safety() -> None:
    baseline = _metrics((10, 8), (11, 9), (7, 6))
    better = _metrics((10, 8), (11, 9), (7, 7))

    assert selection_key(better, baseline) > selection_key(baseline, baseline)
