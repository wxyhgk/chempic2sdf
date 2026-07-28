from chemvector.train_joint_bond_set_multidomain import balanced_batches


def test_balanced_batch_size_must_match_domain_count() -> None:
    try:
        next(
            balanced_batches(
                samples_per_domain=1,
                batch_size=4,
                config=None,
                seed=1,
                shuffle_buffer=1,
            )
        )
    except ValueError as error:
        assert "divisible" in str(error)
    else:
        raise AssertionError("Expected invalid balanced batch size")
