import unittest

from chemvector.training.atoms.train_g3_p1k_component_set_ranker import (
    STEP0_ROUTE_EXACT,
    STEP0_TRAIN_EXACT,
    _gate,
    build_parser,
    validate_args,
)


def metrics(exact, recall):
    return {"known_representable_count": {"exact_rate_6px": exact, "recall_6px": recall}}


class G3P1KComponentSetRankerTrainingTests(unittest.TestCase):
    def test_gate_separates_causal_and_capability(self) -> None:
        step0_train = metrics(STEP0_TRAIN_EXACT, 0.9818)
        step0_route = metrics(STEP0_ROUTE_EXACT, 0.9747)
        causal = _gate(
            {"protocol": True}, step0_train, step0_route,
            metrics(0.70, 0.982), metrics(0.57, 0.975)
        )
        capability = _gate(
            {"protocol": True}, step0_train, step0_route,
            metrics(0.78, 0.995), metrics(0.65, 0.985)
        )

        self.assertEqual(causal["decision"], "causal_go_capability_no_go")
        self.assertEqual(capability["decision"], "causal_go_capability_go")

    def test_cli_is_fixed(self) -> None:
        args = build_parser().parse_args([])
        validate_args(args)
        with self.assertRaisesRegex(ValueError, "fixed max_steps"):
            validate_args(build_parser().parse_args(["--max-steps", "1"]))


if __name__ == "__main__":
    unittest.main()
