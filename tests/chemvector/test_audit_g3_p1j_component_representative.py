import unittest

import torch

from chemvector.evaluation.audits.atoms.audit_g3_p1j_component_representative import (
    component_representatives,
)


class G3P1JComponentRepresentativeTests(unittest.TestCase):
    def test_base_max_uses_score_and_identity_tie_break(self) -> None:
        anchors = torch.tensor([[0.0, 0.0], [1.0, 0.0], [5.0, 5.0]])
        scores = torch.tensor([2.0, 2.0, 1.0])
        ids = torch.tensor([7, 3, 9])
        labels = torch.tensor([0, 0, 1])

        representatives, component_scores = component_representatives(
            anchors, scores, ids, labels, "base_max"
        )

        torch.testing.assert_close(representatives, torch.tensor([[1.0, 0.0], [5.0, 5.0]]))
        torch.testing.assert_close(component_scores, torch.tensor([2.0, 1.0]))

    def test_medoid_and_centroid_are_target_free(self) -> None:
        anchors = torch.tensor([[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]])
        scores = torch.zeros(3)
        ids = torch.arange(3)
        labels = torch.zeros(3, dtype=torch.long)

        medoid, _ = component_representatives(
            anchors, scores, ids, labels, "medoid"
        )
        centroid, _ = component_representatives(
            anchors, scores, ids, labels, "centroid"
        )

        torch.testing.assert_close(medoid[0], torch.tensor([1.0, 0.0]))
        torch.testing.assert_close(centroid[0], torch.tensor([5.0 / 3.0, 0.0]))

    def test_unknown_scheme_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown"):
            component_representatives(
                torch.zeros(1, 2),
                torch.zeros(1),
                torch.zeros(1, dtype=torch.long),
                torch.zeros(1, dtype=torch.long),
                "oracle",
            )


if __name__ == "__main__":
    unittest.main()
