import torch

from chemvector.audit_hybrid_proposal_representation import rank_match_scores


def test_rank_match_scores_preserves_student_order_and_teacher_values() -> None:
    student = torch.tensor([[0.2, 0.9, 0.5, -1.0]])
    student_mask = torch.tensor([[True, True, True, False]])
    teacher = torch.tensor([[0.8, 0.3, 0.6, -1.0]])
    teacher_mask = torch.tensor([[True, True, True, False]])

    result = rank_match_scores(student, student_mask, teacher, teacher_mask)

    assert torch.allclose(result, torch.tensor([[0.3, 0.8, 0.6, -1.0]]))
