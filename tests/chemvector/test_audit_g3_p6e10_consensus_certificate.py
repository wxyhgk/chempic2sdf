from chemvector.evaluation.audits.atoms.audit_g3_p6e10_consensus_certificate import calibrate_zero_harmful


def test_consensus_certificate_prefers_safe_rescues() -> None:
    rows=[
        {"consensus":True,"changed":True,"certificate_score":0.9,"domain":"hard","selected_correct":True},
        {"consensus":True,"changed":True,"certificate_score":0.8,"domain":"hard","selected_correct":True},
        {"consensus":True,"changed":True,"certificate_score":0.7,"domain":"easy","selected_correct":False},
    ]
    winner=calibrate_zero_harmful(rows)
    assert winner["rescues"]==2 and winner["harmful"]==0 and winner["executed"]==2
