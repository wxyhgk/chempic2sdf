from tools.check_display_label_candidate import compare_labels


def atom(text, **overrides):
    return dict(dict(text=text, visible_text=True, element='N', formal_charge=0,
        isotope=0, explicit_h_count=0, implicit_h_count=1, radical_electrons=0), **overrides)


def test_truth_scores_after_parse_and_never_corrects_the_input():
    r = compare_labels([atom('HO')])  # Wrong element, not silently repaired using truth N.
    assert r['newly_supported'] == {'HO':1}
    assert r['field_mismatches'] == {'HO:core':1}


def test_hydrogen_count_uses_total_and_does_not_assume_implicit_zero():
    r = compare_labels([atom('H2N', implicit_h_count=2), atom('13C', element='C', isotope=13, implicit_h_count=3)])
    assert r['field_mismatches'] == {}
    assert r['counts']['positive_visible_H_checked'] == 1
    assert compare_labels([atom('H2N')])['field_mismatches'] == {'H2N:hydrogens':1}


def test_ambiguous_isotope_is_not_decided_by_truth():
    r = compare_labels([atom('H215N', isotope=15, implicit_h_count=2)])
    assert r['abstained'] == {'H215N':1}
    assert r['counts'].get('candidate_parsed',0) == 0
