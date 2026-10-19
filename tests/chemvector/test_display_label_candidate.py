import pytest

from chemvector.core.ocr import _ELEMENT_SYMBOLS, parse_atom_display_text
from chemvector.components.ocr.display_label_candidate import parse_display_label_candidate as parse


@pytest.mark.parametrize('left,right', [('HN','NH'), ('HO','OH'), ('HS','SH'),
    ('H2N','NH2'), ('H2N+','NH2+'), ('H3N+','NH3+'), ('N+H2','NH2+')])
def test_clear_display_orders_share_all_fields(left, right):
    assert parse(left) == parse(right) == parse_atom_display_text(right)
    assert parse_atom_display_text(left) is None  # Frozen default is untouched.


@pytest.mark.parametrize('text', ['H215N','H18O','NH22+','HN2+H','H0N','N0+',
    'NH0','HH','R1','Ph','Me','tBu','NaCl','COOH','N H','N\n','N++','',None])
def test_ambiguous_or_non_atom_strings_abstain(text):
    assert parse(text) is None


def test_periodic_table_symbols_do_not_imply_attached_hydrogen():
    for symbol in _ELEMENT_SYMBOLS:
        assert parse(symbol) == dict(element=symbol, formal_charge=0, isotope=0,
                                     explicit_h_count=0, radical_electrons=0)


@pytest.mark.parametrize('text,expected', [
    ('15NH2+.', ('N',15,2,1,1)), ('18OH', ('O',18,1,0,0)),
    ('N2+H2', ('N',0,2,2,0)), ('Fe3+', ('Fe',0,0,3,0)),
    ('Cl-', ('Cl',0,0,-1,0)), ('13C', ('C',13,0,0,0))])
def test_charge_isotope_hydrogen_radical_semantics(text, expected):
    r = parse(text)
    assert tuple(r[k] for k in ('element','isotope','explicit_h_count','formal_charge','radical_electrons')) == expected
