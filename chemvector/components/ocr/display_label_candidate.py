"""Opt-in display-label parser; deliberately not wired into default inference.

Only text is inspected. No truth graph, model prediction, bbox or identity can
resolve an ambiguous flattened string. In particular H215N and H18O abstain:
hydrogen subscripts and isotope superscripts require separate visual evidence.
This parses label syntax, not molecular valence or OCR confidence.
"""

import re

from chemvector.core.ocr import _ELEMENT_SYMBOLS


_ELEMENT = '(?:' + '|'.join(map(re.escape, _ELEMENT_SYMBOLS)) + ')'
_CHARGE = r'(?P<charge>(?:[1-9][0-9]*)?[+-])?'
_ATOM = rf'(?P<isotope>[1-9][0-9]*)?(?P<element>{_ELEMENT})'
_H = r'(?P<hydrogen>H(?P<hydrogens>[1-9])?)'
_RADICALS = r'(?P<radicals>\.*)'
_STANDARD = re.compile(_ATOM + _H + '?' + _CHARGE + _RADICALS)
_LEFT_H = re.compile(_H + rf'(?P<element>{_ELEMENT})' + _CHARGE + _RADICALS)
_CHARGE_BEFORE_H = re.compile(_ATOM + r'(?P<charge>(?:[1-9][0-9]*)?[+-])' + _H + _RADICALS)


def parse_display_label_candidate(text):
    """Return atom label fields or None; abstain on ambiguous numeric placement.

Hydrogen counts are a single nonzero digit. Thus NH22+ cannot silently become
22 hydrogens or NH2 with charge +2. Left-side hydrogen plus an isotope (H18O,
H215N) is not supported without glyph-position evidence. H, He, Hf, Hg, Ho and
Hs are elements, not evidence of an attached hydrogen.
"""
    if not isinstance(text, str) or not 0 < len(text) <= 24:
        return None
    # Two adjacent numeric roles are ambiguous after flattening typography.
    if re.search(r'H[0-9]{2,}', text):
        return None
    for pattern in (_STANDARD, _LEFT_H, _CHARGE_BEFORE_H):
        match = pattern.fullmatch(text)
        if match is None:
            continue
        groups = match.groupdict()
        charge_text = groups.get('charge')
        charge = 0 if not charge_text else (1 if charge_text[-1] == '+' else -1) * int(charge_text[:-1] or 1)
        hydrogen = int(groups.get('hydrogens') or 1) if groups.get('hydrogen') else 0
        if groups['element'] == 'H' and hydrogen:
            return None  # H2/HH denotes more than one atom, not this atom label.
        return dict(element=groups['element'], formal_charge=charge,
                    isotope=int(groups.get('isotope') or 0), explicit_h_count=hydrogen,
                    radical_electrons=len(groups['radicals']))
    return None
