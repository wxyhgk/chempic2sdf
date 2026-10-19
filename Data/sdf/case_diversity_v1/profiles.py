"""16 paired style cases cover all pairs of five four-level factors."""

from collections import Counter
from functools import lru_cache
from hashlib import sha256
from itertools import combinations
from pathlib import Path
import subprocess

from Data.sdf.chemvector_curriculum_v0_1.rendering import Profile


FAMILIES = ('DejaVu Sans', 'DejaVu Serif', 'Liberation Sans', 'Nimbus Roman')
SIZES = (12, 18, 24, 32)
ROTATIONS = (0., 90., 180., 270.)
LENGTHS = (28., 40., 52., 64.)
STROKES = (.8, 1.6, 2.4, 3.2)
FACTORS = ('font_family', 'font_size', 'rotation_degrees', 'bond_length_cap', 'bond_line_width')


def profiles():
    result = []
    for a in range(4):
        for b in range(4):
            # GF(4) with x^2+x+1; OA(16,5,4,2), not integer mod-4.
            twice = (b << 1) ^ (7 if b & 2 else 0)
            levels = (a, b, a ^ b, a ^ twice, a ^ twice ^ b)
            values = [domain[level] for domain, level in zip((FAMILIES, SIZES, ROTATIONS, LENGTHS, STROKES), levels)]
            result.append(Profile(profile_id=f'oa{a}{b}', **dict(zip(FACTORS, values))))
    return result


def pair_coverage(selected):
    return {left + ' x ' + right: len({(getattr(p, left), getattr(p, right)) for p in selected})
            for left, right in combinations(FACTORS, 2)}


@lru_cache(maxsize=4)
def font_record(family):
    if family not in FAMILIES:
        raise ValueError('Unplanned font family')
    result = subprocess.run(['fc-match', '-f', '%{family}|%{file}', family],
                            capture_output=True, text=True, check=True, timeout=10)
    matched, filename = result.stdout.split('|', 1)
    if matched != family:
        raise ValueError('Font fallback forbidden: ' + family + ' -> ' + matched)
    return dict(family=matched, path=filename, sha256=sha256(Path(filename).read_bytes()).hexdigest())
