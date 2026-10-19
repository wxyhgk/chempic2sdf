"""Synthetic raster-footprint geometry; no training images."""

import numpy as np
import pytest

from tools.pair_anchor_visual_review import footprint, polygon


def test_footprint_interior_excludes_endpoints_and_far_pixels():
    mask = footprint((20., 32.), (70., 32.), size=96)
    assert mask[32, 40] and mask[20, 40] and not mask[19, 40]
    assert not mask[32, 20] and not mask[32, 69]
    assert int(mask.sum()) == 46 * 24


def test_footprint_reversing_pair_is_exactly_equivalent():
    a, b = (20., 20.), (60., 70.)
    assert np.array_equal(footprint(a, b, size=96), footprint(b, a, size=96))


def test_polygon_matches_fixed_width_and_longitudinal_range():
    assert polygon((20., 32.), (70., 32.)) == [(22., 44.), (68., 44.), (68., 20.), (22., 20.)]


def test_zero_length_segment_rejected():
    with pytest.raises(ValueError): footprint((20., 20.), (20., 20.))
