"""Small checkpoint helper for the chemical-head probe."""

from .bundle import load_bundle


def load_bundle_for_probe(path, expected_sha256):
    """Keep probe loading explicit and separate from production profile loading."""
    return load_bundle(path, expected_sha256=expected_sha256)
