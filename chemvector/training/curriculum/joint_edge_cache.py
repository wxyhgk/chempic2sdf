"""Immutable cache contract for the masked joint edge objective."""

from __future__ import annotations

from copy import deepcopy

import torch

from chemvector.bond_object import BOND_TYPES
from .interface_diagnostic import tensor_tree_digest
from .pair_cached_trial import make_cache, verify_cache


SCHEMA = "chemvector-joint-edge-cache-0.1"


def make_joint_cache(forward, batch, config, *, source_binding: dict) -> dict:
    """Build the existing frozen pair cache plus immutable source bindings.

    ``source_binding`` is metadata only; it must contain hashes for the exact
    initial checkpoint, release manifest and matcher/config recipe.  Labels are
    still derived after predicted candidates and unknown rows remain excluded.
    """
    if not isinstance(source_binding, dict):
        raise ValueError("Explicit source binding is required")
    required = ("initial_sha256", "manifest_sha256", "config_sha256", "source", "sample_key")
    if any(not isinstance(source_binding.get(key), str) or not source_binding[key] for key in required):
        raise ValueError("Incomplete joint cache source binding")
    cache = make_cache(forward, batch, config)
    labels = cache["tensors"]["labels"]
    known = cache["tensors"]["known"]
    presence = labels["presence"]
    reachable_types = {
        name: int((known & presence & (labels["chemical"] == index)).sum())
        for index, name in enumerate(BOND_TYPES)
    }
    target_types = {
        name: int((batch["bond_chemical"][batch["bond_mask"]] == index).sum())
        for index, name in enumerate(BOND_TYPES)
    }
    coverage = deepcopy(cache["coverage"])
    coverage.update({
        "reachable_bond_types": reachable_types,
        "target_bond_types": target_types,
        "unreachable_bond_types": {
            name: target_types[name] - reachable_types[name] for name in BOND_TYPES
        },
    })
    binding = dict(source_binding)
    cache["schema_version"] = SCHEMA
    cache["source_binding"] = binding
    cache["coverage"] = coverage
    cache["source_binding_sha256"] = tensor_tree_digest({
        "binding": binding,
        "coverage": coverage,
    })
    cache["cache_sha256"] = tensor_tree_digest({
        "tensors": cache["tensors"],
        "source_binding_sha256": cache["source_binding_sha256"],
    })
    return cache


def verify_joint_cache(cache: dict) -> None:
    if cache.get("schema_version") != SCHEMA:
        raise ValueError("Unsupported joint cache schema")
    binding = cache.get("source_binding")
    required = ("initial_sha256", "manifest_sha256", "config_sha256", "source", "sample_key")
    if not isinstance(binding, dict) or any(not binding.get(key) for key in required):
        raise ValueError("Joint cache source binding is incomplete")
    if tensor_tree_digest({"binding": binding, "coverage": cache["coverage"]}) != cache.get("source_binding_sha256"):
        raise ValueError("Joint cache source binding changed")
    if tensor_tree_digest({
        "tensors": cache["tensors"],
        "source_binding_sha256": cache["source_binding_sha256"],
    }) != cache.get("cache_sha256"):
        raise ValueError("Joint cache digest changed")
    verify_cache({"tensors": cache["tensors"], "cache_sha256": tensor_tree_digest(cache["tensors"])})
    coverage = cache["coverage"]
    for field in ("reachable_bond_types", "target_bond_types", "unreachable_bond_types"):
        if set(coverage.get(field, {})) != set(BOND_TYPES):
            raise ValueError("Joint cache bond-type coverage is incomplete")


__all__ = ["SCHEMA", "make_joint_cache", "verify_joint_cache"]
