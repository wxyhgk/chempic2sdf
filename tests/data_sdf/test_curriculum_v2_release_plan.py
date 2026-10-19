"""Synthetic split/replay/alias checks; never query identities or launch a job."""

from copy import deepcopy
from hashlib import sha256
import json

import pytest
import torch

from tools import curriculum_v02_release_policy as policy
from tools import prepare_curriculum_v02_release_plan as planner


def molecule(mid, group=None, prefix=None, split=None):
    row = dict(molecule_id=mid, group_key=group or "g_" + mid,
               connectivity_prefix=prefix or "c_" + mid, canonical_smiles=mid,
               sample_keys=[mid + "__reference"], image_count=1)
    if split:
        row["split"] = split
    return row


def inventory(candidates):
    return {k: dict(sample_key=k, molecule_id=r["molecule_id"], profile_id="reference",
                   png_sha256="png_" + k, decoded_rgb_sha256="rgb_" + k,
                   meta_sha256="a" * 64)
            for r in candidates for k in r["sample_keys"]}


def old_pixel(smiles, split="train"):
    return dict(source="old", sample_key="old_key", split=split,
                canonical_smiles=smiles, meta_sha256="b" * 64)


def test_existing_validation_wins_even_for_diagnosed_component_and_all_variants():
    candidates = [molecule("seen", "group"), molecule("isotope", "group")]
    old = [molecule("old", "group", split="validation")]
    snapshot = deepcopy((candidates, old))
    result = policy.grouped_proposal(candidates, old, {"seen"})
    assert all(r["proposed_split"] == "excluded" for r in result)
    assert all(r["component_previously_diagnosed"] for r in result)
    assert (candidates, old) == snapshot


def test_union_connectivity_transitive_to_core_and_preserves_old_train():
    candidates = [molecule("a", "ga", "same"), molecule("b", "gb", "same"), molecule("c", "gb", "different")]
    old = [molecule("old", "ga", split="train")]
    rows = policy.grouped_proposal(candidates, old, set())
    assert {r["proposed_split"] for r in rows} == {"train"}
    assert len({r["component_key"] for r in rows}) == 1


def test_union_cannot_repair_conflicting_old_splits():
    old = [molecule("a", "g", split="train"), molecule("b", "g", split="validation")]
    with pytest.raises(ValueError, match="Existing train/validation"):
        policy.grouped_proposal([], old, set())


def test_split_order_independent_and_diagnosed_component_never_holdout():
    candidates = [molecule(str(i)) for i in range(30)]
    a = policy.grouped_proposal(candidates, [], {"0"})
    b = policy.grouped_proposal(list(reversed(candidates)), [], {"0"})
    assert {r["molecule_id"]: r for r in a} == {r["molecule_id"]: r for r in b}
    assert a[0]["proposed_split"] == "train" and a[0]["reason"].startswith("previously_diagnosed")
    assert {r["proposed_split"] for r in a} == {"train", "validation"}
    with pytest.raises(ValueError, match="Invalid candidate"):
        policy.grouped_proposal(candidates, [], {"foreign"})


def test_png_duplicate_not_counted_as_new_but_preserves_admitted_alias():
    c = [molecule("a")]
    inv = inventory(c)
    decisions = policy.grouped_proposal(c, [molecule("old", "g_a", split="train")], set())
    rows = policy.partition_samples(c, decisions, inv, {"png_a__reference": [old_pixel("a")]})
    assert rows[0]["proposed_split"] == "duplicate_old_train"
    assert rows[0]["old_pixel_matches"][0]["sample_key"] == "old_key"
    assert rows[0]["training_authorized"] is False


@pytest.mark.parametrize("fault", ["chemistry", "old_validation", "duplicate_rgb"])
def test_pixel_conflict_fails_closed(fault):
    candidates = [molecule("a"), molecule("b")]
    inv = inventory(candidates)
    decisions = policy.grouped_proposal(candidates, [], {"a", "b"})
    old = {}
    if fault == "chemistry":
        old["png_a__reference"] = [old_pixel("other")]
    elif fault == "old_validation":
        old["png_a__reference"] = [old_pixel("a", "validation")]
    else:
        inv["b__reference"]["decoded_rgb_sha256"] = inv["a__reference"]["decoded_rgb_sha256"]
    with pytest.raises(ValueError):
        policy.partition_samples(candidates, decisions, inv, old)


def test_fixed_trial_excludes_holdout_no_replacement_and_uses_exact_alias():
    candidates = [molecule("a"), molecule("b"), molecule("c")] + [molecule(str(i)) for i in range(40)]
    old = [molecule("oa", "g_a", split="train"), molecule("ob", "g_b", split="validation")]
    decisions = policy.grouped_proposal(candidates, old, {"a", "b", "c"})
    rows = policy.partition_samples(candidates, decisions, inventory(candidates), {"png_a__reference": [old_pixel("a")]})
    diag = [{**r, "sample_key": r["sample_keys"][0]} for r in candidates[:3]]
    replay = [dict(split="train", domain="full_molecule")]
    trial = policy.select_trial(diag, rows, replay, validation_count=3)
    assert [r["molecule_id"] for r in trial["fitting_witnesses"]] == ["a", "c"]
    assert [r["molecule_id"] for r in trial["excluded_diagnostic_witnesses"]] == ["b"]
    assert trial["fitting_origins"] == {"identical_admitted_training_alias": 1, "new_quarantine_needs_publication": 1}
    assert len({r["component_key"] for r in trial["new_group_validation"]}) == 3
    assert not any(r["component_previously_diagnosed"] for r in trial["new_group_validation"])
    with pytest.raises(ValueError, match="Too few"):
        policy.select_trial(diag, rows, replay, validation_count=100)
    with pytest.raises(ValueError, match="Replay must"):
        policy.select_trial(diag, rows, [dict(split="validation", domain="full_molecule")], validation_count=3)


def test_unseen_validation_rejects_previous_diagnostic():
    c = molecule("a")
    row = {**c, **inventory([c])["a__reference"], "proposed_split": "validation", "component_key": "a",
           "component_previously_diagnosed": False}
    with pytest.raises(ValueError, match="Already diagnosed"):
        policy.select_trial([{**c, "sample_key": "a__reference"}], [row], [], validation_count=1)


def test_exact_target_alias_guard(monkeypatch):
    sample = dict(graph={"canonical_smiles": "CC"}, meta={"sha256": {".png": "same"}}, targets=torch.ones(2))
    monkeypatch.setattr(planner.data, "CurriculumCollator", lambda **kw: lambda s: {"targets": s[0]["targets"]})
    assert planner.check_alias(sample, deepcopy(sample)) == ["targets"]
    other = deepcopy(sample)
    other["targets"][0] = 0
    with pytest.raises(ValueError, match="target tensors"):
        planner.check_alias(sample, other)
    other = deepcopy(sample)
    other["meta"]["sha256"][".png"] = "different"
    with pytest.raises(ValueError, match="pixels/chemistry"):
        planner.check_alias(sample, other)


def test_verify_bundle_reuses_target_evidence_only_for_identical_assets(tmp_path):
    root = tmp_path / "data"
    directory = root / "samples"
    directory.mkdir(parents=True)
    key = "a__reference"
    payloads = {s: b"payload" for s in planner.SUFFIXES if s != ".meta.json"}
    hashes = {s: sha256(raw).hexdigest() for s, raw in payloads.items()}
    meta = dict(sample_key=key, molecule_id="a", canonical_smiles="a", split="quarantine",
                training_authorized=False, external_overlap_status="unchecked", sha256=hashes)
    payloads[".meta.json"] = json.dumps(meta).encode()
    corpus = {}
    for suffix, raw in payloads.items():
        (directory / (key + suffix)).write_bytes(raw)
        corpus["samples/" + key + suffix] = sha256(raw).hexdigest()
    quality = dict(sample_key=key, molecule_id="a", png_sha256=hashes[".png"])
    result, count = planner.verify_bundle(root, key, corpus, quality, molecule("a"), lambda: None)
    assert count == 11 and result["meta_sha256"] == corpus["samples/" + key + ".meta.json"]
    (directory / (key + ".sdf")).write_bytes(b"changed")
    with pytest.raises(ValueError, match="asset changed"):
        planner.verify_bundle(root, key, corpus, quality, molecule("a"), lambda: None)


def test_cpu_and_directory_guards(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="CPU-only"):
        planner.prepare(tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(ValueError, match="Fresh budgeted"):
        planner.prepare(tmp_path / "missing")
    (tmp_path / "report.json").write_text("{}")
    with pytest.raises(ValueError, match="Fresh budgeted"):
        planner.prepare(tmp_path)
