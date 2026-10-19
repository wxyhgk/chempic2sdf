"""Selector-only fitting and fixed finite-output gates; no files or launch authority."""

from hashlib import sha256
import random

import torch

from chemvector.proposal_object_set import proposal_object_set_loss
from chemvector.train_direct_atom_pair_bond import match_atom_targets
from tools.selector_retention_prototype import positive_retention_loss

THRESHOLD = -.25
CORE_KEYS = ("curr_8c1ec124155f2e0f__font_large", "curr_ddc8766e5cc2a898__reference",
             "curr_85680a509563d789__font_large", "curr_04a03f058eccd278__reference")
SEED = 20260920


def tensors_to(tree, device):
    if isinstance(tree, torch.Tensor):
        return tree.detach().to(device)
    if isinstance(tree, dict):
        return {k: tensors_to(v, device) for k, v in tree.items()}
    return tree


def configure(model):
    model.eval().requires_grad_(False)
    model.set_head.requires_grad_(True)
    return parameters(model)


def parameters(model):
    result = tuple(model.set_head.parameters())
    if (not result or any(m.training for m in model.modules())
            or any(p.requires_grad != n.startswith("set_head.") for n, p in model.named_parameters())
            or any(p.grad is not None for n, p in model.named_parameters() if not n.startswith("set_head."))):
        raise ValueError("Selector-only eval/frozen scope changed")
    return result


def frozen_digest(model):
    digest = sha256()
    for name, value in sorted(model.state_dict().items()):
        if not name.startswith("set_head."):
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def call_selector(head, raw, bonds):
    return head(raw["atom_features"], raw["atom_geometry"], raw["atom_scores"], raw["atom_mask"],
        bonds["bond_features"], bonds["bond_endpoints"], bonds["bond_types"], bonds["bond_scores"], bonds["bond_mask"])


def objective(output, raw, bonds, labels):
    original = proposal_object_set_loss(output, labels["atom_presence_targets"], labels["bond_presence_targets"],
        raw["atom_mask"], bonds["bond_mask"], labels["target_atom_refs"], labels["bond_supervision_mask"],
        atom_decision_threshold=THRESHOLD, bond_decision_threshold=-.5)["loss"]
    keep = positive_retention_loss(output["atom_presence_logits"], labels["atom_presence_targets"],
                                  raw["atom_mask"], threshold=THRESHOLD)
    return original + keep, original, keep


def require_training_cache(cache, manifest_sha):
    if (cache.get("purpose") != "training" or cache.get("training_authorized") is not True
            or cache.get("manifest_sha256") != manifest_sha or cache.get("fit_domain") not in {"witness", "old_full_molecule"}):
        raise ValueError("Diagnostic/unadmitted/unbound cache cannot train")
    for tree in (cache["raw"], cache["bonds"], cache["labels"]):
        if any(not isinstance(t, torch.Tensor) or t.requires_grad or t.device.type != "cpu" for t in tree.values()):
            raise ValueError("Expected detached CPU prediction/label caches")


def training_order(caches, manifest_sha):
    if not caches or len({r["key"] for r in caches}) != len(caches):
        raise ValueError("Distinct nonempty fitting caches required")
    for cache in caches:
        require_training_cache(cache, manifest_sha)
    pools = {name: [c for c in caches if c["fit_domain"] == name] for name in ("old_full_molecule", "witness")}
    if any(not rows for rows in pools.values()):
        raise ValueError("Both fitting domains required")
    generator, queues = random.Random(SEED), {k: [] for k in pools}
    while True:
        for name in ("old_full_molecule", "old_full_molecule", "witness"):
            if not queues[name]:
                queues[name] = list(pools[name])
                generator.shuffle(queues[name])
            yield queues[name].pop()


def update(model, cache, optimizer, limit, manifest_sha, device, reserve):
    require_training_cache(cache, manifest_sha)
    ps = parameters(model)
    ids = [id(p) for g in optimizer.param_groups for p in g["params"]]
    if len(ids) != len(set(ids)) or set(ids) != {id(p) for p in ps}:
        raise ValueError("Optimizer scope differs from selector")
    if not isinstance(optimizer, torch.optim.AdamW) or any(
            g["lr"] != 5e-6 or g["eps"] != 1e-8 or g["weight_decay"] != 0
            or g["betas"] != (.9, .999) or g["amsgrad"] or g["maximize"]
            for g in optimizer.param_groups):
        raise ValueError("Optimizer differs from fixed AdamW recipe")
    optimizer.zero_grad(set_to_none=True)
    raw, bonds, labels = [tensors_to(cache[k], device) for k in ("raw", "bonds", "labels")]
    output = call_selector(model.set_head, raw, bonds)
    loss, full, keep = objective(output, raw, bonds, labels)
    if not bool(torch.isfinite(loss)):
        raise ValueError("Nonfinite loss; no optimizer update")
    loss.backward()
    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in ps):
        raise ValueError("Nonfinite gradient; no optimizer update")
    if not any(p.grad is not None for p in ps):
        raise ValueError("Missing selector gradients")
    norm = torch.nn.utils.clip_grad_norm_(ps, 1., error_if_nonfinite=True)
    limit.consume()
    reserve(limit.consumed)  # Durable intent precedes optimizer.step.
    optimizer.step()
    if any(not bool(torch.isfinite(p).all()) for p in ps):
        raise ValueError("Nonfinite selector weights")
    return dict(loss=float(loss.detach()), original_full_loss=float(full.detach()),
                retention_loss=float(keep.detach()), gradient_norm=float(norm))


@torch.no_grad()
def metrics(raw, selected, output, labels, batch, config):
    """CPU truth scoring AFTER selection; no truth enters selector/predictor."""
    _, matched = match_atom_targets(selected, batch, config)
    atom_ids = matched[0].nonzero().flatten().tolist()
    refs = batch["bond_atom_refs"][0, batch["bond_mask"][0]]
    reachable = matched[0, refs].all(-1).nonzero().flatten().tolist()
    positive = labels["atom_presence_targets"][0]
    kept = selected["atom_mask"][0]
    target_atoms = int(batch["atom_mask"].sum())
    logits = output["atom_presence_logits"][0]
    return dict(target_atoms=target_atoms, target_bonds=int(batch["bond_mask"].sum()),
        selected_atoms=int(kept.sum()), matched_target_ids=atom_ids, reachable_bond_ids=reachable,
        retained_positive_slots=(kept & positive).nonzero().flatten().tolist(),
        selected_negative_slots=(kept & ~positive).nonzero().flatten().tolist(),
        positive_slots=positive.nonzero().flatten().tolist(),
        complete_endpoints=len(atom_ids) == target_atoms,
        unmatched_selected=int(kept.sum()) - len(atom_ids),
        logits={str(i): float(logits[i]) for i in raw["atom_mask"][0].nonzero().flatten().tolist()})


def index_rows(rows):
    result = {r["key"]: r for r in rows}
    if len(result) != len(rows) or not rows:
        raise ValueError("Nonempty unique assessment rows required")
    return result


def compare(before, after):
    base, final = index_rows(before), index_rows(after)
    if base.keys() != final.keys():
        raise ValueError("Assessment panel changed")
    broken, lost_bonds = [], []
    for key, old in base.items():
        new = final[key]
        if (old["sample_key"], old["group"]) != (new["sample_key"], new["group"]):
            raise ValueError("Assessment row identity changed")
        if old["prediction"] is not None and old["prediction"]["exact"] and not new["prediction"]["exact"]:
            broken.append(key)
        missing = sorted(set(old["metrics"]["reachable_bond_ids"]) - set(new["metrics"]["reachable_bond_ids"]))
        if missing:
            lost_bonds.append(dict(key=key, target_bond_ids=missing))
    return dict(images=len(after), broken_strict_graphs=broken, lost_reachable_bonds=lost_bonds,
        complete_endpoints_before=sum(r["metrics"]["complete_endpoints"] for r in before),
        complete_endpoints_after=sum(r["metrics"]["complete_endpoints"] for r in after),
        matched_atoms_before=sum(len(r["metrics"]["matched_target_ids"]) for r in before),
        matched_atoms_after=sum(len(r["metrics"]["matched_target_ids"]) for r in after),
        reachable_bonds_before=sum(len(r["metrics"]["reachable_bond_ids"]) for r in before),
        reachable_bonds_after=sum(len(r["metrics"]["reachable_bond_ids"]) for r in after),
        false_selections_before=sum(len(r["metrics"]["selected_negative_slots"]) for r in before),
        false_selections_after=sum(len(r["metrics"]["selected_negative_slots"]) for r in after))


def local_gate(baseline, current, step):
    def group(rows, name):
        return [r for r in rows if r["group"] == name]
    old = compare(group(baseline, "old_full_molecule"), group(current, "old_full_molecule"))
    fit = compare(group(baseline, "witness"), group(current, "witness"))
    stop = []
    if old["broken_strict_graphs"]:
        stop.append("old_strict_graph_regression")
    if old["lost_reachable_bonds"]:
        stop.append("old_reachable_bond_regression")
    if step >= 200 and fit["complete_endpoints_after"] <= fit["complete_endpoints_before"]:
        stop.append("no_fitting_complete_endpoint_improvement_by_step200")
    base = {r["sample_key"]: r for r in baseline}
    now = {r["sample_key"]: r for r in current}
    core_errors = []
    for key in CORE_KEYS:
        if key not in base or key not in now:
            raise ValueError("Missing fixed core witness")
        a, b = base[key]["metrics"], now[key]["metrics"]
        if set(a["retained_positive_slots"]) - set(b["retained_positive_slots"]):
            core_errors.append(dict(sample_key=key, reason="lost_baseline_retained_positive"))
        if set(b["selected_negative_slots"]) - set(a["selected_negative_slots"]):
            core_errors.append(dict(sample_key=key, reason="new_false_selection"))
    ethyne = now[CORE_KEYS[0]]["metrics"]
    ethyne_pass = (ethyne["positive_slots"] == [1, 2] and {1, 2} <= set(ethyne["retained_positive_slots"])
                   and 0 not in ethyne["selected_negative_slots"])
    return dict(step=step, stop_reasons=stop, old=old, fitting=fit, core_errors=core_errors,
        ethyne_both_endpoints_kept_wrong_candidate_rejected=ethyne_pass,
        local_pass=not stop and not core_errors and ethyne_pass)


def final_gate(local, baseline_dev, final_dev):
    comparison = compare(baseline_dev, final_dev)
    safe = (comparison["matched_atoms_after"] >= comparison["matched_atoms_before"]
            and comparison["reachable_bonds_after"] >= comparison["reachable_bonds_before"]
            and comparison["false_selections_after"] <= comparison["false_selections_before"]
            and comparison["complete_endpoints_after"] >= comparison["complete_endpoints_before"])
    return dict(local=local, new_group_development=comparison,
                development_nonregression=safe, full_regression_allowed=local["local_pass"] and safe)
