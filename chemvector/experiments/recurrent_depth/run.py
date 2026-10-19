"""Fixed-step paired-seed training; held-out scoring only after all fitting."""

import hashlib
import time

import numpy as np
import torch
from torch.nn import functional as F

from chemvector.external_ocr_r0_contracts import sha256_file, validate_frozen_file, write_new_json, write_new_jsonl
from .model import CandidateRefiner
from .metrics import decide, score_records
from .protocol import load


def prepare(records, device):
    return {name: torch.stack([r[name] for r in records]).to(device) for name in ("x", "mask", "base", "labels")}


def fit(output, protocol, cache_receipt):
    validate_frozen_file(output / "cache.pt", cache_receipt["sha256"], "feature cache")
    records = torch.load(output / "cache.pt", weights_only=False)["records"]
    train = [r for r in records if r["split"] == "train"]
    validation = [r for r in records if r["split"] == "validation"]
    data = prepare(train, "cuda")
    valid_features = data["x"][data["mask"]]
    mean, std = valid_features.mean(0), valid_features.std(0).clamp_min(.01)
    data["x"] = (data["x"] - mean) / std
    torch.save({"mean": mean.cpu(), "std": std.cpu(), "fit_split": "train_only"}, output / "normalization.pt")
    started, trained = time.perf_counter(), []
    for seed in protocol["seeds"]:
        generator = torch.Generator().manual_seed(seed + 1000)
        order = torch.randint(len(train), (protocol["steps"], protocol["batch_size"]), generator=generator)
        order_hash = hashlib.sha256(order.numpy().tobytes()).hexdigest()
        for arm, config in protocol["arms"].items():
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            model = CandidateRefiner(data["x"].shape[-1], protocol["width"], protocol["heads"], config["untied_depth"]).cuda()
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=protocol["learning_rate"], weight_decay=protocol["weight_decay"])
            losses = []
            torch.cuda.synchronize()
            arm_started = time.perf_counter()
            for step, indices in enumerate(order):
                if time.perf_counter() - started > protocol["max_training_seconds"]:
                    raise TimeoutError("Pilot training time budget exceeded; do not treat partial trials as a comparison")
                indices = indices.cuda()
                mask = data["mask"][indices]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    prediction = model(data["x"][indices], mask, data["base"][indices], config["loops"])
                loss = (F.binary_cross_entropy_with_logits(prediction.float(), data["labels"][indices], reduction="none") * mask).sum() / mask.sum().clamp_min(1)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach()))
                if (step + 1) % 200 == 0:
                    print(f"seed={seed} arm={arm} step={step+1}/{protocol['steps']} loss={losses[-1]:.5f}", flush=True)
            torch.cuda.synchronize()
            path = output / f"{arm}-seed{seed}.pt"
            torch.save(model.cpu().state_dict(), path)
            trained.append({"arm": arm, "seed": seed, "trained_loops": config["loops"],
                            "parameters": sum(p.numel() for p in model.parameters()), "checkpoint": path.name,
                            "checkpoint_sha256": sha256_file(path), "batch_order_sha256": order_hash,
                            "training_seconds": time.perf_counter() - arm_started,
                            "loss_first": losses[0], "loss_last": losses[-1]})
            write_new_json(output / f"{arm}-seed{seed}-training.json", {**trained[-1], "losses": losses})
            del model, optimizer
    # No model selection or scheduling based on the held-out population.
    write_new_json(output / "training_receipt.json", {"protocol_sha256": sha256_file(output / "protocol.json"),
                                                       "trials": trained, "validation_used_during_fit": False})
    held = prepare(validation, "cuda")
    held["x"] = (held["x"] - mean) / std
    source, source_rows = score_records(validation, np.stack([r["base"].numpy() for r in validation]))
    write_new_jsonl(output / "source_samples.jsonl", source_rows)
    trials = []
    for trial in trained:
        config = protocol["arms"][trial["arm"]]
        path = output / trial["checkpoint"]
        validate_frozen_file(path, trial["checkpoint_sha256"], "fixed last-step model")
        model = CandidateRefiner(data["x"].shape[-1], protocol["width"], protocol["heads"], config["untied_depth"]).cuda()
        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()
        counts = protocol["tied4_evaluation_loops"] if trial["arm"] == "tied4" else [config["loops"]]
        for loops in counts:
            predictions = []
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for start in range(0, len(validation), protocol["batch_size"]):
                    end = start + protocol["batch_size"]
                    predictions.append(model(held["x"][start:end], held["mask"][start:end], held["base"][start:end], loops).float().cpu().numpy())
            predictions = np.concatenate(predictions)
            metrics, rows = score_records(validation, predictions)
            name = f"{trial['arm']}-seed{trial['seed']}-loops{loops}"
            np.save(output / f"{name}-logits.npy", predictions)
            write_new_jsonl(output / f"{name}-samples.jsonl", rows)
            # Warmed fixed-batch timing; not end-to-end latency.
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for _ in range(3):
                    model(held["x"][:32], held["mask"][:32], held["base"][:32], loops)
                torch.cuda.synchronize()
                begin = time.perf_counter()
                for _ in range(20):
                    model(held["x"][:32], held["mask"][:32], held["base"][:32], loops)
                torch.cuda.synchronize()
            trials.append({**trial, "evaluation_loops": loops, "metrics": metrics,
                           "head_ms_per_image_batch32": (time.perf_counter() - begin) * 1000 / (20 * 32)})
            print(f"eval {name}: exact={metrics['exact']:.4f} fp/image={metrics['fp_per_image']:.3f}", flush=True)
        del model
    load(output)
    report = {"protocol_sha256": sha256_file(output / "protocol.json"), "cache": cache_receipt,
              "train_images": len(train), "validation_images": len(validation), "source": source,
              "trials": trials, "decision": decide(trials, source, protocol["gate"]),
              "limitations": [protocol["scope"], "not scaffold/font disjoint", "three seeds, one bounded training budget",
                              "atom-presence only: no bond connectivity, OCR, or SDF measured",
                              "canonical all-candidate matching labels; scoring rematches selected candidates",
                              "fixed 4-loop training: other depths are untrained-depth diagnostics, not guaranteed extrapolation"],
              "artifacts": {p.name: sha256_file(p) for p in output.iterdir() if p.is_file()}}
    write_new_json(output / "report.json", report)
    return report
