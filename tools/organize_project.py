#!/usr/bin/env python3
"""Organize generated artifacts without touching datasets or source files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil


OUTPUT_CATEGORIES = (
    "checkpoints",
    "evaluations",
    "predictions",
    "diagnostics",
    "experiments",
)


def artifact_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def output_category(path: Path) -> tuple[str, str]:
    name = path.name.lower()
    if (path / "last.pt").is_file():
        return "checkpoints", "contains last.pt"
    if (
        "eval" in name
        or "validation" in name
        or "external" in name
        or (path / "summary.json").is_file()
        or (path / "records.jsonl").is_file()
        or (path / "generation_metrics.json").is_file()
    ):
        return "evaluations", "evaluation or validation artifacts"
    if (
        "gd906" in name
        or "test1" in name
        or (path / "metrics.json").is_file()
        or (path / "prediction.json").is_file()
        or (path / "prediction-repaired.sdf").is_file()
    ):
        return "predictions", "single-image prediction artifacts"
    return "diagnostics", "preview, compatibility, or diagnostic artifacts"


def output_directories(output_root: Path) -> list[Path]:
    ignored = {*OUTPUT_CATEGORIES}
    return sorted(
        path
        for path in output_root.iterdir()
        if path.is_dir() and path.name not in ignored
    )


def loose_output_files(output_root: Path) -> list[Path]:
    retained = {"README.md", "MANIFEST.csv"}
    return sorted(
        path
        for path in output_root.iterdir()
        if path.is_file() and path.name not in retained
    )


def organize_outputs(output_root: Path, *, apply: bool) -> list[dict[str, str | int]]:
    rows = []
    for source in output_directories(output_root):
        category, reason = output_category(source)
        destination = output_root / category / source.name
        if destination.exists():
            raise FileExistsError(f"Artifact destination already exists: {destination}")
        rows.append(
            {
                "name": source.name,
                "category": category,
                "old_path": str(Path(output_root.name) / source.name),
                "new_path": str(
                    Path(output_root.name) / category / source.name
                ),
                "size_bytes": artifact_size(source),
                "reason": reason,
            }
        )
        if apply:
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.rename(destination)
    for source in loose_output_files(output_root):
        destination = output_root / "diagnostics" / "loose-files" / source.name
        if destination.exists():
            raise FileExistsError(f"Artifact destination already exists: {destination}")
        rows.append(
            {
                "name": source.name,
                "category": "diagnostics",
                "old_path": str(Path(output_root.name) / source.name),
                "new_path": str(
                    Path(output_root.name) / "diagnostics" / "loose-files" / source.name
                ),
                "size_bytes": artifact_size(source),
                "reason": "loose file from an early compatibility check",
            }
        )
        if apply:
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.rename(destination)
    return rows


def write_manifest(output_root: Path, rows: list[dict[str, str | int]]) -> None:
    manifest_path = output_root / "MANIFEST.csv"
    fields = ("name", "category", "old_path", "new_path", "size_bytes", "reason")
    merged: dict[str, dict[str, str | int]] = {}
    if manifest_path.is_file():
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                for field in ("old_path", "new_path"):
                    value = Path(row[field])
                    if value.is_absolute():
                        row[field] = str(value.relative_to(output_root.parent))
                merged[row["old_path"]] = row
    for row in rows:
        merged[str(row["old_path"])] = row
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(merged.values(), key=lambda row: str(row["old_path"])))


def clean_project_caches(project_root: Path) -> list[Path]:
    removed = []
    for path in sorted(project_root.rglob(".ipynb_checkpoints"), reverse=True):
        if ".git" in path.parts or any(part.startswith(".venv") for part in path.parts):
            continue
        shutil.rmtree(path)
        removed.append(path)
    for relative in ("Data", "chemvector", "tests", "tools"):
        base = project_root / relative
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("__pycache__"), reverse=True):
            if any(part.startswith(".venv") for part in path.parts):
                continue
            shutil.rmtree(path)
            removed.append(path)
    return removed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--clean-caches", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    output_root = project_root / "outputs"
    rows = organize_outputs(output_root, apply=args.apply)
    if args.apply:
        write_manifest(output_root, rows)
    for row in rows:
        print(f"{row['category']}: {row['old_path']} -> {row['new_path']}")
    if not args.apply:
        print(f"Dry run: {len(rows)} artifacts would be moved. Add --apply to migrate.")
    elif not rows:
        print("No top-level output artifacts required migration.")
    else:
        print(f"Moved {len(rows)} artifacts and wrote {output_root / 'MANIFEST.csv'}")
    if args.clean_caches:
        removed = clean_project_caches(project_root)
        print(f"Removed {len(removed)} cache directories.")


if __name__ == "__main__":
    main()
