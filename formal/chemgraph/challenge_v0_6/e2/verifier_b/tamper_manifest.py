#!/usr/bin/env python3
"""仅供 verifier B 负向测试制造篡改输入，不参与验证。"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import sys


def dump(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True),
        encoding="ascii",
    )


def main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit("usage: tamper_manifest.py MODE INPUT OUTPUT ARTIFACT_ROOT")
    mode, source, output, root = sys.argv[1:]
    source_path, output_path, root_path = Path(source), Path(output), Path(root)
    if mode == "corrupt_json":
        output_path.write_bytes(b'{"pairs":[')
        return
    data = json.loads(source_path.read_text(encoding="ascii"))
    pair = data["pairs"][0]
    if mode == "tuple_id":
        pair["left"]["tuple"]["tuple_id"] += 1
    elif mode == "axis":
        pair["left"]["tuple"]["half_length"] += 1
    elif mode == "missing_pair":
        data["pairs"].pop()
    elif mode == "extra_pair":
        data["pairs"].append(copy.deepcopy(pair))
    elif mode == "duplicate_pair":
        data["pairs"][-1] = copy.deepcopy(pair)
    elif mode == "wrong_cap":
        pair["style"]["cap"] = "round"
    elif mode == "wrong_width":
        pair["style"]["width_px"] = 3
    elif mode == "digest":
        pair["left"]["raw_gray"]["sha256"] = "0" * 64
    elif mode == "same_graph":
        pair["right"]["graph"] = copy.deepcopy(pair["left"]["graph"])
    elif mode in {"differing_bytes_same_digest", "png_trailing"}:
        descriptor = pair["right"]["canonical_gray" if mode == "differing_bytes_same_digest" else "canonical_png"]
        artifact = root_path / descriptor["path"]
        payload = bytearray(artifact.read_bytes())
        if mode == "differing_bytes_same_digest":
            payload[len(payload) // 2] ^= 1
            # Keep the declared digest equal to the left side: a digest-only verifier
            # could accept the pair without checking the actual right-side bytes.
            descriptor["sha256"] = pair["left"]["canonical_gray"]["sha256"]
        else:
            payload.append(0)
            descriptor["byte_length"] = len(payload)
            descriptor["sha256"] = hashlib.sha256(payload).hexdigest()
            data["artifact_total_bytes"] += 1
        temporary = artifact.with_suffix(artifact.suffix + ".tampered")
        temporary.write_bytes(payload)
        os.replace(temporary, artifact)
    else:
        raise SystemExit(f"unknown mode: {mode}")
    dump(output_path, data)


if __name__ == "__main__":
    main()
