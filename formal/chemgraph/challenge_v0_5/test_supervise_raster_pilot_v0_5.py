#!/usr/bin/env python3
"""Fail-closed unit tests for the privileged raster-pilot supervisor."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
import unittest
import zlib

try:
    from .supervise_raster_pilot_v0_5 import (
        FULL_WALL_LIMIT, PILOT_TUPLES, TEMPORARY_LIMIT, TMPFS_SIZE, SupervisorError,
        artifact_tree_root, evaluate_gates, oom_events_increased, reject_unsafe_tree,
        self_virtual_memory_upper_bound, strict_zlib_decompress, sum_memory_events,
    )
except ImportError:
    from supervise_raster_pilot_v0_5 import (  # type: ignore
        FULL_WALL_LIMIT, PILOT_TUPLES, TEMPORARY_LIMIT, TMPFS_SIZE, SupervisorError,
        artifact_tree_root, evaluate_gates, oom_events_increased, reject_unsafe_tree,
        self_virtual_memory_upper_bound, strict_zlib_decompress, sum_memory_events,
    )


def valid_runner() -> dict:
    return {
        "tuple_count": PILOT_TUPLES,
        "padded_clipping_certificates": PILOT_TUPLES,
        "deterministic_replay_count": PILOT_TUPLES,
    }


class ResourceGateTests(unittest.TestCase):
    def test_pre_migration_memory_bound_and_event_sum(self) -> None:
        self.assertGreater(self_virtual_memory_upper_bound(), 0)
        self.assertEqual(
            sum_memory_events(
                {"oom": 1, "max": 2}, {"oom": 3, "oom_kill": 4},
            ),
            {"oom": 4, "max": 2, "oom_kill": 4},
        )
        self.assertFalse(oom_events_increased({"oom": 1}, {"oom": 1, "max": 2}))
        self.assertTrue(oom_events_increased({"oom": 1}, {"oom": 2}))

    def test_artifact_tree_rejects_symlink_hardlink_and_special_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            regular = root / "regular"
            regular.write_bytes(b"content")
            reject_unsafe_tree(root)
            symlink = root / "symlink"
            symlink.symlink_to(regular)
            with self.assertRaises(SupervisorError):
                reject_unsafe_tree(root)
            symlink.unlink()
            hardlink = root / "hardlink"
            hardlink.hardlink_to(regular)
            with self.assertRaises(SupervisorError):
                reject_unsafe_tree(root)
            hardlink.unlink()
            fifo = root / "fifo"
            os.mkfifo(fifo)
            with self.assertRaises(SupervisorError):
                reject_unsafe_tree(root)

    def test_strict_zlib_decoder_rejects_truncation_and_trailing_stream_data(self) -> None:
        payload = zlib.compress(b"abcd")
        self.assertEqual(strict_zlib_decompress(payload, 4), b"abcd")
        with self.assertRaises(SupervisorError):
            strict_zlib_decompress(payload[:-1], 4)
        with self.assertRaises(SupervisorError):
            strict_zlib_decompress(payload + b"trailing", 4)
        with self.assertRaises(SupervisorError):
            strict_zlib_decompress(payload, 3)

    def test_frozen_hard_upper_bounds_pass_when_projection_fits(self) -> None:
        gates, projected_wall, projected_temp = evaluate_gates(
            wall_upper=1000, memory_peak_before=1024, memory_peak_upper=1024,
            memory_oom=0, temporary_upper=TMPFS_SIZE,
            runner_validation=valid_runner(), exit_code=0,
        )
        self.assertTrue(all(gates.values()))
        self.assertLess(projected_wall, FULL_WALL_LIMIT)
        self.assertLess(projected_temp, TEMPORARY_LIMIT)

    def test_wall_oom_and_projection_fail_closed(self) -> None:
        gates, _, _ = evaluate_gates(
            wall_upper=10_000, memory_peak_before=1024, memory_peak_upper=1024,
            memory_oom=1, temporary_upper=16 * 1024**3,
            runner_validation=valid_runner(), exit_code=0,
        )
        self.assertFalse(gates["projected_full_wall_within_96h"])
        self.assertFalse(gates["memory_oom_zero"])
        self.assertFalse(gates["projected_temporary_within_2tib"])

    def test_missing_runner_counts_fail_closed(self) -> None:
        broken = valid_runner()
        broken["padded_clipping_certificates"] -= 1
        gates, _, _ = evaluate_gates(
            wall_upper=100, memory_peak_before=1024, memory_peak_upper=1024,
            memory_oom=0, temporary_upper=TMPFS_SIZE,
            runner_validation=broken, exit_code=1,
        )
        self.assertFalse(gates["runner_exit_zero"])
        self.assertFalse(gates["padded_certificates_exact"])

    def test_memory_measurement_and_upper_bound_fail_closed(self) -> None:
        gates, _, _ = evaluate_gates(
            wall_upper=100, memory_peak_before=86_000_000_000,
            memory_peak_upper=86_000_000_000, memory_oom=0,
            temporary_upper=TMPFS_SIZE, runner_validation=valid_runner(), exit_code=0,
        )
        self.assertFalse(gates["memory_peak_before_receipt_within_80gib"])
        self.assertFalse(gates["memory_hard_upper_bound_within_80gib"])

    def test_artifact_tree_root_is_content_and_path_sensitive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "a").write_bytes(b"one")
            first = artifact_tree_root(root)
            (root / "a").write_bytes(b"two")
            self.assertNotEqual(first, artifact_tree_root(root))
            (root / "a").rename(root / "b")
            self.assertNotEqual(first, artifact_tree_root(root))


if __name__ == "__main__":
    unittest.main()
