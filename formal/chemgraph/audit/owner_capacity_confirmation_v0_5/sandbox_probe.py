"""Non-constructor sandbox probe; this file must never import project modules."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import resource
import socket
from pathlib import Path
from typing import Any


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _attempt_outside_write() -> int:
    try:
        descriptor = os.open("/outside-probe", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except OSError as error:
        return error.errno
    os.close(descriptor)
    return 0


def _attempt_network() -> int:
    connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connection.settimeout(0.25)
    try:
        connection.connect(("198.51.100.1", 9))
    except OSError as error:
        return error.errno or errno.EIO
    finally:
        connection.close()
    return 0


def _attempt_fork() -> int:
    resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))
    try:
        child = os.fork()
    except OSError as error:
        return error.errno
    if child == 0:
        os._exit(97)
    os.waitpid(child, 0)
    return 0


def main() -> None:
    result = {
        "constructor_imported": False,
        "fork_errno": _attempt_fork(),
        "network_errno": _attempt_network(),
        "outside_write_errno": _attempt_outside_write(),
        "schema_version": "owner-capacity-sandbox-probe-v0.5",
        "training_authorized": False,
    }
    raw = _canonical(result)
    path = Path("/out/probe.json")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        os.write(descriptor, raw)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(raw).digest():
        raise SystemExit(98)
    if not all(result[key] != 0 for key in ("fork_errno", "network_errno", "outside_write_errno")):
        raise SystemExit(99)


if __name__ == "__main__":
    main()
