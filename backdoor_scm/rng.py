"""Stable local keyed random streams for task and row isolation."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


RNG_ALGORITHM = "PCG64"
RNG_VERSION = "bdpfn-keyed-pcg64-v1"
NUMPY_RUNTIME_VERSION = np.__version__


def keyed_rng(*key_parts: Any) -> np.random.Generator:
    encoded = json.dumps(
        key_parts,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).digest()
    entropy = [
        int.from_bytes(digest[offset : offset + 4], "big")
        for offset in range(0, len(digest), 4)
    ]
    seed_sequence = np.random.SeedSequence(entropy)
    return np.random.Generator(np.random.PCG64(seed_sequence))
