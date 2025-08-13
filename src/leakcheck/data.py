"""Deterministic synthetic datasets with planted leaks, for docs and tests."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_leaky_dataset(
    n: int = 600,
    seed: int = 0,
    overlap: int = 30,
    test_frac: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build a binary-classification dataset that contains every failure mode.

    The returned ``full`` frame has:

    * ``target``          -- the binary label,
    * ``clean_signal``    -- a genuinely predictive but imperfect feature,
    * ``noise_a/noise_b`` -- pure noise,
    * ``leaky_probe``     -- the target with 1% label noise (planted leak),
    * ``customer_id``     -- a unique identifier (ID-like),
    * ``batch_flag``      -- a constant column.

    It is then split into ``train`` and ``test`` such that ``overlap`` train
    rows are copied verbatim into test (planted contamination).

    Returns:
        ``(full, train, test)``.
    """
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    rng = np.random.default_rng(seed)

    latent = rng.normal(size=n)
    target = (latent + rng.normal(scale=0.5, size=n) > 0).astype(int)

    clean_signal = latent + rng.normal(scale=0.8, size=n)
    noise_a = rng.normal(size=n)
    noise_b = rng.integers(0, 5, size=n).astype(float)

    # Planted target leak: equal to the label except for a 1% flip.
    leaky_probe = target.astype(float).copy()
    flip_idx = rng.choice(n, size=max(1, n // 100), replace=False)
    leaky_probe[flip_idx] = 1.0 - leaky_probe[flip_idx]

    full = pd.DataFrame(
        {
            "customer_id": [f"C{100000 + i}" for i in range(n)],
            "clean_signal": clean_signal,
            "noise_a": noise_a,
            "noise_b": noise_b,
            "leaky_probe": leaky_probe,
            "batch_flag": np.ones(n, dtype=int),
            "target": target,
        }
    )

    n_test = round(n * test_frac)
    perm = rng.permutation(n)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    train = full.iloc[train_idx].reset_index(drop=True)
    test = full.iloc[test_idx].reset_index(drop=True)

    if overlap > 0:
        take = min(overlap, len(train))
        leaked_rows = train.iloc[:take]
        test = pd.concat([test, leaked_rows], ignore_index=True)

    return full, train, test
