"""Shared class-balance logging — used wherever we need to *verify*
(not just assume) that resampling or stratification produced the
balance we intended."""

from typing import Optional
import logging

import numpy as np
import pandas as pd


def class_counts_dict(y) -> dict:
    y = pd.Series(y)
    return {int(k): int(v) for k, v in y.value_counts().sort_index().items()}


def imbalance_ratio(y) -> float:
    """majority count / minority count — 1.0 is perfectly balanced."""
    counts = class_counts_dict(y)
    if len(counts) < 2 or min(counts.values()) == 0:
        return float("inf")
    return max(counts.values()) / min(counts.values())


def log_class_balance(
    logger: logging.Logger, context: str, y, note: str = "", level: int = logging.INFO,
) -> dict:
    """Logs class counts + imbalance ratio for `y`, returns the counts dict
    so callers can assert on it in tests without re-parsing log text."""
    counts = class_counts_dict(y)
    ratio = imbalance_ratio(y)
    suffix = f" ({note})" if note else ""
    logger.log(level, f"{context}: class counts={counts} | imbalance_ratio={ratio:.2f}{suffix}")
    return counts


def warn_if_class_missing(
    logger: logging.Logger, context: str, y, expected_classes, level: int = logging.WARNING,
) -> None:
    present = set(class_counts_dict(y).keys())
    missing = set(expected_classes) - present
    if missing:
        logger.log(level, f"{context}: missing class(es) {sorted(missing)} — not represented here.")