"""Canonical class-name schemes, keyed by class count.

Single source of truth for every consumer (metrics, post-training
evaluation, EDA charts) that used to hardcode its own 4-name list —
those lists silently went stale once n_classes stopped being fixed at 4.
"""
from typing import List

CLASS_NAME_SCHEMES = {
    3: ["Low", "Medium", "High"],
    4: ["Low", "Medium", "High", "Very High"],
    5: ["Very Low", "Low", "Medium", "High", "Very High"],
}


def class_names_for(n_classes: int) -> List[str]:
    try:
        return CLASS_NAME_SCHEMES[n_classes]
    except KeyError:
        raise ValueError(
            f"No class-name scheme defined for n_classes={n_classes}. "
            f"Supported: {sorted(CLASS_NAME_SCHEMES)}"
        )
