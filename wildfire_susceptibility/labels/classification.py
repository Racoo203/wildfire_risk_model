"""Backward-compatible re-export.

LabelCleaner's real implementation now lives in modeling/label_cleaning.py
(Section 7.3 of V2_ARCHITECTURE_BLUEPRINT.md). This module is kept as a
thin re-export so existing code — notably
wildfire_susceptibility/pipeline/preprocessor.py, which does
`from ..labels.classification import LabelCleaner` — continues to work
without any changes.
"""

from ..modeling.label_cleaning import LabelCleaner, HIGH_RISK_CLASSES

__all__ = ["LabelCleaner", "HIGH_RISK_CLASSES"]