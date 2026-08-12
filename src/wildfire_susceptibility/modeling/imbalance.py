# wildfire_susceptibility/modeling/imbalance.py
"""Resolves modeling.imbalance_strategy (+ per-model override) to which
imbalance-handling mechanism applies for a given model: SMOTE resampling
("smote"), cost-weighted sample_weight ("cost_weighted"), or neither
("none"). Mutually exclusive by construction — "cost_weighted" makes
SMOTEResampler.resample() a no-op regardless of use_smote, so the two
mechanisms never stack and double-correct the same imbalance."""

from typing import Optional

import numpy as np
from sklearn.utils.class_weight import compute_sample_weight


class ImbalanceStrategy:
    def __init__(self, config: dict):
        modeling_cfg = config["modeling"]
        self.default = modeling_cfg.get("imbalance_strategy", "smote")
        self.overrides = modeling_cfg.get("imbalance_strategy_by_model", {}) or {}

    def resolve(self, model_name: str) -> str:
        return self.overrides.get(model_name, self.default)

    def smote_allowed(self, model_name: str) -> bool:
        return self.resolve(model_name) == "smote"

    def sample_weight_for(self, model_name: str, y) -> Optional[np.ndarray]:
        """Balanced (inverse-frequency) per-sample weight, computed fresh
        from `y` — the training labels actually being fit on, whatever
        fold/subsample/refit this call is for — or None if this model
        isn't resolved to 'cost_weighted'."""
        if self.resolve(model_name) != "cost_weighted":
            return None
        return compute_sample_weight("balanced", y)
