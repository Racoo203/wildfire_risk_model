# wildfire_susceptibility/modeling/imbalance.py
"""Resolves modeling.imbalance_strategy (+ per-model override) to which
imbalance-handling mechanism applies for a given model: SMOTE resampling
("smote"), cost-weighted sample_weight ("cost_weighted"), each model's own
native mechanism ("native_balanced" — see modeling/models/random_forest.py
and modeling/models/catboost_model.py for what that means per model), or
neither ("none"). Mutually exclusive by construction — both
"cost_weighted" and "native_balanced" make SMOTEResampler.resample() a
no-op regardless of use_smote, and neither ever produces a sample_weight
for the other to double-correct against."""

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
