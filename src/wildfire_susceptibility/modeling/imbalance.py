# wildfire_susceptibility/modeling/imbalance.py
"""Resolves modeling.imbalance_strategy (+ per-model override) to which
imbalance-handling mechanism applies for a given model: SMOTE resampling
("smote"), cost-weighted sample_weight ("cost_weighted"), each model's own
native mechanism ("native_balanced" — see modeling/models/random_forest.py
and modeling/models/catboost_model.py for what that means per model), or
neither ("none"). Mutually exclusive by construction — both
"cost_weighted" and "native_balanced" make SMOTEResampler.resample() a
no-op regardless of use_smote, and neither ever produces a sample_weight
for the other to double-correct against. An explicit imbalance_strategy
(or imbalance_strategy_by_model) entry of "smote" is sufficient on its
own to enable resampling — see smote_explicitly_enabled() — matching how
"cost_weighted"/"native_balanced" need no second flag; use_smote and
friends remain load-bearing only for configs that never mention
imbalance_strategy at all."""

from typing import Optional

import numpy as np
from sklearn.utils.class_weight import compute_sample_weight


class ImbalanceStrategy:
    def __init__(self, config: dict):
        modeling_cfg = config["modeling"]
        # schema.py deliberately leaves the pydantic default at None (not
        # "smote") so this None-vs-real-value distinction survives
        # model_dump() — see that field's docstring. The "smote" fallback
        # lives here, not in the schema, precisely so it doesn't count as
        # "explicit" for smote_explicitly_enabled() below.
        configured = modeling_cfg.get("imbalance_strategy")
        self._default_explicit = configured is not None
        self.default = configured or "smote"
        self.overrides = modeling_cfg.get("imbalance_strategy_by_model", {}) or {}

    def resolve(self, model_name: str) -> str:
        return self.overrides.get(model_name, self.default)

    def smote_allowed(self, model_name: str) -> bool:
        return self.resolve(model_name) == "smote"

    def smote_explicitly_enabled(self, model_name: str) -> bool:
        """True when this model resolves to 'smote' via a config setting
        someone actually wrote down — either a per-model
        imbalance_strategy_by_model entry, or a top-level
        imbalance_strategy: "smote" — as opposed to resolving to 'smote'
        only because imbalance_strategy was never mentioned at all (the
        legacy default). Unlike 'cost_weighted'/'native_balanced', which
        need no second flag, SMOTE historically also required use_smote
        (or search_resample_target_size/smote_sampling_strategy) to
        actually activate; that back-compat requirement still governs
        the legacy-default case (see SMOTEResampler), but an explicit
        imbalance_strategy: "smote" should be sufficient on its own —
        this is what lets it be."""
        if model_name in self.overrides:
            return self.overrides[model_name] == "smote"
        return self._default_explicit and self.default == "smote"

    def sample_weight_for(self, model_name: str, y) -> Optional[np.ndarray]:
        """Balanced (inverse-frequency) per-sample weight, computed fresh
        from `y` — the training labels actually being fit on, whatever
        fold/subsample/refit this call is for — or None if this model
        isn't resolved to 'cost_weighted'."""
        if self.resolve(model_name) != "cost_weighted":
            return None
        return compute_sample_weight("balanced", y)
