import numpy as np

if not hasattr(np, "int"):
    # NumPy >=1.24 removed the deprecated `np.int` alias; mord 0.6 (the version
    # pinned via conda-forge — 0.7 exists on PyPI but not on conda-forge, so
    # pulling it in would mean mixing pip into an otherwise pure conda-forge
    # env) still calls `np.int` internally in LogisticAT.fit()/threshold_predict.
    # NumPy's own deprecation notice for this removal says `np.int` was just an
    # alias for the builtin `int` and that using `int` directly "will not modify
    # any behavior and is safe" — this restores exactly that, guarded so it's a
    # no-op if a future numpy reintroduces the name, and it does not change
    # numpy's behavior anywhere else in the process.
    np.int = int

import mord

from ...core.registry import MODELS


@MODELS.register("ordinal_lr")
class OrdinalLogisticModel:
    """Proportional-odds ordinal logistic regression (mord.LogisticAT) for the
    4-class ordinal susceptibility target. mord.fit() natively accepts a
    sample_weight kwarg — not used yet (cost-weighted learning is a later
    branch), but confirmed to work, unlike statsmodels' OrderedModel which has
    no weighting path at all.

    Note: mord's own .predict() uses the proper cumulative-threshold ordinal
    decision rule, which can disagree with argmax(predict_proba(X)) — the two
    are computed from the same latent score but via different reductions.
    This wrapper only exposes predict_proba, per the shared model contract
    (every caller derives hard labels via argmax(predict_proba), uniformly
    across all models) — so ordinal LR's hard-label predictions in this
    pipeline are the argmax-derived ones, not mord's native threshold rule.
    """

    def __init__(self, **kwargs):
        self.params = kwargs
        self.model: mord.LogisticAT | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OrdinalLogisticModel":
        self.model = mord.LogisticAT(**self.params)
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def param_space(self, trial) -> dict:
        return {
            "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
        }

    def needs_scaling(self) -> bool:
        return True
