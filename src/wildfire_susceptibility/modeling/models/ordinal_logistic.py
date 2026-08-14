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
    sample_weight kwarg, threaded straight into the internal threshold_fit()
    solver call — confirmed via source inspection, unlike statsmodels'
    OrderedModel which has no weighting path at all. Populated by
    modeling.imbalance.ImbalanceStrategy when imbalance_strategy resolves to
    'cost_weighted' for this model; None otherwise.

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

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "OrdinalLogisticModel":
        self.model = mord.LogisticAT(**self.params)
        # Labels reach us as float (raster stacking casts every column,
        # including labels, to float32 for NaN/nodata support). mord derives
        # n_class_ from y's own dtype (classes_.max() - classes_.min() + 1)
        # rather than casting it, so a float y makes n_class_ a numpy.float64
        # and mord's internal np.zeros((n_class_ - 1, ...)) raises TypeError.
        self.model.fit(X, y.astype(int), sample_weight=sample_weight)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def param_space(self, trial) -> dict:
        return {
            "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
        }

    def needs_scaling(self) -> bool:
        return True

    def native_categorical_support(self) -> bool:
        return False
