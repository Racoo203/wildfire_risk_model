"""Shared CV mechanics every strategy inherits: fold scoring and the
mean-AUC convenience wrapper. Subclasses implement only make_folds()."""

from abc import ABC, abstractmethod
from math import ceil
from typing import List, Optional, Tuple
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, cohen_kappa_score

from ..categorical import cat_features_of, to_model_array
from ..resampling import SMOTEResampler
from ..imbalance import ImbalanceStrategy

logger = logging.getLogger(__name__)


def score_multiclass_fold(y_va: np.ndarray, proba: np.ndarray, y_tr: np.ndarray, context: str = "") -> dict:
    """AUC, F1-macro, PR-AUC-macro, and QWK for one fold's validation
    predictions, shared by every CVStrategy (base and
    StratifiedSpatialBlockCV alike — this replaces two independent copies
    of the same scoring logic that had independently drifted out of sync
    with what `proba`'s columns actually mean).

    `proba`'s columns don't always correspond to class labels positionally:
    - RF/CatBoost/mord (ordinal_lr) only emit one column per class actually
      present in `y_tr` (whatever `model.fit()` was called with), so a fold
      whose training split is missing a class has fewer columns than the
      full configured label space.
    - The neural_net wrapper (ann.py's `n_classes = max(y) + 1`) always
      builds a gapless `0..max(y_tr)` output layer regardless of gaps, so a
      missing *middle* class does NOT shrink its column count — that column
      still exists, just untrained/meaningless.

    `trained_classes` below is inferred from shape alone — correct for
    both conventions without knowing which model produced `proba`: if the
    column count matches the number of distinct classes in `y_tr`, columns
    are `sorted(unique(y_tr))` in order (first convention); otherwise
    columns are `range(proba.shape[1])` (gapless convention). This is also
    what fixes `pred = argmax(proba)`, previously a raw column *position*
    treated as if it were the class label — silently wrong for F1-macro/QWK
    whenever a fold's training split was missing a non-maximal class, even
    though that path never raised (so it never showed up in the original
    "scoring as 0.0" log lines the AUC/PR-AUC fix below was diagnosed from).

    AUC/PR-AUC-macro are further restricted to `observed` — classes in
    both `trained_classes` and this fold's `y_va` — since a class the model
    never learned a column for has no meaningful probability to rank, and a
    class the model can score but this fold's ground truth never contains
    contributes nothing to either metric anyway. A fold with fewer than 2
    observed classes has undefined AUC/PR-AUC and is scored NaN (logged),
    not 0.0 — 0.0 reads as "the model failed," which misrepresents a fold
    that was never scorable to begin with.
    """
    unique_y_tr = sorted(np.unique(y_tr).tolist())
    if proba.shape[1] == len(unique_y_tr):
        trained_classes = unique_y_tr
    else:
        trained_classes = list(range(proba.shape[1]))
    trained_classes_arr = np.array(trained_classes)

    pred = trained_classes_arr[np.argmax(proba, axis=1)]
    f1_macro = float(f1_score(y_va, pred, average="macro"))
    qwk = float(cohen_kappa_score(y_va, pred, weights="quadratic"))

    y_va_classes = set(np.unique(y_va).tolist())
    observed = [c for c in trained_classes if c in y_va_classes]

    if len(observed) < 2:
        logger.warning(
            f"{context}: fold has fewer than 2 classes in common between validation "
            f"labels and trained classes (trained={trained_classes}, "
            f"validation={sorted(y_va_classes)}, observed={observed}); AUC/PR-AUC are "
            f"undefined for this fold, scoring as NaN."
        )
        return {"auc": float("nan"), "f1_macro": f1_macro, "pr_auc_macro": float("nan"), "qwk": qwk}

    row_mask = np.isin(y_va, observed)
    col_idx = [trained_classes.index(c) for c in observed]
    y_va_r = y_va[row_mask]
    proba_r = proba[row_mask][:, col_idx]

    try:
        if len(observed) == 2:
            # sklearn's roc_auc_score routes exactly-2-label y_true through
            # its binary path regardless of multi_class="ovr", which expects
            # a 1D positive-class score, not a 2-column matrix — observed is
            # ascending, so column 1 (the higher-valued class) is the
            # positive class under sklearn's own binary convention. The
            # binary path is rank-based only, so proba_r's raw column
            # doesn't need renormalizing the way the multiclass path below
            # does.
            auc_score = float(roc_auc_score(y_va_r, proba_r[:, 1]))
        else:
            # roc_auc_score's multiclass path requires each row to sum to
            # 1.0 across the given `labels` ("Target scores need to be
            # probabilities for multiclass roc_auc"). When `observed` is a
            # strict subset of `trained_classes` (e.g. the model was
            # trained on 5 classes but this fold's validation split only
            # exercises 3 of them), dropping the unobserved columns leaves
            # each row summing to less than 1 — renormalize the remaining
            # probability mass across just the observed classes so the
            # constraint still holds.
            proba_r_norm = proba_r / proba_r.sum(axis=1, keepdims=True)
            auc_score = float(roc_auc_score(y_va_r, proba_r_norm, multi_class="ovr", labels=observed))
    except Exception as exc:
        logger.warning(
            f"{context}: AUC scoring failed even after restricting to observed classes "
            f"{observed} ({exc}); scoring as NaN."
        )
        auc_score = float("nan")

    try:
        y_va_bin = (y_va_r[:, None] == np.array(observed)[None, :]).astype(int)
        pr_auc_macro = float(average_precision_score(y_va_bin, proba_r, average="macro"))
    except Exception as exc:
        logger.warning(
            f"{context}: PR-AUC scoring failed even after restricting to observed classes "
            f"{observed} ({exc}); scoring as NaN."
        )
        pr_auc_macro = float("nan")

    return {"auc": auc_score, "f1_macro": f1_macro, "pr_auc_macro": pr_auc_macro, "qwk": qwk}


class CVStrategy(ABC):
    """One CV strategy = one way to build folds + one way to fit/score a fold.

    Subclasses embody a *specific* strategy (standard / spatial / stratified
    spatial block) rather than being selected via a string switch at call
    time — callers get the right strategy from cv.factory.get_cv_strategy().
    """

    name: str = "base"

    def __init__(self, config: dict, resampler: SMOTEResampler):
        self.config = config
        self.cv_folds = config["modeling"]["cv_folds"]
        self.resampler = resampler
        self.spatial_buffer_m = config["modeling"].get("spatial_buffer_m", 0.0) or 0.0
        self.spatial_block_size_m = config["modeling"].get("spatial_block_size_m", 5000.0)
        self._imbalance = ImbalanceStrategy(config)

    @abstractmethod
    def make_folds(
        self, X: pd.DataFrame, y: pd.Series, groups: Optional[pd.Series] = None,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Return a list of (train_idx, test_idx) positional-index pairs."""
        ...

    def _buffer_radius_blocks(self) -> int:
        """How many grid cells out from a test block to exclude, so no kept
        train point can be closer than spatial_buffer_m to a test block's
        edge. Blocks are spatial_block_size_m squares, so a ring of radius r
        guarantees a minimum gap of (r-1)*spatial_block_size_m; ceil() rounds
        up so the configured buffer is always met, never just approached."""
        if self.spatial_buffer_m <= 0:
            return 0
        return ceil(self.spatial_buffer_m / self.spatial_block_size_m)

    def _apply_spatial_buffer(
        self, train_idx: np.ndarray, test_idx: np.ndarray, groups: pd.Series,
    ) -> np.ndarray:
        """Drop from train_idx any row whose (block_x, block_y) falls within
        the buffer ring around any test block for this fold. test_idx itself
        is never touched — buffering only removes near-boundary train rows
        that would otherwise sit right next to a test block with zero
        enforced separation (see cv/spatial.py, cv/stratified_spatial.py).
        No-op when spatial_buffer_m is 0 (the default), so existing configs
        and results are unaffected unless a buffer is explicitly set."""
        radius = self._buffer_radius_blocks()
        if radius <= 0:
            return train_idx

        test_blocks = {groups.iloc[i] for i in test_idx}
        excluded_blocks = {
            (bx + dx, by + dy)
            for bx, by in test_blocks
            for dx in range(-radius, radius + 1)
            for dy in range(-radius, radius + 1)
        }

        train_blocks = groups.iloc[train_idx].to_numpy()
        keep_mask = np.fromiter(
            (block not in excluded_blocks for block in train_blocks),
            dtype=bool, count=len(train_blocks),
        )
        dropped = int((~keep_mask).sum())
        if dropped:
            logger.info(
                f"[{self.name}] spatial buffer ({self.spatial_buffer_m:.0f}m, "
                f"radius={radius} block(s)): dropped {dropped:,}/{len(train_idx):,} "
                f"train rows within the buffer ring of this fold's test blocks"
            )
        return train_idx[keep_mask]

    def fit_and_score_full(
        self, model_cls, params: dict, X: pd.DataFrame, y: pd.Series,
        train_idx: np.ndarray, test_idx: np.ndarray, context: str = "",
        model_name: Optional[str] = None,
    ) -> dict:
        """Fit once on train_idx (resampled per self.resampler), score AUC,
        F1-macro, PR-AUC-macro, and QWK on test_idx untouched — all
        multiclass-safe (roc_auc_score/f1_score use ovr/macro averaging,
        average_precision_score is computed on one-hot-encoded y, QWK uses
        quadratic-weighted Cohen's kappa since the 4 classes are an ordinal
        risk discretization — distant misclassifications are penalized more
        than adjacent ones). Shared by every strategy; strategies that need
        different resampling semantics (e.g. stratified-spatial-block's
        extra class-balance logging) override this. `fit_and_score`
        delegates here so a fold's model is only ever fit once even when
        both the scalar and the full metric set are needed (see
        trainer.py's optimism-gap logging).

        `model_name`, when given, resolves modeling.imbalance_strategy for
        this model (see modeling/imbalance.py): SMOTE is disabled here
        regardless of self.resampler's own config if the resolved strategy
        isn't 'smote', and a balanced sample_weight is computed from y_tr
        (post-resample) and passed to model.fit() if the resolved strategy
        is 'cost_weighted'. model_name=None (e.g. ad-hoc/test callers)
        preserves old behavior: resampler runs as configured, no weighting."""
        logger.info(f"{context}: fitting on {len(train_idx):,} rows...")
        cat_positions = cat_features_of(model_cls)
        X_tr = to_model_array(X.iloc[train_idx], cat_positions)
        y_tr = y.iloc[train_idx].values
        X_tr, y_tr = self.resampler.resample(X_tr, y_tr, context=context, model_name=model_name)

        sample_weight = self._imbalance.sample_weight_for(model_name, y_tr) if model_name else None
        model = model_cls(**params)
        model.fit(X_tr, y_tr, sample_weight=sample_weight)

        y_va = y.iloc[test_idx].values
        proba = model.predict_proba(to_model_array(X.iloc[test_idx], cat_positions))

        return score_multiclass_fold(y_va, proba, y_tr, context=context)

    def fit_and_score(
        self, model_cls, params: dict, X: pd.DataFrame, y: pd.Series,
        train_idx: np.ndarray, test_idx: np.ndarray, context: str = "",
        model_name: Optional[str] = None,
    ) -> float:
        """Scalar-AUC convenience wrapper around fit_and_score_full, kept
        for callers (HyperparamSearch's Optuna objective) that only need
        the one number."""
        return self.fit_and_score_full(
            model_cls, params, X, y, train_idx, test_idx, context=context, model_name=model_name,
        )["auc"]

    def mean_auc_across_folds(
        self, model_cls, params, X, y, folds, context: str = "", model_name: Optional[str] = None,
    ) -> float:
        scores = []
        for i, (train_idx, test_idx) in enumerate(folds):
            fold_context = f"{context} fold {i + 1}/{len(folds)}"
            auc = self.fit_and_score(
                model_cls, params, X, y, train_idx, test_idx, context=fold_context, model_name=model_name,
            )
            logger.info(f"{fold_context} AUC={auc:.4f}")
            scores.append(auc)
        return float(np.mean(scores))