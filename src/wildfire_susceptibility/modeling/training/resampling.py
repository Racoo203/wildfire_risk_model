# wildfire_susceptibility/modeling/training/resampling.py
"""SMOTE oversampling (and, when a target size is configured, combined
over+under sampling to a fixed total), isolated from the training loop
so ModelTrainer doesn't need to know anything about imblearn or
degenerate-fold edge cases."""

from typing import Optional
import logging

import numpy as np

from .balance import log_class_balance, imbalance_ratio

logger = logging.getLogger(__name__)


class SMOTEResampler:
    def __init__(
        self,
        config: dict,
        target_size: Optional[float] = None,
        is_final: bool = False,
    ):
        modeling_cfg = config["modeling"]
        self.k_neighbors = modeling_cfg.get("smote_k_neighbors", 5)
        self.sampling_strategy = modeling_cfg.get("smote_sampling_strategy", "auto")
        self.is_final = is_final

        # Final refit NEVER inherits search_resample_target_size, regardless
        # of whether it's set in config — that knob controls "how big/balanced
        # is the fold the model gets fit on during HPO", not the final refit.
        # Only a caller that explicitly passes target_size (search-time callers)
        # can enable fixed-size resampling.
        if is_final:
            self.target_size = None
        else:
            self.target_size = (
                target_size if target_size is not None
                else modeling_cfg.get("search_resample_target_size")
            )

        use_smote_explicit = modeling_cfg.get("use_smote")
        implicit_enable = (
            (not is_final) and use_smote_explicit is None and self.target_size is not None
        ) or (
            is_final and use_smote_explicit is None and modeling_cfg.get("smote_sampling_strategy") is not None
        )
        self.enabled = bool(use_smote_explicit) or implicit_enable

        if self.enabled and not use_smote_explicit and self.target_size is not None:
            logger.info(
                f"SMOTEResampler: use_smote not explicitly set but "
                f"search_resample_target_size={self.target_size} was provided — "
                f"enabling resampling implicitly."
            )

    def resample(self, X, y, context=""):
        if not self.enabled:
            log_class_balance(logger, context, y, note="SMOTE disabled — distribution unchanged", level=logging.DEBUG)
            return X, y

        if self.target_size:
            return self._resample_to_target_size(X, y, self.target_size, context)
        return self._resample_majority_relative(X, y, context)

    # -----------------------------------------------------------------
    # Mode 1: majority-relative (original behavior) — auto or fixed dict
    # -----------------------------------------------------------------

    def _resample_majority_relative(self, X, y, context=""):
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError:
            logger.warning(f"{context}: use_smote=True but imbalanced-learn is not installed; skipping.")
            return X, y

        before_counts = log_class_balance(logger, context, y, note="before SMOTE")
        classes, counts = np.unique(y, return_counts=True)
        min_class_count = int(counts.min())
        if min_class_count <= 1:
            logger.warning(f"{context}: smallest class has {min_class_count} sample(s); skipping SMOTE.")
            return X, y

        majority_count = int(counts.max())
        strategy = self.sampling_strategy
        if isinstance(strategy, dict):
            current = {int(c): n for c, n in zip(classes.tolist(), counts.tolist())}
            strategy = {}
            for cls, target in self.sampling_strategy.items():
                cls = int(cls)
                if cls not in current:
                    continue
                capped_target = min(target, majority_count)
                if capped_target > current[cls]:
                    strategy[cls] = capped_target
            if not strategy:
                return X, y

        k = min(self.k_neighbors, min_class_count - 1)
        smote = SMOTE(sampling_strategy=strategy, k_neighbors=k, random_state=42)
        X_res, y_res = smote.fit_resample(X, y)

        after_counts = log_class_balance(logger, context, y_res, note="after SMOTE")
        after_ratio = imbalance_ratio(y_res)
        logger.info(
            f"{context}: SMOTE applied — {before_counts} -> {after_counts} "
            f"(k={k}, strategy={strategy}, resulting imbalance_ratio={after_ratio:.2f})"
        )
        if after_ratio > 1.5:
            logger.warning(
                f"{context}: post-SMOTE imbalance_ratio={after_ratio:.2f} is still notably "
                f"unbalanced — check smote_sampling_strategy targets against this fold's sizes."
            )
        return X_res, y_res

    # -----------------------------------------------------------------
    # Mode 2: fixed-total, evenly-split-across-classes resample
    # -----------------------------------------------------------------

    def _resample_to_target_size(self, X, y, target_size: float, context: str = ""):
        try:
            from imblearn.over_sampling import SMOTE
            from imblearn.under_sampling import RandomUnderSampler
        except ImportError:
            logger.warning(f"{context}: use_smote=True but imbalanced-learn is not installed; skipping.")
            return X, y

        classes, counts = np.unique(y, return_counts=True)
        n_classes = len(classes)
        if n_classes == 0:
            return X, y

        resolved_total = int(target_size) if target_size > 1 else int(round(target_size * len(y)))
        per_class_target = max(resolved_total // n_classes, 2)  # SMOTE needs >=2 to synthesize from


        before_counts = log_class_balance(
            logger, context, y, note=f"before size-target resample (target={per_class_target}/class, total~{resolved_total})",
        )
        current = dict(zip(classes.tolist(), counts.tolist()))

        # Step 1: undersample any class currently ABOVE the per-class target.
        under_targets = {c: per_class_target for c, n in current.items() if n > per_class_target}
        if under_targets:
            rus = RandomUnderSampler(sampling_strategy=under_targets, random_state=42)
            X, y = rus.fit_resample(X, y)

        # Step 2: oversample any class still BELOW the per-class target.
        classes2, counts2 = np.unique(y, return_counts=True)
        current2 = dict(zip(classes2.tolist(), counts2.tolist()))

        skipped = [c for c in classes.tolist() if current2.get(c, 0) < 2]
        if skipped:
            logger.warning(f"{context}: class(es) {skipped} have <2 samples after undersampling; cannot SMOTE to target, left as-is.")

        over_targets = {
            c: per_class_target for c in classes.tolist()
            if current2.get(c, 0) < per_class_target and current2.get(c, 0) >= 2
        }
        if over_targets:
            min_count = min(current2[c] for c in over_targets)
            k = min(self.k_neighbors, max(min_count - 1, 1))
            smote = SMOTE(sampling_strategy=over_targets, k_neighbors=k, random_state=42)
            X, y = smote.fit_resample(X, y)

        after_counts = log_class_balance(logger, context, y, note="after size-target resample")
        after_ratio = imbalance_ratio(y)
        logger.info(
            f"{context}: size-target resample — {before_counts} -> {after_counts} "
            f"(target={per_class_target}/class, total={len(y):,}, imbalance_ratio={after_ratio:.2f})"
        )
        if after_ratio > 1.5:
            logger.warning(
                f"{context}: post-resize imbalance_ratio={after_ratio:.2f} — likely caused by a "
                f"class with too few source samples to reach its target via SMOTE (see 'skipped' above)."
            )
        return X, y