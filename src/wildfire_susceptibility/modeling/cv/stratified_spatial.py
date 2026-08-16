# src/wildfire_susceptibility/modeling/cv/stratified_spatial.py
"""Stratified Spatial Block Cross-Validation with In-Fold Resampling.

This is the dissertation's headline CV method — its methodological
contribution, alongside seasonal stratification. The other two strategies
(`standard.py`, `spatial.py`) are baselines it is compared against.

Steps, mapped to methods:

    1. Grid Partitioning       -> DatasetPrep.assign_spatial_blocks (block_size_m)
    2. Class Stratification    -> _stratify_blocks_into_folds, itself two
                                   passes (see that method's docstring):
                                     1a. _seed_every_fold_with_rarest_class_coverage
                                     1b. _greedily_fill_remaining_blocks
    3. Boundary Buffering      -> CVStrategy._apply_spatial_buffer (spatial_buffer_m,
                                   optional, 0 by default) drops train blocks within
                                   the buffer ring of each fold's test blocks
    4. Strict Validation       -> make_folds returns held-out fold UNTOUCHED
       Isolation                 (no resampling ever applied to test_idx —
                                   see the "untouched" naming in
                                   fit_and_score_full below)
    5. In-Fold Training         -> _fit_resampled_model resamples ONLY the
       Resampling                 training side, using self.resampler
                                   (never disabled for this strategy, unlike
                                   the plain spatial baseline's search-time
                                   SMOTE-off convention)
    6. Model Training/Eval      -> fit_and_score_full reports PR-AUC-macro
                                    (the Optuna HPO objective) + AUC/F1-macro/
                                    QWK, scored by _score_fold_metrics, as
                                    documented in the dissertation methodology

What "stratified" means here (and what it does not)
-----------------------------------------------------
"Stratified" is at the *block* level, not the row level: whole spatial
blocks are assigned to folds so that each fold's aggregate class
proportions approximate the dataset's overall class proportions —
including the rarest class, which real fire-susceptibility data confines
to a handful of blocks. This is deliberately not row-level stratification
(e.g. sklearn's `StratifiedKFold`), which would require splitting a single
block's rows across folds and would reintroduce the spatial leakage that
motivates using blocks as the CV unit in the first place.

This class does NOT address feature-space / covariate shift between
folds — i.e. it does not guarantee that a fold's *predictor* distribution
resembles the others', only its *label* distribution. That is a distinct,
still-open problem; see `spatial.py`'s module docstring for the
covariate-shift-agnostic baseline (verde.BlockKFold) this class is
compared against in the dissertation's CV comparison table.

Why this is bespoke code, not a library call
-----------------------------------------------
Class-stratified spatial blocking sits at the intersection of two
literatures that don't talk to each other. Spatial CV (Roberts et al. 2017;
Brenning's `sperrorest`; `verde`) comes from geostatistics/remote sensing
and targets continuous spatial autocorrelation, with no native concept of
class balance. Class-imbalance handling (SMOTE, cost-sensitive learning)
comes from ML and assumes i.i.d. data with no spatial structure. Balancing
K folds simultaneously on (a) spatial block contiguity and (b) every
class's representation — especially a rare class confined to a handful of
blocks — is a constrained multi-objective assignment problem with no
closed-form solution; any implementation is necessarily a domain-specific
heuristic, not a generalizable algorithm, which is exactly why it stays
bespoke code rather than becoming a package dependency. The closest
published candidate, SP-CV / "Spatial+" (Wang et al. 2023), was evaluated
twice in this project — under both its original framing and its later
renamed one — and confirmed both times to solve feature-space shift
between folds via agglomerative clustering, not the class-imbalance-
within-block problem this class exists to solve.

Worked example
-----------------
Six blocks, three rows each, one rare-class ("2") block. Split into 3 folds:

    block:        A    B    C    D    E    F
    class 0 rows:  1    2    0    1    3    0
    class 1 rows:  2    1    0    2    0    3
    class 2 rows:  0    0    3    0    0    0   <- only block C has class 2

Class totals: class 2 = 3, class 0 = 7, class 1 = 8, so
rarity_order = [2, 0, 1] (rarest first).

Pass 1 (`_seed_every_fold_with_rarest_class_coverage`) walks classes in
that order:

  - class 2: only block C carries it -> C seeded into fold 0.
  - class 0: candidates ranked by class-0 row count, largest first
    (E=3, B=2, A=1, D=1) -> the top 3 (E, B, A) are round-robined into
    folds 0, 1, 2 respectively.
  - class 1: of the two blocks left unassigned (D, F), ranked by
    class-1 count (F=3, D=2) -> F into fold 0, D into fold 1.

Every block is already placed after Pass 1 in this example — {C, E, F} ->
fold 0, {B, D} -> fold 1, {A} -> fold 2 — so Pass 2
(`_greedily_fill_remaining_blocks`) has nothing left to do here. Pass 2's
fill only activates once some class has more contributing blocks than
there are folds — see
`test_greedily_fill_remaining_blocks_places_every_leftover_block`, which
uses a larger synthetic fixture specifically to exercise blocks left over
after Pass 1: each is assigned to whichever fold currently has the least
of that block's own dominant class, tie-broken by which fold has fewest
rows total so far.

See `test_seed_every_fold_with_rarest_class_coverage_places_the_only_rare_block`
for this exact worked example executed as assertions, not just prose.
"""
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, cohen_kappa_score

from .base import CVStrategy
from ..balance import log_class_balance, warn_if_class_missing
from ..categorical import cat_features_of, to_model_array

logger = logging.getLogger(__name__)

# One block's per-class row tally, e.g. {0: 12, 1: 4, 2: 0}.
BlockClassCounts = Dict[int, int]


class StratifiedSpatialBlockCV(CVStrategy):
    """Blocks (not rows) are the CV unit; blocks are greedily assigned to
    K folds to balance every class's representation — including the
    rarest ("very high") class — across folds. See the module docstring
    for the full algorithm walkthrough and worked example."""

    name = "stratified_spatial_block"

    def make_folds(self, X, y, groups: Optional[pd.Series] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        if groups is None:
            raise ValueError("Stratified spatial block CV requested but no spatial groups (blocks) were provided.")
        groups = pd.Series(groups).reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)

        fold_of_block = self._stratify_blocks_into_folds(groups, y, self.cv_folds)
        row_fold = groups.map(fold_of_block).to_numpy()

        folds = []
        for k in range(self.cv_folds):
            test_idx = np.where(row_fold == k)[0]
            train_idx = np.where(row_fold != k)[0]
            # test_idx is passed in only to *exclude* nearby train blocks —
            # it is never itself modified. The buffer ring is centered on
            # this fold's test blocks, so all filtering happens on the
            # train side.
            train_idx = self._apply_spatial_buffer(train_idx, test_idx, groups)
            if len(test_idx) == 0 or len(train_idx) == 0:
                logger.warning(f"[stratified_spatial_block] fold {k} is degenerate (empty split); skipping.")
                continue

            # Verifies Step 2 (Class Stratification): confirm every fold —
            # including the smallest, "very high" class — is represented,
            # and log each fold's raw (pre-resample) proportions so a
            # reader can visually confirm stratification worked rather
            # than just trusting the assignment algorithm ran.
            log_class_balance(
                logger, f"[stratified_spatial_block] fold {k} (validation, untouched)", y.iloc[test_idx],
            )
            warn_if_class_missing(
                logger, f"[stratified_spatial_block] fold {k} (validation)", y.iloc[test_idx], expected_classes=sorted(y.unique()),
            )
            folds.append((train_idx, test_idx))

        self._log_cross_fold_balance_summary(y, folds)
        return folds

    @staticmethod
    def _log_cross_fold_balance_summary(y: pd.Series, folds) -> None:
        """One-line-per-class summary of how evenly each class's rows were
        spread across folds — the actual thing 'balanced stratification'
        means, distinct from in-fold SMOTE balance."""
        classes = sorted(y.unique())
        per_fold_props = []
        for _, test_idx in folds:
            fold_y = y.iloc[test_idx]
            fold_counts = fold_y.value_counts()
            per_fold_props.append({c: fold_counts.get(c, 0) / max(len(fold_y), 1) for c in classes})

        for c in classes:
            props = [p[c] for p in per_fold_props]
            spread = (max(props) - min(props)) if props else 0.0
            logger.info(
                f"[stratified_spatial_block] class {c} proportion across folds: "
                f"{[round(p, 3) for p in props]} (spread={spread:.3f})"
            )
            if spread > 0.15:
                logger.warning(
                    f"[stratified_spatial_block] class {c} spread across folds ({spread:.3f}) "
                    f"exceeds 0.15 — block-level stratification may be too coarse for this class "
                    f"at the current spatial_block_size_m."
                )

    # ------------------------------------------------------------------
    # Step 2: Class Stratification (block -> fold assignment)
    # ------------------------------------------------------------------
    #
    # Exact K-fold class balance at the block level is a constrained
    # assignment problem — see the module docstring's "why this is bespoke
    # code" section — so this is a deliberate two-pass *greedy* heuristic,
    # not an optimal solver:
    #
    #   Pass 1 (_seed_every_fold_with_rarest_class_coverage) solves the
    #   one constraint that actually matters for the dissertation's
    #   results: every fold must see every class, including the rarest
    #   one, or that fold's AUC/PR-AUC for the rare class is undefined
    #   (see test_fit_and_score_full_survives_a_fold_that_lost_a_class_in_training
    #   for what happens downstream when a class is missing). A single
    #   greedy pass that just optimizes overall balance can concentrate a
    #   scarce class's few blocks into fewer than n_folds folds, because
    #   its "which fold needs this block most" tie-breaking is driven by
    #   whichever class happens to dominate that block — which is rarely
    #   the rare class itself. Seeding rarest-class-first, before any
    #   other assignment happens, guarantees coverage before anything else
    #   is optimized.
    #
    #   Pass 2 (_greedily_fill_remaining_blocks) is an ordinary greedy
    #   load-balancing fill for every block Pass 1 didn't already place.
    #   Exact (non-greedy) balancing across the *remaining* blocks was not
    #   worth the complexity here: block counts are small enough (tens to
    #   low hundreds) that a greedy fill's imbalance is within the
    #   spread<=0.15 tolerance _log_cross_fold_balance_summary checks for,
    #   and an exact solve would need to be re-run through the same
    #   heuristic reasoning as Pass 1 anyway to keep the rarest class
    #   protected — i.e. it would still be bespoke, just slower.
    @staticmethod
    def _stratify_blocks_into_folds(groups: pd.Series, y: pd.Series, n_folds: int) -> dict:
        """Orchestrates the two-pass block -> fold assignment described
        above. Returns {block_id: fold_index}, covering every block
        exactly once."""
        block_class_counts = StratifiedSpatialBlockCV._count_rows_per_block_and_class(groups, y)

        # Rarest-class-first ordering drives both passes: Pass 1 walks
        # classes in this order so scarce classes get first claim on fold
        # slots; Pass 2 uses the same order to pick each remaining block's
        # "dominant" class consistently with Pass 1's priorities.
        rarity_order = y.value_counts().sort_values().index.tolist()

        fold_of_block, fold_class_counts = StratifiedSpatialBlockCV._seed_every_fold_with_rarest_class_coverage(
            block_class_counts, rarity_order, n_folds,
        )
        StratifiedSpatialBlockCV._greedily_fill_remaining_blocks(
            block_class_counts, rarity_order, n_folds, fold_of_block, fold_class_counts,
        )
        return fold_of_block

    @staticmethod
    def _count_rows_per_block_and_class(groups: pd.Series, y: pd.Series) -> Dict[object, BlockClassCounts]:
        """{block_id: {class: row_count}} — the raw tally both passes below
        make their decisions from. A block with zero rows of a given class
        still gets an explicit 0 entry (not a missing key), so downstream
        lookups like `counts[c]` never need a `.get(c, 0)`."""
        classes = sorted(y.unique())
        block_class_counts: Dict[object, BlockClassCounts] = defaultdict(lambda: {c: 0 for c in classes})
        for block, cls in zip(groups, y):
            block_class_counts[block][cls] += 1
        return dict(block_class_counts)

    @staticmethod
    def _seed_every_fold_with_rarest_class_coverage(
        block_class_counts: Dict[object, BlockClassCounts], rarity_order: list, n_folds: int,
    ) -> Tuple[dict, List[BlockClassCounts]]:
        """Pass 1 (coverage guarantee): for each class, rarest first,
        round-robin its top-contributing blocks across every fold. This is
        what actually enforces "every fold has the rarest class
        represented" — see the Pass 1/Pass 2 design note above the caller.

        Returns (fold_of_block, fold_class_counts) covering only the
        blocks placed in this pass; `_greedily_fill_remaining_blocks`
        extends both in place for the blocks left over.
        """
        fold_of_block: dict = {}
        fold_class_counts = [{c: 0 for c in rarity_order} for _ in range(n_folds)]

        for c in rarity_order:
            # Only blocks that (a) actually carry class c and (b) haven't
            # already been claimed by an earlier, rarer class this pass.
            blocks_with_c = [
                b for b, counts in block_class_counts.items()
                if counts[c] > 0 and b not in fold_of_block
            ]
            # Largest-count-first: the blocks that carry the *most* of
            # class c are the ones whose placement most affects whether
            # every fold ends up with a meaningful share of it, so they're
            # placed while every fold still needs coverage.
            blocks_with_c.sort(key=lambda b: -block_class_counts[b][c])
            # One block per fold, in order — this is what "round-robin"
            # means here. Only the first n_folds blocks are seeded this
            # way; any further blocks carrying class c are left for Pass 2,
            # since coverage (>=1 block per fold) is already satisfied.
            for i, block in enumerate(blocks_with_c[:n_folds]):
                target_fold = i % n_folds
                fold_of_block[block] = target_fold
                for cc in rarity_order:
                    fold_class_counts[target_fold][cc] += block_class_counts[block][cc]

        return fold_of_block, fold_class_counts

    @staticmethod
    def _greedily_fill_remaining_blocks(
        block_class_counts: Dict[object, BlockClassCounts], rarity_order: list, n_folds: int,
        fold_of_block: dict, fold_class_counts: List[BlockClassCounts],
    ) -> None:
        """Pass 2 (general greedy fill): every block Pass 1 didn't already
        place gets assigned to whichever fold currently has the least of
        that block's own dominant class, tie-broken by which fold has
        fewest rows total so far (a plain load-balancing tie-break, not a
        class-aware one — Pass 1 already handled class-aware placement for
        the blocks that mattered most for coverage).

        Mutates `fold_of_block` and `fold_class_counts` in place, since
        this pass is inherently sequential: each placement changes the
        fold totals the next placement's greedy choice depends on.
        """
        remaining_blocks = [b for b in block_class_counts if b not in fold_of_block]

        def _priority(block):
            # Same "largest contribution first" idea as Pass 1: blocks
            # dominated by a rare class are placed before blocks that are
            # mostly common-class rows, so the greedy fill still leans
            # toward protecting rare-class balance even though Pass 1
            # already guaranteed baseline coverage.
            counts = block_class_counts[block]
            return tuple(-counts[c] for c in rarity_order)

        remaining_blocks.sort(key=_priority)

        for block in remaining_blocks:
            counts = block_class_counts[block]
            # This block's single largest-count class, rarest-class ties
            # broken toward the rarer class (rarity_order is walked in
            # rarest-first order, and min() keeps the first minimum it
            # sees) — i.e. a block split 50/50 between classes 0 and 2 is
            # treated as a class-2 block for placement purposes, since
            # protecting the rare class matters more than the common one.
            dominant_class = min(rarity_order, key=lambda c: -counts[c] if counts[c] else 0)
            target_fold = min(
                range(n_folds),
                key=lambda k: (fold_class_counts[k][dominant_class], sum(fold_class_counts[k].values())),
            )
            fold_of_block[block] = target_fold
            for c in rarity_order:
                fold_class_counts[target_fold][c] += counts[c]

    # ------------------------------------------------------------------
    # Steps 5-6: in-fold resampling, fitting, and scoring
    # ------------------------------------------------------------------

    def _fit_resampled_model(
        self, model_cls, params: dict, X: pd.DataFrame, y: pd.Series,
        train_idx: np.ndarray, context: str, model_name: Optional[str],
    ):
        """Step 5 (In-Fold Resampling): `self.resampler.resample` is called
        on `train_idx`-derived data ONLY. `test_idx` never appears in this
        method at all — that absence, not just a comment, is what makes
        the isolation guarantee checkable by reading the code: there is no
        variable here `self.resampler` could reach that traces back to the
        held-out fold.

        `model_name`, when given, resolves modeling.imbalance_strategy for
        this model (see modeling/imbalance.py) — a balanced sample_weight
        is computed from y_tr (post-resample) and passed to model.fit()
        when the resolved strategy is 'cost_weighted'. model_name=None
        (e.g. ad-hoc/test callers) skips weighting.

        Returns (fitted_model, cat_positions) — cat_positions is returned
        alongside the model because the caller needs the same categorical-
        feature encoding to build the validation array consistently.
        """
        y_tr_raw = y.iloc[train_idx]
        log_class_balance(logger, context, y_tr_raw, note="train, pre-resample", level=logging.DEBUG)
        warn_if_class_missing(
            logger, f"{context} (train, pre-resample)", y_tr_raw, expected_classes=sorted(y.unique()),
        )

        cat_positions = cat_features_of(model_cls)
        X_tr = to_model_array(X.iloc[train_idx], cat_positions)
        y_tr = y_tr_raw.values
        X_tr, y_tr = self.resampler.resample(X_tr, y_tr, context=context, model_name=model_name)
        log_class_balance(logger, context, y_tr, note="train, post-resample (used to fit)")

        sample_weight = self._imbalance.sample_weight_for(model_name, y_tr) if model_name else None
        model = model_cls(**params)
        model.fit(X_tr, y_tr, sample_weight=sample_weight)
        return model, cat_positions

    @staticmethod
    def _score_fold_metrics(y_va: np.ndarray, proba: np.ndarray, pred: np.ndarray, all_classes: list, context: str) -> dict:
        """AUC, F1-macro, PR-AUC-macro, and QWK for one fold's validation
        predictions. AUC and PR-AUC both need `proba` to have one column
        per class in `all_classes`; when spatial buffering strips a class
        out of a fold's *training* split entirely (its rows survive only
        in the untouched validation split — see
        test_fit_and_score_full_survives_a_fold_that_lost_a_class_in_training),
        `predict_proba` returns fewer columns than `len(all_classes)` and
        these calls raise. Rather than let one degenerate fold crash an
        entire Optuna trial spanning many folds, each is caught and logged
        as a warning, scoring that one metric as 0.0 for this fold only —
        the other metrics for the same fold are unaffected.
        """
        try:
            auc = float(roc_auc_score(y_va, proba, multi_class="ovr", labels=all_classes))
        except Exception as exc:
            logger.warning(f"{context}: AUC scoring failed ({exc}); scoring as 0.0")
            auc = 0.0

        f1_macro = float(f1_score(y_va, pred, average="macro"))

        try:
            y_va_bin = np.eye(proba.shape[1])[y_va.astype(int)]
            pr_auc_macro = float(average_precision_score(y_va_bin, proba, average="macro"))
        except Exception as exc:
            logger.warning(f"{context}: PR-AUC scoring failed ({exc}); scoring as 0.0")
            pr_auc_macro = 0.0

        qwk = float(cohen_kappa_score(y_va, pred, weights="quadratic"))

        return {"auc": auc, "f1_macro": f1_macro, "pr_auc_macro": pr_auc_macro, "qwk": qwk}

    def fit_and_score_full(
        self, model_cls, params, X, y, train_idx, test_idx, context: str = "",
        model_name: Optional[str] = None,
    ) -> dict:
        """Fit once, return the full metric set the dissertation reports
        (AUC, F1-macro, PR-AUC-macro, QWK). `fit_and_score` delegates here
        so a fold's model is only ever fit once even when both the scalar
        and the full metric set are needed (see trainer.py's optimism-gap
        logging)."""
        model, cat_positions = self._fit_resampled_model(
            model_cls, params, X, y, train_idx, context, model_name,
        )

        # Step 4 (Strict Validation Isolation): X_va/y_va are built
        # directly from test_idx and passed straight to predict_proba —
        # they never touch self.resampler.resample, which only ever saw
        # train_idx inside _fit_resampled_model above. The "_untouched"
        # suffix names that guarantee at the call site rather than leaving
        # it to be inferred from control flow.
        X_va_untouched = to_model_array(X.iloc[test_idx], cat_positions)
        y_va_untouched = y.iloc[test_idx].values
        log_class_balance(logger, context, y_va_untouched, note="validation, untouched — isolation check")

        proba = model.predict_proba(X_va_untouched)
        pred = np.argmax(proba, axis=1)

        all_classes = sorted(y.unique())
        metrics = self._score_fold_metrics(y_va_untouched, proba, pred, all_classes, context)

        logger.info(
            f"{context} AUC={metrics['auc']:.4f} F1_macro={metrics['f1_macro']:.4f} "
            f"PR_AUC_macro={metrics['pr_auc_macro']:.4f} QWK={metrics['qwk']:.4f} "
            f"(validation fold untouched/imbalanced)"
        )
        return metrics

    def fit_and_score(
        self, model_cls, params, X, y, train_idx, test_idx, context: str = "",
        model_name: Optional[str] = None,
    ) -> float:
        return self.fit_and_score_full(
            model_cls, params, X, y, train_idx, test_idx, context=context, model_name=model_name,
        )["auc"]
