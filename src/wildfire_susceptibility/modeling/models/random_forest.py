"""Random Forest wrapper. Under imbalance_strategy="native_balanced", swaps
sklearn's RandomForestClassifier for imbalanced-learn's
BalancedRandomForestClassifier — plain sample_weight only reweights the
split criterion, not which rows get bootstrapped per tree, so it can't fix
severe imbalance on its own.
"""
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
import numpy as np

from ...core.registry import MODELS


@MODELS.register("random_forest")
class RandomForestModel:
    def __init__(self, **kwargs):
        self.params = kwargs
        self.model: RandomForestClassifier | BalancedRandomForestClassifier | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "RandomForestModel":
        params = dict(self.params)
        # native_balanced is bound onto model_cls via functools.partial in
        # trainer.py (same mechanism as cat_features), not an Optuna-tuned
        # hyperparameter — pop it before it reaches either sklearn
        # constructor, neither of which accepts it directly.
        native_balanced = params.pop("native_balanced", False)

        if native_balanced:
            # imbalance_strategy="native_balanced" (modeling/imbalance.py):
            # sklearn's RandomForestClassifier sample_weight only reweights
            # the split criterion and leaf vote, NOT which rows get
            # bootstrapped per tree — with a class this rare, it can be
            # statistically thinned to a handful of examples per tree
            # before weighting ever gets a say. BalancedRandomForestClassifier
            # fixes this by having each tree bootstrap a class-balanced
            # sample directly. ImbalanceStrategy.sample_weight_for() never
            # returns a weight for 'native_balanced', so sample_weight is
            # expected to be None here — no class_weight/sample_weight
            # double-correction risk to guard against.
            self.model = BalancedRandomForestClassifier(
                random_state=42,
                n_jobs=-1,
                criterion="gini",
                bootstrap=True,
                replacement=True,
                sampling_strategy="all",
                **params,
            )
            self.model.fit(X, y)
            return self

        if sample_weight is not None:
            # An externally-computed sample_weight (cost_weighted imbalance
            # strategy) takes over class balancing — sklearn multiplies
            # class_weight-derived weights by sample_weight elementwise, so
            # leaving class_weight="balanced" here (Optuna's HPO choice, see
            # param_space below) would silently compound the two.
            params["class_weight"] = None
        self.model = RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            criterion="gini",
            **params
        )
        self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def param_space(self, trial) -> dict:
        # class_weight is deliberately NOT tunable here: imbalance handling
        # is a resolver-level config choice (modeling.imbalance_strategy),
        # not something Optuna should pick. Leaving it in the search space
        # meant a trial could sample class_weight="balanced" while
        # imbalance_strategy resolved to "smote" for this model, stacking
        # class weighting on top of SMOTE-resampled training data with
        # nothing preventing it (the fit()-level guard below only fires for
        # the "cost_weighted" sample_weight path).
        # Range tightened 08/16/2026: deployed models were consistently
        # landing at/near the old max_depth=25 ceiling and min_samples_leaf=1
        # floor (e.g. depth=23/leaf=2), with standard-CV AUC ~0.99 collapsing
        # to ~0.5-0.55 on spatial CV and true validation — a severe
        # overfitting signature the old range let Optuna reach even though
        # search is spatial-CV-scored. max_samples added as a new tunable
        # (bootstrap row-subsample fraction, default was unset ->
        # RandomForestClassifier's full-bootstrap default) since row
        # subsampling decorrelates trees more effectively than depth/leaf
        # constraints alone on spatially autocorrelated features.
        #
        # Range re-widened 08/17/2026 (max_depth 12->20, min_samples_leaf
        # floor 5->2, min_samples_split floor 10->5): the 08/16 tightening
        # was diagnosed under labels.classify_method="jenks"/n_classes=4,
        # where the rarest class was ~0.1-0.2% of the data — exactly the
        # kind of minority a deep tree can exploit by carving one leaf
        # around a handful of memorized points. Under
        # classify_method="gmm"/n_classes=3 (configs/experiment/
        # baseline.yaml), the rarest class is ~9.8% of training data, and
        # the standard-vs-spatial CV optimism gap that motivated the
        # tightening has independently collapsed from ~0.35 to ~0.02-0.03
        # PR-AUC-macro at the OLD tight bounds — evidence the specific
        # overfitting exploit may no longer be reachable, not proof the
        # current bounds are still the right ceiling. This widening is the
        # deliberate empirical test of that: rerun and check whether the
        # optimism gap stays low as depth/leaf constraints loosen (bounds
        # were fine, more capacity was being left on the table) or reopens
        # (the exploit is still there, just needs deeper trees to reach
        # under the new label balance too). max_features/max_samples left
        # untouched to isolate the depth/leaf effect from other knobs.
        # See tests/unit/modeling/models/test_model_contracts.py::
        # test_random_forest_search_space_widened_for_deeper_learning for
        # the regression pin (supersedes the old tightened-bounds pin).
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 5, 40),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "max_samples": trial.suggest_float("max_samples", 0.3, 0.8),
        }

    def needs_scaling(self) -> bool:
        return False

    def native_categorical_support(self) -> bool:
        return False