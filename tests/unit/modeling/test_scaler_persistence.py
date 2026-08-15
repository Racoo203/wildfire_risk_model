"""Regression coverage for the ordinal_lr/neural_net scaler-persistence fix
(fix-ordinal-lr-scaler-persistence).

Before this fix, trainer.py fit a StandardScaler once at training time
(dataset_prep.py::scale_for_model_family) and immediately discarded it
(`X_tr, X_va, _ = ...`). stage_evaluate.py reloads test/full-domain data
fresh from CSV and never applied any scaler to it, so any scale-sensitive
model (needs_scaling() -> True: ordinal_lr, neural_net) was evaluated
against raw, unscaled features -- completely different from what it was fit
on. RF/CatBoost (needs_scaling() -> False) were unaffected.

apply_scaling mirrors the fit/apply split already established for
landuse_class categorical encoding (dataset_prep.py::fit_categorical_encoding
/ apply_categorical_encoding, see test_categorical_encoding.py): a scaler
fit on train data only (scale_for_model_family) must be replayed
transform-only against any other split, never re-fit.

See tests/integration/test_scaler_persistence_roundtrip.py for the
end-to-end stage_train -> stage_evaluate path (manifest persistence,
backward-compat failure on pre-fix artifacts).
"""
import numpy as np
import pandas as pd

from wildfire_susceptibility.modeling.dataset_prep import DatasetPrep


def _frame(n, seed, loc=0.0, scale=1.0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "elevation": rng.normal(loc, scale, size=n),
        "d_roads": rng.normal(loc, scale, size=n),
    })


class TestScaleForModelFamily:
    def test_needs_scaling_false_returns_no_scaler(self):
        prep = DatasetPrep(config={})
        X_train, X_test = _frame(50, seed=1), _frame(20, seed=2)

        X_tr, X_va, scaler = prep.scale_for_model_family(X_train, X_test, needs_scaling=False)

        assert scaler is None
        pd.testing.assert_frame_equal(X_tr, X_train)
        pd.testing.assert_frame_equal(X_va, X_test)


class TestApplyScaling:
    def test_matches_manual_scaler_transform(self):
        prep = DatasetPrep(config={})
        X_train, X_test = _frame(100, seed=1), _frame(30, seed=2)

        _, _, scaler = prep.scale_for_model_family(X_train, X_test, needs_scaling=True)
        X_test_scaled = prep.apply_scaling(X_test, scaler)

        expected = pd.DataFrame(
            scaler.transform(X_test), columns=X_test.columns, index=X_test.index
        )
        pd.testing.assert_frame_equal(X_test_scaled, expected)

    def test_transform_only_uses_train_statistics_not_test(self):
        """The actual regression this fix targets: apply_scaling must use
        the persisted train-fit scaler's mean/std, not silently re-fit on
        whatever data it's given -- a shifted-distribution test/full-domain
        split must come out shifted after scaling, not re-centered to 0."""
        prep = DatasetPrep(config={})
        X_train = _frame(200, seed=1, loc=0.0, scale=1.0)
        X_test = _frame(50, seed=2, loc=50.0, scale=1.0)  # deliberately shifted

        _, _, scaler = prep.scale_for_model_family(X_train, X_test, needs_scaling=True)
        X_test_scaled = prep.apply_scaling(X_test, scaler)

        # If apply_scaling had re-fit on X_test, the scaled mean would land
        # near 0. Correctly transform-only (using train's fit mean ~0,
        # std ~1), it stays far from 0.
        assert X_test_scaled["elevation"].mean() > 20

    def test_preserves_columns_and_index(self):
        prep = DatasetPrep(config={})
        X_train, X_test = _frame(100, seed=1), _frame(10, seed=2)
        X_test.index = np.arange(1000, 1010)

        _, _, scaler = prep.scale_for_model_family(X_train, X_test, needs_scaling=True)
        X_test_scaled = prep.apply_scaling(X_test, scaler)

        assert list(X_test_scaled.columns) == list(X_test.columns)
        assert list(X_test_scaled.index) == list(X_test.index)
