"""Regression coverage for the landuse_class categorical-encoding fix.

Before this fix, landuse_class (features/proximity.py::_write_landuse_class
— an arbitrary 1..N position in the configured human_fclasses list, 0 = no
human-activity land use) reached every model as a plain numeric column, with
no model ever declaring it categorical. Confirmed at runtime against an
actual trained model: CatBoost's own split conditions were ordinal
thresholds ("landuse_class > 3.5") rather than category-set membership, and
`cb.get_cat_feature_indices()` was empty.

The encoding logic itself is a fit/apply split (fit_categorical_encoding on
train data only, apply_categorical_encoding against any other split using
the persisted encoding_meta) rather than one fused train+test function, so
stage_evaluate.py can reload a persisted encoding_meta from manifest.json
and apply it to test/full-domain data reloaded fresh from CSV — see
tests/integration/test_stage_train_evaluate_categorical_roundtrip.py for
that end-to-end path.

These tests cover:
  (i)   CatBoost receives landuse_class via cat_features, not as a raw
        numeric split target.
  (ii)  RF/ordinal-LR/NN receive a one-hot-expanded version with the
        correct number of dummy columns and no accidental ordinal leakage
        (0 preserved as its own distinct category, not collapsed).
  (iii) The DataFrame -> model-input-array conversion (modeling/categorical.py)
        preserves the categorical column's int dtype only when a model
        actually declared cat_features, and is a no-op (identical to the
        pre-fix `.values` behavior) otherwise.
  (iv)  apply_categorical_encoding, given a persisted encoding_meta, fills
        an all-zero dummy for a persisted category absent from the new
        split, and raises loudly (rather than silently misaligning
        columns) if the new split contains a category never seen at fit
        time.
"""
import functools

import numpy as np
import pandas as pd
import pytest

from wildfire_susceptibility import modeling  # noqa: F401
from wildfire_susceptibility.modeling import models as _models  # noqa: F401 — registers wrappers
from wildfire_susceptibility.core.registry import MODELS
from wildfire_susceptibility.modeling.categorical import cat_features_of, to_model_array
from wildfire_susceptibility.modeling.dataset_prep import DatasetPrep


def _synthetic_frame(n=120, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "elevation": rng.normal(size=n),
        "d_roads": rng.normal(size=n),
        # 0 = no human activity, 1/2/5 = arbitrary human_fclasses config
        # positions (mirrors the real data's non-contiguous codes, e.g.
        # fall training data has {0,1,2,5,6,7}).
        "landuse_class": rng.choice([0, 1, 2, 5], size=n).astype("float64"),
    })
    return df


class TestFitCategoricalEncoding:
    def test_catboost_keeps_single_int_column_and_reports_it(self):
        prep = DatasetPrep(config={})
        X = _synthetic_frame()
        X_tr, encoding_meta = prep.fit_categorical_encoding(X, MODELS["catboost"])

        assert encoding_meta["kind"] == "native"
        assert encoding_meta["columns"] == ["landuse_class"]
        assert list(X_tr.columns) == list(X.columns), "CatBoost path must not add/remove columns"
        assert X_tr["landuse_class"].dtype == np.int64 or pd.api.types.is_integer_dtype(X_tr["landuse_class"])
        # 0 ("no human activity") must survive as a real value, not become NaN.
        assert set(X_tr["landuse_class"].unique()) == {0, 1, 2, 5}
        assert encoding_meta["categories"]["landuse_class"] == [0, 1, 2, 5]

    @pytest.mark.parametrize("model_name", ["random_forest", "ordinal_lr", "neural_net"])
    def test_non_native_models_get_one_hot_expansion_with_no_raw_column(self, model_name):
        prep = DatasetPrep(config={})
        X = _synthetic_frame()
        X_tr, encoding_meta = prep.fit_categorical_encoding(X, MODELS[model_name])

        assert encoding_meta["kind"] == "onehot"
        assert "landuse_class" not in X_tr.columns, (
            f"{model_name}: raw landuse_class column leaked through — must be one-hot encoded, "
            f"never passed as a raw numeric/ordinal column."
        )
        dummy_cols = [c for c in X_tr.columns if c.startswith("landuse_class_")]
        # 4 distinct train-side categories observed: {0, 1, 2, 5}.
        assert len(dummy_cols) == 4
        # 0 ("no human activity") gets its own distinct dummy, not folded
        # into class 1 or dropped.
        assert "landuse_class_0" in dummy_cols
        # Exactly one dummy is hot per row (a valid one-hot encoding).
        assert (X_tr[dummy_cols].sum(axis=1) == 1).all()
        assert encoding_meta["categories"]["landuse_class"] == [0, 1, 2, 5]

    def test_absent_categorical_column_is_a_no_op(self):
        """Configs without landuse_class in feature_cols (e.g. excluded)
        must pass through completely unchanged for every model family."""
        prep = DatasetPrep(config={})
        X = _synthetic_frame().drop(columns=["landuse_class"])
        X_tr, encoding_meta = prep.fit_categorical_encoding(X, MODELS["catboost"])

        assert encoding_meta["kind"] == "none"
        pd.testing.assert_frame_equal(X_tr, X)


class TestApplyCategoricalEncoding:
    @pytest.mark.parametrize("model_name", ["random_forest", "ordinal_lr", "neural_net"])
    def test_one_hot_train_apply_columns_match(self, model_name):
        """apply_categorical_encoding, given the persisted encoding_meta,
        must reproduce the exact same dummy-column layout fit_categorical_encoding
        produced on train — the contract stage_evaluate.py depends on when
        it reloads test/full data fresh from CSV."""
        prep = DatasetPrep(config={})
        X_train = _synthetic_frame(seed=1)
        X_test = _synthetic_frame(seed=2)

        X_tr, encoding_meta = prep.fit_categorical_encoding(X_train, MODELS[model_name])
        X_te = prep.apply_categorical_encoding(X_test, encoding_meta)

        assert list(X_tr.columns) == list(X_te.columns)

    @pytest.mark.parametrize("model_name", ["random_forest", "ordinal_lr", "neural_net"])
    def test_persisted_category_absent_from_split_still_gets_a_zero_dummy(self, model_name):
        """A category seen at fit time but absent from this particular
        split (e.g. a small test set that happens to have zero rows of
        landuse_class=5) must still get its own all-zero dummy column, not
        silently disappear."""
        prep = DatasetPrep(config={})
        X_train = _synthetic_frame(seed=1)
        X_test = _synthetic_frame(seed=2)
        X_test["landuse_class"] = X_test["landuse_class"].replace(5, 2)  # drop category 5 from this split

        _, encoding_meta = prep.fit_categorical_encoding(X_train, MODELS[model_name])
        X_te = prep.apply_categorical_encoding(X_test, encoding_meta)

        assert "landuse_class_5" in X_te.columns
        assert (X_te["landuse_class_5"] == 0).all()

    @pytest.mark.parametrize("model_name", ["random_forest", "ordinal_lr", "neural_net", "catboost"])
    def test_unseen_category_at_apply_time_raises_instead_of_silently_misaligning(self, model_name):
        """The actual regression this fit/apply split targets: a category
        value that never appeared in training must raise loudly at
        evaluate time, not silently produce a mismatched/all-zero encoding
        or feed CatBoost a category it has no learned statistics for."""
        prep = DatasetPrep(config={})
        X_train = _synthetic_frame(seed=1)
        X_test = _synthetic_frame(seed=2)
        X_test.loc[X_test.index[0], "landuse_class"] = 99.0  # never seen in X_train

        _, encoding_meta = prep.fit_categorical_encoding(X_train, MODELS[model_name])

        with pytest.raises(ValueError, match="99"):
            prep.apply_categorical_encoding(X_test, encoding_meta)

    def test_absent_categorical_column_is_a_no_op(self):
        prep = DatasetPrep(config={})
        X = _synthetic_frame().drop(columns=["landuse_class"])
        encoding_meta = {"kind": "none", "columns": [], "categories": {}}
        X_out = prep.apply_categorical_encoding(X, encoding_meta)
        pd.testing.assert_frame_equal(X_out, X)


class TestCatFeaturesOf:
    def test_from_partial_bound_model_class(self):
        bound = functools.partial(MODELS["catboost"], cat_features=[2])
        assert cat_features_of(bound) == [2]

    def test_from_fitted_model_instance(self):
        model = MODELS["catboost"](cat_features=[2], iterations=5)
        assert cat_features_of(model) == [2]

    def test_empty_for_models_never_bound_with_cat_features(self):
        assert cat_features_of(MODELS["random_forest"]) == []
        assert cat_features_of(MODELS["random_forest"]()) == []


class TestToModelArray:
    def test_no_cat_positions_matches_plain_values(self):
        X = _synthetic_frame()
        arr = to_model_array(X, [])
        np.testing.assert_array_equal(arr, X.values)
        assert arr.dtype == np.float64

    def test_cat_positions_preserve_int_values_in_object_array(self):
        X = _synthetic_frame()
        X["landuse_class"] = X["landuse_class"].astype(int)
        pos = X.columns.get_loc("landuse_class")

        arr = to_model_array(X, [pos])
        assert arr.dtype == object
        assert all(isinstance(v, (int, np.integer)) for v in arr[:, pos])
        np.testing.assert_array_equal(arr[:, pos].astype(int), X["landuse_class"].to_numpy())
        # Non-categorical columns are untouched numerically.
        np.testing.assert_allclose(arr[:, 0].astype(float), X["elevation"].to_numpy())


class TestCatBoostEndToEndSyntheticExample:
    def test_fit_with_cat_features_produces_categorical_splits_not_ordinal_thresholds(self):
        """The actual regression this fix targets: before the fix,
        CatBoost split on landuse_class as a float threshold
        ("landuse_class > 2.5"); after the fix, cat_features must be
        registered on the fitted model and the raw int values must have
        reached CatBoost unmodified (no ordinal/one-hot encoding applied)."""
        rng = np.random.default_rng(42)
        n = 300
        X = pd.DataFrame({
            "elevation": rng.normal(size=n),
            "d_roads": rng.normal(size=n),
            "landuse_class": rng.choice([0, 1, 2, 5], size=n).astype("float64"),
        })
        # Make landuse_class actually predictive so the trained model has
        # a reason to split on it (mirrors real signal, not required for
        # the dtype/wiring assertions but keeps the test meaningful).
        y = (X["landuse_class"].isin([1, 2])).astype(int).to_numpy()
        y[: n // 4] = rng.integers(0, 2, size=n // 4)  # add noise so it's not trivially separable

        prep = DatasetPrep(config={})
        X_tr, encoding_meta = prep.fit_categorical_encoding(X, MODELS["catboost"])
        cat_positions = [X_tr.columns.get_loc(c) for c in encoding_meta["columns"]]
        bound_cls = functools.partial(MODELS["catboost"], cat_features=cat_positions)

        model = bound_cls(iterations=20, depth=3)
        model.fit(to_model_array(X_tr, cat_positions), y)

        assert model.model.get_cat_feature_indices() == cat_positions
        # predict_proba must also work post-fit through the same conversion
        # path (this is the call CatBoost previously could not validate
        # against a plain float ndarray with cat_features set).
        proba = model.predict_proba(to_model_array(X_tr, cat_positions))
        assert proba.shape == (n, 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-3)
