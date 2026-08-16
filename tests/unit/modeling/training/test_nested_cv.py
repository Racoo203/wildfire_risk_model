# tests/unit/modeling/training/test_nested_cv.py
"""Coverage for nested_cv.run_nested_cv (branch
refactor-nested-cv-optuna-schratz-method): genuinely nested CV, where the
OUTER loop (whatever CVStrategy is configured) provides train/test splits
used only for performance estimation, and the INNER loop (always
StandardKFoldCV, built fresh from each outer fold's training data only)
tunes hyperparameters without ever seeing that outer fold's held-out test
rows.

The single most important correctness property this module adds is the
leakage boundary: no row from an outer test fold may ever appear in any
inner fold used to tune the model that gets scored against that same
outer test fold. That gets its own explicit test below, using a
"row_id" identity column to track rows through subsampling/reset_index
rather than relying on positional indices staying comparable across the
outer/inner boundary."""

import numpy as np
import pandas as pd
import pytest


N_ROWS_PER_CLASS = 20
N_CLASSES = 4
N_OUTER_FOLDS = 3


@pytest.fixture(autouse=True)
def _isolated_optuna_storage(tmp_path, monkeypatch):
    """Every test in this file drives real HyperparamSearch/Optuna study
    creation (one study per outer fold) — without this, they'd write into
    the real data/silver/dbs/optuna_studies.db, same isolation pattern
    test_search_objective.py's study-naming test already uses."""
    db_path = tmp_path / "optuna_test.db"
    monkeypatch.setattr(
        "wildfire_susceptibility.modeling.training.search.OPTUNA_STORAGE",
        f"sqlite:///{db_path}",
    )


@pytest.fixture
def tiny_dataset():
    rng = np.random.default_rng(0)
    rows, labels = [], []
    for cls in range(N_CLASSES):
        for _ in range(N_ROWS_PER_CLASS):
            rows.append([cls * 10.0 + rng.normal(0, 0.5), rng.normal(0, 0.5)])
            labels.append(cls)
    X = pd.DataFrame(rows, columns=["f1", "f2"])
    X["row_id"] = np.arange(len(X))  # unique identity, survives .iloc/.reset_index
    y = pd.Series(labels)
    return X, y


def _model_cls():
    from wildfire_susceptibility.core.registry import MODELS
    from wildfire_susceptibility.modeling import models  # noqa: F401 — registers model wrappers
    return MODELS["random_forest"]


def test_outer_test_fold_never_leaks_into_inner_tuning_folds(tiny_dataset, fast_modeling_config, monkeypatch):
    """For every outer fold, the set of rows ever visible to the inner
    tuning search (HyperparamSearch.prepare_search_data's population,
    before any further subsampling) must be disjoint from that same outer
    fold's held-out test rows (the ones outer_strategy.fit_and_score_full
    is refit-and-scored on)."""
    from wildfire_susceptibility.modeling.cv.factory import get_cv_strategy
    from wildfire_susceptibility.modeling.resampling import SMOTEResampler
    from wildfire_susceptibility.modeling.training import nested_cv
    from wildfire_susceptibility.modeling.training.search import HyperparamSearch

    X, y = tiny_dataset
    config = fast_modeling_config
    config["modeling"]["cv_folds"] = N_OUTER_FOLDS
    config["modeling"]["cv_strategy"] = "standard"

    resampler = SMOTEResampler(config)
    outer_strategy = get_cv_strategy("standard", config, resampler)

    inner_populations = []
    original_prepare = HyperparamSearch.prepare_search_data

    def _spy_prepare(self, X_tr, *args, **kwargs):
        inner_populations.append(set(X_tr["row_id"]))
        return original_prepare(self, X_tr, *args, **kwargs)

    monkeypatch.setattr(HyperparamSearch, "prepare_search_data", _spy_prepare)

    outer_test_sets = []
    original_fit_score = outer_strategy.fit_and_score_full

    def _spy_fit_score(model_cls, params, X, y_arg, train_idx, test_idx, **kwargs):
        outer_test_sets.append(set(X.iloc[test_idx]["row_id"]))
        return original_fit_score(model_cls, params, X, y_arg, train_idx, test_idx, **kwargs)

    monkeypatch.setattr(outer_strategy, "fit_and_score_full", _spy_fit_score)

    model_cls = _model_cls()
    nested_cv.run_nested_cv(
        outer_strategy, resampler, model_cls, "random_forest", "test_season", config,
        X, y, None,
    )

    assert len(inner_populations) == N_OUTER_FOLDS
    assert len(outer_test_sets) == N_OUTER_FOLDS
    for i, (inner_rows, outer_test_rows) in enumerate(zip(inner_populations, outer_test_sets)):
        leaked = inner_rows & outer_test_rows
        assert not leaked, (
            f"outer fold {i}: {len(leaked)} row(s) leaked from the outer test "
            f"fold into the inner tuning population: {leaked}"
        )


def test_inner_loop_never_uses_block_aware_subsampling_regardless_of_outer_strategy(
    tiny_dataset, fast_modeling_config, monkeypatch,
):
    """The inner loop is always StandardKFoldCV (requires_spatial_groups is
    False for it), so HyperparamSearch.prepare_search_data must always take
    the plain _subsample_rows path for inner searches — _subsample_blocks
    (block-aware search subsampling) is only relevant to an OUTER strategy
    building its own folds with groups, never to the inner tuning loop,
    regardless of what the outer strategy is."""
    from wildfire_susceptibility.modeling.cv.factory import get_cv_strategy
    from wildfire_susceptibility.modeling.resampling import SMOTEResampler
    from wildfire_susceptibility.modeling.training import nested_cv
    from wildfire_susceptibility.modeling.training.search import HyperparamSearch

    X, y = tiny_dataset
    groups = pd.Series([f"blk_{i % 6}" for i in range(len(X))])

    config = fast_modeling_config
    config["modeling"]["cv_folds"] = N_OUTER_FOLDS
    config["modeling"]["cv_strategy"] = "stratified_spatial_block"
    config["modeling"].setdefault("spatial_block_size_m", 5000.0)

    resampler = SMOTEResampler(config)
    outer_strategy = get_cv_strategy("stratified_spatial_block", config, resampler)

    block_subsample_calls = []
    original_subsample_blocks = HyperparamSearch._subsample_blocks

    def _spy_subsample_blocks(self, *args, **kwargs):
        block_subsample_calls.append(True)
        return original_subsample_blocks(self, *args, **kwargs)

    monkeypatch.setattr(HyperparamSearch, "_subsample_blocks", _spy_subsample_blocks)

    model_cls = _model_cls()
    nested_cv.run_nested_cv(
        outer_strategy, resampler, model_cls, "random_forest", "test_season", config,
        X, y, groups,
    )

    assert block_subsample_calls == [], (
        "the inner tuning loop must never take the block-aware subsampling "
        "path, even when the outer strategy itself requires spatial groups"
    )


def test_study_names_carry_outer_fold_identity_and_inner_standard_marker(
    tiny_dataset, fast_modeling_config, monkeypatch,
):
    """Each outer fold's inner search must get its own, distinctly-named
    Optuna study — encoding which outer fold and which outer strategy it
    belongs to, plus an explicit inner-standard marker — so it can never
    collide with (or accidentally resume via load_if_exists=True) the
    unrelated production study, another outer fold's study, or (under
    cv_strategy="both") the other side's nested studies."""
    from wildfire_susceptibility.modeling.cv.factory import get_cv_strategy
    from wildfire_susceptibility.modeling.resampling import SMOTEResampler
    from wildfire_susceptibility.modeling.training import nested_cv
    from wildfire_susceptibility.modeling.training.search import HyperparamSearch

    X, y = tiny_dataset
    config = fast_modeling_config
    config["modeling"]["cv_folds"] = N_OUTER_FOLDS
    config["modeling"]["cv_strategy"] = "standard"

    resampler = SMOTEResampler(config)
    outer_strategy = get_cv_strategy("standard", config, resampler)

    seasons_seen = []
    original_get_or_create = HyperparamSearch.get_or_create_study

    def _spy_get_or_create(self, season, model_name):
        seasons_seen.append(season)
        return original_get_or_create(self, season, model_name)

    monkeypatch.setattr(HyperparamSearch, "get_or_create_study", _spy_get_or_create)

    model_cls = _model_cls()
    nested_cv.run_nested_cv(
        outer_strategy, resampler, model_cls, "random_forest", "test_season", config,
        X, y, None,
    )

    assert len(seasons_seen) == N_OUTER_FOLDS
    assert len(set(seasons_seen)) == N_OUTER_FOLDS, "each outer fold must get a distinct study identity"
    for i, season_label in enumerate(seasons_seen):
        assert f"outerfold{i}of{N_OUTER_FOLDS}" in season_label
        assert "outer-standard" in season_label
        assert "inner-standard" in season_label
        assert season_label != "test_season", "must not collide with the plain production-search study name"


def test_inner_config_overrides_cv_folds_without_mutating_original(fast_modeling_config):
    """_inner_config must read optuna_inner_cv_folds into the inner
    strategy's cv_folds without mutating the caller's config dict (which
    is also used, unmodified, for the outer strategy and the production
    search)."""
    from wildfire_susceptibility.modeling.training.nested_cv import _inner_config

    config = fast_modeling_config
    config["modeling"]["cv_folds"] = 5
    config["modeling"]["optuna_inner_cv_folds"] = 2

    inner_cfg = _inner_config(config)

    assert inner_cfg["modeling"]["cv_folds"] == 2
    assert config["modeling"]["cv_folds"] == 5, "original config's outer cv_folds must be untouched"
