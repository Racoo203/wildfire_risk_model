# tests/unit/modeling/training/test_optuna_study_invalidation.py
"""Confirms the OBJECTIVE_VERSION bump for the landuse_class categorical-
encoding fix actually invalidates the 12 pre-fix studies (spring/summer/
fall x random_forest/catboost/ordinal_lr/neural_net) currently persisted in
data/silver/dbs/optuna_studies.db: the new study name for every
(season, model) pair must not already exist in that database, so the next
run does real re-search instead of resuming old, encoding-broken trials.

Runs against a tmp_path *copy* of the real production DB (never mutates the
committed file) so it reflects whatever is actually on disk right now,
rather than a hardcoded snapshot that would go stale."""

import shutil
from pathlib import Path

import optuna
import pytest

PROD_DB = Path("data/silver/dbs/optuna_studies.db")


def test_new_objective_version_study_names_absent_from_existing_db(
    tmp_path, monkeypatch, fast_modeling_config
):
    if not PROD_DB.exists():
        pytest.skip(f"no production optuna DB at {PROD_DB} in this environment")

    from wildfire_susceptibility.modeling.training.search import HyperparamSearch, OBJECTIVE_VERSION
    from wildfire_susceptibility.modeling.cv import get_cv_strategy
    from wildfire_susceptibility.modeling.resampling import SMOTEResampler
    from wildfire_susceptibility.core.registry import MODELS
    from wildfire_susceptibility.modeling import models  # noqa: F401 — registers all model classes

    db_copy = tmp_path / "optuna_studies_snapshot.db"
    shutil.copy(PROD_DB, db_copy)
    storage_url = f"sqlite:///{db_copy}"
    monkeypatch.setattr(
        "wildfire_susceptibility.modeling.training.search.OPTUNA_STORAGE", storage_url
    )

    storage = optuna.storages.RDBStorage(url=storage_url)
    existing_names = {s.study_name for s in optuna.study.get_all_study_summaries(storage)}

    # Sanity check the snapshot is actually the pre-fix DB we think it is —
    # otherwise this test would trivially pass for the wrong reason.
    assert any(name.endswith("_objv2-prauc") for name in existing_names), (
        "expected the production DB to contain old objv2-prauc studies; "
        "if this fails, the DB has already been migrated and this test's "
        "premise is stale."
    )

    config = fast_modeling_config
    config["modeling"]["cv_strategy"] = "standard"
    resampler = SMOTEResampler(config)
    cv_strategy = get_cv_strategy("standard", config, resampler)
    search = HyperparamSearch(config, cv_strategy)

    season = "spring"
    for model_name in ["random_forest", "catboost", "ordinal_lr", "neural_net"]:
        expected_sig = HyperparamSearch._param_space_signature(MODELS[model_name])
        expected_name = f"{season}_{model_name}_{expected_sig}_{OBJECTIVE_VERSION}"

        assert expected_name not in existing_names, (
            f"{expected_name!r} already exists in the pre-fix DB — the "
            f"OBJECTIVE_VERSION bump did not produce a fresh study name for "
            f"({season}, {model_name})."
        )

        study = search.get_or_create_study(season, model_name)
        assert study.study_name == expected_name
        assert study.trials == [], (
            "a newly-invalidated study must start empty, not resume trials "
            "from a differently-named old study."
        )

    # Old studies must survive untouched under their own names — invalidation
    # is "orphan, don't delete."
    post_names = {s.study_name for s in optuna.study.get_all_study_summaries(storage)}
    assert existing_names <= post_names, (
        "pre-existing objv2 studies must still be present after creating "
        "the new objv3 studies alongside them."
    )
