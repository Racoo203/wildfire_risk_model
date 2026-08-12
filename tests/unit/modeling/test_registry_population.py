"""Guards against the exact failure mode described in
tests/integration/test_train_smoke.py: a model class existing on disk but
never actually landing in MODELS because its module isn't imported by
modeling/models/__init__.py (the registration is an import side effect,
not automatic discovery)."""

import pytest

from wildfire_susceptibility.core.registry import MODELS
from wildfire_susceptibility import modeling  # noqa: F401
from wildfire_susceptibility.modeling import models as _models  # noqa: F401 — registers wrappers
from wildfire_susceptibility.config.schema import WildfireConfig


def test_all_expected_models_are_registered():
    expected = {"random_forest", "svm", "xgboost", "neural_net", "catboost", "ordinal_lr"}
    assert expected <= set(MODELS.keys())


@pytest.mark.parametrize("model_name,expected_cls_name", [
    ("catboost", "CatBoostModel"),
    ("ordinal_lr", "OrdinalLogisticModel"),
])
def test_config_driven_model_selection_instantiates_new_models(model_name, expected_cls_name):
    """Confirm a config listing the new model name round-trips through the
    pydantic schema (Literal accepts it) and resolves, via the registry, to
    the correct wrapper class — the same path stage_train.py uses."""
    cfg = WildfireConfig(modeling={"models": [model_name]})
    assert cfg.modeling.models == [model_name]

    model_cls = MODELS[model_name]
    assert model_cls.__name__ == expected_cls_name

    instance = model_cls()
    assert hasattr(instance, "fit")
    assert hasattr(instance, "predict_proba")


def test_default_roster_matches_finalized_decision():
    """Finalized roster: ordinal LR replaces SVC, CatBoost replaces XGBoost.
    svm/xgboost stay registered (reversible) but drop out of the default."""
    cfg = WildfireConfig()
    assert cfg.modeling.models == ["random_forest", "catboost", "ordinal_lr", "neural_net"]
