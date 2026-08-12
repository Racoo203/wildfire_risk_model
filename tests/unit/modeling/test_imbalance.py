# tests/unit/modeling/test_imbalance.py
"""Coverage for ImbalanceStrategy (modeling/imbalance.py): the resolver
that decides, per model, whether SMOTE or cost-weighted sample_weight
applies — and guarantees the two are mutually exclusive (see
resampling.py's model_name gating and cv/base.py's fit_and_score_full)."""

import numpy as np
import pytest

from wildfire_susceptibility.modeling.imbalance import ImbalanceStrategy


def test_default_strategy_is_smote_when_unset(minimal_modeling_config):
    """Legacy back-compat: existing configs that never mention
    imbalance_strategy must resolve to 'smote' for every model, so
    use_smote/search_resample_target_size keep governing behavior exactly
    as before this feature existed."""
    strategy = ImbalanceStrategy(minimal_modeling_config)
    assert strategy.resolve("random_forest") == "smote"
    assert strategy.smote_allowed("random_forest") is True
    assert strategy.sample_weight_for("random_forest", np.array([0, 1, 2, 3])) is None


def test_per_model_override_wins_over_default(minimal_modeling_config):
    config = minimal_modeling_config
    config["modeling"]["imbalance_strategy"] = "smote"
    config["modeling"]["imbalance_strategy_by_model"] = {"catboost": "cost_weighted"}
    strategy = ImbalanceStrategy(config)

    assert strategy.resolve("catboost") == "cost_weighted"
    assert strategy.resolve("random_forest") == "smote"  # falls back to default
    assert strategy.smote_allowed("catboost") is False
    assert strategy.smote_allowed("random_forest") is True


def test_cost_weighted_produces_balanced_inverse_frequency_weights(minimal_modeling_config):
    from sklearn.utils.class_weight import compute_sample_weight

    config = minimal_modeling_config
    config["modeling"]["imbalance_strategy"] = "cost_weighted"
    strategy = ImbalanceStrategy(config)

    y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 2, 3])  # deliberately skewed
    weight = strategy.sample_weight_for("random_forest", y)

    expected = compute_sample_weight("balanced", y)
    np.testing.assert_allclose(weight, expected)
    # Majority class (0) must be weighted below the minority classes.
    assert weight[y == 0][0] < weight[y == 1][0]
    assert weight[y == 1][0] < weight[y == 3][0]


def test_none_strategy_disables_both_mechanisms(minimal_modeling_config):
    config = minimal_modeling_config
    config["modeling"]["imbalance_strategy"] = "none"
    strategy = ImbalanceStrategy(config)

    assert strategy.smote_allowed("random_forest") is False
    assert strategy.sample_weight_for("random_forest", np.array([0, 1, 2, 3])) is None
