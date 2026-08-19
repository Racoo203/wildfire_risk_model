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


def test_smote_explicitly_enabled_true_only_when_someone_wrote_it(minimal_modeling_config):
    """The bug this guards against: configs/experiment/dissertation.yaml
    sets imbalance_strategy: "smote" but never touches use_smote (still
    False, inherited from configs/modeling.yaml) — under the old
    schema-level default="smote", ImbalanceStrategy couldn't tell that
    apart from a config that never mentions imbalance_strategy at all, so
    SMOTEResampler stayed gated behind use_smote regardless and silently
    never resampled. schema.py now leaves the pydantic default at None so
    this distinction survives model_dump(); smote_explicitly_enabled()
    is what SMOTEResampler.resample() consults to let an explicit setting
    activate resampling on its own, no use_smote needed."""
    config = minimal_modeling_config
    assert config["modeling"]["imbalance_strategy"] is None  # nobody set it yet

    strategy_unset = ImbalanceStrategy(config)
    assert strategy_unset.resolve("random_forest") == "smote"  # still falls back to "smote"
    assert strategy_unset.smote_explicitly_enabled("random_forest") is False  # but not explicitly

    config["modeling"]["imbalance_strategy"] = "smote"
    strategy_explicit = ImbalanceStrategy(config)
    assert strategy_explicit.smote_explicitly_enabled("random_forest") is True

    # Per-model override follows the same rule, independent of the default.
    config["modeling"]["imbalance_strategy"] = None
    config["modeling"]["imbalance_strategy_by_model"] = {"catboost": "smote"}
    strategy_override = ImbalanceStrategy(config)
    assert strategy_override.smote_explicitly_enabled("catboost") is True
    assert strategy_override.smote_explicitly_enabled("random_forest") is False


def test_native_balanced_produces_no_sample_weight_and_disables_smote(minimal_modeling_config):
    """'native_balanced' (modeling/models/random_forest.py's
    BalancedRandomForestClassifier, modeling/models/catboost_model.py's
    auto_class_weights='Balanced') is mutually exclusive with both SMOTE
    and the external cost_weighted sample_weight — each model handles
    balancing internally, at model-construction time (via the
    native_balanced kwarg trainer.py binds onto model_cls), not through
    anything ImbalanceStrategy hands back here."""
    config = minimal_modeling_config
    config["modeling"]["use_smote"] = True  # would enable SMOTE under the legacy/default path
    config["modeling"]["imbalance_strategy_by_model"] = {
        "random_forest": "native_balanced", "catboost": "native_balanced",
    }
    strategy = ImbalanceStrategy(config)

    for model_name in ("random_forest", "catboost"):
        assert strategy.resolve(model_name) == "native_balanced"
        assert strategy.smote_allowed(model_name) is False
        assert strategy.sample_weight_for(model_name, np.array([0, 1, 2, 3])) is None
