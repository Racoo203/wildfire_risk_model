"""Label builder tests. Scope: currently covers
KernelDensityClassifier.classify()'s label_path attachment (fixed
alongside the Phase 1 stage-contract work). Broader label-builder
coverage remains Phase 7 debt."""

from wildfire_susceptibility.labels.kernel_density import KernelDensityClassifier


def _classifier(minimal_config, synthetic_reference_raster):
    minimal_config["labels"] = {
        "density_method": "convolution",
        "classify_method": "percentile",
        "percentiles": [60, 75, 90],
        "conv_sigma_cells": 1,
        "random_state": 42,
    }
    return KernelDensityClassifier(minimal_config, synthetic_reference_raster)


def test_classify_attaches_label_path_on_fresh_build(
    minimal_config, synthetic_reference_raster, synthetic_fire_points
):
    kde = _classifier(minimal_config, synthetic_reference_raster)
    density = kde.compute_density(synthetic_fire_points, season="test_train")
    labels, fit_artifact = kde.classify(density, season="test_train")

    assert "label_path" in fit_artifact
    assert fit_artifact["label_path"].exists()
    assert fit_artifact["label_path"] == kde.output_dir / "risk_labels_convolution_percentile_test_train.tif"


def test_classify_attaches_label_path_on_cache_hit(
    minimal_config, synthetic_reference_raster, synthetic_fire_points
):
    kde = _classifier(minimal_config, synthetic_reference_raster)
    density = kde.compute_density(synthetic_fire_points, season="test_train")
    kde.classify(density, season="test_train")  # first call: fresh build

    # Second call with an identically-configured classifier: cache hit path
    kde2 = _classifier(minimal_config, synthetic_reference_raster)
    labels, fit_artifact = kde2.classify(density, season="test_train")

    assert "label_path" in fit_artifact
    assert fit_artifact["label_path"].exists()


def test_classify_frozen_fit_gets_own_split_label_path_not_trains(
    minimal_config, synthetic_reference_raster, synthetic_fire_points
):
    """The test split's frozen-fit application must report ITS OWN
    label_path (test's raster), not silently inherit train's — this is
    exactly the aliasing bug the dict-copy fix in classify() guards
    against."""
    kde = _classifier(minimal_config, synthetic_reference_raster)
    density_train = kde.compute_density(synthetic_fire_points, season="test_train")
    _, train_fit = kde.classify(density_train, season="test_train")
    train_label_path = train_fit["label_path"]

    density_test = kde.compute_density(synthetic_fire_points, season="test_test")
    _, test_fit = kde.classify(density_test, season="test_test", fitted=train_fit)

    assert test_fit["label_path"] != train_label_path
    assert test_fit["label_path"].exists()
    # Original train_fit must be untouched by the frozen-apply call
    assert train_fit["label_path"] == train_label_path