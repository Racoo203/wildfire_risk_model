"""Label builder tests. Scope: currently covers
KernelDensityClassifier.classify()'s label_path attachment (fixed
alongside the Phase 1 stage-contract work), plus the KDE zero-threshold
regression coverage below. Broader label-builder coverage remains
Phase 7 debt."""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

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


# --- KDE zero-threshold regression coverage -----------------------------
#
# Regression coverage for the KDE+Jenks bug: an absolute kde_zero_threshold
# (e.g. 1e-9) admits nearly the entire raster as "valid" for KDE, because KDE's
# gaussian tails have infinite support and rarely hit exact zero -- unlike
# convolution's finite-kernel output, which gives true zeros far from any fire
# point. The fix (kde_zero_percentile) replaces the absolute cutoff with a
# percentile of the raster's own density distribution.

_KDE_GRID_SIZE = 200


@pytest.fixture
def large_kde_reference_raster(tmp_path):
    """A 200x200 all-valid reference raster -- large enough for percentile
    trimming and Jenks to behave meaningfully (the 10x10 grid used elsewhere
    in this file is too small for that)."""
    path = tmp_path / "large_reference.tif"
    transform = from_origin(0, 100_000, 30.0, 30.0)
    data = np.ones((_KDE_GRID_SIZE, _KDE_GRID_SIZE), dtype="float32")
    meta = {
        "driver": "GTiff", "height": _KDE_GRID_SIZE, "width": _KDE_GRID_SIZE,
        "count": 1, "dtype": "float32", "crs": "EPSG:27700",
        "transform": transform, "nodata": np.nan,
    }
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(data[np.newaxis, :, :])
    return path


@pytest.fixture
def kde_scale_density():
    """Synthetic KDE-scale density: 95% background (~1e-9-5e-8, mimicking the
    far-field tail of a gaussian-kernel KDE) and 5% signal (~1e-7-1e-5,
    mimicking cells actually near a fire cluster). Raw KDE probability
    densities live several orders of magnitude below convolution's smoothed
    counts, which is what breaks an absolute zero_threshold."""
    rng = np.random.default_rng(42)
    n = _KDE_GRID_SIZE * _KDE_GRID_SIZE
    n_signal = int(n * 0.05)
    n_background = n - n_signal
    values = np.concatenate([
        rng.uniform(1e-9, 5e-8, n_background),
        rng.uniform(1e-7, 1e-5, n_signal),
    ])
    rng.shuffle(values)
    return values.reshape(_KDE_GRID_SIZE, _KDE_GRID_SIZE).astype("float32")


def _kde_jenks_classifier(minimal_config, ref_path):
    minimal_config["labels"] = {
        "density_method": "kde",
        "classify_method": "jenks",
        "kde_zero_percentile": 90.0,
        "trim_bottom_pct": 0.1,
        "n_classes": 4,
        "random_state": 42,
    }
    return KernelDensityClassifier(minimal_config, ref_path)


def test_kde_jenks_thresholds_nonzero_and_distinct(
    minimal_config, large_kde_reference_raster, kde_scale_density
):
    kde = _kde_jenks_classifier(minimal_config, large_kde_reference_raster)
    _, fit_artifact = kde.classify(kde_scale_density, season="kde_jenks_thresholds")

    thresholds = fit_artifact["thresholds"]
    assert np.all(thresholds > 0)
    assert np.all(np.diff(thresholds) > 0), "thresholds must be strictly increasing, not collapsed"


def test_kde_jenks_class_distribution_not_pathological(
    minimal_config, large_kde_reference_raster, kde_scale_density
):
    """Guards against the dilution bug specifically: with the old absolute
    kde_zero_threshold, 95%+ of cells landed in the bottom class because the
    'valid' population was overwhelmingly KDE background noise. With a
    percentile-based cutoff, the background tail is excluded and all four
    classes should carry a meaningful share of cells."""
    kde = _kde_jenks_classifier(minimal_config, large_kde_reference_raster)
    labels, _ = kde.classify(kde_scale_density, season="kde_jenks_balance")

    valid_labels = labels[~np.isnan(labels)].astype(int)
    counts = np.bincount(valid_labels)

    assert (counts > 0).sum() == 4, "all 4 classes should be populated"
    largest_class_fraction = counts.max() / counts.sum()
    assert largest_class_fraction < 0.9, (
        f"largest class holds {largest_class_fraction:.1%} of cells -- "
        "this is the dilution signature of an under-calibrated zero cutoff"
    )


# --- GMM scale-collapse regression coverage ------------------------------
#
# Regression coverage for the GMM scale-collapse bug: GaussianMixture's
# default reg_covar (1e-6) dominates KDE-scale density variance
# (~1e-10-1e-12), collapsing every component to near-identical
# means/covariances regardless of the underlying structure. The fix
# standardizes the values fed into GaussianMixture.fit() (and predict/
# threshold computation) to the fit data's own mean/std, inverting the
# transform wherever downstream code needs original-scale values.

@pytest.fixture
def kde_scale_bimodal_density():
    """Synthetic KDE-scale density (~1e-9-1e-5) with two well-separated
    gaussian bumps, mimicking two distinct fire-density regions. Values are
    signal-only (no diluting background), matching the diagnosis session's
    finding that the collapse reproduces even with background already
    excluded -- ruling out kde_zero_threshold as the cause."""
    rng = np.random.default_rng(11)
    n = _KDE_GRID_SIZE * _KDE_GRID_SIZE
    n_low = n // 2
    n_high = n - n_low
    low = rng.normal(1e-7, 2e-8, n_low)
    high = rng.normal(1e-5, 2e-6, n_high)
    values = np.clip(np.concatenate([low, high]), 1e-9, None)
    rng.shuffle(values)
    return values.reshape(_KDE_GRID_SIZE, _KDE_GRID_SIZE).astype("float32")


def _kde_gmm_classifier(minimal_config, ref_path):
    minimal_config["labels"] = {
        "density_method": "kde",
        "classify_method": "gmm",
        "kde_zero_percentile": 0.0,
        "trim_bottom_pct": 0.0,
        "n_classes": 2,
        "gmm_random_state": 42,
        "random_state": 42,
    }
    return KernelDensityClassifier(minimal_config, ref_path)


def test_kde_gmm_does_not_collapse_to_single_class(
    minimal_config, large_kde_reference_raster, kde_scale_bimodal_density
):
    """Before the fix, GaussianMixture.fit() on raw KDE-scale values
    (~1e-9-1e-5) collapsed every component to the same mean/covariance
    because reg_covar (1e-6) swamped the signal's variance -- regardless of
    how clearly separated the underlying clusters were."""
    kde = _kde_gmm_classifier(minimal_config, large_kde_reference_raster)
    labels, fit_artifact = kde.classify(kde_scale_bimodal_density, season="kde_gmm_bimodal")

    valid_labels = labels[~np.isnan(labels)].astype(int)
    counts = np.bincount(valid_labels)
    assert (counts > 0).sum() > 1, "GMM collapsed to a single populated class"

    gmm, mean, std = fit_artifact["gmm"], fit_artifact["mean"], fit_artifact["std"]
    means_original_scale = gmm.means_.flatten() * std + mean
    assert np.ptp(means_original_scale) > 1e-6, (
        f"GMM component means are degenerate/near-identical: {means_original_scale}"
    )


def _conv_gmm_classifier(minimal_config, ref_path):
    minimal_config["labels"] = {
        "density_method": "convolution",
        "classify_method": "gmm",
        "conv_sigma_cells": 3.0,
        "n_classes": 4,
        "gmm_random_state": 42,
        "random_state": 42,
    }
    return KernelDensityClassifier(minimal_config, ref_path)


def test_convolution_gmm_still_produces_well_formed_classes(
    minimal_config, large_kde_reference_raster, synthetic_fire_points
):
    """Regression guard: convolution-scale density (O(1e-3)-O(1e-1)) sits
    well above reg_covar, so GMM classification already worked correctly
    there before the standardization fix. Confirms that fix doesn't
    reintroduce a collapse or break monotonic thresholds for this path."""
    kde = _conv_gmm_classifier(minimal_config, large_kde_reference_raster)
    density = kde.compute_density(synthetic_fire_points, season="conv_gmm_check")
    labels, fit_artifact = kde.classify(density, season="conv_gmm_check")

    valid_labels = labels[~np.isnan(labels)].astype(int)
    counts = np.bincount(valid_labels)
    assert (counts > 0).sum() > 1, "convolution+GMM collapsed to a single populated class"

    thresholds = fit_artifact["thresholds"]
    assert np.all(np.diff(thresholds) > 0), "thresholds must be strictly increasing, not collapsed"