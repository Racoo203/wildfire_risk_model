"""Regression coverage: the susceptibility raster must predict a class
for every in-domain pixel, not just ones with a ground-truth density
label. prepare_full_domain() is the dataset_prep step that keeps
NaN-label rows (instead of dropping them like prepare_test() does) so
stage_evaluate can predict over the full raster domain."""

import numpy as np
import pandas as pd
import pytest

from wildfire_susceptibility.modeling.dataset_prep import DatasetPrep


@pytest.fixture
def prep(minimal_modeling_config):
    return DatasetPrep(minimal_modeling_config)


def _df_with_nan_labels(n=20, n_nan=5):
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 4, size=n).astype(float)
    labels[:n_nan] = np.nan
    return pd.DataFrame({
        "elevation": rng.normal(size=n),
        "ndvi": rng.normal(size=n),
        "label": labels,
        "_x": rng.uniform(0, 1000, size=n),
        "_y": rng.uniform(0, 1000, size=n),
    })


def test_prepare_full_domain_keeps_nan_label_rows(prep):
    df = _df_with_nan_labels()
    out = prep.prepare_full_domain(df, season="summer", climate_vars=())
    assert len(out) == len(df)
    assert out["label"].isna().sum() == 5


def test_prepare_test_still_drops_nan_label_rows(prep):
    """Contrast case: prepare_test (used for metrics) must keep dropping
    NaN-label rows -- only the full-domain prediction path changed."""
    df = _df_with_nan_labels()
    out = prep.prepare_test(df, season="summer", climate_vars=())
    assert len(out) == len(df) - 5
    assert out["label"].isna().sum() == 0


def test_prepare_full_domain_still_imputes_ndvi(prep):
    df = _df_with_nan_labels()
    df.loc[0, "ndvi"] = np.nan
    out = prep.prepare_full_domain(df, season="summer", climate_vars=())
    assert not out["ndvi"].isna().any()
