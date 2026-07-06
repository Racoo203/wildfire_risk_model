import numpy as np
from wildfire_susceptibility.modeling.training.resampling import SMOTEResampler

def test_smote_disabled_passthrough():
    resampler = SMOTEResampler({"modeling": {"use_smote": False}})
    X, y = np.zeros((10, 3)), np.array([0]*8 + [1]*2)
    X_out, y_out = resampler.resample(X, y)
    assert X_out is X and y_out is y

def test_smote_skips_on_degenerate_class():
    resampler = SMOTEResampler({"modeling": {"use_smote": True, "smote_k_neighbors": 5}})
    X, y = np.zeros((10, 3)), np.array([0]*9 + [1])  # minority class has 1 sample
    X_out, y_out = resampler.resample(X, y, context="test")
    assert len(y_out) == len(y)  # unchanged, not resampled