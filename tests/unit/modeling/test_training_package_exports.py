# tests/test_modeling/test_training_package_exports.py
"""Guards against the training/ subpackage silently becoming a namespace
package (missing __init__.py) or losing its re-exports — this is the
'ImportError: cannot import name ModelTrainer ... unknown location' failure
mode, distinct from the registry-population bug."""

import importlib
import pytest


def test_training_package_has_real_init():
    """A namespace package has no __file__ on its __init__ — this catches
    a missing __init__.py specifically, before even checking exports."""
    import wildfire_susceptibility.modeling.training as training_pkg

    assert training_pkg.__file__ is not None, (
        "wildfire_susceptibility.modeling.training has no __init__.py "
        "(imported as a namespace package) — add __init__.py with the "
        "expected re-exports."
    )


@pytest.mark.parametrize("name", [
    "ModelTrainer", "SMOTEResampler", "FoldStrategy", "HyperparamSearch", "PostTrainingEvaluator",
])
def test_training_package_exports(name):
    training_pkg = importlib.import_module("wildfire_susceptibility.modeling.training")
    assert hasattr(training_pkg, name), (
        f"'{name}' not exported from wildfire_susceptibility.modeling.training "
        f"— check __init__.py's imports"
    )