"""Regression coverage for the class-name-scheme lookup: this replaces
four independent hardcoded 4-name lists (metrics.py, evaluate.py,
evaluation.py, charts.py) that silently truncated once n_classes stopped
being fixed at 4."""

import pytest

from wildfire_susceptibility.modeling.class_labels import class_names_for


@pytest.mark.parametrize("n_classes,expected", [
    (3, ["Low", "Medium", "High"]),
    (4, ["Low", "Medium", "High", "Very High"]),
    (5, ["Very Low", "Low", "Medium", "High", "Very High"]),
])
def test_class_names_for_known_schemes(n_classes, expected):
    assert class_names_for(n_classes) == expected


def test_class_names_for_unsupported_count_raises():
    with pytest.raises(ValueError, match="n_classes=7"):
        class_names_for(7)
