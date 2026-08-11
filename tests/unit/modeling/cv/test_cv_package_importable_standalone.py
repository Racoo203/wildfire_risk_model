# tests/unit/modeling/cv/test_cv_package_importable_standalone.py
"""Regression test for a circular-import bug: `modeling.cv` used to fail to
import unless `modeling.training` had already finished initializing first
(cv/base.py -> training.resampling -> training/__init__.py -> trainer.py
-> back to ..cv, which was still mid-init). Fixed by moving resampling.py
and balance.py out of the training/ package to modeling/, so cv no longer
needs to trigger training/__init__.py at all.

Must run in a fresh subprocess: within one pytest process, some other test
module almost certainly imports `wildfire_susceptibility.modeling.training`
before this one runs, which caches it in sys.modules and hides the bug -
that's exactly why the original bug went uncaught by the existing suite."""

import subprocess
import sys


def test_modeling_cv_importable_as_first_import():
    result = subprocess.run(
        [sys.executable, "-c", "import wildfire_susceptibility.modeling.cv"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"`import wildfire_susceptibility.modeling.cv` failed as a fresh, "
        f"first import (no other wildfire_susceptibility module imported "
        f"first):\n{result.stderr}"
    )
