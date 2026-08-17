"""Full pairwise Spearman rho + significance across every column in the
fully-prepared EDA dataset (see stage_eda.py) -- not just the modeling
feature subset plot_vif_correlation's heatmap covers. CSV only, no plot;
additive alongside the existing correlation_spearman matrix/heatmap.

IMPORTANT -- does NOT call scipy.stats.spearmanr (nor numpy.corrcoef/cov,
which spearmanr calls internally): on this project's numpy 2.4.6 / scipy
1.18.0 combination, any call into numpy.corrcoef/cov crashes the whole
Python process with a hard OS-level exception (Windows
"0xc06d007f") rather than raising a catchable Python exception --
reproduced with plain `np.corrcoef(a, b)` on two ordinary float64 arrays,
unrelated to ties, NaNs, or one-hot columns specifically. This is almost
certainly the same class of issue as the "Found Intel OpenMP (libiomp) and
LLVM OpenMP (libomp) loaded at the same time" RuntimeWarning already seen
elsewhere in this repo's test output (threadpoolctl, via
test_stage_labels_comparison.py). pandas.DataFrame.corr(method="spearman")
-- already used safely elsewhere in this codebase, e.g.
plot_vif_correlation in charts.py -- uses its own internal routine and does
not hit this crash, so it's used here for rho; the p-value is computed
separately via the standard analytic t-distribution approximation
(algebraically the same formula scipy.stats.spearmanr itself has long used
internally for n >= 3: t = rho * sqrt((n-2) / (1-rho**2)), two-sided
p = 2 * t.sf(|t|, n-2)), using only scipy.stats.t.sf (confirmed not to
trigger the crash) rather than spearmanr's own p-value machinery.
"""

from itertools import combinations
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..modeling.categorical import CATEGORICAL_FEATURES
from .one_hot import one_hot_encode_categoricals

ALPHA = 0.05


def _p_value_from_rho(rho: float, n_eff: int) -> float:
    """Two-sided p-value for a Spearman rho computed from n_eff pairwise-
    complete observations, via the analytic t-distribution approximation.
    NaN when there's not enough data (n_eff < 3) or rho itself is NaN
    (zero-variance column); 0.0 for |rho| == 1 (perfect correlation, e.g.
    a duplicated column or a 2-category one-hot pair) rather than dividing
    by zero."""
    if n_eff < 3 or pd.isna(rho):
        return float("nan")
    if abs(rho) >= 1.0:
        return 0.0
    t_stat = rho * np.sqrt((n_eff - 2) / (1 - rho**2))
    return float(2 * stats.t.sf(abs(t_stat), n_eff - 2))


def compute_spearman_significance_export(
    df: pd.DataFrame,
    figures_dir: Path,
    season: Optional[str] = None,
    categorical_cols: Tuple[str, ...] = CATEGORICAL_FEATURES,
    alpha: float = ALPHA,
) -> dict:
    """
    Pairwise Spearman's rho + significance for every column in `df`
    (categorical columns one-hot encoded first, so dummies participate as
    separate binary columns rather than the raw categorical code). Writes
    one flat CSV to figures_dir/<season>/eda/spearman_significance_full.csv:

        Var1, Var2, Spearman's rho, test statistic, significant? (yes/no, alpha=0.95)

    "test statistic" reports a p-value (see module docstring for why this
    isn't scipy.stats.spearmanr's own p-value); a pair is flagged
    significant when p < alpha (alpha=0.05, i.e. the 95% confidence
    framing in the column header).

    One row per unique unordered pair -- N*(N-1)/2 rows for N columns; no
    self-pairs, no duplicated (B, A) rows. NaNs are handled pairwise (each
    pair's rho comes from pandas' own pairwise-complete correlation, and
    its p-value's degrees of freedom come from that same pair's actual
    non-null overlap), so missingness in unrelated columns never drops a
    row from a pair that's otherwise complete.

    Returns a summary dict -- {"out_path", "n_cols", "n_rows",
    "onehot_parent_groups"} -- for the caller to log or report; does not
    itself log to figures/manifest.json (that manifest is for rendered
    figures, not flat diagnostic tables -- matches vif.csv/
    correlation_spearman.csv's existing convention).
    """
    encoded_df, parent_map = one_hot_encode_categoricals(df, categorical_cols)
    columns = list(encoded_df.columns)

    rho_matrix = encoded_df.corr(method="spearman")

    valid = encoded_df.notna().astype(np.int64)
    pairwise_n = valid.T.dot(valid)

    records = []
    for col_a, col_b in combinations(columns, 2):
        rho = rho_matrix.loc[col_a, col_b]
        n_eff = int(pairwise_n.loc[col_a, col_b])
        p_value = _p_value_from_rho(rho, n_eff)
        significant = "yes" if pd.notna(p_value) and p_value < alpha else "no"
        records.append({
            "Var1": col_a,
            "Var2": col_b,
            "Spearman's rho": rho,
            "test statistic": p_value,
            "significant? (yes/no, alpha=0.95)": significant,
        })

    result_df = pd.DataFrame.from_records(
        records,
        columns=["Var1", "Var2", "Spearman's rho", "test statistic", "significant? (yes/no, alpha=0.95)"],
    )

    out_path = Path(figures_dir) / (season or "static") / "eda" / "spearman_significance_full.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_path, index=False)

    return {
        "out_path": out_path,
        "n_cols": len(columns),
        "n_rows": len(result_df),
        "onehot_parent_groups": parent_map,
    }
