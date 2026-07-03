from typing import Literal

FigureCategory = Literal[
    "factor_map", "susceptibility_map",
    "vif", "correlation", "correlation_spearman",
    "class_balance", "nan_coverage",
    "roc_curve", "shap_summary", "cv_comparison",
    "label_density_diagnostic", "label_cleaning_comparison",
]