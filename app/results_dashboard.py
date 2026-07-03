import streamlit as st
import json
from pathlib import Path
import mlflow
import pandas as pd

from wildfire_susceptibility import viz  # ensures category constants importable if needed

FIGURES_DIR = Path("./figures")
MLFLOW_EXPERIMENT = "wildfire-susceptibility-essex"

st.set_page_config(page_title="Wildfire Susceptibility — Results", layout="wide")

@st.cache_data
def load_manifest():
    path = FIGURES_DIR / "manifest.json"
    return json.loads(path.read_text()) if path.exists() else []

manifest = load_manifest()
seasons = sorted({e["season"] for e in manifest if e["season"]})
season = st.sidebar.selectbox("Season", ["all"] + seasons)

def entries_for(category):
    e = [x for x in manifest if x["category"] == category]
    if season != "all":
        e = [x for x in e if x["season"] in (season, None)]
    return e

tab_maps, tab_eda, tab_models, tab_susceptibility, tab_validation = st.tabs(
    ["Feature Maps", "EDA", "Model Comparison", "Susceptibility Maps", "Validation"]
)

with tab_maps:
    for e in entries_for("factor_map"):
        st.image(str(FIGURES_DIR / e["path"]), caption=e["params"].get("factor", e["path"]))

with tab_eda:
    for cat in ["vif", "correlation", "correlation_spearman", "class_balance", "nan_coverage"]:
        for e in entries_for(cat):
            st.image(str(FIGURES_DIR / e["path"]), caption=cat)

with tab_models:
    st.subheader("mlflow run comparison")
    try:
        runs = mlflow.search_runs(experiment_names=[MLFLOW_EXPERIMENT])
        if season != "all":
            runs = runs[runs["tags.season"] == season]
        st.dataframe(runs[[c for c in runs.columns if c.startswith("metrics.") or c.startswith("tags.")]])
    except Exception as ex:
        st.warning(f"Could not load mlflow runs: {ex}")

    for e in entries_for("roc_curve") + entries_for("cv_comparison"):
        st.image(str(FIGURES_DIR / e["path"]), caption=e["category"])

    for e in entries_for("shap_summary"):
        st.image(str(FIGURES_DIR / e["path"]), caption=e["params"].get("model", "SHAP"))

with tab_susceptibility:
    for e in entries_for("susceptibility_map"):
        label = e["params"].get("model", "") 
        st.image(str(FIGURES_DIR / e["path"]), caption=f"{e['season']} — {label}")

with tab_validation:
    st.info("Time-forward validation figures — populate once evaluate.py's time-forward check is wired to viz/.")