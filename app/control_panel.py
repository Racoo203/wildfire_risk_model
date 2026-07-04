import streamlit as st
import yaml
from pathlib import Path
import logging
import io

from wildfire_susceptibility.config.loader import ConfigLoader
from wildfire_susceptibility.pipeline.preprocessor import WildfirePreprocessor
from wildfire_susceptibility.modeling.train import ModelTrainer
from wildfire_susceptibility.modeling.dataset_prep import DatasetPrep
from wildfire_susceptibility import viz

from ..scripts.generate_report_figures import generate_all, ALL_CATEGORIES

CONFIG_PATH = Path("./wildfire_susceptibility/config/defaults.yaml")
WORKING_CONFIG_PATH = Path("./wildfire_susceptibility/config/_working.yaml")

st.set_page_config(page_title="Wildfire Susceptibility — Control Panel", layout="wide")

# --- log capture: tail into the UI without touching the logger setup ------
class _StreamlitLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.buffer = io.StringIO()

    def emit(self, record):
        self.buffer.write(self.format(record) + "\n")

if "log_handler" not in st.session_state:
    handler = _StreamlitLogHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger("wildfire_susceptibility").addHandler(handler)
    logging.getLogger("wildfire_susceptibility.scripts.generate_report_figures").addHandler(handler)
    st.session_state.log_handler = handler

# --- load working config (a scratch copy, never overwrites defaults.yaml) -
def load_working_config() -> dict:
    src = WORKING_CONFIG_PATH if WORKING_CONFIG_PATH.exists() else CONFIG_PATH
    with open(src) as f:
        return yaml.safe_load(f)

def save_working_config(cfg: dict):
    with open(WORKING_CONFIG_PATH, "w") as f:
        yaml.safe_dump(cfg, f)

cfg = load_working_config()

# --- sidebar: config editor, plain widgets bound to known keys ------------
st.sidebar.header("Config")

cfg["seasons"]["active"] = st.sidebar.multiselect(
    "Active seasons", ["spring", "summer", "fall", "winter"], default=cfg["seasons"]["active"]
)
cfg["labels"]["density_method"] = st.sidebar.selectbox(
    "Density method", ["convolution", "kde"],
    index=["convolution", "kde"].index(cfg["labels"]["density_method"])
)
cfg["labels"]["classify_method"] = st.sidebar.selectbox(
    "Classify method", ["percentile", "jenks", "gmm"],
    index=["percentile", "jenks", "gmm"].index(cfg["labels"]["classify_method"])
)
cfg["labels"]["include_d_fires_as_feature"] = st.sidebar.checkbox(
    "Include d_fires as feature (leaky variant)", value=cfg["labels"]["include_d_fires_as_feature"]
)
cfg["modeling"]["models"] = st.sidebar.multiselect(
    "Models", ["random_forest", "svm", "xgboost", "neural_net"], default=cfg["modeling"]["models"]
)
cfg["modeling"]["cv_strategy"] = st.sidebar.selectbox(
    "CV strategy", ["standard", "spatial", "both"],
    index=["standard", "spatial", "both"].index(cfg["modeling"].get("cv_strategy", "both"))
)
cfg["modeling"]["optuna_n_trials"] = st.sidebar.number_input(
    "Optuna trials", min_value=1, max_value=200, value=cfg["modeling"]["optuna_n_trials"]
)
cfg["processing"]["force_recompute"] = st.sidebar.checkbox(
    "Force recompute (bypass cache)", value=cfg["processing"]["force_recompute"]
)

figure_categories = st.sidebar.multiselect(
    "Figure categories to generate", list(ALL_CATEGORIES), default=list(ALL_CATEGORIES)
)

if st.sidebar.button("Save config"):
    save_working_config(cfg)
    st.sidebar.success(f"Saved to {WORKING_CONFIG_PATH.name}")

save_working_config(cfg)  # keep scratch copy current even without explicit save

# --- validate against schema before allowing any run ----------------------
try:
    validated = ConfigLoader.load(WORKING_CONFIG_PATH)
    st.sidebar.success("Config valid")
except Exception as ex:
    st.sidebar.error(f"Config invalid: {ex}")
    st.stop()

# --- main area: stage list with cached/stale status, one button each ------
st.title("Pipeline stages")

STAGES = [
    ("Features + labels (per active season)", "features_labels"),
    ("Dataset assembly (Gold)", "dataset_assembly"),
    ("Model training", "training"),
    ("Figure generation", "figures"),
]

def output_dir_status(cfg: dict, season: str) -> str:
    p = Path(cfg["base"]["output_dir"]) / f"risk_labels_clean_{season}.tif"
    return "cached" if p.exists() and not cfg["processing"]["force_recompute"] else "stale"

for label, key in STAGES:
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.write(f"**{label}**")
        if key == "features_labels":
            for s in cfg["seasons"]["active"]:
                st.caption(f"{s}: {output_dir_status(cfg, s)}")
    with col2:
        run_clicked = st.button("Run", key=f"run_{key}")
    with col3:
        st.caption("")

    if run_clicked:
        with st.spinner(f"Running {label}..."):
            try:
                if key == "features_labels":
                    wf = WildfirePreprocessor(WORKING_CONFIG_PATH)
                    dataset_paths = wf.run_full_pipeline()
                    st.session_state["dataset_paths"] = dataset_paths
                    st.success(f"Done: {dataset_paths}")

                elif key == "dataset_assembly":
                    st.info("Dataset assembly runs as part of the features/labels stage in the current pipeline.")

                elif key == "training":
                    dataset_paths = st.session_state.get("dataset_paths")
                    if not dataset_paths:
                        st.error("Run features/labels first.")
                    else:
                        for season, path in dataset_paths.items():
                            import pandas as pd
                            df = pd.read_csv(path)
                            prep = DatasetPrep(cfg)
                            X_train, X_test, y_train, y_test = prep.stratified_split(df)
                            groups_train = prep.assign_spatial_blocks(X_train) if cfg["modeling"].get("cv_strategy") in ("spatial", "both") else None
                            feature_cols = [c for c in X_train.columns if not c.startswith("_")]
                            trainer = ModelTrainer(cfg)
                            results = trainer.train_all(
                                season, X_train[feature_cols], y_train, X_test[feature_cols], y_test,
                            )
                            st.success(f"[{season}] training complete: {list(results.keys())}")

                elif key == "figures":
                    generate_all(
                        config_path=WORKING_CONFIG_PATH,
                        seasons=cfg["seasons"]["active"],
                        categories=figure_categories,
                    )
                    st.success(f"Figures regenerated for {cfg['seasons']['active']}: {figure_categories}")

            except Exception as ex:
                st.error(f"Failed: {ex}")

st.divider()
st.subheader("Logs")
st.text_area("Log tail", st.session_state.log_handler.buffer.getvalue()[-5000:], height=300)