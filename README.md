# wildfire_susceptibility

A wildfire susceptibility mapping pipeline for Essex, UK — an MSc dissertation
(MA981, University of Essex) done in partnership with the Essex Centre for Data
Analytics, and an operational deliverable for Essex County Council.

Adapts the 14-feature, 4-model ML susceptibility pipeline from
[Bouzeraa et al. (2025), *Applied Sciences* 15, 12188](https://doi.org/10.3390/app152212188)
— originally applied to a Mediterranean context — to a temperate UK county,
adding seasonal stratification (spring/summer/fall models), spatially-aware
cross-validation, and literature-informed feature engineering.

## Architecture

- **Data tiers**: medallion architecture — `data/{bronze,silver,gold}/` for
  raw, cleaned, and model-ready data respectively.
- **Pipeline**: a ten-stage pipeline (`static, seasonal, labels, integration,
  preprocessing, eda, temporal_eda, train, evaluate, selection`), orchestrated
  by [`run_pipeline.py`](src/wildfire_susceptibility/pipeline/run_pipeline.py),
  resumable mid-run via JSON state.
- **Config**: pydantic v2 schema
  ([`config/schema.py`](src/wildfire_susceptibility/config/schema.py)), YAML-driven,
  base files in [`configs/`](configs) with per-run overrides in
  [`configs/experiment/`](configs/experiment).
- **Models**: Random Forest, CatBoost, Ordinal Logistic Regression
  (proportional odds, `mord.LogisticAT`), and a feed-forward neural network
  (PyTorch), trained per season. SVM/XGBoost remain registered for
  reproducing earlier results but aren't in the default roster.
- **Labels**: susceptibility classes come from crossing a fire-density
  estimate (KDE or convolution) with a classification method (percentile,
  Jenks natural breaks, or GMM) — all combinations are supported as a
  documented methodological comparison.
- **Experiment tracking**: MLflow (SQLite backend) for runs, Optuna (SQLite
  backend) for hyperparameter search.
- **Spatial stats**: `esda`, `libpysal`, `verde`, `scikit-gstat`, `jenkspy` for
  spatial cross-validation and spatial-autocorrelation-aware components.

## Setup

```bash
conda env create -f env.yaml
conda activate wildfire
pip install -e .
```

## Running the pipeline

```bash
python -m wildfire_susceptibility.pipeline.run_pipeline --stage all
python -m wildfire_susceptibility.pipeline.run_pipeline --stage train --config-file configs/experiment/dissertation.yaml
```

Individual stages (`static`, `seasonal`, `labels`, `integration`,
`preprocessing`, `eda`, `temporal_eda`, `train`, `evaluate`, `selection`) can
be run one at a time with `--stage <name>`; progress is checkpointed to
`models/.pipeline_state.json` so a later invocation resumes rather than
recomputing.

## Tests

```bash
pytest                    # full suite
pytest -m "not slow"      # skip tests that hit MLflow/Optuna/real model fits
```

## Project status

Bug-fixing-before-final-results phase: correctness of the reported numbers
matters more than feature completeness right now.
