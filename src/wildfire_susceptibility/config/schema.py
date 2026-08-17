from pydantic import BaseModel, ConfigDict, Field
from pathlib import Path
from typing import Literal, List, Dict, Tuple, Optional

class BaseConfig(BaseModel):
    region: str = Field(default="Essex")
    boundary_shapefile: Path = Field(default=Path("./data/silver/layers/boundary.shp"))
    output_dir: Path = Field(default=Path("./data/silver/layers"))
    gold_dir: Path = Field(default=Path("./data/gold"))
    model_data_dir: Path = Field(default=Path("./data/silver/model_datasets"))
    figures_dir: Path = Field(default=Path("./figures"))

class ProcessingConfig(BaseModel):
    cell_size_m: float = Field(default=30.0)
    crs: str = Field(default="EPSG:27700")
    training_years: Tuple[int, int] = Field(default=(2009, 2022), description="[start, end)")
    test_years: Tuple[int, int] = Field(default=(2022, 2026), description="[start, end)")
    force_recompute: bool = Field(default=False)

class SeasonsConfig(BaseModel):
    active: List[Literal["spring", "summer", "fall", "winter"]] = Field(
        default_factory=lambda: ["spring", "summer", "fall"],
        description="Seasons to process this run",
    )
    seasonal_ndvi: bool = Field(
        default=True,
        description="If true, NDVI is recomputed per season instead of once",
    )
    definitions: Dict[str, List[int]] = Field(
        default_factory=lambda: {
            "spring": [3, 4, 5],
            "summer": [6, 7, 8],
            "fall": [9, 10, 11],
            "winter": [12, 1, 2],
        }
    )

class DataSourceConfig(BaseModel):
    class Cua(BaseModel):
        data_dir: Path = Path("./data/bronze/boundary")

    class Srtm(BaseModel):
        data_dir: Path = Path("./data/bronze/topography")
        tiles: List[str] = [
            "n51_e000_1arc_v3.tif",
            "n51_e001_1arc_v3.tif",
            "n51_w001_1arc_v3.tif",
            "n52_e000_1arc_v3.tif",
            "n52_e001_1arc_v3.tif",
        ]

    class Haduk(BaseModel):
        data_dir: Path = Path("./data/bronze/climate_haduk")
        sources: List[str] = ["tas", "tasmax", "tasmin", "rainfall", "sfcWind", "hurs"]

    class Modis(BaseModel):
        data_dir: Path = Path("./data/bronze/vegetation")

    class Proximity(BaseModel):
        data_dir: Path = Path("./data/bronze/proximity")
        sources: List[str] = ["oproad_gpkg_gb", "oprvrs_gpkg_gb", "essex-260607-free."]
        human_fclasses: List[str] = [
            "residential", "farmland", "farm", "farm_auxiliary", "industrial",
            "commercial", "retail", "allotmets", "orchard", "greenhouse_horticulture",
        ]

    class Incidents(BaseModel):
        data_dir: Path = Path("./data/bronze/incidents")
        date_col: str = "CallDateID"
        geo_cols: List[str] = ["IncidentEasting", "IncidentNorthing"]
        severity_cols: List[str] = ["NumberVehiclesDeployed"]

    cua: Cua = Cua()
    srtm: Srtm = Srtm()
    haduk: Haduk = Haduk()
    modis_nvdi: Modis = Modis()
    proximity: Proximity = Proximity()
    incidents: Incidents = Incidents()

class LabelsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    density_method: Literal["convolution", "kde"] = "convolution"
    classify_method: Literal["percentile", "jenks", "gmm"] = "gmm"
    compare_classify_methods: List[str] = Field(default_factory=lambda: ["percentile", "jenks", "gmm"])
    kmeans_n_init: int = 10
    kmeans_max_iter: int = 300
    random_state: int = 42

    max_sample_per_class: int = 200_000
    percentiles: List[int] = Field(default_factory=lambda: [60, 75, 90])
    conv_sigma_cells: int = 3
    kde_bandwidth_m: int = 500
    kde_zero_percentile: float = 90.0
    trim_bottom_pct: float = 0.1

    n_classes: int = Field(
        default=4,
        description="Class count actually used by KernelDensityClassifier.classify() "
                     "for whichever classify_method is dispatched (percentile/jenks/gmm "
                     "all read this single generic field, not a per-method one) when "
                     "auto_find_k is False. See auto_find_k for the K-selection axis.",
    )
    auto_find_k: bool = Field(
        default=False,
        description="If true, KernelDensityClassifier.find_optimal_k() picks n_classes "
                     "via silhouette score over k_search_range instead of using the "
                     "fixed n_classes value. K-selection behavior itself is out of scope "
                     "for this change -- see CLAUDE.md known bug #2.",
    )
    k_search_range: Tuple[int, int] = (2, 6)

    jenks_n_classes: int = 4
    jenks_max_sample: int = 200_000
    gmm_n_components: int = 4
    gmm_max_sample: int = 200_000
    gmm_random_state: int = 42

    kmeans_k: int = 2

    clean_labels: bool = True
    run_sensitivity: bool = True
    sensitivity_n_inits: List[int] = Field(default_factory=lambda: [10, 20])
    sensitivity_seeds: List[int] = Field(default_factory=lambda: [7, 123])
    sensitivity_n_reweight_trials: int = 3
    sensitivity_feature_noise_std: float = 0.05

    clustering_exclude_features: List[str] = Field(
        default_factory=lambda: ["tas", "tasmin", "landuse_class", "d_fires"]
    )
    clustering_variance_floor_pct: float = 0.01

# wildfire_susceptibility/config/schema.py — ModelingConfig

class ModelingConfig(BaseModel):
    models: List[Literal[
        "random_forest", "svm", "xgboost", "neural_net", "catboost", "ordinal_lr"
    ]] = Field(
        default_factory=lambda: ["random_forest", "catboost", "ordinal_lr", "neural_net"]
    )
    optuna_n_trials: int = 30
    cv_folds: int = 5
    mlflow_experiment: str = "wildfire-susceptibility-essex"
    selection_rule: Literal["best_auc", "best_pr_auc", "most_conservative"] = "best_pr_auc"
    cv_strategy: Literal[
        "standard", "spatial", "stratified_spatial_block", "both"
    ] = "both"
    spatial_block_size_m: float = 5000.0
    spatial_buffer_m: float = Field(
        default=0.0,
        description="Minimum enforced distance (metres) between any train "
                     "and test point under 'spatial'/'stratified_spatial_block' "
                     "CV. Implemented by excluding, from both train and test, "
                     "every spatial block within ceil(spatial_buffer_m / "
                     "spatial_block_size_m) grid cells of a test block for "
                     "that fold. 0.0 (default) preserves prior behavior: no "
                     "buffer, fold separation only at the block-ID level.",
    )
    log_full_cv_diagnostics: bool = Field(
        default=False,
        description="Only used when cv_strategy='both'. When True, the "
                     "PRIMARY CV strategy also gets a full AUC/F1-macro/"
                     "PR-AUC-macro re-scoring pass on the full training set "
                     "(not HyperparamSearch's subsample) with the winning "
                     "hyperparameters fixed — needed to compute the "
                     "F1-macro/PR-AUC-macro standard-vs-spatial optimism "
                     "gap and to populate cv_auc_spatial_folds (consumed by "
                     "stage_selection's Kruskal-Wallis test), neither of "
                     "which Optuna's search alone can produce. Roughly "
                     "doubles this stage's extra CV-fold fitting cost, so "
                     "it defaults False for routine iteration; set True for "
                     "the run whose numbers are actually being reported "
                     "(see configs/experiment/baseline.yaml). The "
                     "diagnostic-side AUC gap (cv_auc_optimism_gap) is "
                     "always logged regardless of this flag — that pass "
                     "already existed before this flag was introduced.",
    )
    imbalance_strategy: Literal["none", "smote", "cost_weighted", "native_balanced"] = Field(
        default="smote",
        description="Primary imbalance-handling switch, resolved per model "
                     "via imbalance_strategy_by_model (falls back to this "
                     "default). 'smote' defers entirely to the existing "
                     "use_smote/smote_during_search/search_resample_target_"
                     "size/smote_sampling_strategy fields below (unchanged "
                     "legacy behavior — this is the default specifically so "
                     "existing configs are unaffected). 'cost_weighted' "
                     "computes balanced (inverse-frequency) sample_weight "
                     "from each fit's own training y and passes it into the "
                     "model's native fit(sample_weight=...); SMOTE is "
                     "force-disabled for that model regardless of "
                     "use_smote, to avoid double-correcting for the same "
                     "imbalance. 'native_balanced' uses each model's own "
                     "best-available native imbalance mechanism instead of "
                     "an externally-computed sample_weight — for "
                     "random_forest this is imbalanced-learn's "
                     "BalancedRandomForestClassifier (each tree bootstraps a "
                     "class-balanced sample, fixing sklearn's RF sample_weight "
                     "not being bootstrap-aware — see "
                     "scripts/experiment_imbalance_native_vs_costweighted.py); "
                     "for catboost this is auto_class_weights='Balanced' (see "
                     "scripts/experiment_catboost_weighting_variants.py). No "
                     "other model currently implements 'native_balanced'; SMOTE "
                     "is force-disabled the same as under 'cost_weighted'. "
                     "'none' does neither.",
    )
    imbalance_strategy_by_model: Dict[str, Literal["none", "smote", "cost_weighted", "native_balanced"]] = Field(
        default_factory=dict,
        description="Per-model override of imbalance_strategy, same "
                     "fallback-to-default convention as "
                     "optuna_search_subsample_by_model.",
    )
    use_smote: Optional[bool] = Field(
        default=None,
        description="Explicit override. If left unset (None) and "
                     "search_resample_target_size is configured, resampling "
                     "is enabled implicitly — see SMOTEResampler. Set to "
                     "False explicitly to force resampling off even when "
                     "a target size is configured.",
    )
    smote_during_search: bool = True
    search_resample_target_size: Optional[float] = Field(
        default=None,
        description="If set, search-time resampling targets this TOTAL "
                     "row count, split evenly across present classes "
                     "(over+under sampling to hit the target), instead of "
                     "the auto/dict majority-relative behavior used for "
                     "the final refit. <=1.0 is a fraction of the fold's "
                     "own row count; >1.0 is an absolute total.",
    )
    smote_k_neighbors: int = 5
    smote_sampling_strategy: Optional[Dict[int, int]] = None
    optuna_search_subsample: Optional[float] = Field(
        default=None,
        description="Search-time subsample size: values <= 1.0 are treated "
                     "as a fraction of the training population, values > 1.0 "
                     "as an absolute row count.",
    )
    optuna_search_subsample_by_model: Dict[str, float] = Field(
        default_factory=lambda: {"svm": 6000},
        description="Per-model override, same fraction-vs-absolute convention "
                     "as optuna_search_subsample.",
    )
    excluded_features: List[str] = Field(
        default_factory=lambda: ["tas", "tasmin"],
        description="Feature columns dropped from X before model training "
                     "(distinct from clustering_exclude_features, which only "
                     "governs label-cleaning k-means geometry).",
    )

class LoggingConfig(BaseModel):
    level: str = "INFO"
    log_path: Path = Path("./data/logs/pipeline.log")

class WildfireConfig(BaseModel):
    base: BaseConfig = Field(default_factory=BaseConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    seasons: SeasonsConfig = Field(default_factory=SeasonsConfig)
    data_sources: DataSourceConfig = Field(default_factory=DataSourceConfig)
    labels: LabelsConfig = Field(default_factory=LabelsConfig)
    modeling: ModelingConfig = Field(default_factory=ModelingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)