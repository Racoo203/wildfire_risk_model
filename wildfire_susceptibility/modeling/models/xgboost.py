from ..registry import MODELS

@MODELS.register("xgboost")
class XGBoostModel(BaseWildfireModel):
    def param_space(self, trial):        # Optuna hook
        return {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        }