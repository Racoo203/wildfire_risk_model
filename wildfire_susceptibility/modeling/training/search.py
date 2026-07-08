from pathlib import Path
from typing import Callable, Optional
import logging

import numpy as np
import optuna
import hashlib
from tqdm import tqdm

from .cv import FoldStrategy

logger = logging.getLogger(__name__)

OPTUNA_STORAGE = "sqlite:///data/silver/dbs/optuna_studies.db"
FINISHED_STATES = {optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED}


class HyperparamSearch:
    def __init__(self, config: dict, fold_strategy: FoldStrategy):
        self.n_trials = config["modeling"]["optuna_n_trials"]
        self.fold_strategy = fold_strategy
        self._ensure_storage_dir()

    @staticmethod
    def _param_space_signature(model_cls) -> str:
        """
        A short hash capturing the *shape* of param_space — distribution
        types and, for categoricals, their choices — so that changing a
        model's search space (e.g. dropping 'poly' from SVM's kernel
        choices) produces a new study name instead of colliding with an
        old study's incompatible persisted distributions.

        Uses a dummy optuna trial to introspect what suggest_* calls the
        model's param_space() makes, without needing a live study.
        """
        import optuna

        probe_study = optuna.create_study()
        probe_trial = probe_study.ask()
        space = model_cls().param_space(probe_trial)

        # Capture each param's distribution signature from the probe trial.
        parts = []
        for name in sorted(space.keys()):
            dist = probe_trial.distributions[name]
            parts.append(f"{name}:{dist!r}")
        signature = "|".join(parts)
        return hashlib.sha1(signature.encode()).hexdigest()[:8]

    @staticmethod
    def _ensure_storage_dir() -> None:
        db_path = Path(OPTUNA_STORAGE.replace("sqlite:///", ""))
        db_path.parent.mkdir(parents=True, exist_ok=True)

    def get_or_create_study(self, season: str, model_name: str) -> optuna.Study:
        from ...core.registry import MODELS

        model_cls = MODELS[model_name]
        sig = self._param_space_signature(model_cls)
        study_name = f"{season}_{model_name}_{sig}"

        storage = optuna.storages.RDBStorage(
            url=OPTUNA_STORAGE,
            engine_kwargs={"connect_args": {"timeout": 30}},
        )
        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
        )

        incomplete = [t for t in study.trials if t.state not in FINISHED_STATES]
        if incomplete:
            logger.info(
                f"[{season}][{model_name}] removing {len(incomplete)} incomplete trial(s) "
                f"from a previous run: {[t.number for t in incomplete]}"
            )
            for t in incomplete:
                try:
                    study.tell(t.number, state=optuna.trial.TrialState.FAIL)
                except Exception as e:
                    logger.warning(f"Could not mark trial {t.number} as FAILED: {e}")

        return study

    def run(
        self,
        study: optuna.Study,
        model_cls,
        X_search, y_search,
        folds,
        season: str,
        model_name: str,
        progress_callback: Optional[Callable[[str, int, float], None]] = None,
    ) -> None:
        n_finished = len([t for t in study.trials if t.state in FINISHED_STATES])
        n_remaining = max(0, self.n_trials - n_finished)

        logger.info(f"[{season}][{model_name}] finished={n_finished} target={self.n_trials} remaining={n_remaining}")
        if n_remaining == 0:
            logger.info(f"[{season}][{model_name}] target trial count already reached; skipping search")
            return

        objective = self._make_objective(model_cls, X_search, y_search, folds, season, model_name)

        with tqdm(total=n_remaining, desc=f"[{season}] {model_name}") as pbar:
            def _callback(study, trial):
                pbar.update(1)
                if progress_callback is not None:
                    progress_callback(model_name, trial.number, trial.value or 0.0)

            study.optimize(objective, n_trials=n_remaining, callbacks=[_callback])

    def _make_objective(self, model_cls, X_search, y_search, folds, season, model_name):
        def objective(trial):
            params = model_cls().param_space(trial)
            fold_aucs = []

            for fold_idx, (train_idx, test_idx) in enumerate(folds):
                context = f"[{season}][{model_name}] trial {trial.number} fold {fold_idx + 1}"
                auc = self.fold_strategy.fit_and_score(
                    model_cls, params, X_search, y_search, train_idx, test_idx, context=context,
                )
                fold_aucs.append(auc)
                logger.info(f"{context} AUC={auc:.4f} | params={params}")

                trial.report(float(np.mean(fold_aucs)), step=fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return float(np.mean(fold_aucs))

        return objective