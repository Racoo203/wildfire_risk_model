"""PyTorch feed-forward classifier, wrapped to expose the same fit/predict_proba
interface as the sklearn/xgboost wrappers so it drops into the same
Optuna/mlflow harness without special-casing (Section 8)."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

from ...core.registry import MODELS


class _FeedForward(nn.Module):
    def __init__(self, n_features: int, n_classes: int, hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()
        layers = []
        in_dim = n_features
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


@MODELS.register("neural_net")
class NeuralNetModel:
    def __init__(
        self, hidden_dim=64, n_layers=2, dropout=0.2, lr=1e-3,
        epochs=50, batch_size=256, weight_decay=0.0,
        early_stopping_patience=10, early_stopping_val_fraction=0.15, **kwargs,
    ):
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        # Not exposed via param_space(): this is an architectural safety
        # net (added 08/17/2026 alongside that method's regularization
        # tightening — see its comment), not a hyperparameter Optuna
        # should be free to weaken, same rationale as random_forest.py
        # excluding class_weight from its own search space.
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_val_fraction = early_stopping_val_fraction
        self.model = None
        self.n_classes = None

    def fit(self, X, y, sample_weight=None):
        # Every other model wrapper fixes random_state=42; torch has no
        # equivalent constructor kwarg, so weight init and the DataLoader's
        # shuffle order have to be seeded explicitly here to match that
        # convention — otherwise search/refit results for this model alone
        # aren't reproducible run-to-run.
        torch.manual_seed(42)
        generator = torch.Generator().manual_seed(42)

        self.n_classes = int(np.max(y)) + 1
        n_features = X.shape[1]
        self.model = _FeedForward(n_features, self.n_classes, self.hidden_dim, self.n_layers, self.dropout)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # reduction="none" + manual per-sample multiply (rather than
        # CrossEntropyLoss's per-CLASS weight= kwarg) so this generalizes
        # to any sample_weight array, not just ones that are constant
        # within a class — sample_weight_for() currently only ever produces
        # the latter (balanced/inverse-frequency), but nothing here assumes
        # that.
        criterion = nn.CrossEntropyLoss(reduction="none")

        if sample_weight is None:
            sample_weight = np.ones(len(y), dtype="float32")

        # Carved out of the training data (not a caller-supplied held-out
        # set) purely to give fit() a stopping signal — this model was the
        # only one of the four with no mechanism to stop before it
        # overfits (the others stop implicitly via tree count/depth).
        # Fixed random_state=42, same reproducibility convention as above.
        train_idx, val_idx = train_test_split(
            np.arange(len(y)), test_size=self.early_stopping_val_fraction,
            random_state=42, stratify=y,
        )

        X_train_t = torch.as_tensor(X[train_idx], dtype=torch.float32)
        y_train_t = torch.as_tensor(y[train_idx], dtype=torch.long)
        w_train_t = torch.as_tensor(sample_weight[train_idx], dtype=torch.float32)
        X_val_t = torch.as_tensor(X[val_idx], dtype=torch.float32)
        y_val_t = torch.as_tensor(y[val_idx], dtype=torch.long)
        w_val_t = torch.as_tensor(sample_weight[val_idx], dtype=torch.float32)

        dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t, w_train_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, generator=generator)

        best_val_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0
        self.n_epochs_trained = 0

        for epoch_idx in range(self.epochs):
            self.model.train()
            for xb, yb, wb in loader:
                optimizer.zero_grad()
                per_sample_loss = criterion(self.model(xb), yb)
                loss = (per_sample_loss * wb).mean()
                loss.backward()
                optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val_loss = (criterion(self.model(X_val_t), y_val_t) * w_val_t).mean().item()

            self.n_epochs_trained = epoch_idx + 1

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.early_stopping_patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_t = torch.as_tensor(X, dtype=torch.float32)
            logits = self.model(X_t)
            probs = torch.softmax(logits, dim=1)
        return probs.numpy()

    def param_space(self, trial) -> dict:
        # Range tightened 08/17/2026, same overfitting diagnosis that drove
        # random_forest.py's and catboost_model.py's 08/16/2026 param_space
        # cuts (see those files' comments) applied here: this model wasn't
        # touched in that pass despite having the most capacity of the four
        # (up to 4 layers x 256 units) and the weakest regularization floor
        # (dropout could search down to 0.0, weight_decay down to 1e-6,
        # i.e. trials could disable both forms of regularization at once).
        # hidden_dim/n_layers ceilings lowered and dropout/weight_decay
        # floors raised so Optuna can no longer reach a near-unregularized
        # trial, matching RF's max_depth/min_samples_leaf and CatBoost's
        # l2_leaf_reg treatment. epochs left as-is: fit() has no early
        # stopping (no validation split during training), so capping epochs
        # here would be a partial substitute for that, not a fix for it —
        # see the separate early-stopping suggestion raised alongside this
        # change.
        return {
            "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 128]),
            "n_layers": trial.suggest_int("n_layers", 1, 3),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
            "epochs": trial.suggest_int("epochs", 20, 80),
        }

    def needs_scaling(self) -> bool:
        return True

    def native_categorical_support(self) -> bool:
        return False