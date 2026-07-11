"""PyTorch feed-forward classifier, wrapped to expose the same fit/predict_proba
interface as the sklearn/xgboost wrappers so it drops into the same
Optuna/mlflow harness without special-casing (Section 8)."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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
        epochs=50, batch_size=256, weight_decay=0.0, **kwargs,
    ):
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.model = None
        self.n_classes = None

    def fit(self, X, y):
        self.n_classes = int(np.max(y)) + 1
        n_features = X.shape[1]
        self.model = _FeedForward(n_features, self.n_classes, self.hidden_dim, self.n_layers, self.dropout)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()

        X_t = torch.as_tensor(X, dtype=torch.float32)
        y_t = torch.as_tensor(y, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimizer.step()

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_t = torch.as_tensor(X, dtype=torch.float32)
            logits = self.model(X_t)
            probs = torch.softmax(logits, dim=1)
        return probs.numpy()

    def param_space(self, trial) -> dict:
        return {
            "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 128, 256]),
            "n_layers": trial.suggest_int("n_layers", 1, 4),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
            "epochs": trial.suggest_int("epochs", 20, 80),
        }

    def needs_scaling(self) -> bool:
        return True