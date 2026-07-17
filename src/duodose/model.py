"""DuoDose classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


CLASS_LABELS = ("clean", "heterotypic_doublet", "homotypic_doublet", "low_quality")


def _numeric_frame(X: pd.DataFrame | np.ndarray, columns: Sequence[str] | None = None) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        frame = X.copy()
    else:
        frame = pd.DataFrame(X, columns=columns)
    frame = frame.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    return frame


class DuoDoseBaseClassifier:
    """Interpretable baseline classifier for DuoDose engineered features."""

    def __init__(
        self,
        n_estimators: int = 200,
        random_state: int = 0,
        class_labels: Sequence[str] = CLASS_LABELS,
    ) -> None:
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.class_labels = tuple(class_labels)
        self.feature_names_: list[str] | None = None
        self.pipeline_: Pipeline | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: Sequence[str]) -> "DuoDoseBaseClassifier":
        """Fit the base random forest classifier."""

        frame = _numeric_frame(X)
        y_arr = np.asarray(y, dtype=object)
        if np.unique(y_arr).size < 2:
            raise ValueError("DuoDoseBaseClassifier needs at least two training classes.")
        self.feature_names_ = list(frame.columns)
        clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            class_weight="balanced_subsample",
            random_state=self.random_state,
            n_jobs=-1,
            min_samples_leaf=2,
        )
        self.pipeline_ = Pipeline([("imputer", SimpleImputer(strategy="median")), ("classifier", clf)])
        self.pipeline_.fit(frame, y_arr)
        self.classes_ = np.asarray(self.pipeline_.named_steps["classifier"].classes_, dtype=object)
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return probabilities for all DuoDose base classes."""

        if self.pipeline_ is None or self.feature_names_ is None:
            raise RuntimeError("Classifier is not fitted.")
        frame = _numeric_frame(X).reindex(columns=self.feature_names_, fill_value=0.0)
        raw = self.pipeline_.predict_proba(frame)
        classes = np.asarray(self.pipeline_.named_steps["classifier"].classes_, dtype=object)
        probs = pd.DataFrame(0.0, index=frame.index, columns=list(self.class_labels))
        for i, label in enumerate(classes):
            probs[str(label)] = raw[:, i]
        row_sum = probs.sum(axis=1).replace(0.0, 1.0)
        return probs.div(row_sum, axis=0)


try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover - torch is optional at import time
    torch = None
    nn = object
    DataLoader = None
    TensorDataset = None


if torch is not None:

    class DuoDoseNet(nn.Module):
        """Lightweight MLP for optional DuoDose-Net predictions."""

        def __init__(self, input_dim: int, output_dim: int = 4, dropout: float = 0.15) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, output_dim),
            )

        def forward(self, x):  # type: ignore[override]
            return self.network(x)

else:

    class DuoDoseNet:  # type: ignore[no-redef]
        """Placeholder raised when PyTorch is unavailable."""

        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("PyTorch is required for DuoDoseNet.")


@dataclass
class DuoDoseNetResult:
    model: DuoDoseNet
    scaler: StandardScaler
    imputer: SimpleImputer
    label_encoder: LabelEncoder
    feature_names: list[str]


def fit_duodose_net(
    X: pd.DataFrame,
    y: Sequence[str],
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    random_state: int = 0,
    dropout: float = 0.15,
) -> DuoDoseNetResult:
    """Fit the optional PyTorch MLP classifier."""

    if torch is None:
        raise ImportError("PyTorch is required for fit_duodose_net.")
    torch.manual_seed(random_state)
    frame = _numeric_frame(X)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_np = scaler.fit_transform(imputer.fit_transform(frame)).astype(np.float32)
    encoder = LabelEncoder()
    y_np = encoder.fit_transform(np.asarray(y, dtype=object)).astype(np.int64)
    model = DuoDoseNet(input_dim=X_np.shape[1], output_dim=len(encoder.classes_), dropout=dropout)

    counts = np.bincount(y_np)
    weights = counts.sum() / np.maximum(counts, 1)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataset = TensorDataset(torch.tensor(X_np), torch.tensor(y_np))
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
    return DuoDoseNetResult(model=model, scaler=scaler, imputer=imputer, label_encoder=encoder, feature_names=list(frame.columns))


def predict_proba_duodose_net(result: DuoDoseNetResult, X: pd.DataFrame, class_labels: Iterable[str] = CLASS_LABELS) -> pd.DataFrame:
    """Predict probabilities with a fitted DuoDose-Net result."""

    if torch is None:
        raise ImportError("PyTorch is required for predict_proba_duodose_net.")
    frame = _numeric_frame(X).reindex(columns=result.feature_names, fill_value=0.0)
    X_np = result.scaler.transform(result.imputer.transform(frame)).astype(np.float32)
    result.model.eval()
    with torch.no_grad():
        logits = result.model(torch.tensor(X_np))
        raw = torch.softmax(logits, dim=1).cpu().numpy()
    probs = pd.DataFrame(0.0, index=frame.index, columns=list(class_labels))
    for i, label in enumerate(result.label_encoder.classes_):
        probs[str(label)] = raw[:, i]
    row_sum = probs.sum(axis=1).replace(0.0, 1.0)
    return probs.div(row_sum, axis=0)

