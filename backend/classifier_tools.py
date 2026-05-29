from __future__ import annotations

import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from backend.csgo_dataset import infer_feature_types

ROOT = Path(__file__).resolve().parents[1]
STORAGE_DIR = ROOT / "storage" / "classifiers"
REGISTRY_PATH = STORAGE_DIR / "registry.txt"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Metrics:
    auc: float
    ap: float
    f1: float

    def as_dict(self) -> dict[str, float]:
        return {"auc": self.auc, "ap": self.ap, "f1": self.f1}


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols, cat_cols = infer_feature_types(X)
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
    )


def proba_from_estimator(est, X: pd.DataFrame) -> np.ndarray:
    if hasattr(est, "predict_proba"):
        return np.asarray(est.predict_proba(X))[:, 1]
    if hasattr(est, "decision_function"):
        score = np.asarray(est.decision_function(X), dtype=float)
        return 1.0 / (1.0 + np.exp(-score))
    raise TypeError("Модель должна поддерживать predict_proba или decision_function")


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, thr: float = 0.5) -> Metrics:
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)
    y_pred = (y_proba >= float(thr)).astype(int)
    return Metrics(
        auc=float(roc_auc_score(y_true, y_proba)),
        ap=float(average_precision_score(y_true, y_proba)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
    )


def save_pickle(obj, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _cast_value(raw: str):
    if raw.isdigit():
        return int(raw)
    try:
        return float(raw)
    except ValueError:
        return raw


def format_registry(meta: dict) -> str:
    lines = [
        "# Реестр обученных классификаторов CS:GO",
        f"target={meta['target']}",
        f"rows={meta['rows']}",
        f"positive_rate={meta['positive_rate']}",
        "",
        "# id | название | файл | auc | ap | f1",
    ]
    for model_id, info in meta["models"].items():
        m = info["metrics"]
        lines.append(
            f"{model_id} | {info['name']} | {info['file']} | "
            f"{m['auc']:.6f} | {m['ap']:.6f} | {m['f1']:.6f}"
        )
    return "\n".join(lines) + "\n"


def parse_registry(text: str) -> dict:
    meta: dict = {"models": {}}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "|" in line:
            parts = [part.strip() for part in line.split("|")]
            if len(parts) < 6:
                continue
            model_id, title, file_name, auc, ap, f1 = parts[:6]
            meta["models"][model_id] = {
                "name": title,
                "file": file_name,
                "metrics": {"auc": float(auc), "ap": float(ap), "f1": float(f1)},
            }
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            meta[key.strip()] = _cast_value(value.strip())
    return meta


def save_registry(meta: dict) -> None:
    REGISTRY_PATH.write_text(format_registry(meta), encoding="utf-8")


@lru_cache(maxsize=1)
def load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        return {}
    return parse_registry(REGISTRY_PATH.read_text(encoding="utf-8"))
