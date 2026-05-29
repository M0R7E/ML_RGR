from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "csgo_done.csv"
TARGET = "bomb_planted"
DROP_COLS = ("Unnamed: 0",)
CAT_COLS = ("map",)


@dataclass(frozen=True)
class Split:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def load_raw(path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or DATA_PATH)


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in DROP_COLS:
        if col in out.columns:
            out = out.drop(columns=[col])
    return out


def make_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET not in df.columns:
        raise ValueError(f"Нет целевого столбца `{TARGET}`")
    y = pd.to_numeric(df[TARGET], errors="raise").astype(int)
    return df.drop(columns=[TARGET]), y


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Split:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return Split(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def infer_feature_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    cat = [c for c in CAT_COLS if c in df.columns]
    num = [c for c in df.columns if c not in set(cat)]
    return num, cat


def safe_clip_time_left(df: pd.DataFrame) -> pd.DataFrame:
    if "time_left" not in df.columns:
        return df
    out = df.copy()
    out["time_left"] = np.clip(pd.to_numeric(out["time_left"], errors="coerce"), 0.0, None)
    return out
