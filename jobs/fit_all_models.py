from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from backend.classifier_tools import (
    STORAGE_DIR,
    compute_metrics,
    proba_from_estimator,
    save_pickle,
    save_registry,
    build_preprocessor,
)
from backend.csgo_dataset import TARGET, basic_clean, load_raw, make_xy, safe_clip_time_left, split_train_test

RANDOM_STATE = 42


def make_pipeline(pre, estimator) -> Pipeline:
    return Pipeline([("prep", pre), ("clf", estimator)])


def build_model_catalog(pre) -> dict[str, tuple[str, Pipeline]]:
    return {
        "logreg": (
            "Логистическая регрессия",
            make_pipeline(pre, LogisticRegression(max_iter=1500, class_weight="balanced")),
        ),
        "random_forest": (
            "Случайный лес",
            make_pipeline(
                pre,
                RandomForestClassifier(
                    n_estimators=120,
                    max_depth=12,
                    class_weight="balanced_subsample",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ),
        "catboost": (
            "CatBoost",
            make_pipeline(
                pre,
                CatBoostClassifier(
                    iterations=400,
                    depth=6,
                    learning_rate=0.06,
                    random_seed=RANDOM_STATE,
                    verbose=False,
                ),
            ),
        ),
        "gbm": (
            "Градиентный бустинг",
            make_pipeline(pre, GradientBoostingClassifier(random_state=RANDOM_STATE)),
        ),
        "adaboost": (
            "AdaBoost",
            make_pipeline(
                pre,
                AdaBoostClassifier(
                    estimator=DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE),
                    n_estimators=80,
                    learning_rate=0.8,
                    random_state=RANDOM_STATE,
                ),
            ),
        ),
    }


def main() -> None:
    df = basic_clean(load_raw())
    df = safe_clip_time_left(df)
    X, y = make_xy(df)

    split = split_train_test(X, y, test_size=0.2, random_state=RANDOM_STATE)
    pre = build_preprocessor(split.X_train)
    catalog = build_model_catalog(pre)

    registry_models: dict[str, dict] = {}
    best_id = ""
    best_ap = -1.0

    for model_id, (title, pipe) in catalog.items():
        print(f"Обучение: {title} ({model_id})")
        pipe.fit(split.X_train, split.y_train)

        metrics = compute_metrics(split.y_test.to_numpy(), proba_from_estimator(pipe, split.X_test))
        file_name = f"{model_id}.pkl"
        save_pickle(pipe, STORAGE_DIR / file_name)

        registry_models[model_id] = {
            "name": title,
            "file": file_name,
            "metrics": metrics.as_dict(),
        }
        print(model_id, metrics.as_dict())

        if metrics.ap > best_ap:
            best_ap = metrics.ap
            best_id = model_id

    save_registry(
        {
            "target": TARGET,
            "rows": int(len(df)),
            "positive_rate": round(float(y.mean()), 4),
            "models": registry_models,
        }
    )

    best_pipe = catalog[best_id][1]
    y_hat = (proba_from_estimator(best_pipe, split.X_test) >= 0.5).astype(int)
    print(f"\nЛучшая модель по AP: {best_id}")
    print(classification_report(split.y_test, y_hat, digits=3, zero_division=0))


if __name__ == "__main__":
    main()
