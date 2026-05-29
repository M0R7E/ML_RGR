import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import path_setup  # noqa: F401

import pandas as pd
import streamlit as st

from backend.classifier_tools import load_registry
from backend.csgo_dataset import DATA_PATH, TARGET, basic_clean, load_raw

st.set_page_config(page_title="Датасет", layout="wide")
st.title("Датасет")

st.markdown(f"Целевая переменная: **{TARGET}** (0 — бомба не установлена, 1 — установлена).")

registry = load_registry()
if registry:
    st.subheader("Сводка")
    st.write(
        {
            "строк": registry.get("rows"),
            "доля класса 1": registry.get("positive_rate"),
            "моделей": len(registry.get("models", {})),
        }
    )

    rows = []
    for model_id, info in registry.get("models", {}).items():
        m = info.get("metrics", {})
        rows.append(
            {
                "id": model_id,
                "модель": info.get("name"),
                "auc": m.get("auc"),
                "ap": m.get("ap"),
                "f1": m.get("f1"),
            }
        )
    if rows:
        st.subheader("Метрики на тесте")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.subheader("Пример данных")
df = basic_clean(load_raw(DATA_PATH))
st.dataframe(df.head(20), use_container_width=True)
st.dataframe(df.describe().T, use_container_width=True)
