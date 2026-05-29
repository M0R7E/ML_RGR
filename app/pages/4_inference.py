import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import path_setup  # noqa: F401

import io
import pickle

import pandas as pd
import streamlit as st

from backend.classifier_tools import STORAGE_DIR, load_registry, proba_from_estimator
from backend.csgo_dataset import TARGET, basic_clean

st.set_page_config(page_title="Инференс", layout="wide")
st.title("Инференс")

registry = load_registry()
models = registry.get("models") or {}
if not models:
    st.error("Сначала обучите модели: `python jobs/fit_all_models.py`")
    st.stop()


def load_model(file_name: str):
    with open(STORAGE_DIR / file_name, "rb") as f:
        return pickle.load(f)


model_id = st.selectbox(
    "Модель",
    list(models.keys()),
    format_func=lambda key: models[key]["name"],
)
model = load_model(models[model_id]["file"])

metrics = models[model_id].get("metrics", {})
if metrics:
    c1, c2, c3 = st.columns(3)
    c1.metric("ROC-AUC (test)", f"{metrics.get('auc', 0):.3f}")
    c2.metric("AP (test)", f"{metrics.get('ap', 0):.3f}")
    c3.metric("F1 (test)", f"{metrics.get('f1', 0):.3f}")

tab_csv, tab_form = st.tabs(["CSV", "Ручной ввод"])

with tab_csv:
    up = st.file_uploader("Загрузить CSV", type=["csv"])
    if up is not None:
        raw = basic_clean(pd.read_csv(io.BytesIO(up.read())))
        if TARGET in raw.columns:
            raw = raw.drop(columns=[TARGET])
        st.dataframe(raw.head(10), use_container_width=True)
        if st.button("Предсказать", key="pred_csv"):
            proba = proba_from_estimator(model, raw)
            out = raw.copy()
            out["proba_bomb_planted"] = proba
            out["pred_bomb_planted"] = (proba >= 0.5).astype(int)
            st.dataframe(out.head(30), use_container_width=True)
            st.download_button(
                "Скачать результат",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
            )

with tab_form:
    st.caption("Заполните параметры одного раунда и нажмите «Рассчитать прогноз».")

    MAPS = [
        "de_dust2",
        "de_mirage",
        "de_inferno",
        "de_nuke",
        "de_overpass",
        "de_vertigo",
        "de_cache",
        "de_train",
    ]

    with st.form("round_form", border=True):
        st.markdown("**Состояние раунда**")
        r1, r2, r3 = st.columns(3)
        with r1:
            time_left = st.slider("Оставшееся время, с", 0.0, 200.0, 115.0, step=1.0)
        with r2:
            ct_score = st.number_input("Счёт CT", 0, 30, 0)
        with r3:
            t_score = st.number_input("Счёт T", 0, 30, 0)
        map_name = st.selectbox("Карта", MAPS, index=MAPS.index("de_dust2"))

        st.divider()
        st.markdown("**Команда CT**")
        ct1, ct2, ct3 = st.columns(3)
        with ct1:
            ct_players_alive = st.selectbox("Игроков в живых", [0, 1, 2, 3, 4, 5], index=5)
            ct_health = st.number_input("Суммарное HP", 0, 500, 500, step=10)
        with ct2:
            ct_armor = st.number_input("Броня", 0, 500, 0, step=10)
            ct_helmets = st.selectbox("Шлемы", [0, 1, 2, 3, 4, 5], index=0)
        with ct3:
            ct_money = st.number_input("Деньги, $", 0, 50000, 4000, step=100)
            ct_defuse_kits = st.selectbox("Наборы для разминирования", [0, 1, 2, 3, 4, 5], index=0)

        st.divider()
        st.markdown("**Команда T**")
        t1, t2, t3 = st.columns(3)
        with t1:
            t_players_alive = st.selectbox("Игроков в живых ", [0, 1, 2, 3, 4, 5], index=5, key="t_alive")
            t_health = st.number_input("Суммарное HP ", 0, 500, 500, step=10, key="t_hp")
        with t2:
            t_armor = st.number_input("Броня ", 0, 500, 0, step=10, key="t_armor")
            t_helmets = st.selectbox("Шлемы ", [0, 1, 2, 3, 4, 5], index=0, key="t_helmets")
        with t3:
            t_money = st.number_input("Деньги, $ ", 0, 50000, 4000, step=100, key="t_money")

        submitted = st.form_submit_button("Рассчитать прогноз", use_container_width=True)

    if submitted:
        row = pd.DataFrame(
            [
                {
                    "time_left": time_left,
                    "ct_score": ct_score,
                    "t_score": t_score,
                    "map": map_name,
                    "ct_health": ct_health,
                    "t_health": t_health,
                    "ct_armor": ct_armor,
                    "t_armor": t_armor,
                    "ct_money": ct_money,
                    "t_money": t_money,
                    "ct_helmets": ct_helmets,
                    "t_helmets": t_helmets,
                    "ct_defuse_kits": ct_defuse_kits,
                    "ct_players_alive": ct_players_alive,
                    "t_players_alive": t_players_alive,
                }
            ]
        )

        p = float(proba_from_estimator(model, row)[0])
        pred = int(p >= 0.5)

        st.markdown("#### Результат")
        out1, out2, out3 = st.columns(3)
        out1.metric("P(bomb_planted = 1)", f"{p:.1%}")
        out2.metric("Класс (порог 0.5)", "Установлена" if pred else "Не установлена")
        out3.metric("Уверенность", f"{max(p, 1 - p):.1%}")

        if pred:
            st.warning("Модель считает, что бомба с высокой вероятностью будет установлена в этом раунде.")
        else:
            st.info("Модель считает, что бомба, скорее всего, не будет установлена в этом раунде.")
