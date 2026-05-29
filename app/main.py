import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import path_setup  # noqa: F401

import streamlit as st

st.set_page_config(
    page_title="CS:GO — bomb_planted",
    page_icon="💣",
    layout="wide",
)

st.title("Дашборд классификации CS:GO")
st.markdown(
    """
Приложение предсказывает вероятность установки бомбы (`bomb_planted`) по состоянию раунда.

**Разделы:**
- сведения о разработчике;
- описание датасета и метрик моделей;
- визуализации;
- инференс по CSV или ручному вводу.

Обучение моделей выполняется отдельным заданием: `python jobs/fit_all_models.py`.
"""
)
