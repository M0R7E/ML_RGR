from __future__ import annotations

from pathlib import Path

import streamlit as st

st.set_page_config(page_title="О разработчике", layout="wide")

st.title("Страница 1. Сведения о разработчике")

PHOTO_PATH = Path(__file__).resolve().parents[2] / "assets" / "dev.png"
if PHOTO_PATH.exists():
    st.image(str(PHOTO_PATH), width=280)

st.markdown(
    """
**ФИО:** Завьялов Егор Сергеевич  
**Группа:** ФИТ-242  
**Кафедра:** ПМФИ (Прикладная математика и фундаментальная информатика)  
**Тема РГР:** разработка веб-приложения (дашборда) для инференса моделей машинного обучения  
**Задача:** классификация события `bomb_planted` по состоянию раунда CS:GO  
"""
)
