import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import path_setup  # noqa: F401

import plotly.express as px
import streamlit as st

from backend.csgo_dataset import DATA_PATH, TARGET, basic_clean, load_raw

st.set_page_config(page_title="Визуализации", layout="wide")
st.title("Визуализации")

df = basic_clean(load_raw(DATA_PATH))

st.subheader("Распределение класса")
st.plotly_chart(px.histogram(df, x=TARGET, text_auto=True), use_container_width=True)

st.subheader("Доля bomb_planted=1 по картам")
rate_by_map = df.groupby("map")[TARGET].mean().sort_values(ascending=False).reset_index()
st.plotly_chart(px.bar(rate_by_map, x="map", y=TARGET), use_container_width=True)

st.subheader("time_left по классам")
st.plotly_chart(px.box(df, x=TARGET, y="time_left", points="outliers"), use_container_width=True)

st.subheader("Распределение признака")
feature = st.selectbox(
    "Признак",
    ["time_left", "ct_money", "t_money", "ct_health", "t_health", "ct_players_alive", "t_players_alive"],
)
st.plotly_chart(
    px.histogram(df, x=feature, color=TARGET, barmode="overlay", nbins=50, opacity=0.6),
    use_container_width=True,
)

st.subheader("Корреляции")
num_cols = [c for c in df.columns if c != "map"]
st.plotly_chart(
    px.imshow(df[num_cols].corr(numeric_only=True), color_continuous_scale="RdBu_r", zmin=-1, zmax=1),
    use_container_width=True,
)
