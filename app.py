# streamlit_app/app.py
from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Investor Happiness – Model Board", layout="wide")
st.title("Investor Happiness · 评测结果与可视化（离线产物）")

ART = Path(__file__).parent / "artifacts"
base_path   = ART / "base_models_base.csv"
cov_path    = ART / "base_models_cov.csv"
aurc_path   = ART / "base_models_aurc.csv"
extj_path   = ART / "base_models_ext_reports.json"

# 侧栏：允许用户手动替换文件（可选）
with st.sidebar:
    st.header("数据来源")
    st.caption("默认读取 repo 内 artifacts。也可上传新文件临时查看。")
    up_base = st.file_uploader("base_models_base.csv", type=["csv"])
    up_cov  = st.file_uploader("base_models_cov.csv", type=["csv"])
    up_aurc = st.file_uploader("base_models_aurc.csv", type=["csv"])
    up_ext  = st.file_uploader("base_models_ext_reports.json", type=["json"])

def read_csv(default_path, upload):
    if upload is not None:
        return pd.read_csv(upload)
    return pd.read_csv(default_path)

def read_json(default_path, upload):
    if upload is not None:
        return json.load(upload)
    with open(default_path, "r", encoding="utf-8") as f:
        return json.load(f)

# 读取
base_df = read_csv(base_path, up_base)
cov_df  = read_csv(cov_path,  up_cov)
aurc_df = read_csv(aurc_path, up_aurc)
ext_reports = read_json(extj_path, up_ext)

# 主看板
st.subheader("📊 基础指标（已含 TabPFN）")
st.dataframe(base_df.style.format(precision=2))

# AURC 排名
st.subheader("🏁 AURC（越小越好）")
st.dataframe(aurc_df)
try:
    import altair as alt
    chart = alt.Chart(aurc_df).mark_bar().encode(
        x=alt.X("AURC:Q", sort="ascending"),
        y=alt.Y("model:N", sort="-x"),
        color="policy:N",
        tooltip=list(aurc_df.columns)
    ).properties(height=400)
    st.altair_chart(chart, use_container_width=True)
except Exception:
    st.caption("Altair 不可用时将仅显示表格。")

# 覆盖率下的表现
st.subheader("🎯 覆盖率曲线（acc/macro_f1/qwk）")
model_sel = st.selectbox("选择模型", sorted(cov_df["model"].unique()))
sub = cov_df[cov_df["model"] == model_sel]
pivot = sub.pivot_table(index="coverage", columns="policy", values=["acc%","macro_f1%","qwk"])
st.dataframe(pivot)

# 极端类报告（百分制）
st.subheader("🧪 极端类报告（ext2）")
name_sel = st.selectbox("选择极端报告模型", sorted(ext_reports.keys()))
st.json(ext_reports[name_sel])
