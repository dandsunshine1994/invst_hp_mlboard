# streamlit_app/app.py
from __future__ import annotations
from pathlib import Path
import io, os, json, importlib, sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ==== 基本设置 ====
st.set_page_config(page_title="Investor Happiness · Model Board", layout="wide")
def _ver(m):
    try: return importlib.import_module(m).__version__
    except Exception: return "N/A"
st.caption(f"Py {sys.version.split()[0]} | streamlit {_ver('streamlit')} | "
           f"pandas {_ver('pandas')} | numpy {_ver('numpy')} | altair {_ver('altair')}")

ART = Path(__file__).parent / "artifacts"
PROB_DIR = ART / "probas"
MODEL_DIR = ART / "models"
MET_DIR = ART / "metrics"

# ==== 读文件的薄封装 ====
@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource(show_spinner=False)
def lazy_load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as d:
        payload = {k: d[k] for k in d.files}
    return payload

def file_exists(p: Path) -> bool:
    try:
        return p.exists()
    except Exception:
        return False

# ==== 尝试加载核心产物 ====
missing = []
base_path = ART / "base.csv"
cov_path  = ART / "cov.csv"
aurc_path = ART / "aurc.csv"
ext_path  = ART / "ext_reports.json"
ext3_meta = MET_DIR / "ext3_meta.json"
fi_path   = MET_DIR / "feature_importance_lgb.csv"

for p in [base_path, cov_path, aurc_path, ext_path]:
    if not file_exists(p): missing.append(str(p))

if missing:
    st.error("缺少以下必要产物，请确认仓库 `streamlit_app/artifacts/` 中存在：\n" + "\n".join(missing))
    st.stop()

base_df = load_csv(base_path)
cov_df  = load_csv(cov_path)
aurc_df = load_csv(aurc_path)
ext_reports = load_json(ext_path)
ext3_info = load_json(ext3_meta) if file_exists(ext3_meta) else None
fi_df = load_csv(fi_path) if file_exists(fi_path) else None

# 可用模型列表（来自 probas/*.npz）
model_npzs = sorted([p.stem for p in PROB_DIR.glob("*.npz")]) if PROB_DIR.exists() else []

# ==== 侧栏 ====
with st.sidebar:
    st.header("选项")
    metric_cols = [c for c in base_df.columns if c not in ("model",)]
    default_metric = "base_macro_f1%" if "base_macro_f1%" in metric_cols else metric_cols[0]
    topk = st.slider("Top-K（排行榜）", 3, min(10, len(base_df)), min(5, len(base_df)))
    sort_col = st.selectbox("排序指标", metric_cols, index=metric_cols.index(default_metric))
    sort_asc = st.checkbox("升序", value=False)
    st.divider()
    model_pick = st.selectbox("选择模型（概率诊断/详情）", sorted(base_df["model"].unique()))
    st.caption("若要概率诊断，请确保 artifacts/probas/{模型名}.npz 已存在。")
    st.divider()
    st.caption("下载原始产物")
    c1,c2,c3 = st.columns(3)
    with c1:
        st.download_button("⬇ base.csv", data=base_path.read_bytes(), file_name="base.csv")
    with c2:
        st.download_button("⬇ cov.csv",  data=cov_path.read_bytes(),  file_name="cov.csv")
    with c3:
        st.download_button("⬇ aurc.csv", data=aurc_path.read_bytes(), file_name="aurc.csv")
    st.download_button("⬇ ext_reports.json", data=ext_path.read_bytes(), file_name="ext_reports.json")

st.title("Investor Happiness · 离线评测看板")

# ==== Tab 结构 ====
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏆 排行榜", "🎯 覆盖率曲线", "🏁 AURC", "🔍 模型详情", "📈 概率诊断", "🧭 特征重要性"
])

# === Tab1: 排行榜 ===
with tab1:
    st.subheader("🏆 模型排行榜")
    board = base_df.copy()
    board = board.sort_values(sort_col, ascending=sort_asc).reset_index(drop=True)
    st.dataframe(board.style.format(precision=2), use_container_width=True)
    st.caption("注：百分制列如 base_acc%、base_macro_f1%、prauc_c0% 等已为 0-100 尺度。")

    st.markdown("**Top-K（用于下游导出/对比）**")
    topk_df = board.head(topk)
    st.dataframe(topk_df, use_container_width=True)

# === Tab2: 覆盖率曲线 ===
with tab2:
    import altair as alt
    st.subheader("🎯 覆盖率 vs 指标（policy: top1 / margin）")
    pols = st.multiselect("选择 policy", sorted(cov_df["policy"].unique()), default=list(cov_df["policy"].unique()))
    metric_choice = st.selectbox("指标", ["acc%", "macro_f1%", "qwk"], index=1)
    cov_plot_df = cov_df[cov_df["policy"].isin(pols)]
    cov_plot_df["coverage"] = cov_plot_df["coverage"].astype(float)
    cov_plot_df = cov_plot_df[cov_plot_df["model"].isin(topk_df["model"])]
    base = alt.Chart(cov_plot_df).encode(
        x=alt.X("coverage:Q", title="Coverage (%)"),
        y=alt.Y(f"{metric_choice}:Q"),
        color="model:N",
        strokeDash="policy:N",
        tooltip=list(cov_plot_df.columns)
    )
    st.altair_chart(base.mark_line().interactive().properties(height=420), use_container_width=True)

# === Tab3: AURC ===
with tab3:
    import altair as alt
    st.subheader("🏁 RC 曲线面积（AURC，越小越好）")
    aurc_plot = aurc_df.copy()
    aurc_plot["AURC"] = aurc_plot["AURC"].astype(float)
    pick_policy = st.multiselect("选择 policy", sorted(aurc_plot["policy"].unique()),
                                 default=list(aurc_plot["policy"].unique()))
    sub = aurc_plot[aurc_plot["policy"].isin(pick_policy)]
    chart = alt.Chart(sub).mark_bar().encode(
        x=alt.X("AURC:Q", sort="ascending"),
        y=alt.Y("model:N", sort="-x"),
        color="policy:N",
        tooltip=list(sub.columns)
    ).properties(height=420)
    st.altair_chart(chart, use_container_width=True)

# === Tab4: 模型详情（极端类报告 + ext3 meta） ===
with tab4:
    st.subheader(f"🔍 模型详情：{model_pick}")
    # 极端类报告（如果存在）
    if model_pick in ext_reports:
        st.markdown("**极端类报告（百分制）**")
        st.json(ext_reports[model_pick])
    else:
        st.info("没有找到该模型的极端类报告（ext_reports.json）。")

    # ext3 阈值元信息
    if ext3_info is not None and model_pick == "ext3":
        st.markdown("**ext3 阈值/约束**")
        st.json(ext3_info)
    elif model_pick == "ext3":
        st.info("未找到 ext3_meta.json。")

    # 双模型对比（与另一个模型差值）
    st.markdown("**模型对比（差值：当前 - 备选）**")
    other = st.selectbox("选择备选模型", [m for m in base_df["model"].unique() if m != model_pick])
    cur = base_df[base_df["model"] == model_pick].set_index("model")
    oth = base_df[base_df["model"] == other].set_index("model")
    common_cols = sorted(set(cur.columns) & set(oth.columns))
    delta = (cur[common_cols].iloc[0] - oth[common_cols].iloc[0]).to_frame(name="Δ(Current - Other)")
    st.dataframe(delta.style.format(precision=2), use_container_width=True)

# === Tab5: 概率诊断（需要 probas/{model}.npz） ===
with tab5:
    st.subheader(f"📈 概率诊断：{model_pick}")
    npz_path = PROB_DIR / f"{model_pick}.npz"
    if not file_exists(npz_path):
        st.info(f"未找到 {npz_path.name}，请确认导出。")
    else:
        data = lazy_load_npz(npz_path)
        proba = data.get("proba", None)
        if proba is None:
            st.info("该模型没有存概率（proba），仅保存了 y_pred。")
        else:
            pmax = proba.max(axis=1)
            psort = np.sort(proba, axis=1)
            margin = psort[:, -1] - psort[:, -2]

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Top-1 置信度分布**")
                fig, ax = plt.subplots()
                ax.hist(pmax, bins=25)
                ax.set_xlabel("max probability")
                ax.set_ylabel("count")
                st.pyplot(fig, clear_figure=True)
            with c2:
                st.markdown("**Top1-Top2 间距（margin）分布**")
                fig, ax = plt.subplots()
                ax.hist(margin, bins=25)
                ax.set_xlabel("p1 - p2")
                ax.set_ylabel("count")
                st.pyplot(fig, clear_figure=True)

            st.caption("提示：若你也保存了 y_true，可扩展此页做校准曲线、PR/ROC 等。")

# === Tab6: 特征重要性（LGBM 示例） ===
with tab6:
    st.subheader("🧭 特征重要性（LGBM 示例）")
    if fi_df is None or fi_df.empty:
        st.info("未找到 metrics/feature_importance_lgb.csv。仅当导出的模型为带预处理的 LGBM pipeline 且保存了重要性才显示。")
    else:
        import altair as alt
        topn = st.slider("展示前 N", 10, min(50, len(fi_df)), min(25, len(fi_df)))
        plot_df = fi_df.sort_values("importance", ascending=False).head(topn)
        chart = alt.Chart(plot_df).mark_bar().encode(
            x=alt.X("importance:Q"),
            y=alt.Y("feature:N", sort="-x"),
            tooltip=["feature","importance"]
        ).properties(height=600)
        st.altair_chart(chart, use_container_width=True)

# ==== 可选：离线推断（仅当 models/*.joblib 存在时启用） ====
if MODEL_DIR.exists() and any(MODEL_DIR.glob("*.joblib")):
    st.divider()
    st.subheader("🧪 离线推断（演示）")
    st.caption("上传少量 CSV（列需与训练一致）。本功能依赖你导出的 pipeline（含预处理）。")
    up = st.file_uploader("上传 CSV（小于 5MB）", type=["csv"])
    if up is not None:
        import joblib
        pick_model = st.selectbox(
            "选择已导出模型", [p.name for p in MODEL_DIR.glob("*.joblib")]
        )
        mdl = joblib.load(MODEL_DIR / pick_model)
        df = pd.read_csv(up)
        try:
            yhat = mdl.predict(df)
            st.write("预测前 20 行：", pd.DataFrame({"pred": yhat}).head(20))
            if hasattr(mdl, "predict_proba"):
                proba = mdl.predict_proba(df)
                st.write("概率（前 5 行）：", pd.DataFrame(proba).head(5))
        except Exception as e:
            st.error(f"推断失败：{e}")
