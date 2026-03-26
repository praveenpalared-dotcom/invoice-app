import os
import sys
import tempfile
import warnings
import traceback

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from parser import parse_invoices_to_df
from anomaly_detector import detect_anomalies

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Invoice Anomaly Detector",
    page_icon="🧾",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'IBM Plex Mono', monospace !important;
}
.stApp {
    background: #0f1117;
    color: #e8e8e8;
}
.metric-card {
    background: #1a1d27;
    border: 1px solid #2a2d3e;
    border-radius: 8px;
    padding: 20px;
    text-align: center;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.2rem;
    font-weight: 600;
    color: #7eb8f7;
}
.metric-label {
    font-size: 0.8rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}
.anomaly-badge {
    background: #3d1a1a;
    border: 1px solid #c0392b;
    color: #e74c3c;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-family: 'IBM Plex Mono', monospace;
}
.normal-badge {
    background: #1a3d2a;
    border: 1px solid #27ae60;
    color: #2ecc71;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-family: 'IBM Plex Mono', monospace;
}
.upload-hint {
    color: #888;
    font-size: 0.85rem;
    margin-top: -10px;
}
.section-divider {
    border: none;
    border-top: 1px solid #2a2d3e;
    margin: 30px 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown("## 🧾 Invoice Anomaly Detector")
st.markdown("Upload invoice images — the pipeline extracts fields, runs ML, and flags suspicious invoices.")
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Upload Invoice Images",
    type=["png", "jpg", "jpeg", "tiff", "bmp"],
    accept_multiple_files=True,
    help="PNG, JPG, TIFF, BMP supported. Upload as many as you like.",
)
st.markdown('<p class="upload-hint">Supports PNG, JPG, TIFF, BMP — upload multiple files at once</p>',
            unsafe_allow_html=True)

if not uploaded_files:
    st.info("👆 Upload one or more invoice images above to get started.")
    st.stop()

if st.button("⚡ Run Pipeline", type="primary", use_container_width=True):
    with tempfile.TemporaryDirectory() as tmpdir:
        image_paths = []
        for uf in uploaded_files:
            path = os.path.join(tmpdir, uf.name)
            with open(path, "wb") as f:
                f.write(uf.read())
            image_paths.append(path)

        progress = st.progress(0, text="Starting pipeline…")

        try:
            progress.progress(20, text="📄 Running OCR and parsing fields…")
            df = parse_invoices_to_df(image_paths)

            progress.progress(60, text="🤖 Detecting anomalies…")
            df = detect_anomalies(df)

            progress.progress(100, text="✅ Done!")
            progress.empty()

        except Exception as e:
            progress.empty()
            st.error(f"Pipeline error: {e}")
            st.code(traceback.format_exc())
            st.stop()

    st.session_state["df"] = df

if "df" not in st.session_state:
    st.stop()

df = st.session_state["df"]

st.markdown("### 📊 Summary")
c1, c2, c3, c4 = st.columns(4)

total_invoices = len(df)
total_anomalies = int(df["is_anomaly"].sum())
total_value = df["grand_total"].sum()
unique_vendors = df["vendor_name"].nunique()

for col, value, label in [
    (c1, total_invoices,                                                       "Invoices Processed"),
    (c2, total_anomalies,                                                      "Anomalies Flagged"),
    (c3, f"${total_value:,.2f}" if pd.notna(total_value) else "N/A",          "Total Invoice Value"),
    (c4, unique_vendors,                                                       "Unique Vendors"),
]:
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

st.markdown("### 📋 Invoice Results")

display_cols = ["invoice_number", "invoice_date", "vendor_name",
                "grand_total", "calculated_total", "is_anomaly", "anomaly_reason"]
display_df = df[[c for c in display_cols if c in df.columns]].copy()

def highlight_anomaly(row):
    if row.get("is_anomaly", False):
        return ["background-color: #2d1515; color: #f5b7b1"] * len(row)
    return [""] * len(row)

st.dataframe(
    display_df.style.apply(highlight_anomaly, axis=1),
    use_container_width=True,
    height=400,
)

csv_data = display_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download Results as CSV",
    data=csv_data,
    file_name="invoice_results.csv",
    mime="text/csv",
    use_container_width=True,
)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

flagged = df[df["is_anomaly"]]
if not flagged.empty:
    st.markdown("### ⚠️ Flagged Invoice Details")
    for _, row in flagged.iterrows():
        with st.expander(f"🔴 {row['invoice_number']}  —  {row['vendor_name']}"):
            col1, col2 = st.columns(2)
            col1.metric("Grand Total",       f"${row['grand_total']:,.2f}"      if pd.notna(row["grand_total"])      else "N/A")
            col1.metric("Calculated Total",  f"${row['calculated_total']:,.2f}" if pd.notna(row["calculated_total"]) else "N/A")
            col2.metric("Invoice Date",      row["invoice_date"])
            col2.metric("Z-Score",           f"{row['z_score']:.2f}"            if pd.notna(row.get("z_score"))      else "N/A")
            st.markdown(f"**Reason(s):** `{row['anomaly_reason']}`")
else:
    st.success("✅ No anomalies detected in the uploaded invoices.")

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

st.markdown("### 📈 Visualisations")

amounts = df[df["grand_total"].notna()].copy()

if not amounts.empty:
    tab1, tab2, tab3 = st.tabs(["Box Plot", "Z-Score Scatter", "Anomaly Reasons"])

    with tab1:
        sns.set_theme(style="darkgrid")
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor("#1a1d27")
        ax.set_facecolor("#1a1d27")

        vendor_names = amounts["vendor_name"].unique()
        data_per = [amounts[amounts["vendor_name"] == v]["grand_total"].values for v in vendor_names]
        bp = ax.boxplot(data_per, patch_artist=True,
                        medianprops={"color": "#f1c40f", "linewidth": 2})

        colours = sns.color_palette("cool", len(vendor_names))
        for patch, c in zip(bp["boxes"], colours):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        for element in ["whiskers", "caps", "fliers"]:
            for item in bp[element]:
                item.set_color("#aaa")

        for i, vendor in enumerate(vendor_names, 1):
            vdf = amounts[amounts["vendor_name"] == vendor]
            normal = vdf[~vdf["is_anomaly"]]["grand_total"]
            anomalous = vdf[vdf["is_anomaly"]]["grand_total"]
            rng = np.random.default_rng(42)
            ax.scatter([i + j for j in rng.uniform(-0.15, 0.15, len(normal))],
                       normal, alpha=0.5, s=20, color="#7eb8f7", zorder=3)
            ax.scatter([i + j for j in rng.uniform(-0.15, 0.15, len(anomalous))],
                       anomalous, alpha=0.95, s=60, color="#e74c3c",
                       marker="D", zorder=4)

        ax.set_xticks(range(1, len(vendor_names) + 1))
        ax.set_xticklabels(vendor_names, rotation=15, ha="right", color="#ccc")
        ax.set_ylabel("Invoice Amount ($)", color="#ccc")
        ax.set_title("Invoice Amounts per Vendor  (◆ = anomaly)", color="#eee")
        ax.tick_params(colors="#aaa")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2d3e")
        normal_p = mpatches.Patch(color="#7eb8f7", alpha=0.7, label="Normal")
        anomaly_p = mpatches.Patch(color="#e74c3c", alpha=0.9, label="Anomaly")
        ax.legend(handles=[normal_p, anomaly_p], facecolor="#1a1d27",
                  labelcolor="white", edgecolor="#2a2d3e")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tab2:
        z_data = amounts[amounts["z_score"].notna()].copy() if "z_score" in amounts else pd.DataFrame()
        if not z_data.empty:
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            fig2.patch.set_facecolor("#1a1d27")
            ax2.set_facecolor("#1a1d27")

            sc = ax2.scatter(range(len(z_data)), z_data["grand_total"],
                             c=z_data["z_score"], cmap="RdYlGn_r", s=60, alpha=0.85)
            cb = plt.colorbar(sc, ax=ax2)
            cb.set_label("|Z-score|", color="#ccc")
            cb.ax.yaxis.set_tick_params(color="#ccc")
            plt.setp(cb.ax.yaxis.get_ticklabels(), color="#ccc")

            top = z_data.nlargest(3, "z_score")
            for _, r in top.iterrows():
                loc = z_data.index.get_loc(r.name)
                ax2.annotate(r["invoice_number"], (loc, r["grand_total"]),
                             textcoords="offset points", xytext=(0, 8),
                             fontsize=8, color="#e74c3c", ha="center")

            ax2.set_xlabel("Invoice Index", color="#ccc")
            ax2.set_ylabel("Grand Total ($)", color="#ccc")
            ax2.set_title("Invoice Amounts coloured by Z-Score", color="#eee")
            ax2.tick_params(colors="#aaa")
            for spine in ax2.spines.values():
                spine.set_edgecolor("#2a2d3e")
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)
        else:
            st.info("Not enough data per vendor to compute Z-scores.")

    with tab3:
        if flagged.empty:
            st.success("No anomalies — nothing to chart.")
        else:
            reasons = []
            for r in flagged["anomaly_reason"]:
                reasons.extend([x.strip() for x in r.split(",")])
            reason_counts = pd.Series(reasons).value_counts()

            fig3, ax3 = plt.subplots(figsize=(8, 3))
            fig3.patch.set_facecolor("#1a1d27")
            ax3.set_facecolor("#1a1d27")
            reason_counts.sort_values().plot(
                kind="barh", ax=ax3, color="#e74c3c", edgecolor="#1a1d27", alpha=0.8)
            ax3.set_xlabel("Count", color="#ccc")
            ax3.set_title("Anomaly Flags by Type", color="#eee")
            ax3.tick_params(colors="#aaa")
            for spine in ax3.spines.values():
                spine.set_edgecolor("#2a2d3e")
            for bar in ax3.patches:
                ax3.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                         f"{int(bar.get_width())}", va="center", color="#eee", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)
