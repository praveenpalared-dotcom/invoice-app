import sys
import os
import glob
import warnings
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from parser import parse_invoices_to_df
from anomaly_detector import detect_anomalies

warnings.filterwarnings("ignore")

DEFAULT_IMAGE_DIR = "sample_invoices"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def discover_images(directory: str) -> list[str]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.tiff", "*.tif", "*.bmp")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(set(paths))


def print_summary(df: pd.DataFrame) -> None:
    sep = "─" * 60
    print(f"\n{sep}")
    print("  PIPELINE SUMMARY")
    print(sep)
    print(f"  Total invoices processed : {len(df)}")
    print(f"  Anomalies flagged        : {df['is_anomaly'].sum()}")
    print(f"  Vendors found            : {df['vendor_name'].nunique()}")
    if df["grand_total"].notna().any():
        print(f"  Total invoice value      : ${df['grand_total'].sum():,.2f}")
        print(f"  Mean invoice amount      : ${df['grand_total'].mean():,.2f}")

    if df["is_anomaly"].any():
        print(f"\n  ⚠  Flagged invoices:")
        flagged = df[df["is_anomaly"]][["invoice_number", "vendor_name",
                                        "grand_total", "anomaly_reason"]]
        for _, row in flagged.iterrows():
            reason = textwrap.shorten(row["anomaly_reason"], width=55, placeholder="…")
            amt = f"${row['grand_total']:,.2f}" if pd.notna(row["grand_total"]) else "N/A"
            print(f"    [{row['invoice_number']}] {row['vendor_name']} | {amt}")
            print(f"      Reason: {reason}")
    print(sep)


def _style():
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({"figure.dpi": 120, "font.size": 10})


def plot_amount_distribution(df: pd.DataFrame) -> None:
    _style()
    amounts = df[df["grand_total"].notna()].copy()
    if amounts.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    vendor_names = amounts["vendor_name"].unique()
    data_per_vendor = [amounts[amounts["vendor_name"] == v]["grand_total"].values
                       for v in vendor_names]

    bp = ax.boxplot(data_per_vendor, patch_artist=True, notch=False,
                    medianprops={"color": "black", "linewidth": 2})

    colours = sns.color_palette("muted", len(vendor_names))
    for patch, colour in zip(bp["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.7)

    for i, vendor in enumerate(vendor_names, start=1):
        vendor_df = amounts[amounts["vendor_name"] == vendor]
        normal = vendor_df[~vendor_df["is_anomaly"]]["grand_total"]
        anomalous = vendor_df[vendor_df["is_anomaly"]]["grand_total"]

        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(normal))
        ax.scatter([i + j for j in jitter], normal,
                   alpha=0.5, s=25, zorder=3, color="steelblue")
        jitter2 = np.random.default_rng(99).uniform(-0.15, 0.15, len(anomalous))
        ax.scatter([i + j for j in jitter2], anomalous,
                   alpha=0.9, s=60, zorder=4, color="crimson",
                   marker="D", label="Anomaly" if i == 1 else "")

    ax.set_xticks(range(1, len(vendor_names) + 1))
    ax.set_xticklabels(vendor_names, rotation=15, ha="right")
    ax.set_ylabel("Invoice Amount ($)")
    ax.set_title("Invoice Amount Distribution per Vendor\n(Red diamonds = flagged anomalies)")
    normal_patch = mpatches.Patch(color="steelblue", alpha=0.7, label="Normal")
    anomaly_patch = mpatches.Patch(color="crimson", alpha=0.9, label="Anomaly")
    ax.legend(handles=[normal_patch, anomaly_patch])
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "boxplot_amounts.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  📊  Saved: {path}")


def plot_scatter_zscore(df: pd.DataFrame) -> None:
    _style()
    amounts = df[df["grand_total"].notna() & df["z_score"].notna()].copy()
    if amounts.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    sc = ax.scatter(
        range(len(amounts)),
        amounts["grand_total"],
        c=amounts["z_score"],
        cmap="RdYlGn_r",
        s=60,
        alpha=0.8,
        zorder=3,
    )
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("|Z-score| (per vendor)")

    top = amounts.nlargest(3, "z_score")
    for _, row in top.iterrows():
        loc = amounts.index.get_loc(row.name)
        ax.annotate(
            row["invoice_number"],
            (loc, row["grand_total"]),
            textcoords="offset points",
            xytext=(0, 8),
            fontsize=8,
            color="crimson",
            ha="center",
        )

    ax.set_xlabel("Invoice (sorted by detection order)")
    ax.set_ylabel("Grand Total ($)")
    ax.set_title("Invoice Amounts vs Z-Score\n(Red = high Z-score outliers)")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "scatter_zscore.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  📊  Saved: {path}")


def plot_anomaly_reasons(df: pd.DataFrame) -> None:
    _style()
    flagged = df[df["is_anomaly"]].copy()
    if flagged.empty:
        return

    reasons = []
    for r in flagged["anomaly_reason"]:
        reasons.extend([x.strip() for x in r.split(",")])
    reason_counts = pd.Series(reasons).value_counts()

    fig, ax = plt.subplots(figsize=(8, 4))
    reason_counts.sort_values().plot(kind="barh", ax=ax, color="salmon", edgecolor="black")
    ax.set_xlabel("Count")
    ax.set_title("Anomaly Flags by Type")
    for bar in ax.patches:
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{int(bar.get_width())}", va="center")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "anomaly_reasons.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  📊  Saved: {path}")


def run_pipeline(image_dir: str) -> pd.DataFrame:
    images = discover_images(image_dir)
    if not images:
        raise FileNotFoundError(
            f"No invoice images found in '{image_dir}'.\n"
            "Run:  python generate_sample_invoices.py  first."
        )
    print(f"\n🔍  Found {len(images)} invoice image(s) in '{image_dir}'")

    print("\n📄  Parsing invoices (OCR + extraction) …")
    df = parse_invoices_to_df(images)

    print("\n🤖  Running anomaly detection …")
    df = detect_anomalies(df)

    csv_cols = [
        "file_path", "invoice_number", "invoice_date", "vendor_name",
        "grand_total", "calculated_total", "num_line_items",
        "z_score", "is_anomaly", "anomaly_reason",
    ]
    out_csv = os.path.join(OUTPUT_DIR, "invoice_results.csv")
    df[csv_cols].to_csv(out_csv, index=False)
    print(f"\n💾  Results saved to: {out_csv}")

    print("\n📊  Generating visualisations …")
    plot_amount_distribution(df)
    plot_scatter_zscore(df)
    plot_anomaly_reasons(df)

    print_summary(df)

    return df


if __name__ == "__main__":
    image_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE_DIR
    run_pipeline(image_dir)
