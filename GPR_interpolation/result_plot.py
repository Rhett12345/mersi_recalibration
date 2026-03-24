# -*- coding: utf-8 -*-
"""Diagnostic plotting for the GP interpolation pipeline."""

from __future__ import annotations

import argparse
import glob
import logging
import os
import traceback

import matplotlib
n = matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

BAND_WAVELENGTHS = {
    1: "412 nm",
    2: "443 nm",
    3: "490 nm",
    4: "555 nm",
    5: "660 nm",
    6: "680 nm",
    7: "745 nm",
}


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "legend.frameon": False,
    })


def date_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))


def load_cv(result_dir: str) -> pd.DataFrame:
    path = os.path.join(result_dir, "cross_validation_results.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


def load_manifest(result_dir: str) -> pd.DataFrame:
    path = os.path.join(result_dir, "interpolation_manifest.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


def cv_text(cv_df: pd.DataFrame, channel: int, cal_type: str, parameter: str) -> str:
    if cv_df.empty:
        return ""
    key = f"ch{channel}_{cal_type}_{parameter}"
    sub = cv_df[cv_df["series"] == key]
    if sub.empty:
        return ""
    return f"CV MAE={sub['MAE'].mean():.4f}  RMSE={sub['RMSE'].mean():.4f}"


def mode_text(manifest_df: pd.DataFrame, channel: int, cal_type: str, parameter: str) -> str:
    if manifest_df.empty:
        return ""
    key = f"ch{channel}_{cal_type}_{parameter}"
    sub = manifest_df[manifest_df["series"] == key]
    if sub.empty:
        return ""
    row = sub.iloc[0]
    return f"train={row['train_mode']} | obs={int(row['n_observed_rows'])} | syn={int(row['n_synthetic_rows'])}"


def plot_results(data_dir: str, result_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    configure_style()

    obs_path = os.path.join(data_dir, "preprocessed_data.csv")
    if not os.path.exists(obs_path):
        raise FileNotFoundError(f"Missing {obs_path}")

    obs_df = pd.read_csv(obs_path)
    obs_df["date"] = pd.to_datetime(obs_df["date"])
    if "is_synthetic" not in obs_df.columns:
        obs_df["is_synthetic"] = False

    cv_df = load_cv(result_dir)
    manifest_df = load_manifest(result_dir)

    for cal_type in ["RAD", "REF"]:
        for parameter in ["Intercept", "Slope"]:
            try:
                fig, axes = plt.subplots(7, 1, figsize=(11, 18), sharex=False)
                for i, ch in enumerate(range(1, 8)):
                    ax = axes[i]
                    interp_path = os.path.join(result_dir, f"interp_ch{ch}_{cal_type}_{parameter}.csv")
                    if not os.path.exists(interp_path):
                        ax.text(0.5, 0.5, "No interpolation result", transform=ax.transAxes, ha="center", va="center")
                        continue

                    gp_df = pd.read_csv(interp_path)
                    gp_df["date"] = pd.to_datetime(gp_df["date"])
                    sub = obs_df[(obs_df["channel"] == ch) & (obs_df["type"] == cal_type) & (obs_df["parameter"] == parameter)].copy()
                    real_obs = sub[sub["is_synthetic"] == False]
                    syn_obs = sub[sub["is_synthetic"] == True]

                    if "std" in gp_df.columns:
                        ax.fill_between(gp_df["date"], gp_df["value"] - 1.96 * gp_df["std"], gp_df["value"] + 1.96 * gp_df["std"], alpha=0.25, linewidth=0)
                    ax.plot(gp_df["date"], gp_df["value"], linewidth=1.8)
                    if not real_obs.empty:
                        ax.scatter(real_obs["date"], real_obs["value"], s=22, marker="o", edgecolors="white", linewidths=0.5, zorder=5)
                    if not syn_obs.empty:
                        ax.scatter(syn_obs["date"], syn_obs["value"], s=18, marker="^", alpha=0.8, edgecolors="white", linewidths=0.5, zorder=4)

                    ax.text(0.02, 0.96, f"B{i+1:02d} ({BAND_WAVELENGTHS[i+1]})", transform=ax.transAxes, va="top", ha="left", fontweight="bold")
                    txt = cv_text(cv_df, ch, cal_type, parameter)
                    if txt:
                        ax.text(0.98, 0.96, txt, transform=ax.transAxes, va="top", ha="right", fontsize=7.5)
                    txt2 = mode_text(manifest_df, ch, cal_type, parameter)
                    if txt2:
                        ax.text(0.98, 0.86, txt2, transform=ax.transAxes, va="top", ha="right", fontsize=7.2)
                    ax.set_ylabel(parameter)
                    date_axis(ax)
                    if i < 6:
                        ax.set_xticklabels([])
                    else:
                        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

                legend_items = [
                    Line2D([0], [0], marker="o", color="w", markerfacecolor="C0", markersize=6, label="Observed (used for main training)"),
                    Line2D([0], [0], marker="^", color="w", markerfacecolor="C1", markersize=6, label="Synthetic / regularised"),
                    Line2D([0], [0], color="C2", linewidth=2, label="GP posterior mean"),
                    plt.Rectangle((0, 0), 1, 1, fc="C2", alpha=0.25, label="95% credible interval"),
                ]
                fig.legend(handles=legend_items, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01), fontsize=8)
                fig.suptitle(f"{cal_type} — {parameter}", fontsize=12, fontweight="bold", y=1.0)
                plt.tight_layout(rect=[0, 0.03, 1, 0.98])
                out_path = os.path.join(output_dir, f"{cal_type}_{parameter}_interpolation.png")
                fig.savefig(out_path, bbox_inches="tight")
                plt.close(fig)
                logging.info("Saved %s", out_path)
            except Exception:
                logging.error("Failed to plot %s %s:\n%s", cal_type, parameter, traceback.format_exc())
                plt.close("all")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot GP interpolation diagnostics.")
    parser.add_argument("--data_dir", required=True, help="Directory containing preprocessed_data.csv")
    parser.add_argument("--result_dir", required=True, help="Directory containing interpolation CSV files")
    parser.add_argument("--output_dir", required=True, help="Directory for output figures")
    args = parser.parse_args()
    plot_results(args.data_dir, args.result_dir, args.output_dir)


if __name__ == "__main__":
    main()
