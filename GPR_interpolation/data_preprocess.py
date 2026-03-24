# -*- coding: utf-8 -*-
"""
Module 1: Data loading, quality control, and temporal regularisation.

Design goals for the public release:
1. Keep observed and synthetic points explicitly separated.
2. Emit an audit trail for every QC and regularisation action.
3. Use one consistent temporal reference per (channel, type, parameter) series.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

FILE_RE = re.compile(
    r"^CAL_(?P<sensor>.+?)_(?P<cal_type>REF|RAD)_ch01toch07_(?P<date>\d{8})\.csv$",
    re.IGNORECASE,
)

TARGET_FREQ = "D"
OUTLIER_THRESHOLD = 3.5
MAX_FILL_GAP_DAYS = 30
SPARSE_SERIES_THRESHOLD = 20
MIN_POINTS_FOR_NEIGHBOUR_SUB = 2


@dataclass
class AuditEvent:
    channel: int
    cal_type: str
    parameter: str
    date: str
    event_type: str
    original_value: Optional[float]
    replacement_value: Optional[float]
    note: str


@dataclass
class SeriesSummary:
    channel: int
    cal_type: str
    parameter: str
    n_observed_input: int
    n_outliers_flagged: int
    n_substituted: int
    n_synthetic_regularised: int
    n_training_eligible_observed: int
    start_date: str
    end_date: str


def load_data(data_root: str) -> pd.DataFrame:
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"Input directory not found: {data_root}")

    frames: List[pd.DataFrame] = []
    file_count = 0
    for root, _, files in os.walk(data_root):
        for fname in files:
            m = FILE_RE.match(fname)
            if not m:
                continue
            file_count += 1
            cal_type = m.group("cal_type").upper()
            obs_date = datetime.strptime(m.group("date"), "%Y%m%d")
            fpath = os.path.join(root, fname)
            try:
                tmp = pd.read_csv(
                    fpath,
                    comment="#",
                    header=None,
                    names=["Channel", "Intercept", "Slope"],
                )
            except Exception as exc:
                logging.error("Failed to read %s: %s", fpath, exc)
                continue

            tmp["date"] = pd.Timestamp(obs_date)
            tmp["type"] = cal_type
            tmp["source_file"] = fname
            frames.append(tmp)

    if not frames:
        raise RuntimeError(f"No matching calibration files found under {data_root}")

    df = pd.concat(frames, ignore_index=True)
    df["Channel"] = pd.to_numeric(df["Channel"], errors="coerce")
    df = df.dropna(subset=["Channel", "Intercept", "Slope", "date", "type"])
    df["Channel"] = df["Channel"].astype(int)
    df = df.sort_values(["Channel", "type", "date"]).reset_index(drop=True)

    logging.info(
        "Loaded %d records from %d files. Date range: %s to %s",
        len(df),
        file_count,
        df["date"].min().date(),
        df["date"].max().date(),
    )
    return df


def add_static_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["doy"] = df["date"].dt.dayofyear
    df["sin_doy"] = np.sin(2.0 * np.pi * df["doy"] / 365.25)
    df["cos_doy"] = np.cos(2.0 * np.pi * df["doy"] / 365.25)
    df["sin_mon"] = np.sin(2.0 * np.pi * df["month"] / 12.0)
    df["cos_mon"] = np.cos(2.0 * np.pi * df["month"] / 12.0)
    return df


def robust_outlier_mask(values: np.ndarray, threshold: float = OUTLIER_THRESHOLD) -> np.ndarray:
    if len(values) < 4:
        return np.zeros(len(values), dtype=bool)

    values = np.asarray(values, dtype=float)
    med = float(np.median(values))
    mad = float(median_abs_deviation(values, scale="normal"))
    if mad < 1e-12:
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        scale = iqr if iqr > 1e-12 else np.std(values)
        if scale < 1e-12:
            return np.zeros(len(values), dtype=bool)
        return np.abs(values - med) > threshold * scale

    modified_z = 0.6745 * np.abs(values - med) / mad
    return modified_z > threshold


def neighbour_substitution(clean_df: pd.DataFrame, target_date: pd.Timestamp) -> Tuple[Optional[float], str]:
    if clean_df.empty:
        return None, "no_clean_observations"

    prev_obs = clean_df.loc[clean_df["date"] < target_date].sort_values("date").tail(1)
    next_obs = clean_df.loc[clean_df["date"] > target_date].sort_values("date").head(1)

    if prev_obs.empty and next_obs.empty:
        return None, "no_neighbours"
    if prev_obs.empty:
        return float(next_obs["value"].iloc[0]), "single_forward_neighbour"
    if next_obs.empty:
        return float(prev_obs["value"].iloc[0]), "single_backward_neighbour"

    t_prev = pd.Timestamp(prev_obs["date"].iloc[0])
    t_next = pd.Timestamp(next_obs["date"].iloc[0])
    v_prev = float(prev_obs["value"].iloc[0])
    v_next = float(next_obs["value"].iloc[0])

    d_prev = max((target_date - t_prev).days, 1)
    d_next = max((t_next - target_date).days, 1)
    w_prev = 1.0 / d_prev
    w_next = 1.0 / d_next
    val = (w_prev * v_prev + w_next * v_next) / (w_prev + w_next)
    return float(val), "distance_weighted_neighbours"


def regularise_gaps(clean_df: pd.DataFrame, max_gap_days: int) -> List[Tuple[pd.Timestamp, float, str]]:
    results: List[Tuple[pd.Timestamp, float, str]] = []
    if len(clean_df) < 2:
        return results

    clean_df = clean_df.sort_values("date").reset_index(drop=True)
    for i in range(len(clean_df) - 1):
        left = clean_df.iloc[i]
        right = clean_df.iloc[i + 1]
        gap = int((right["date"] - left["date"]).days)
        if gap <= 1 or gap > max_gap_days:
            continue
        total_gap = float(gap)
        for step in range(1, gap):
            d = pd.Timestamp(left["date"]) + pd.Timedelta(days=step)
            w_left = 1.0 / step
            w_right = 1.0 / (gap - step)
            value = (w_left * float(left["value"]) + w_right * float(right["value"])) / (w_left + w_right)
            results.append((d, float(value), f"gap_fill_{int(total_gap)}d"))
    return results


def build_parameter_series(
    base_group: pd.DataFrame,
    parameter: str,
    enable_regularisation: bool,
    max_fill_gap_days: int,
) -> Tuple[pd.DataFrame, List[AuditEvent], SeriesSummary]:
    grp = base_group[["Channel", "type", "date", parameter, "source_file"]].copy()
    grp = grp.rename(columns={parameter: "value"}).sort_values("date").reset_index(drop=True)
    grp["channel"] = grp["Channel"].astype(int)
    grp["parameter"] = parameter
    grp["is_synthetic"] = False
    grp["is_outlier"] = robust_outlier_mask(grp["value"].to_numpy())
    grp["qc_status"] = np.where(grp["is_outlier"], "flagged_outlier", "observed")
    grp["provenance"] = np.where(grp["is_outlier"], "original_flagged", "original_observed")
    grp["used_for_training"] = ~grp["is_outlier"]

    audit: List[AuditEvent] = []
    outliers = grp[grp["is_outlier"]].copy()
    clean_obs = grp[~grp["is_outlier"]].copy()

    substituted_rows: List[Dict[str, object]] = []
    if len(clean_obs) >= MIN_POINTS_FOR_NEIGHBOUR_SUB:
        for _, row in outliers.iterrows():
            replacement, note = neighbour_substitution(clean_obs[["date", "value"]], pd.Timestamp(row["date"]))
            audit.append(
                AuditEvent(
                    channel=int(row["channel"]),
                    cal_type=str(row["type"]),
                    parameter=parameter,
                    date=pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
                    event_type="outlier_flagged_and_substituted" if replacement is not None else "outlier_flagged_not_substituted",
                    original_value=float(row["value"]),
                    replacement_value=None if replacement is None else float(replacement),
                    note=note,
                )
            )
            if replacement is None:
                continue
            substituted_rows.append(
                {
                    "Channel": int(row["channel"]),
                    "type": row["type"],
                    "date": pd.Timestamp(row["date"]),
                    "value": float(replacement),
                    "source_file": row["source_file"],
                    "channel": int(row["channel"]),
                    "parameter": parameter,
                    "is_synthetic": True,
                    "is_outlier": False,
                    "qc_status": "substituted_for_outlier",
                    "provenance": "synthetic_substitution",
                    "used_for_training": False,
                }
            )
    else:
        for _, row in outliers.iterrows():
            audit.append(
                AuditEvent(
                    channel=int(row["channel"]),
                    cal_type=str(row["type"]),
                    parameter=parameter,
                    date=pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
                    event_type="outlier_flagged_not_substituted",
                    original_value=float(row["value"]),
                    replacement_value=None,
                    note="insufficient_clean_neighbours",
                )
            )

    model_df = clean_obs.copy()
    if substituted_rows:
        model_df = pd.concat([model_df, pd.DataFrame(substituted_rows)], ignore_index=True)

    regularised_rows: List[Dict[str, object]] = []
    if enable_regularisation and len(model_df) < SPARSE_SERIES_THRESHOLD:
        for d, value, note in regularise_gaps(model_df[["date", "value"]], max_fill_gap_days=max_fill_gap_days):
            regularised_rows.append(
                {
                    "Channel": int(grp["channel"].iloc[0]),
                    "type": grp["type"].iloc[0],
                    "date": d,
                    "value": value,
                    "source_file": "synthetic_regularisation",
                    "channel": int(grp["channel"].iloc[0]),
                    "parameter": parameter,
                    "is_synthetic": True,
                    "is_outlier": False,
                    "qc_status": "regularised_gap",
                    "provenance": note,
                    "used_for_training": False,
                }
            )
            audit.append(
                AuditEvent(
                    channel=int(grp["channel"].iloc[0]),
                    cal_type=str(grp["type"].iloc[0]),
                    parameter=parameter,
                    date=pd.Timestamp(d).strftime("%Y-%m-%d"),
                    event_type="gap_regularised",
                    original_value=None,
                    replacement_value=float(value),
                    note=note,
                )
            )

    all_rows = [clean_obs]
    if substituted_rows:
        all_rows.append(pd.DataFrame(substituted_rows))
    if regularised_rows:
        reg_df = pd.DataFrame(regularised_rows)
        reg_df = reg_df[~reg_df["date"].isin(model_df["date"])]
        if not reg_df.empty:
            all_rows.append(reg_df)

    final_df = pd.concat(all_rows, ignore_index=True).sort_values("date").reset_index(drop=True)
    final_df = add_static_features(final_df)
    ref_date = final_df["date"].min()
    final_df["t_index"] = (final_df["date"] - ref_date).dt.days

    summary = SeriesSummary(
        channel=int(grp["channel"].iloc[0]),
        cal_type=str(grp["type"].iloc[0]),
        parameter=parameter,
        n_observed_input=int(len(grp)),
        n_outliers_flagged=int(grp["is_outlier"].sum()),
        n_substituted=int(len(substituted_rows)),
        n_synthetic_regularised=int(len(regularised_rows)),
        n_training_eligible_observed=int((final_df["used_for_training"] == True).sum()),
        start_date=final_df["date"].min().strftime("%Y-%m-%d"),
        end_date=final_df["date"].max().strftime("%Y-%m-%d"),
    )
    return final_df, audit, summary


def save_sample_data(df: pd.DataFrame, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    grouped = df.groupby(["date", "type"])
    for (date, cal_type), group in grouped:
        coeff_matrix = []
        channels = list(range(1, 8))
        for ch in channels:
            ch_data = group[group["Channel"] == ch]
            if len(ch_data) > 0:
                row = ch_data.iloc[0]
                coeff_matrix.append([row["Intercept"], row["Slope"], 0.0])
            else:
                coeff_matrix.append([np.nan, np.nan, 0.0])

        df_out = pd.DataFrame(
            coeff_matrix,
            index=[f"ch{str(i).zfill(2)}" for i in channels],
            columns=["cal0", "cal1", "cal2"],
        )
        csv_path = os.path.join(output_dir, f"{cal_type}_{date.strftime('%Y%m%d')}.csv")
        df_out.to_csv(csv_path)


def run_pipeline(data_root: str, output_dir: str, sample_dir: str, enable_regularisation: bool, max_fill_gap_days: int) -> None:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    raw_df = load_data(data_root)
    save_sample_data(raw_df, sample_dir)

    processed_frames: List[pd.DataFrame] = []
    audit_events: List[AuditEvent] = []
    summaries: List[SeriesSummary] = []

    for (channel, cal_type), group in raw_df.groupby(["Channel", "type"]):
        logging.info("Processing channel=%s type=%s", channel, cal_type)
        for parameter in ["Intercept", "Slope"]:
            final_df, audit, summary = build_parameter_series(
                group,
                parameter=parameter,
                enable_regularisation=enable_regularisation,
                max_fill_gap_days=max_fill_gap_days,
            )
            processed_frames.append(final_df)
            audit_events.extend(audit)
            summaries.append(summary)

    preprocessed = pd.concat(processed_frames, ignore_index=True)
    preprocessed = preprocessed.sort_values(["channel", "type", "parameter", "date"]).reset_index(drop=True)
    preprocessed_path = os.path.join(output_dir, "preprocessed_data.csv")
    preprocessed.to_csv(preprocessed_path, index=False)

    audit_df = pd.DataFrame([asdict(x) for x in audit_events])
    audit_path = os.path.join(output_dir, "preprocessing_audit.csv")
    audit_df.to_csv(audit_path, index=False)

    summary_df = pd.DataFrame([asdict(x) for x in summaries])
    summary_path = os.path.join(output_dir, "series_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    logging.info("Saved preprocessed data to %s", preprocessed_path)
    logging.info("Saved preprocessing audit to %s", audit_path)
    logging.info("Saved series summary to %s", summary_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Data preprocessing and QC for calibration coefficient series.")
    parser.add_argument("--data_root", required=True, help="Directory containing raw calibration CSV files.")
    parser.add_argument("--output_dir", required=True, help="Directory for preprocessed outputs.")
    parser.add_argument("--sample_dir", required=True, help="Directory for per-date sample matrices.")
    parser.add_argument("--disable_regularisation", action="store_true", help="Disable synthetic gap regularisation.")
    parser.add_argument("--max_fill_gap_days", type=int, default=MAX_FILL_GAP_DAYS, help="Maximum gap length eligible for regularisation.")
    args = parser.parse_args()

    try:
        run_pipeline(
            data_root=args.data_root,
            output_dir=args.output_dir,
            sample_dir=args.sample_dir,
            enable_regularisation=not args.disable_regularisation,
            max_fill_gap_days=args.max_fill_gap_days,
        )
    except Exception:
        logging.error("Preprocessing failed:\n%s", traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
