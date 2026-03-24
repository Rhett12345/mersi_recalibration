# -*- coding: utf-8 -*-
"""
Module 2: Gaussian-process interpolation for calibration coefficients.

Public-release safeguards:
1. Use one consistent temporal reference per series.
2. Default to training and evaluating on observed QC-passed points only.
3. Preserve provenance of synthetic points for plotting and audit, but do not
   include them in training unless explicitly requested.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import traceback
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel as C,
    ExpSineSquared,
    RBF,
    RationalQuadratic,
    WhiteKernel,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

FEATURE_COLS = ["t_index", "sin_doy", "cos_doy", "sin_mon", "cos_mon"]
MIN_TRAIN_POINTS = 6
DEFAULT_PAD_DAYS = 15


@dataclass
class CVRecord:
    series: str
    fold: int
    channel: int
    cal_type: str
    parameter: str
    n_train: int
    n_test: int
    MAE: float
    RMSE: float
    train_mode: str


@dataclass
class SeriesManifest:
    series: str
    channel: int
    cal_type: str
    parameter: str
    n_total_rows: int
    n_observed_rows: int
    n_training_rows: int
    n_synthetic_rows: int
    train_mode: str
    ref_date: str
    start_date: str
    end_date: str


def build_kernel() -> object:
    return (
        C(1.0, (1e-3, 1e3))
        * (
            1.0 * RBF(length_scale=30.0, length_scale_bounds=(1.0, 1e3))
            + 0.8 * ExpSineSquared(length_scale=10.0, periodicity=365.25, length_scale_bounds=(1.0, 1e3), periodicity_bounds=(300.0, 450.0))
            + 0.5 * RationalQuadratic(length_scale=30.0, alpha=1.0, length_scale_bounds=(1.0, 1e3), alpha_bounds=(1e-3, 1e3))
        )
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1.0))
    )


def prepare_features(df: pd.DataFrame, ref_date: pd.Timestamp) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["month"] = out["date"].dt.month
    out["doy"] = out["date"].dt.dayofyear
    out["sin_doy"] = np.sin(2.0 * np.pi * out["doy"] / 365.25)
    out["cos_doy"] = np.cos(2.0 * np.pi * out["doy"] / 365.25)
    out["sin_mon"] = np.sin(2.0 * np.pi * out["month"] / 12.0)
    out["cos_mon"] = np.cos(2.0 * np.pi * out["month"] / 12.0)
    out["t_index"] = (out["date"] - ref_date).dt.days
    return out


def build_prediction_grid(ref_date: pd.Timestamp, start_date: pd.Timestamp, end_date: pd.Timestamp, pad_days: int) -> pd.DataFrame:
    grid_dates = pd.date_range(start=start_date - pd.Timedelta(days=pad_days), end=end_date + pd.Timedelta(days=pad_days), freq="D")
    grid = pd.DataFrame({"date": grid_dates})
    return prepare_features(grid, ref_date)


def fit_gp(X: np.ndarray, y: np.ndarray) -> GaussianProcessRegressor:
    gp = GaussianProcessRegressor(
        kernel=build_kernel(),
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=0,
    )
    gp.fit(X, y)
    return gp


def blocked_cross_validate(train_df: pd.DataFrame, channel: int, cal_type: str, parameter: str, train_mode: str) -> List[CVRecord]:
    if len(train_df) < 2 * MIN_TRAIN_POINTS:
        return []

    n_splits = min(5, max(2, len(train_df) // MIN_TRAIN_POINTS))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    X = train_df[FEATURE_COLS].to_numpy(dtype=float)
    y = train_df["value"].to_numpy(dtype=float)

    records: List[CVRecord] = []
    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X), start=1):
        if len(tr_idx) < MIN_TRAIN_POINTS or len(te_idx) < 2:
            continue
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx])
        X_te = scaler.transform(X[te_idx])
        gp = fit_gp(X_tr, y[tr_idx])
        pred = gp.predict(X_te)
        records.append(
            CVRecord(
                series=f"ch{channel}_{cal_type}_{parameter}",
                fold=fold,
                channel=channel,
                cal_type=cal_type,
                parameter=parameter,
                n_train=len(tr_idx),
                n_test=len(te_idx),
                MAE=float(mean_absolute_error(y[te_idx], pred)),
                RMSE=float(np.sqrt(mean_squared_error(y[te_idx], pred))),
                train_mode=train_mode,
            )
        )
    return records


def save_kernel_summary(path: str, kernel_map: Dict[str, str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(kernel_map, f, indent=2, ensure_ascii=False)


def interpolate_series(df: pd.DataFrame, output_dir: str, pad_days: int, include_synthetic_in_training: bool) -> None:
    os.makedirs(output_dir, exist_ok=True)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    if "used_for_training" not in df.columns:
        df["used_for_training"] = ~df.get("is_synthetic", False)
    if "is_synthetic" not in df.columns:
        df["is_synthetic"] = False

    cv_records: List[CVRecord] = []
    manifests: List[SeriesManifest] = []
    kernel_map: Dict[str, str] = {}

    for (channel, cal_type, parameter), grp in df.groupby(["channel", "type", "parameter"]):
        grp = grp.sort_values("date").reset_index(drop=True)
        ref_date = pd.Timestamp(grp["date"].min())
        grp = prepare_features(grp, ref_date)

        observed = grp[(grp["is_synthetic"] == False) & (grp.get("is_outlier", False) == False)].copy()
        if include_synthetic_in_training:
            train_df = grp[(grp["used_for_training"] == True) | (grp["is_synthetic"] == True)].copy()
            train_mode = "observed_plus_synthetic"
        else:
            train_df = grp[grp["used_for_training"] == True].copy()
            train_mode = "observed_only"

        if len(train_df) < MIN_TRAIN_POINTS:
            logging.warning("Skipping ch%s %s %s: only %d eligible training rows", channel, cal_type, parameter, len(train_df))
            continue

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[FEATURE_COLS].to_numpy(dtype=float))
        y_train = train_df["value"].to_numpy(dtype=float)
        gp = fit_gp(X_train, y_train)

        pred_grid = build_prediction_grid(ref_date, grp["date"].min(), grp["date"].max(), pad_days)
        X_pred = scaler.transform(pred_grid[FEATURE_COLS].to_numpy(dtype=float))
        pred_mean, pred_std = gp.predict(X_pred, return_std=True)
        pred_grid["value"] = pred_mean
        pred_grid["std"] = pred_std
        pred_grid["channel"] = channel
        pred_grid["type"] = cal_type
        pred_grid["parameter"] = parameter
        pred_grid["train_mode"] = train_mode

        out_csv = os.path.join(output_dir, f"interp_ch{channel}_{cal_type}_{parameter}.csv")
        pred_grid[["date", "channel", "type", "parameter", "value", "std", "train_mode"]].to_csv(out_csv, index=False)

        cv_records.extend(blocked_cross_validate(train_df.sort_values("date"), channel, cal_type, parameter, train_mode))
        series_key = f"ch{channel}_{cal_type}_{parameter}"
        kernel_map[series_key] = str(gp.kernel_)
        manifests.append(
            SeriesManifest(
                series=series_key,
                channel=int(channel),
                cal_type=str(cal_type),
                parameter=str(parameter),
                n_total_rows=int(len(grp)),
                n_observed_rows=int(len(observed)),
                n_training_rows=int(len(train_df)),
                n_synthetic_rows=int((grp["is_synthetic"] == True).sum()),
                train_mode=train_mode,
                ref_date=ref_date.strftime("%Y-%m-%d"),
                start_date=pd.Timestamp(grp["date"].min()).strftime("%Y-%m-%d"),
                end_date=pd.Timestamp(grp["date"].max()).strftime("%Y-%m-%d"),
            )
        )
        logging.info("Saved interpolation for ch%s %s %s (%s)", channel, cal_type, parameter, train_mode)

    cv_df = pd.DataFrame([asdict(r) for r in cv_records])
    cv_df.to_csv(os.path.join(output_dir, "cross_validation_results.csv"), index=False)
    pd.DataFrame([asdict(m) for m in manifests]).to_csv(os.path.join(output_dir, "interpolation_manifest.csv"), index=False)
    save_kernel_summary(os.path.join(output_dir, "kernel_hyperparameters.json"), kernel_map)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gaussian-process interpolation for calibration coefficients.")
    parser.add_argument("--input_csv", required=True, help="Path to preprocessed_data.csv")
    parser.add_argument("--output_dir", required=True, help="Directory for interpolation outputs")
    parser.add_argument("--pad_days", type=int, default=DEFAULT_PAD_DAYS, help="Padding days on each boundary for interpolation grid")
    parser.add_argument("--include_synthetic_in_training", action="store_true", help="Explicitly include synthetic rows in GP training and CV")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input_csv)
        interpolate_series(
            df=df,
            output_dir=args.output_dir,
            pad_days=args.pad_days,
            include_synthetic_in_training=args.include_synthetic_in_training,
        )
    except Exception:
        logging.error("Interpolation failed:\n%s", traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
