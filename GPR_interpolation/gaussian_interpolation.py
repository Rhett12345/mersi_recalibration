# -*- coding: utf-8 -*-
"""
程序2：高斯过程插值
"""

import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel, ExpSineSquared, RationalQuadratic
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
import traceback
import argparse
from datetime import datetime, timedelta
import os
import h5py
import json

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def enhanced_gaussian_process(X_train, y_train, X_pred):
    """优化的高斯过程回归"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_pred_scaled = scaler.transform(X_pred)

    kernel = (
            C(1.0, (1e-3, 1e3)) *
            RBF(length_scale=100.0, length_scale_bounds=(10.0, 1000.0)) +
            C(0.5, (1e-2, 10)) *
            ExpSineSquared(length_scale=1.0, periodicity=365.25, periodicity_bounds=(300, 400)) *
            RBF(length_scale=10.0) +
            RationalQuadratic(alpha=0.1, length_scale=1.0) +
            WhiteKernel(noise_level=0.05, noise_level_bounds=(1e-5, 0.1))
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=25,
        alpha=0.02,
        normalize_y=True,
        random_state=42
    )

    try:
        gp.fit(X_train_scaled, y_train)
        y_pred, sigma = gp.predict(X_pred_scaled, return_std=True)
        return y_pred, sigma, gp.kernel_
    except Exception as e:
        logging.error(f"高斯过程优化失败: {str(e)}")
        simple_kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=100.0, length_scale_bounds=(1, 1000))
        gp = GaussianProcessRegressor(
            kernel=simple_kernel,
            n_restarts_optimizer=15,
            alpha=0.05,
            normalize_y=True,
            random_state=42
        )
        gp.fit(X_train_scaled, y_train)
        y_pred, sigma = gp.predict(X_pred_scaled, return_std=True)
        return y_pred, sigma, gp.kernel_


def cross_validate(data, n_splits=5):
    """执行时间序列交叉验证"""
    if len(data) < 2:
        return pd.DataFrame()

    tscv = TimeSeriesSplit(n_splits=min(n_splits, len(data) - 1))
    metrics = []
    data = data.sort_values('date')

    for fold, (train_index, test_index) in enumerate(tscv.split(data)):
        train_data = data.iloc[train_index]
        test_data = data.iloc[test_index]

        X_train = train_data[['days_since_start', 'sin_day', 'cos_day', 'sin_month', 'cos_month']].values
        y_train = train_data['value'].values
        X_test = test_data[['days_since_start', 'sin_day', 'cos_day', 'sin_month', 'cos_month']].values
        y_test = test_data['value'].values

        try:
            y_pred, _, _ = enhanced_gaussian_process(X_train, y_train, X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            metrics.append({
                'fold': fold + 1,
                'MAE': mae,
                'RMSE': rmse,
                'n_train': len(train_index),
                'n_test': len(test_index)
            })
        except Exception as e:
            logging.error(f"交叉验证出错: {str(e)}")

    return pd.DataFrame(metrics)


def save_daily_results(results, date, cal_type, output_dir):
    """保存每日结果"""
    os.makedirs(output_dir, exist_ok=True)

    channels = sorted(results.keys())
    coeff_matrix = []

    for ch in channels:
        coeff_matrix.append([
            results[ch]['Intercept'],
            results[ch]['Slope'],
            0.0
        ])

    df = pd.DataFrame(
        coeff_matrix,
        index=[f"ch{str(i).zfill(2)}" for i in channels],
        columns=['cal0', 'cal1', 'cal2']
    )

    # 保存CSV
    date_str = date.strftime('%Y%m%d')
    csv_filename = f"{cal_type}_{date_str}.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    df.to_csv(csv_path)

    # 保存HDF
    hdf_filename = f"{cal_type}_{date_str}.h5"
    hdf_path = os.path.join(output_dir, hdf_filename)

    with h5py.File(hdf_path, 'w') as f:
        dset = f.create_dataset("calibration_coeff", data=df.values)
        dset.attrs['date'] = date_str
        dset.attrs['cal_type'] = cal_type
        dset.attrs['channels'] = ','.join(df.index.tolist())
        dset.attrs['columns'] = ','.join(df.columns.tolist())


def main():
    parser = argparse.ArgumentParser(description='高斯过程插值')
    parser.add_argument('--input_dir', required=True, help='预处理数据目录')
    parser.add_argument('--output_dir', required=True, help='插值结果输出目录')
    parser.add_argument('--daily_dir', required=True, help='每日结果输出目录')

    args = parser.parse_args()

    logging.info("开始高斯过程插值...")

    # 加载预处理数据
    input_file = os.path.join(args.input_dir, 'preprocessed_data.csv')
    if not os.path.exists(input_file):
        logging.error(f"预处理数据文件不存在: {input_file}")
        return

    df = pd.read_csv(input_file)
    df['date'] = pd.to_datetime(df['date'])

    # 创建完整日期范围
    min_date = df['date'].min() - timedelta(days=15)
    max_date = df['date'].max() + timedelta(days=15)
    full_dates = pd.date_range(start=min_date, end=max_date, freq='D')

    # 对每个通道-类型-参数组合进行插值
    groups = df.groupby(['channel', 'type', 'parameter'])
    all_interp_results = {}
    kernel_info = {}

    for (channel, cal_type, param), group in groups:
        logging.info(f"处理通道 {channel} - {cal_type} - {param}")

        # 准备特征数据
        feature_cols = ['days_since_start', 'sin_day', 'cos_day', 'sin_month', 'cos_month']
        X_train = group[feature_cols].values
        y_train = group['value'].values

        # 准备完整预测数据
        full_df = pd.DataFrame({'date': full_dates})
        min_start_date = group['date'].min()
        full_df['days_since_start'] = (full_df['date'] - min_start_date).dt.days
        full_df['day_of_year'] = full_df['date'].dt.dayofyear
        full_df['month'] = full_df['date'].dt.month
        full_df['sin_day'] = np.sin(2 * np.pi * full_df['day_of_year'] / 365.25)
        full_df['cos_day'] = np.cos(2 * np.pi * full_df['day_of_year'] / 365.25)
        full_df['sin_month'] = np.sin(2 * np.pi * full_df['month'] / 12)
        full_df['cos_month'] = np.cos(2 * np.pi * full_df['month'] / 12)

        X_full = full_df[feature_cols].values

        # 执行插值
        try:
            y_pred, sigma, kernel = enhanced_gaussian_process(X_train, y_train, X_full)
            kernel_info[f"Channel{channel}_{cal_type}_{param}"] = str(kernel)

            # 存储结果
            for date, value in zip(full_dates, y_pred):
                date_str = date.strftime('%Y-%m-%d')
                if date_str not in all_interp_results:
                    all_interp_results[date_str] = {}
                if cal_type not in all_interp_results[date_str]:
                    all_interp_results[date_str][cal_type] = {}
                if channel not in all_interp_results[date_str][cal_type]:
                    all_interp_results[date_str][cal_type][channel] = {}

                all_interp_results[date_str][cal_type][channel][param] = value

            # 保存单个参数结果
            param_results = pd.DataFrame({
                'date': full_dates,
                'value': y_pred,
                'std': sigma
            })
            param_file = os.path.join(args.output_dir, f'interp_ch{channel}_{cal_type}_{param}.csv')
            param_results.to_csv(param_file, index=False)

        except Exception as e:
            logging.error(f"插值失败: {str(e)}")
            logging.error(traceback.format_exc())

    # 保存每日结果
    for date_str, date_data in all_interp_results.items():
        date = datetime.strptime(date_str, '%Y-%m-%d')
        for cal_type, cal_data in date_data.items():
            save_daily_results(cal_data, date, cal_type, args.daily_dir)

    # 保存核信息
    kernel_file = os.path.join(args.output_dir, 'kernel_info.json')
    with open(kernel_file, 'w') as f:
        json.dump(kernel_info, f, indent=2)

    logging.info("高斯过程插值完成!")


if __name__ == "__main__":
    main()