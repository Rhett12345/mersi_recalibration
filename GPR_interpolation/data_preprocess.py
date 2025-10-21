# -*- coding: utf-8 -*-
"""
程序1：数据预处理和增强
"""

import os
import glob
import re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import traceback
import argparse
from scipy.stats import zscore
import matplotlib

matplotlib.use('Agg')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_data(data_root):
    """加载所有CSV文件到DataFrame"""
    data = []

    if not os.path.exists(data_root):
        logging.error(f"数据根目录不存在: {data_root}")
        return pd.DataFrame()

    # 使用递归查找所有匹配的文件
    file_list = []
    for root, dirs, files in os.walk(data_root):
        for file in files:
            # 更灵活的文件名匹配
            if (file.startswith('CAL_') or file.startswith('cal_')) and \
                    ('ch01toch07' in file.lower()) and \
                    file.endswith('.csv'):
                file_path = os.path.join(root, file)
                file_list.append(file_path)

    # 去重
    file_list = list(set(file_list))

    if not file_list:
        logging.warning(f"在 {data_root} 下未找到任何匹配的文件")
        return pd.DataFrame()

    logging.info(f"总共找到 {len(file_list)} 个文件进行处理...")

    for i, filepath in enumerate(file_list):
        try:
            filename = os.path.basename(filepath)
            if i < 10:  # 只记录前10个文件的详细信息
                logging.info(f"正在处理文件 {i + 1}/{len(file_list)}: {filename}")

            # 解析日期
            date_match = re.search(r'(\d{8})\.csv$', filename)
            if not date_match:
                logging.warning(f"无法从文件名解析日期: {filename}")
                continue

            date_str = date_match.group(1)

            # 判断类型
            filename_upper = filename.upper()
            if 'REF' in filename_upper:
                cal_type = 'REF'
            elif 'RAD' in filename_upper:
                cal_type = 'RAD'
            else:
                logging.warning(f"无法确定文件类型(REF/RAD): {filename}")
                continue

            try:
                date = datetime.strptime(date_str, '%Y%m%d')
            except ValueError:
                logging.warning(f"日期格式无效: {date_str} in {filename}")
                continue

            # 读取CSV文件
            try:
                df = pd.read_csv(filepath, comment='#', header=None,
                                 names=['Channel', 'Intercept', 'Slope'])

                if i < 5:  # 只记录前5个文件的详细信息
                    logging.info(f"  成功读取，形状: {df.shape}")

            except Exception as e:
                logging.error(f"读取文件 {filename} 失败: {str(e)}")
                continue

            df['date'] = date
            df['type'] = cal_type
            df['source_file'] = filename

            data.append(df)

        except Exception as e:
            logging.error(f"处理文件 {filepath} 时出错: {str(e)}")
            logging.error(traceback.format_exc())

    if not data:
        logging.error("未成功加载任何数据")
        return pd.DataFrame()

    # 合并数据
    result_df = pd.concat(data, ignore_index=True)
    logging.info(f"合并后的数据形状: {result_df.shape}")
    logging.info(f"数据日期范围: {result_df['date'].min()} 到 {result_df['date'].max()}")
    logging.info(f"REF数据点数: {len(result_df[result_df['type'] == 'REF'])}")
    logging.info(f"RAD数据点数: {len(result_df[result_df['type'] == 'RAD'])}")
    logging.info(f"通道分布: {result_df['Channel'].value_counts().sort_index().to_dict()}")

    return result_df


def process_data(df):
    """数据预处理和特征工程"""
    if df.empty:
        return df

    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_year'] = df['date'].dt.dayofyear

    min_date = df['date'].min()
    df['days_since_start'] = (df['date'] - min_date).dt.days

    df['sin_day'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['cos_day'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)

    df['is_sample'] = ((df['day'] <= 5) | ((df['day'] >= 14) & (df['day'] <= 18)))
    df['channel_type'] = df['Channel'].astype(str) + '_' + df['type']

    return df


def detect_outliers(data, param, z_threshold=3.0):
    """Z-score"""
    if len(data) < 3:
        data['is_outlier'] = False
        return data

    data = data.sort_values('date').copy()

    z_scores = np.abs(zscore(data[param]))
    data['is_outlier'] = z_scores > z_threshold

    n_outliers = sum(data['is_outlier'])
    if n_outliers > 0:
        outlier_dates = data.loc[data['is_outlier'], 'date'].dt.strftime('%Y-%m-%d').tolist()
        logging.info(
            f"检测到 {n_outliers} 个异常值: {', '.join(outlier_dates[:5])}{'...' if len(outlier_dates) > 5 else ''}")

    return data

def create_pseudo_samples_for_outliers(clean_data, outlier_data):
    """为异常值日期创建伪样本 - 使用前后两个日期的中值"""
    if outlier_data.empty:
        return pd.DataFrame()

    pseudo_samples = []

    for _, outlier_row in outlier_data.iterrows():
        outlier_date = outlier_row['date']

        # 查找前后相邻的日期
        prev_data = clean_data[clean_data['date'] < outlier_date]
        next_data = clean_data[clean_data['date'] > outlier_date]

        if not prev_data.empty and not next_data.empty:
            # 取最近的前一个和后一个日期
            prev_date = prev_data['date'].max()
            next_date = next_data['date'].min()

            prev_value = clean_data[clean_data['date'] == prev_date]['value'].iloc[0]
            next_value = clean_data[clean_data['date'] == next_date]['value'].iloc[0]

            # 计算中值
            pseudo_value = (prev_value + next_value) / 2

            # 创建伪样本
            pseudo_sample = outlier_row.copy()
            pseudo_sample['value'] = pseudo_value
            pseudo_sample['is_augmented'] = True
            pseudo_sample['is_outlier'] = False

            pseudo_samples.append(pseudo_sample)

            logging.info(f"为异常值日期 {outlier_date.strftime('%Y-%m-%d')} 创建伪样本: "
                         f"前值({prev_date.strftime('%Y-%m-%d')})={prev_value:.4f}, "
                         f"后值({next_date.strftime('%Y-%m-%d')})={next_value:.4f}, "
                         f"伪样本值={pseudo_value:.4f}")

    return pd.DataFrame(pseudo_samples)


def augment_data(df):
    """简化版数据增强"""
    if len(df) < 3:
        df['is_augmented'] = False
        return df

    df = df.sort_values('date').reset_index(drop=True)
    augmented = []
    dates = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='D')

    for target_date in dates:
        if target_date in df['date'].values:
            continue

        prev_data = df[df['date'] < target_date]
        next_data = df[df['date'] > target_date]

        if not prev_data.empty and not next_data.empty:
            prev_point = prev_data.iloc[-1]
            next_point = next_data.iloc[0]

            # 简化时间间隔检查
            time_gap = max((target_date - prev_point['date']).days,
                           (next_point['date'] - target_date).days)

            if time_gap <= 30:  # 统一时间阈值
                pseudo_value = (prev_point['value'] + next_point['value']) / 2

                pseudo_point = {
                    'date': target_date,
                    'value': pseudo_value,
                    'is_sample': False,
                    'is_augmented': True,
                    'is_outlier': False,
                    'year': target_date.year,
                    'month': target_date.month,
                    'day': target_date.day,
                    'day_of_year': target_date.timetuple().tm_yday,
                    'days_since_start': (target_date - df['date'].min()).days,
                    'sin_day': np.sin(2 * np.pi * target_date.timetuple().tm_yday / 365.25),
                    'cos_day': np.cos(2 * np.pi * target_date.timetuple().tm_yday / 365.25),
                    'sin_month': np.sin(2 * np.pi * target_date.month / 12),
                    'cos_month': np.cos(2 * np.pi * target_date.month / 12)
                }
                augmented.append(pseudo_point)

    df['is_augmented'] = False

    if augmented:
        augmented_df = pd.DataFrame(augmented)
        combined_df = pd.concat([df, augmented_df], ignore_index=True)
        logging.info(f"生成了 {len(augmented)} 个伪样本")
        return combined_df.sort_values('date').reset_index(drop=True)
    else:
        return df

def save_sample_data(df, output_dir):
    """保存原始样本数据到CSV"""
    os.makedirs(output_dir, exist_ok=True)

    grouped = df.groupby(['date', 'type'])
    for (date, cal_type), group in grouped:
        coeff_matrix = []
        channels = sorted(group['Channel'].unique())

        for ch in channels:
            ch_data = group[group['Channel'] == ch]
            if len(ch_data) > 0:
                row = ch_data.iloc[0]
                coeff_matrix.append([row['Intercept'], row['Slope'], 0.0])
            else:
                coeff_matrix.append([np.nan, np.nan, 0.0])

        df_out = pd.DataFrame(
            coeff_matrix,
            index=[f"ch{str(i).zfill(2)}" for i in channels],
            columns=['cal0', 'cal1', 'cal2']
        )

        date_str = date.strftime('%Y%m%d')
        csv_filename = f"{cal_type}_{date_str}.csv"
        csv_path = os.path.join(output_dir, csv_filename)
        df_out.to_csv(csv_path)

def main():
    parser = argparse.ArgumentParser(description='数据预处理和增强')
    parser.add_argument('--data_root', required=True, help='输入数据根目录')
    parser.add_argument('--output_dir', required=True, help='预处理数据输出目录')
    parser.add_argument('--sample_dir', required=True, help='样本数据输出目录')

    args = parser.parse_args()

    logging.info("开始数据预处理...")

    # 加载和处理数据
    df = load_data(args.data_root)
    if df.empty:
        logging.error("未加载任何数据，程序退出")
        return

    df = process_data(df)

    # 保存原始样本数据
    save_sample_data(df, args.sample_dir)
    logging.info(f"样本数据已保存到: {args.sample_dir}")

    # 对每个通道和类型组合进行处理
    groups = df.groupby(['Channel', 'type'])
    processed_data = []

    for (channel, cal_type), group in groups:
        logging.info(f"处理通道 {channel} - {cal_type}")
        group = group.sort_values('date')

        for param in ['Intercept', 'Slope']:
            param_data = group[['date', 'days_since_start', 'sin_day', 'cos_day',
                                'sin_month', 'cos_month', 'year', param, 'is_sample']].copy()
            param_data = param_data.rename(columns={param: 'value'})

            # 异常值检测
            param_data = detect_outliers(param_data, 'value')

            # 分离异常值和正常数据
            outlier_data = param_data[param_data['is_outlier']].copy()
            clean_data = param_data[~param_data['is_outlier']].copy()

            # 为异常值日期创建伪样本
            pseudo_for_outliers = create_pseudo_samples_for_outliers(clean_data, outlier_data)

            if not pseudo_for_outliers.empty:
                logging.info(f"为 {len(pseudo_for_outliers)} 个异常值日期创建了伪样本")
                # 将异常值的伪样本加入到clean_data中
                clean_data = pd.concat([clean_data, pseudo_for_outliers], ignore_index=True)
                clean_data = clean_data.sort_values('date')

            # 数据增强 - 为缺失日期创建伪样本
            if len(clean_data) < 20:
                augmented_data = augment_data(clean_data)
            else:
                augmented_data = clean_data.copy()
                augmented_data['is_augmented'] = False

            # 添加标识信息
            augmented_data['channel'] = channel
            augmented_data['type'] = cal_type
            augmented_data['parameter'] = param

            processed_data.append(augmented_data)

    # 保存预处理数据
    if processed_data:
        final_df = pd.concat(processed_data, ignore_index=True)
        output_file = os.path.join(args.output_dir, 'preprocessed_data.csv')
        final_df.to_csv(output_file, index=False)

        # 统计信息
        n_original = len(final_df[~final_df['is_augmented']])
        n_augmented = len(final_df[final_df['is_augmented']])
        logging.info(f"预处理数据统计: 原始样本 {n_original} 个, 伪样本 {n_augmented} 个")
        logging.info(f"预处理数据已保存到: {output_file}")

    logging.info("数据预处理完成!")

if __name__ == "__main__":
    main()