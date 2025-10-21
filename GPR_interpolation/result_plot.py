# -*- coding: utf-8 -*-
"""
程序3：结果可视化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import logging
import argparse
import os
import glob
import traceback

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

BAND_WAVELENGTHS = {
    1: "0.47 µm",
    2: "0.55 µm",
    3: "0.65 µm",
    4: "0.865 µm",
    5: "1.38 µm",
    6: "1.64 µm",
    7: "2.13 µm"
}

def plot_results(data_dir, result_dir, output_dir):
    """可视化插值结果 - 四张图片，每张包含7个子图"""
    os.makedirs(output_dir, exist_ok=True)

    logging.info(f"数据目录: {data_dir}")
    logging.info(f"结果目录: {result_dir}")
    logging.info(f"输出目录: {output_dir}")

    # 检查目录是否存在
    if not os.path.exists(data_dir):
        logging.error(f"数据目录不存在: {data_dir}")
        return

    if not os.path.exists(result_dir):
        logging.error(f"结果目录不存在: {result_dir}")
        return

    # 查找所有插值结果文件
    result_files = glob.glob(os.path.join(result_dir, 'interp_ch*_*.csv'))
    logging.info(f"找到 {len(result_files)} 个插值结果文件")

    if not result_files:
        logging.warning("没有找到任何插值结果文件")
        return

    # 加载预处理数据
    data_file = os.path.join(data_dir, 'preprocessed_data.csv')
    if not os.path.exists(data_file):
        logging.error(f"预处理数据文件不存在: {data_file}")
        return

    try:
        data_df = pd.read_csv(data_file)
        data_df['date'] = pd.to_datetime(data_df['date'])
        logging.info(f"成功加载预处理数据，形状: {data_df.shape}")
    except Exception as e:
        logging.error(f"加载预处理数据失败: {str(e)}")
        return

    # 定义要处理的参数组合
    cal_types = ['RAD', 'REF']
    params = ['Slope', 'Intercept']
    channels = range(1, 8)  # 7个通道

    # 设置绘图样式
    plt.style.use('seaborn-whitegrid')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'figure.dpi': 500,
        'figure.figsize': (14, 20),
    })

    processed_count = 0
    error_count = 0

    # 为每种校准类型和参数组合创建图表
    for cal_type in cal_types:
        for param in params:
            try:
                # 创建2列4行的子图布局（共8个位置，使用7个）
                fig, axes = plt.subplots(7, 1, figsize=(10, 16))
                axes = axes.flatten()  # 将二维数组展平为一维

                # 为每个通道创建子图
                for i, channel in enumerate(channels):
                    ax = axes[i]

                    # 查找对应的结果文件
                    result_pattern = os.path.join(result_dir, f'interp_ch{channel}_{cal_type}_{param}.csv')
                    result_files = glob.glob(result_pattern)

                    if not result_files:
                        logging.warning(f"未找到结果文件: {result_pattern}")
                        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(f'Channel {channel}', fontsize=12)
                        continue

                    result_file = result_files[0]

                    # 加载插值结果
                    interp_df = pd.read_csv(result_file)
                    if 'date' not in interp_df.columns:
                        logging.warning(f"文件 {result_file} 缺少日期列")
                        continue

                    interp_df['date'] = pd.to_datetime(interp_df['date'])

                    # 筛选对应通道和类型的数据
                    channel_data = data_df[
                        (data_df['channel'] == int(channel)) &
                        (data_df['type'] == cal_type) &
                        (data_df['parameter'] == param)
                        ]

                    if channel_data.empty:
                        logging.warning(f"没有找到对应的原始数据: 通道 {channel}, 类型 {cal_type}, 参数 {param}")
                        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(f'Channel {channel}', fontsize=12)
                        continue

                    # 绘制原始数据点
                    ax.scatter(
                        channel_data['date'], channel_data['value'],
                        c='#4C72B0', marker='o', s=40,
                        edgecolors='white', linewidth=1.0,
                        zorder=10
                    )

                    # 标记增强的数据点
                    if 'is_augmented' in channel_data.columns:
                        augmented_data = channel_data[channel_data['is_augmented']]
                        if not augmented_data.empty:
                            ax.scatter(
                                augmented_data['date'], augmented_data['value'],
                                c='#2ca02c', marker='^', s=40,
                                edgecolors='white', linewidth=0.8, alpha=0.8,
                                zorder=9
                            )

                    # 绘制插值曲线
                    ax.plot(
                        interp_df['date'].to_numpy(),
                        interp_df['value'].to_numpy(),
                        color='#C44E52', linewidth=2.0, alpha=0.95
                    )

                    # 添加不确定性区间
                    if 'std' in interp_df.columns:
                        ax.fill_between(
                            interp_df['date'],
                            interp_df['value'] - 2 * interp_df['std'],
                            interp_df['value'] + 2 * interp_df['std'],
                            alpha=0.25, color='lightgrey', label='95% CI'
                        )

                    # 设置子图属性
                    ax.text(0.5, 0.97, f'B{i + 1:02d} ({BAND_WAVELENGTHS[i + 1]})',
                            transform=ax.transAxes,
                            fontsize=14, fontweight='bold',
                            horizontalalignment='center', verticalalignment='top'
                            )
                    #ax.set_title(f'B{i+1:02d}({BAND_WAVELENGTHS[i+1]})', fontsize=20)

                    # 设置日期格式
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%-m'))
                    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

                    if i < 6:
                        ax.set_xticklabels([])
                    else:
                        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

                # 调整布局
                plt.tight_layout(rect=[0, 0, 1, 1])

                # 保存图表
                output_file = os.path.join(output_dir, f"{cal_type}_{param}_interpolation.png")
                plt.savefig(output_file, bbox_inches='tight', dpi=500)
                plt.close(fig)

                processed_count += 1
                logging.info(f"已生成 {cal_type} {param} 的图表")

            except Exception as e:
                error_count += 1
                logging.error(f"绘制 {cal_type} {param} 时出错: {str(e)}")
                logging.error(traceback.format_exc())
                continue

    logging.info(f"可视化完成! 成功处理 {processed_count} 个图表，失败 {error_count} 个图表")


def main():
    parser = argparse.ArgumentParser(description='结果可视化')
    parser.add_argument('--data_dir', required=True, help='预处理数据目录')
    parser.add_argument('--result_dir', required=True, help='插值结果目录')
    parser.add_argument('--output_dir', required=True, help='可视化输出目录')

    args = parser.parse_args()

    logging.info("开始结果可视化...")
    plot_results(args.data_dir, args.result_dir, args.output_dir)
    logging.info("可视化完成!")


if __name__ == "__main__":
    main()