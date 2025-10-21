# -*- coding: utf-8 -*-
"""
主控程序 - 协调数据预处理、高斯插值和可视化
"""

import os
import glob
import matplotlib

matplotlib.use('Agg')
from datetime import datetime, timedelta
import logging
import traceback
import argparse
import subprocess
import sys
import json

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class MainController:
    def __init__(self, data_root, output_dir):
        self.data_root = data_root
        self.output_dir = output_dir
        self.setup_directories()

    def setup_directories(self):
        """创建所有必要的输出目录"""
        self.dirs = {
            'preprocess': os.path.join(self.output_dir, 'preprocessed_data'),
            'interpolation': os.path.join(self.output_dir, 'interpolation_results'),
            'visualization': os.path.join(self.output_dir, 'plots'),
            'daily': os.path.join(self.output_dir, 'daily_results'),
            'samples': os.path.join(self.output_dir, 'sample_data'),
            'config': os.path.join(self.output_dir, 'config')
        }

        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)

        logging.info(f"输出目录结构已创建在: {self.output_dir}")

    def run_preprocessing(self):
        """运行数据预处理程序"""
        logging.info("启动数据预处理...")

        cmd = [
            sys.executable, 'data_preprocess.py',
            '--data_root', self.data_root,
            '--output_dir', self.dirs['preprocess'],
            '--sample_dir', self.dirs['samples'],
        ]

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logging.info("数据预处理完成")
            logging.debug(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"数据预处理失败: {e}")
            logging.error(f"错误输出: {e.stderr}")
            return False

    def run_interpolation(self):
        """运行高斯插值程序"""
        logging.info("启动高斯过程插值...")

        cmd = [
            #sys.executable, 'gaussian_interpolation.py',
            sys.executable, 'interpolation_compare.py',
            '--input_dir', self.dirs['preprocess'],
            '--output_dir', self.dirs['interpolation'],
            '--daily_dir', self.dirs['daily']
        ]

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logging.info("高斯插值完成")
            logging.debug(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"高斯插值失败: {e}")
            logging.error(f"错误输出: {e.stderr}")
            return False

    def run_visualization(self):
        """运行可视化程序"""
        logging.info("启动结果可视化...")

        cmd = [
            #sys.executable, 'result_plot.py',
            sys.executable, 'plot_compare.py',
            '--data_dir', self.dirs['preprocess'],
            '--result_dir', self.dirs['interpolation'],
            '--output_dir', self.dirs['visualization']
        ]

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logging.info("可视化完成")
            logging.debug(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"可视化失败: {e}")
            logging.error(f"错误输出: {e.stderr}")
            return False

    def save_configuration(self):
        """保存运行配置"""
        config = {
            'data_root': self.data_root,
            'output_dir': self.output_dir,
            'timestamp': datetime.now().isoformat(),
            'directories': self.dirs
        }

        config_file = os.path.join(self.dirs['config'], 'run_config.json')
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        logging.info(f"配置已保存到: {config_file}")

    def run_pipeline(self):
        """运行完整的数据处理管道"""
        logging.info("=" * 50)
        logging.info("开始运行完整的数据处理管道")
        logging.info(f"数据根目录: {self.data_root}")
        logging.info(f"输出目录: {self.output_dir}")
        logging.info("=" * 50)

        # 检查输入目录是否存在
        if not os.path.exists(self.data_root):
            logging.error(f"输入数据目录不存在: {self.data_root}")
            return False

        # 检查输入目录中是否有文件
        pattern = os.path.join(self.data_root, '*', 'CAL_*_ch01toch07_*.csv')
        file_list = glob.glob(pattern)
        logging.info(f"找到 {len(file_list)} 个匹配的文件")

        if len(file_list) == 0:
            logging.error("没有找到任何匹配的CSV文件，请检查数据路径和文件命名模式")
            # 尝试其他模式
            other_patterns = [
                os.path.join(self.data_root, 'CAL_*_ch01toch07_*.csv'),
                os.path.join(self.data_root, '*', '*CAL*ch01toch07*.csv'),
                os.path.join(self.data_root, '*CAL*ch01toch07*.csv')
            ]
            for pattern in other_patterns:
                files = glob.glob(pattern)
                if files:
                    logging.info(f"使用模式 '{pattern}' 找到 {len(files)} 个文件")
                    file_list.extend(files)

            if not file_list:
                return False

        # 显示前几个文件
        for i, file_path in enumerate(file_list[:5]):
            logging.info(f"文件 {i + 1}: {os.path.basename(file_path)}")
        if len(file_list) > 5:
            logging.info(f"... 还有 {len(file_list) - 5} 个文件")

        self.save_configuration()

        # 运行预处理
        logging.info("启动数据预处理...")
        if not self.run_preprocessing():
            logging.error("预处理阶段失败，终止执行")
            return False

        # 检查预处理是否生成了文件
        preprocess_dir = self.dirs['preprocess']
        preprocess_files = os.listdir(preprocess_dir) if os.path.exists(preprocess_dir) else []
        logging.info(f"预处理目录文件: {preprocess_files}")

        # 运行插值
        logging.info("启动高斯过程插值...")
        if not self.run_interpolation():
            logging.error("插值阶段失败，终止执行")
            return False

        # 检查插值结果
        interpolation_dir = self.dirs['interpolation']
        interpolation_files = os.listdir(interpolation_dir) if os.path.exists(interpolation_dir) else []
        logging.info(f"插值目录文件: {interpolation_files}")

        # 运行可视化
        logging.info("启动结果可视化...")
        if not self.run_visualization():
            logging.warning("可视化阶段失败，但主要处理已完成")

        logging.info("=" * 50)
        logging.info("数据处理管道完成!")
        logging.info("=" * 50)
        return True

def main():
    parser = argparse.ArgumentParser(description='时间序列插值主控程序')
    parser.add_argument('--data_root',
                        default="/share/home/dq014/yuqiang/data/cali_result_last_bak/",
                        help='输入数据根目录')
    parser.add_argument('--output_dir',
                        default="/share/home/dq014/yuqiang/data/inter_gaussian_251017_com",
                        help='输出目录')

    args = parser.parse_args()

    controller = MainController(args.data_root, args.output_dir)
    controller.run_pipeline()


if __name__ == "__main__":
    main()