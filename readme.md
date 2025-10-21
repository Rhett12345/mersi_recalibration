# 🌏 MERSI Recalibration Project | MERSI再定标项目

> 📘 **A supplementary dataset and toolset for the research on satellite cross-calibration and temporal interpolation.**  
> 📘 **卫星交叉定标与时间插值研究的补充数据与工具集。**

本项目为论文的**补充材料（Supplementary Material）**，包含再定标结果与插值代码，  
并将根据研究进展 **持续更新** 数据与方法。

---

## 📂 Project Structure | 项目结构

├── 📁 fy3d_mersi/ # 再定标结果数据
│ ├── 201801/
│ │ ├── RAD_202501.csv
│ │ ├── RAD_202501.h5
│ ├── 201802/
│ │ ├── ...
│ └── ...
│
├──📁 GPR_interpolation/ # 插值算法与分析代码
│ ├── main_control.py
│ ├── data_preprocess.py
│ ├── gaussian_interpolation.py
│ ├── result_plot.py
├── LICENSE
└── README.md


### 🛰️ `fy3d_mersi/`
- Contains **monthly updated** recalibration results in `.csv` and `.h5` formats.  
- 包含**按月更新**的再定标结果，提供 `.csv` 与 `.h5` 两种格式。  
- 每个文件夹对应一个月的再定标输出结果。  

### 📈 `GPR_interpolation/`
- Provides scripts for **daily interpolation** of recalibration coefficients.  
- 包含用于**逐日插值**再定标系数的脚本。  
- 支持数据可视化与结果分析。  

---

## 🧩 How to Use Interpolation | 插值使用方法
 **Clone the repository | 克隆仓库**
cd GPR_interpolation | 进入项目目录
python3 main_control.py | 运行插值脚本

🧪 Purpose | 项目目的
This project serves as a supplementary resource for our research on:

Cross-calibration between satellite sensors

Temporal consistency analysis

Daily recalibration generation via interpolation

本项目旨在作为论文研究的补充资料，用于：

不同卫星传感器的交叉定标研究

时间一致性分析与验证

通过插值生成逐日再定标结果

📅 Updates | 更新计划
🔄 Recalibration results will be updated monthly.

🧠 Interpolation code will include algorithmic improvements over time.

🔄 再定标数据将按月更新；

🧠 插值算法将随着研究进展持续优化与扩展。

📜 Citation | 引用说明
If you use this dataset or code, please cite our paper after publication:

Author(s), “Title,” Journal, Year.

若您在研究中使用了本项目的数据或代码，请在论文发表后引用：

作者, “论文题目,” 期刊名称, 年份。

🤝 Contact | 联系方式
Author / 作者： Min MIn， Qiang Yu

Email / 邮箱： [minm5@mail.sysu.edu.cn，yuqiang6@mail2.sysu.edu.cn]

Institution / 单位： [School of Atmospheric Sciences and Guangdong Province Key Laboratory for Climate Change and Natural Disaster Studies, Sun Yat-sen University and Southern Marine Science and Engineering Guangdong Laboratory (Zhuhai), Zhuhai 519082, China]

⭐ If you find this project useful, please give it a star!
⭐ 如果本项目对您的研究有帮助，请为它点亮一颗星！