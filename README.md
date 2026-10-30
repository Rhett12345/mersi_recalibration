# 🌏 MERSI-II Recalibration Project | MERSI-II 再定标项目

> 📘 **A supplementary dataset for satellite cross-calibration and temporal interpolation research, with the associated code maintained in a companion repository.**
> 📘 **卫星交叉定标与时间插值研究的补充数据集，配套算法与绘图代码由独立代码仓库维护。**

本仓库为以下 **IEEE Transactions on Geoscience and Remote Sensing (TGRS)** 论文研究提供补充材料（Supplementary Material）：

**Spectral-Fidelity-Preserving Recalibration of Reflective Solar Bands for MERSI-II onboard Fengyun-3D Satellite**

FY-3D / MERSI-II 是上述论文研究的主体。本仓库保留这一科研用途，并随研究进展持续更新数据与方法；FY-3F / MERSI-III 作为后续扩展实验单独管理，其结果与验证范围应独立说明，不能直接等同于原 FY-3D 论文结论。

This repository supports the associated TGRS research on FY-3D / MERSI-II recalibration. FY-3F / MERSI-III is maintained separately as a research extension. Its coefficients and validation evidence must be interpreted independently of the original FY-3D study.

- **Coefficient dataset | 系数数据：** [mersi_recalibration](https://github.com/Rhett12345/mersi_recalibration)
- **Algorithms and plotting | 算法与绘图：** [re-calibration](https://github.com/Rhett12345/re-calibration)

---

## 🧪 Purpose | 项目目的

This project serves as a supplementary resource for research on:

- Cross-calibration between satellite sensors.
- Temporal consistency analysis and validation.
- Daily recalibration coefficient estimation through temporal interpolation.

本项目旨在作为论文研究的补充资料，用于不同卫星传感器的交叉定标、时间一致性分析与验证，以及通过时间插值估计逐日再定标系数。**插值得到的每日系数是模型估计，不代表每天都有独立交叉观测，也不直接证明长期辐射稳定性。**

## 📂 Project Structure | 项目结构

FY-3D 与 FY-3F 在最外层分开；每颗卫星内部再区分真实交叉定标系数和插值系数。

```text
mersi_recalibration/
├── FY3D/
│   ├── observed/
│   │   ├── YYYYMMDD/
│   │   │   └── CAL_RAD_ch01toch07_YYYYMMDD.csv
│   │   └── coefficient_manifest.csv
│   └── interpolated/
│       ├── YYYYMMDD/
│       │   └── CAL_RAD_ch01toch07_YYYYMMDD.csv
│       ├── daily_coefficients.csv
│       ├── model_manifest.csv
│       ├── completion_manifest.csv
│       └── paired_date_coefficients.csv
├── FY3F/
│   ├── observed/
│   │   ├── YYYYMMDD/
│   │   │   └── CAL_RAD_ch01toch07_YYYYMMDD.csv
│   │   └── coefficient_manifest.csv
│   └── interpolated/
│       ├── YYYYMMDD/
│       │   └── CAL_RAD_ch01toch07_YYYYMMDD.csv
│       ├── daily_coefficients.csv
│       ├── model_manifest.csv
│       ├── completion_manifest.csv
│       └── paired_date_coefficients.csv
├── LICENSE
└── README.md
```

### 🛰️ `FY3D/observed/` and `FY3F/observed/` | 真实交叉定标系数

These are coefficients fitted from actual matched satellite observations. Original dated coefficient CSVs are preserved without modification. Dated files contain seven actual fitted bands. Partial actual fits are retained in the coefficient manifest; complete usable observation-date sets are provided separately.

这些系数来自真实卫星匹配观测的单日拟合，原始单日 CSV 保持不变。日期文件仅在七个波段均有真实拟合时生成；部分通道的真实拟合结果仍保留于索引。`coefficient_manifest.csv` 记录全精度系数、有效样本数、拟合状态及原因；未能真实拟合的通道不作为独立观测锚点。

### 📈 `FY3D/interpolated/` and `FY3F/interpolated/` | 每日插值系数

These are daily temporal-model estimates within observed support. The actual observation fits remain in `observed/`; model estimates at those same dates are explicitly labelled in the daily table.

这些系数是在真实观测支持范围内得到的每日模型估计，**首尾不外推**。`daily_coefficients.csv` 提供全精度系数、方法、状态、相邻观测间隔及质量标记；`model_manifest.csv` 记录各波段的方法、分段和实际观测范围。模型在真实观测日期给出的估计标为 `model_estimate_at_observed_date`，不能将其另计为新增观测。

The current repository tree contains coefficient CSVs and the necessary CSV indexes, together with this README, the license and ignore rules. Figures, raw observations, pixel caches, JSON diagnostics and source scripts are maintained outside this coefficient repository.

当前版本只保存系数 CSV、必要的 CSV 索引和仓库说明文件；图像、原始观测、像元缓存、JSON 诊断及源代码不混入系数目录。

## 🧩 How to Use | 使用方法

1. **Clone or download | 克隆或下载**

   ```bash
   git clone https://github.com/Rhett12345/mersi_recalibration.git
   ```

2. **Choose the satellite and coefficient source | 选择卫星与系数来源**

   使用实际交叉定标结果时，读取相应卫星的 `observed/`；需要每日估计时，读取 `interpolated/`，同时查看索引中的支持范围、缺测状态和观测间隔。FY-3D 与 FY-3F 的系数不能互换。

   每个配对日期的完整七通道可用系数见 `interpolated/paired_date_coefficients.csv`。仅当原筛选不足时才分级补救；仍无法拟合的通道使用最近日期的真实系数，并以 `source_type`、`source_date` 和质量标记明确区分。此类兜底值不计为新增真实交叉定标观测。

3. **Read the coefficient convention | 核对系数定义**

   单日 CSV 保持七个通道的原有三列格式：

   ```text
   # Channel,Intercept,Slope
   ```

   当前线性再定标关系为 **`Ref* = Intercept + Slope × DN`**。`Ref*` 沿用本研究的约化反射率约定，以反射率百分数数值表示；TOA 反射率转换沿用研究代码中的日地距离和太阳天顶角因子。各传感器的光谱转换与波段定义应使用其对应配置。表格中的 `quadratic = 0` 不代表另外拟合了二次项。

   真实单日 CSV 保留原有七位小数输出，插值单日 CSV 使用十二位小数；需要全精度计算时，使用对应的汇总 CSV 索引。

4. **Reproduce interpolation and figures | 复现插值与绘图**

   插值、验证和绘图代码位于独立代码仓库。运行方法、依赖和数据要求见 [每日系数流程说明](https://github.com/Rhett12345/re-calibration/blob/main/docs/DAILY_COEFFICIENT_PRODUCT.md)，本轮实验的逐波段误差及局限见 [实验结果说明](https://github.com/Rhett12345/re-calibration/blob/main/docs/DAILY_COEFFICIENT_RESULTS_20261030.md)。全量计算应在计算节点或调度作业中运行。

## 📅 Updates | 更新计划

- 🔄 再定标数据计划在新观测与验证完成后按月整理更新，实际覆盖范围以发布版本及其 CSV 索引为准。
- 🧠 插值算法、时间一致性分析与验证方法将随研究进展持续优化；方法变化应与数据版本一并说明。
- 📌 论文复现与引用应记录使用的 Git 提交或正式发布版本，区分原论文材料与后续 FY-3D / FY-3F 扩展实验。

Recalibration releases are intended to be updated monthly as new observations and validation become available. Interpolation methods will continue to evolve with the research. Record the exact dataset revision when reproducing or citing results.

## 📜 Citation | 引用说明

If you use this dataset or its associated code, please cite the corresponding paper when its publication details are available, and identify the dataset revision used:

**“Spectral-Fidelity-Preserving Recalibration of Reflective Solar Bands for MERSI-II onboard Fengyun-3D Satellite.”**

若在研究中使用本数据或配套代码，请在论文正式出版信息明确后引用对应论文，并注明所用数据版本或 Git 提交。作者顺序、年份、卷期页码及 DOI 以正式论文记录为准。后续 FY-3F 扩展实验的引用信息将随研究进展另行更新。

## 🤝 Contact | 联系方式

- **Authors / 作者：** Min Min，Qiang Yu
- **Email / 邮箱：** [minm5@mail.sysu.edu.cn](mailto:minm5@mail.sysu.edu.cn)，[yuqiang6@mail2.sysu.edu.cn](mailto:yuqiang6@mail2.sysu.edu.cn)
- **Institution / 单位：** School of Atmospheric Sciences and Guangdong Province Key Laboratory for Climate Change and Natural Disaster Studies, Sun Yat-sen University and Southern Marine Science and Engineering Guangdong Laboratory (Zhuhai), Zhuhai 519082, China

## 📄 License | 许可

See [LICENSE](LICENSE) for the license terms. / 使用条款见 [LICENSE](LICENSE)。

⭐ If you find this project useful, please give it a star!
⭐ 如果本项目对您的研究有帮助，请为它点亮一颗星！
