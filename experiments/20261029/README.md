# 2026-10-29 FY-3D优化与FY-3F真实交叉定标实验

状态：全部463组观测候选处理完成，主要结果是分原生定标历元后的初步相对定标，尚未替换生产。

- [代码优化报告](CODE_OPTIMIZATION_REPORT.md)
- [FY-3F适配与运行说明](FY3F_ADAPTATION.md)
- [真实定标系数、指标及图件](FY3F_RESULTS.md)
- [科学可靠性与限制](SCIENTIFIC_ASSESSMENT.md)
- [双精度系数、日期簇置信区间和验证指标CSV](coefficients_post_update.csv)
- [代码级科学审查](https://github.com/Rhett12345/re-calibration/blob/main/docs/SCIENCE_AUDIT.md)

`fy3f_full`保留全时期混合诊断，`fy3f_pre_update`和`fy3f_post_update`分别保存原生定标更新前后结果。中间缓存、真实像元数组和下载参考文件均不进入Git。本目录仅保存小型统计、核验记录、报告和图件。

- [FY-3F逐日插值、三种方法比较及反射率提升](daily_interpolation/REPORT.md)

## Joint fixed-coefficient validation

[FY-3D/FY-3F report](fixed_joint/REPORT.md) includes held-out Bias, RMSE, MAE, failure cases, publication figures and numerical comparisons. No daily interpolation is used.

## Current per-date calibration output

[Observed-date results](../20261030/observed_daily/README.md) replace annual/period fit diagnostics as the calibration output. Each actual date has one seven-panel PNG and a companion coefficient CSV.
