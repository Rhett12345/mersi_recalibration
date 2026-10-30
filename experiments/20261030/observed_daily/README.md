# 每个实际观测日独立再定标输出

本目录按原定标输出形式组织。每个 YYYYMMDD 文件夹只使用当天有效匹配
样本，独立进行两遍 OLS/2σ 回归，输出一张含 B1–B7 的 4×2 子图；第八格
留空。它不是年度/分段样本汇总，也没有逐日插值、补日或平滑系数。

```
FY3D/YYYYMMDD/CAL_RAD_ch01toch07_YYYYMMDD.png
FY3D/YYYYMMDD/CAL_RAD_ch01toch07_YYYYMMDD.csv
FY3F/YYYYMMDD/CAL_RAD_ch01toch07_YYYYMMDD.png
FY3F/YYYYMMDD/CAL_RAD_ch01toch07_YYYYMMDD.csv
```

CSV 保持原来的 `# Channel,Intercept,Slope` 及七位小数。每个日期另有
fit_statistics.json，记录全精度系数、回归样本数、R 和不足以拟合的波段。
缺乏足够有效样本的波段明确标记，并保存 NaN，不借用邻日或汇总系数。
没有接受匹配的输入日期不生成虚构结果。

当前选用 SFP 光谱转换 RAD 分支，一天一张图。历史 REF/RAD 原文件未覆盖。
原来每日期目录下还存在直接反射率 REF 分支，本轮未新增该分支输出。
散点颜色以全部保留拟合样本的分箱密度计算，显示点上限 30000/波段；
N/R/方程和系数均使用当天全部保留拟合样本。虚线沿用旧坐标范围参考线，
不是 DN 与反射率的物理 1:1 线。

默认 run_crosscal.py/run_fy3f.py 输出现已改为上述每日结构。
独立日期精度验证由 evaluate_fixed.py 单独执行；不会自动输出年度/分段
七子图。此前 experiments/20261029/fixed_joint 下的年度拟合图保留为
历史实验记录，不能视为本目录的每日定标输出。

本次恢复日分组、单日拟合和文件接口。使用现有已验证匹配缓存和当前
光谱/QC/DN 定义，没有为匹配旧论文数字而调整方法，因此不宣称复现
TGRS Fig.4 的系数。旧生产目录及历史图、CSV 均未覆盖。

运行（计算节点，现有依赖，无需重匹配）：

```sh
python tools/run_daily_calibration.py --sensor FY3D --records /absolute/fy3d_matched/scene_records.json --output /absolute/observed_daily/FY3D --workers 4
python tools/run_daily_calibration.py --sensor FY3F --records /absolute/fy3f_full/scene_records.json --output /absolute/observed_daily/FY3F --workers 4
```

加 `--date 20181003` 可只重绘这一实际观测日。缺测日不补全。
同日拟合图用于回归诊断，不能当成独立验证精度。
