# FY-3F MERSI-III 适配说明

## 传感器与原生输入

FY-3D 搭载 MERSI-II，FY-3F 搭载 MERSI-III。两者都有 25 个波段，250 m/1 km 原生空间分辨率。本轮延续旧流程的七个反射波段：0.47、0.55、0.65、0.865、1.38、1.64、2.13 µm；没有宣称完成全部 25 个波段的定标。

| 项目 | FY-3D 兼容路径 | FY-3F 实验路径 |
|---|---|---|
| 参考仪器 | Aqua MODIS / MYD* | Terra MODIS / MOD* |
| 文件组织 | FY3D_MERSI_GBAL，旧入口假设 L1/GEO 同目录 | FY3F_MERSI_GRAN，1000m 与 GEO 分目录；直接使用联合清单绝对路径 |
| 可见光输入 | 保留旧读取器和旧输出格式 | EV_250_Aggr.1KM_RefSB 的 B1–B4；EV_1KM_RefSB 的 B5–B7 |
| 定标参数 | 保留历史 FY-3D 行为 | 每个文件自己的 VIS_Cal_Coeff 和 Solar_Irradiance |
| 地形字段 | 旧云检测程序曾假设 DEM | FY-3F 提供 Altitude；本次反射波段定标不读取地形，不做未经证实的字段替换 |
| 光谱处理 | 固定旧系数用于兼容性对照 | 官方 FY-3F SRF + NASA Terra SRF + 既有 MODTRAN 重建转换 |
| QA 与时间 | 不暗改旧筛选 | 原生扫描 QA、GEO 日计数及毫秒计数，并与 EV_start_time 交叉检查 |

## 复用和新增模块

`crosscal/optimized.py` 管理公共投影索引、兼容性窗口和一次拼接；`sensors.py` 管理原生平台身份、缩放、QA 和时间；`matching.py` 管理公共 LAEA 投影与显式科学筛选；`spectral.py` 管理 SRF、物理单位和光谱转换；`regression.py` 管理 OLS、共同样本指标和日期簇 bootstrap。工具脚本分别负责实验编排、回归验证、绘图和追溯信息。没有复制一套高度重复的 FY-3D/FY-3F 匹配实现。

FY-3F 原生单位遵循官方 L1 指南第6.1节：

```
dn = DN * Slope + Intercept
Ref_star[%] = Cal0 + Cal1*dn + Cal2*dn^2
L = Ref_star[%] * E0 / (100*pi)
Ref_TOA[%] = Ref_star[%] * distance_au^2 / cos(SZA)
```

归一化 DN 不再减一次 SV。EarthSun Distance Ratio 使用原生带空格的属性名，缺少关键物理属性直接报错。GEO Millisecond_Count 自带0.1缩放；正确读取后才转秒。MODIS TAI93 在本批2017—2026年范围扣除自1993年起增加的10个闰秒，超出已审计时间范围直接拒绝。

## 科学筛选及声明的变化

保留原研究的 MODIS 参考、七波段对应、1 km LAEA 最后写入规则、VZA<30°、观测余弦比差<0.01、SZA<75°、B3 的3×3变异系数≤2%、两次线性拟合及训练残差2σ筛选。FY-3F 增加真正的扫描时间差≤600 s、大圆距离≤1 km、有效完整邻域、原生逐波段QA、MODIS无效不确定度索引剔除和光谱训练域约束。这些修正有明确物理目的，没有作为“性能优化”悄悄加入旧 FY-3D 结果。

MOD35_L2 已配套存在，但本轮没有把 MOD35 当作真值、没有声称完成云检测验证，也没有把其结果用于训练筛选。后续若做晴空限定，需要单独设计并报告样本选择影响。

## 运行

代码仓库 README 给出逐条可运行命令。先在计算节点推导光谱转换，再运行 `tools/run_fy3f.py`；最后复用缓存，按2025-01-22业务定标更新节点分别评价前后时期。所有拟合、评价及绘图数据来自真实观测配对，MODTRAN 只用于光谱转换。

官方依据：[仪器](https://www.nsmc.org.cn/nsmc/en/instrument/MERSI-3.html)、[L1指南](https://img.nsmc.org.cn/PORTAL/NSMC/DATASERVICE/OperatingGuide/FY3F/FY-3F_L1_Data_Instruction_MERSI-III.pdf)、[运行事件](https://www.nsmc.org.cn/portal/cn/event/MERSI.html)、[NASA参数](https://mcst.gsfc.nasa.gov/calibration/parameters)。下载文件及源码SHA-256记录在 run_provenance.json。
