# Retain valid observed pixels and attempt every paired date

The simulation-radiance training range and the per-scene minimum-pixel cut have been removed. Other QA, physical validity, geometric, SZA and CV rules are unchanged. All same-day valid pixels are pooled before calibration. Every paired date is attempted, with explicit NaN/reasons when a channel cannot be fitted. No temporal interpolation or cross-date coefficient substitution is used.

## Coverage and fit support

| Sensor | Paired dates attempted | Seven fitted bands | Partial fits | No fitted band | Dates with a fit: old -> new |
|---|---:|---:|---:|---:|---:|
| FY3D | 261 | 175 | 5 | 81 | 170 -> 180 |
| FY3F | 66 | 45 | 0 | 21 | 45 -> 45 |

No-fit dates remain visible; date coverage is not falsely equated with scientifically valid calibration. SZA and CV can still leave some dates without usable observations.

## Valid pixel support before daily regression

| Sensor | Band | Old candidate N | New candidate N |
|---|---|---:|---:|
| FY3D | B01 | 1,354,456 | 3,765,537 |
| FY3D | B02 | 1,543,414 | 3,766,994 |
| FY3D | B03 | 1,676,536 | 3,765,940 |
| FY3D | B04 | 1,711,480 | 3,682,289 |
| FY3D | B05 | 1,851,406 | 1,907,120 |
| FY3D | B06 | 654,244 | 2,100,411 |
| FY3D | B07 | 687,303 | 3,745,567 |
| FY3F | B01 | 841,512 | 1,093,281 |
| FY3F | B02 | 932,548 | 1,093,281 |
| FY3F | B03 | 940,664 | 1,093,281 |
| FY3F | B04 | 1,019,408 | 1,093,281 |
| FY3F | B05 | 1,076,331 | 1,089,661 |
| FY3F | B06 | 280,528 | 1,090,222 |
| FY3F | B07 | 209,408 | 1,090,397 |

## Independent-date comparison

Original fixed-experiment training and validation dates are frozen. Native calibration, old model and newly fitted model are evaluated on identical expanded held-out observations. New paired dates not present in the original split are processed daily but are not added to this fixed-model comparison. Numbers below are Ref*, with RMSE in reflectance percentage points. Daily fit figures themselves are not independent accuracy tests.

| Sensor | Band | Held-out N | Native RMSE | Old-model RMSE | New-model RMSE | New vs native reduction (%) | New vs old reduction (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| FY3D | B01 | 608,555 | 1.477 | 0.970 | 0.796 | 46.14 | 17.93 |
| FY3D | B02 | 608,559 | 0.893 | 0.787 | 0.727 | 18.55 | 7.63 |
| FY3D | B03 | 608,553 | 1.149 | 0.880 | 0.757 | 34.16 | 14.02 |
| FY3D | B04 | 586,852 | 0.791 | 0.825 | 0.792 | -0.17 | 4.05 |
| FY3D | B05 | 494,593 | 1.174 | 0.673 | 0.702 | 40.24 | -4.26 |
| FY3D | B06 | 341,368 | 1.172 | 1.072 | 0.845 | 27.92 | 21.21 |
| FY3D | B07 | 600,394 | 1.029 | 0.722 | 0.620 | 39.78 | 14.15 |
| FY3F | B01 | 236,910 | 1.744 | 0.638 | 0.642 | 63.20 | -0.64 |
| FY3F | B02 | 236,910 | 1.525 | 0.476 | 0.473 | 68.99 | 0.72 |
| FY3F | B03 | 236,910 | 1.860 | 0.538 | 0.536 | 71.20 | 0.36 |
| FY3F | B04 | 236,910 | 2.159 | 0.637 | 0.637 | 70.49 | -0.04 |
| FY3F | B05 | 236,095 | 0.964 | 0.197 | 0.197 | 79.54 | -0.09 |
| FY3F | B06 | 235,901 | 1.789 | 1.082 | 0.600 | 66.47 | 44.54 |
| FY3F | B07 | 236,090 | 0.746 | 1.006 | 0.442 | 40.74 | 56.06 |

Negative reductions are retained: the new model is not uniformly better. FY-3D B4 remains close to native performance; FY-3D B5 has a modest deterioration relative to the old model on these same held-out pairs. Bias/MAE, TOA metrics and date-bootstrap intervals are in evaluation/metrics.csv and evaluation/evaluation.json. These are MODIS-relative agreement results, not absolute truth.

## Output layout

`SENSOR/daily/YYYYMMDD/CAL_RAD_ch01toch07_YYYYMMDD.png` and `.csv` preserve the daily seven-panel layout and coefficient table schema. Each day also has fit_statistics.json. Missing bands use NaN and a reason. The selected branch is SFP RAD.

## Reproducibility

tools/revalidate_scene_caches.py recomputes only masks using real MODIS UI and verifies exact reconstruction of each original accepted mask. All other stored arrays were checked unchanged; discarded small scenes were rematched from real L1/GEO files. Revalidation summaries remain in matches/. Source records with known corrupt/invalid geometry inputs remain explicit.

The prior policy results in ../observed_daily/ are historical. This directory is the updated policy. Raw satellite data, pixel caches and logs are not committed.

## Inspect coefficient scatter and gaps

[Daily coefficient timelines and date lists](../coefficient_timeline/README.md) show all actual slopes/intercepts and missing paired dates without altering coefficients or QC.
