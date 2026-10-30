# FY-3D / FY-3F MERSI calibration coefficients

This repository contains coefficient CSVs and their provenance/quality tables.
Scientific code is maintained in [re-calibration](https://github.com/Rhett12345/re-calibration),
including interpolation, validation and plotting scripts. Images, HDF/NPZ data,
intermediate results and source scripts are not part of this repository tree.

## Two distinct collections

- **FY3D/observed/** and **FY3F/observed/**: actual single-date cross-calibration coefficients. Original
  per-date CSV bytes are preserved. FY3D has 180 dates (175 complete, 5 partial);
  FY3F has 45 complete dates. A failed paired date appears only in the metadata
  manifest and is never an interpolation anchor. No failed date is relabelled
  as an observed coefficient.
- **FY3D/interpolated/** and **FY3F/interpolated/**: daily temporal-model estimates. FY3D spans 2018-01-03 to
  2025-01-15 (2,570 days); FY3F spans 2024-07-04 to 2026-07-12 (739 days).
  Each band is limited to its own observed support. No boundary extrapolation.
  Estimates at observed dates are labelled model estimates, and never replace
  the original observations. Consult status and gap flags before application.

## Files and interpretation

Each dated CSV keeps `# Channel,Intercept,Slope`, seven channel rows, and
`Ref* = Intercept + Slope * DN`, where Ref* uses the existing reduced-reflectance
percentage-point convention. This is not radiance in SI units. TOA conversion
uses the existing Earth-Sun-distance and solar-zenith factor. FY3D/MERSI-II uses
Aqua MODIS; FY3F/MERSI-III uses Terra MODIS with sensor-specific spectral transfer.
Never exchange their coefficients or channel spectral definitions.

`SENSOR/observed/coefficient_manifest.csv` records full-precision coefficients,
sample counts, fit status and reasons, including failed paired dates. The dated
observed CSVs retain the original seven-decimal output precision.
`SENSOR/interpolated/daily_coefficients.csv` records full-precision daily estimates,
method, source support and gap flags; dated estimated files use twelve decimals.
`model_manifest.csv` records the method, regime, anchor bounds and date counts.
The quadratic field is zero, not an additional fitted parameter.

FY3F B5–B7 are modelled separately across the native calibration update on
2025-01-22. Their 2025-01-13 through 2025-02-02 rows remain unsupported/NaN.
B1–B4 remain available on those days. Never bridge that gap by forward filling.
Long interior gaps are retained and labelled; they are weakly constrained, not
independent daily observations. Missing dates are not filled beyond the first
or last observed anchor of each applicable segment.

The current methods were selected among ten candidates using blocked-date
radiometric RMSE and a conservative one-standard-error rule. Full-anchor models
are linear for FY3D B1–B6 and FY3F B1–B7 (separate regimes where applicable),
and smoothing spline strength 10 for FY3D B7. Model development was exploratory;
internal nested validation does not establish absolute accuracy or externally
confirmed long-term stability. Empirical coefficient-error percentiles describe
held-out observed dates; they are not formal confidence intervals for every day.

## Historical data preservation

The previous tree is preserved in Git history at commit
`92355e59484fc96e36c50bdfdf6225c89cfe3a7a` and in a complete external server
archive. Historical continuous FY3D CSV/HDF series do not have sufficient
per-date provenance to relabel all of them as actual observations. They are
therefore not mixed into the observed collections. The four historical GPR scripts and their
license are preserved in the code repository under
`legacy/results_repository_GPR_interpolation/`.

The current tree contains no images or raw observation files. Git history was
preserved without force pushes; old binary objects remain recoverable in history.
For algorithms, commands, validation limits and reproducibility, see
[the product documentation](https://github.com/Rhett12345/re-calibration/blob/main/docs/DAILY_COEFFICIENT_PRODUCT.md).

FY3D and FY3F have separate top-level directories. All coefficients and indexes
are CSV; this repository contains no JSON diagnostics or figures.
