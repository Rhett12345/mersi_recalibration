# Figure legends

**Figure 1 | Fixed-coefficient calibration on held-out dates.** Each platform
has a native/recalibrated pair of panels for B1–B7. Color indicates log-scaled
counts of all validation pairs per bin, with identical within-band color scales
and axes before and after. Dashed lines indicate 1:1 agreement. Ref* is
radiance-equivalent reflectance in percent. RMSE and MAE are percentage points;
Bias in panel annotations is relative mean bias. n counts matched pixels, not
independent replicates. Combined 0.1/99.9 percentile display bounds omit tails
from the display only; their counts are reported in summary.json, and metrics
retain every eligible pair.

**Figure 2 | Bandwise changes in radiometric agreement.** Top: relative mean
bias. Middle: RMSE. Bottom: 100(1-RMSE_after/RMSE_before). D uses pooled held-out
predictions from prespecified yearly fixed fits; F uses its post-update fixed
fit. Each before/after comparison uses identical observations. Lines indicate
95% percentile intervals from 500 paired whole-date bootstrap replicates,
conditional on fitted coefficients. Orange identifies an interval containing
zero; blue identifies a positive interval. These are not familywise-adjusted
hypothesis tests. Platforms have different periods and scene populations;
bar heights are not a controlled comparison of intrinsic sensor performance.

**Figure 3 | Residual distributions on the same held-out samples.** Residual is
observation minus spectrally transferred reference, in percentage points.
Native and recalibrated distributions share bins. Probability density uses all
eligible pairs as its denominator. Display tails are recorded separately and
are included in summary metrics. No negative values are clipped.

**Figure S1 | Geometry-conditioned residual bias.** Markers show mean residual
in observed 5-degree SZA bins, with circles for native observations and squares
for recalibrated observations. Source tables provide bin counts, VZA and
reference-level diagnostics. No interpolation or fitted trend is shown.
Geometry, season and surface composition may be confounded.
