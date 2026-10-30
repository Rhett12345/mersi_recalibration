# Per-fit seven-band diagnostics

Original 4x2 arrangement: B1-B7 with the eighth panel blank. Each actual
fixed-coefficient fit emits CAL_RAD_ch01toch07_SENSOR_PERIOD.png, .pdf and .csv.
The CSV retains the original Channel,Intercept,Slope columns and seven decimal
places. No daily interpolation or unobserved-date coefficients are generated.
Scatter colors are full retained-population relative bin frequencies rather
than the legacy quadratic-cost KDE; displayed points are deterministically
capped at 30000 per band. N/R/equations use all retained training pairs. Dashed
lines retain the old DN-axis/0-100 range guide and are not physical 1:1 lines.
Axes retain the original limits; out-of-view sample counts are in
fit_statistics.json. These fit diagnostics do not establish held-out accuracy.
