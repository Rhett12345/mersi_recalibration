# Actual daily coefficient scatter and date coverage

FY3D/FY3F_daily_coefficients plots seven band slopes and intercepts from actual
same-day fits. No fit, smoothing, interpolation, connection lines, coefficient
clipping or outlier removal is performed here. All finite accepted coefficients
are displayed, regardless of how unusual they look. Orange crosses at the axis
bottom mark paired dates without a coefficient; their vertical position is a
status marker, not a zero coefficient.

FY3D_FY3F_date_coverage uses real daily calendar spacing. Blue: coefficient
exists; orange: paired input date but no fitted coefficient; light gray: no
paired input in this run. Gray calendar days are not automatically missing
required observations; the original acquisition scheme is sparse.

*_coefficient_source.csv contains full-precision plotted values, N and R.
*_paired_date_status.csv contains all paired dates and per-stage counts.
*_missing_paired_dates.csv lists dates with at least one missing band and the
first stage leaving no pixels for wholly unfitted dates. For partial days the
coefficient source table gives per-band reasons. *_no_paired_input_calendar.csv
lists calendar days without an input pair, without declaring them required.

These are coefficient-availability diagnostics, not independent accuracy or
long-term stability validation. All plotted values were checked against the
same-date legacy-format coefficient CSV within its 7-decimal rounding precision.
