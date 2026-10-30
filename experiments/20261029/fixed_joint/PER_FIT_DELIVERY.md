> Superseded for calibration output: see ../../20261030/observed_daily/. Each actual observation date now has its own independent fit and PNG/CSV. The annual/period figures below are historical diagnostics only.

# Per-fit diagnostic delivery

Code commit: `473036dddf587cff2426b381e08cc0add5d148dc`

Automatic output is enabled in run_crosscal.py / run_fy3f.py and evaluate_fixed.py.
Eight actual fits (seven FY-3D years and one FY-3F period) export seven-panel PNG/PDF and original-schema coefficient CSV.
All 56 per-band coefficients and sample counts match the existing fits; the main-entry smoke preserves prior validation metrics exactly.
20 unit tests passed. Full computations and rendering ran on compute nodes.
Source data, caches and smoke intermediate outputs are excluded from Git.
