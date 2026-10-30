# Observed-date calibration delivery

Code commit: `19686cf4e97814be8be249fde8f09aac8d660cbc`

Default outputs now use YYYYMMDD/CAL_RAD_ch01toch07_YYYYMMDD.png and .csv.
Each date is fitted independently. No interpolation, annual or period grouping.
22 unit tests passed. The main-entry smoke confirmed one date/one PNG and CSV, with no aggregate fit output.
Full fitting/rendering/verification ran on node111/node112.
Missing bands are NaN with explicit reasons; a rendered figure does not imply seven valid channel fits.

```json
{
  "FY3D": {
    "observed_dates": 171,
    "complete_seven_band_dates": 135,
    "partial_fit_dates": 35,
    "no_fit_dates": 1
  },
  "FY3F": {
    "observed_dates": 45,
    "complete_seven_band_dates": 32,
    "partial_fit_dates": 13,
    "no_fit_dates": 0
  }
}
```
