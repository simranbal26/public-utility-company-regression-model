# public-utility-company-regression-model
# use this like a landing page for your project, you can edit the readme like a text cell in colab file
# AFM244 Q8 — Quarterly Revenue Forecasting for Target Corp

A time-series regression project that models and forecasts Target Corporation's
quarterly revenue using a linear trend plus seasonal dummy variables, built in
Google Colab with `pandas` and `statsmodels`.

---

## Overview

The notebook walks through a standard analytics workflow:

1. **Data exploration** — load quarterly sales data, filter to a single firm, plot it
2. **Pattern identification** — identify trend and seasonality visually
3. **Feature engineering** — build a time index and quarterly dummy variables
4. **Train/test split** — 75% training, 25% holdout
5. **Model development** — OLS regression on the training set
6. **Model assessment** — MAPE on the holdout set, plus an actual-vs-predicted chart
7. **Communication** — a written memo addressed to Chapman Wealth Management

---

## Requirements

```python
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import pandas as pd
```

All four are pre-installed in Google Colab. No additional installs are needed.

---

## Data

**File:** `qSales_2024.csv` (must be uploaded to the Colab session, or mounted
from Drive, before running)

**Shape:** 277 rows × 16 columns

The dataset is in Compustat quarterly fundamentals format. Columns include:

| Column | Meaning |
|---|---|
| `gvkey` | Compustat firm identifier |
| `datadate` | Fiscal period end date |
| `fyearq` / `fqtr` / `fyr` | Fiscal year, fiscal quarter, fiscal year-end month |
| `tic` / `conm` | Ticker and company name |
| `saleq` | **Quarterly revenue** (the dependent variable) |
| `curcdq` | Currency (USD throughout) |
| `datacqtr` / `datafqtr` | Calendar quarter and fiscal quarter labels |
| `indfmt`, `consol`, `popsrc`, `datafmt`, `costat` | Compustat format/status flags |

The file covers several firms (Apple and Nintendo also appear); this analysis
filters to `tic == 'TGT'`, which yields **93 quarterly observations** spanning
fiscal periods ending 2001-01-31 through 2024-01-31.

Note that Target's fiscal year ends in January (`fyr = 1`), so fiscal Q4 ends in
late January and captures the holiday shopping season.

---

## Method

### Feature engineering

- `time` — a sequential integer index from 1 to 93, capturing the linear trend
- `dv_q4` — 1 if `fqtr == 4`, else 0 (the visible revenue spikes)
- `dv_q1` — 1 if `fqtr == 1`, else 0 (the visible post-holiday dips)

Q2 and Q3 are the omitted baseline category.

### Split

```python
dt4training = target_sales[:int(0.75*len(target_sales))]   # first 75% of rows
dt4testing  = target_sales[int(0.75*len(target_sales)):]   # last 25% of rows
```

With 93 rows this gives **69 training** observations and **24 testing**
observations. The split is chronological, not random, which is appropriate for
time-series forecasting.

### Model

Ordinary least squares, fit on the training set only:

```
saleq = const + β₁·time + β₂·dv_q4 + β₃·dv_q1
```

Fitted coefficients:

| Term | Coefficient |
|---|---|
| `const` | 9,399.40 |
| `time` | 139.15 |
| `dv_q4` | 4,472.62 |
| `dv_q1` | −232.80 |

**Interpretation:** revenue grows by roughly $139M per quarter on average; Q4
runs about $4,473M above the Q2/Q3 baseline; Q1 runs about $233M below it.
(All figures in millions USD, matching the units of `saleq`.)

---

## Results

**Test-set MAPE: ≈ 0.1197 (about 12.0%)**

Predictions track the seasonal shape of the holdout period well but sit
systematically below actuals from roughly 2020 onward — the linear trend fitted
on 2001–2018 data does not capture the sharper revenue growth Target
experienced afterward. This is the main visible weakness in the actual-vs-
predicted chart.

---

## How to run

1. Open the notebook in Google Colab.
2. Upload `qSales_2024.csv` to the session (Files pane → Upload), or mount Google
   Drive and adjust the path in `pd.read_csv`.
3. Run all cells top to bottom. The cells are order-dependent — the model cell
   relies on `time` and the dummy columns created earlier.

---

## Known issues / notes

- **`SettingWithCopyWarning`** appears in several cells because `target_sales`
  and `dt4testing` are slices of the parent DataFrame. The assignments still
  work, but the warnings can be removed by calling `.copy()` immediately after
  each filter/slice, e.g. `target_sales = qSales.loc[qSales['tic']=='TGT'].copy()`.
- The model is fit once on the training window and never re-fit; there is no
  rolling or expanding-window re-estimation.
- MAPE is the only accuracy metric reported. RMSE or MAE would add useful
  context, since MAPE can be misleading when the series level changes a lot.
- Only Q4 and Q1 are dummied. A full set of quarterly dummies, or a log
  transformation of revenue to allow multiplicative growth, are natural next
  steps.

---

## Files

| File | Description |
|---|---|
| `AFM244_Q8.ipynb` | The Colab notebook containing the full analysis |
| `qSales_2024.csv` | Input dataset (not included in this repo — supplied with the assignment) |

---

