# Residential Electricity Short Term Load Demand Forecasting

A hybrid Prophet + LightGBM forecasting pipeline with a three-method explainable AI framework (SHAP, LIME, Anchors), evaluated across six California ASHRAE climate zones using the NREL ResStock dataset.

**Author:** Chaitra R
**Programme:** MSc Data Science, Liverpool John Moores University
**Final Result:** RMSLE 0.1473, R² 0.913 — 
---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Pipeline Walkthrough](#pipeline-walkthrough)
  - [Step 1 — Select Buildings](#step-1--select-buildings)
  - [Step 2 — Download Data](#step-2--download-data)
  - [Step 3 — Run the Forecasting Pipelines](#step-3--run-the-forecasting-pipelines)
- [Notebook Reference](#notebook-reference)

---

## Overview

This project forecasts hourly residential electricity demand at the **individual building level** across six California ASHRAE climate zones (2B, 3B, 3C, 4B, 4C, 5B), using 100 single-family detached buildings with no rooftop solar, sourced from NREL's ResStock End-Use Load Profiles dataset.

The pipeline progresses through three model configurations, each isolating a specific design decision:

| Configuration | Models | Features | RMSLE | R² |
|---|---|---|---|---|
| **1 — Baseline** | Holt-Winters + Prophet + Gradient Boosting | 58 | 0.1879 | 0.868 |
| **2 — Isolation** | Prophet + Gradient Boosting (no Holt-Winters) | 58 | 0.1856 | 0.863 |
| **3 — Enhanced ★** | Prophet + LightGBM | 93 | **0.1473** | **0.913** |

A separate ablation study quantifies Prophet's exact contribution to the final ensemble, and a three-method XAI framework (SHAP, LIME, Anchors) explains the enhanced model's predictions and produces deployable battery dispatch rules with 97–98% precision.

---

## Project Structure

```
.
├── README.md                              ← you are here
├── extractColumns.py                      ← Step 1: select 100 stratified buildings
├── downloadFiles.py                       ← Step 2: download timeseries + weather from S3
├── selected_buildings.csv                 ← output of Step 1 (committed for reproducibility)
│
├── LoadForecastingPipeline.ipynb          ← Configuration 1 (baseline, 58 features)
├── ProphetGB_Pipeline.ipynb               ← Configuration 2 (Holt-Winters removed)
├── LoadForecastingPipeline_R2improved.ipynb ← Configuration 3 (enhanced, 93 features, LightGBM)
├── GB_Only_Ablation.ipynb                 ← Prophet contribution ablation study
│
├── metadata/
│   └── CA_upgrade0.parquet                ← NREL ResStock building metadata (not committed — see Setup)
│
├── data/                                  ← created by downloadFiles.py
│   ├── timeseries_2018/                   ← per-building hourly consumption (AMY 2018)
│   ├── timeseries_2012/                   ← per-building hourly consumption (AMY 2012)
│   ├── weather_2018/                      ← per-county weather (AMY 2018)
│   ├── weather_2012/                      ← per-county weather (AMY 2012)
│   └── metadata/CA_upgrade0.parquet
│
├── run1_1year/                            ← Configuration 1 outputs, 1-year training
├── run2_2year/                            ← Configuration 1 outputs, 2-year training
├── run1_1year_r2/                         ← Configuration 3 outputs, 1-year training
├── run2_2year_r2/                         ← Configuration 3 outputs, 2-year training
├── run_prophet_gb_2year/                  ← Configuration 2 outputs
├── run_hybrid/                            ← Hybrid (2yr Prophet + 1yr GB) experiment outputs
└── run_ablation_gb_only/                  ← Ablation study outputs
```

Each `run*/` directory follows the same internal structure:

```
run*/
├── processed/        preprocessed_data.parquet, engineered_features.parquet, train/test splits
├── predictions/       per-model prediction parquet files
├── models/            pickled trained models, feature importance CSVs
└── results/
    ├── metrics_summary.csv
    ├── ensemble_weights.csv
    ├── xai/{shap,lime,anchors,plots}/
    └── evaluation/{plots}/
```

---

## Prerequisites

- Python 3.10+
- An AWS account is **not** required — the NREL ResStock data is hosted on a public, unsigned S3 bucket (`oedi-data-lake`)
- Internet access to `oedi-data-lake.s3.amazonaws.com`

### Python packages

```bash
pip install pandas numpy scikit-learn statsmodels prophet lightgbm \
            shap lime anchor-exp boto3 matplotlib --break-system-packages
```

| Package | Used for |
|---|---|
| `pandas`, `numpy` | Data wrangling, feature engineering |
| `scikit-learn` | Gradient Boosting (Config 1/2), metrics |
| `statsmodels` | Holt-Winters exponential smoothing |
| `prophet` | Seasonal decomposition model |
| `lightgbm` | Final enhanced model (Config 3) |
| `shap` | Global feature attribution |
| `lime` | Local instance explanations |
| `anchor-exp` | High-precision rule extraction |
| `boto3` | Unsigned S3 download from NREL's public bucket |
| `matplotlib` | All result plots |

---

## Setup

### 1. Clone / download this repository

Ensure the four notebooks, `extractColumns.py`, and `downloadFiles.py` are all in the same root directory.

### 2. Obtain the ResStock metadata file

Download `CA_upgrade0.parquet` (California building metadata, upgrade scenario 0 / baseline) from the NREL ResStock metadata release and place it at:

```
./metadata/CA_upgrade0.parquet
```

### 3. Obtain the S3 file availability list

`extractColumns.py` expects an `available_files.txt` listing which building IDs actually have timeseries files on S3 (not every metadata row has a corresponding file). Generate this with an `aws s3 ls` listing of the relevant prefix, or request it from the project maintainer.

---

## Pipeline Walkthrough

### Step 1 — Select Buildings

```bash
python extractColumns.py
```

This script:
1. Loads `CA_upgrade0.parquet` metadata
2. Cross-references against `available_files.txt` to confirm S3 availability
3. Filters to **Single-Family Detached** buildings with **no PV** (`in.has_pv == "No"`)
4. Performs a **stratified sample of 100 buildings** across six climate zones using `random_state=42` for reproducibility:

   | Zone | Buildings | Description |
   |---|---|---|
   | 3B | 35 | Los Angeles Basin |
   | 3C | 25 | San Francisco Coastal |
   | 4B | 15 | Fresno / Central Valley |
   | 4C | 10 | Alameda |
   | 5B | 10 | Mountain |
   | 2B | 5 | Mojave Desert (most data-limited) |

5. Saves the result to `selected_buildings.csv`, which every downstream script and notebook reads from.

### Step 2 — Download Data

```bash
python downloadFiles.py
```

This script reads `selected_buildings.csv`, derives the required GISJOIN weather codes **dynamically** (not hardcoded — see [Known Issues](#known-issues--fixes-applied)), and downloads:

- Per-building hourly timeseries parquet files (AMY 2018 and AMY 2012)
- Per-county weather CSV files (AMY 2018 and AMY 2012)
- Building metadata

All files are saved into `./data/` in the exact structure the notebooks expect. By default both 2018 and 2012 are downloaded, since the 2-year training experiments require both. Set `DOWNLOAD_2012 = False` at the top of the script if you only intend to run the 1-year configuration.

**Note:** download time depends on connection speed and S3 throttling; expect 15–40 minutes for the full 100-building, two-year dataset.

### Step 3 — Run the Forecasting Pipelines

Run the notebooks **in this order**, since later notebooks depend on outputs from earlier ones:

```
1. LoadForecastingPipeline.ipynb              (set RUN_MODE = '1year', then '2year')
2. LoadForecastingPipeline_R2improved.ipynb   (set RUN_MODE = '1year', then '2year')
3. ProphetGB_Pipeline.ipynb                   (set RUN_MODE = '2year')
4. GB_Only_Ablation.ipynb
```

Each notebook has a `RUN_MODE` flag near the top — run it once with `'1year'` and once with `'2year'` where applicable, since the thesis compares both training regimes.

---

## Notebook Reference

### `LoadForecastingPipeline.ipynb` — Configuration 1 (Baseline)

The original pipeline. Trains Holt-Winters, Prophet (without yearly seasonality), and Gradient Boosting on 58 features, then optimises per-zone ensemble weights via grid search. Produces the baseline RMSLE 0.1879 result and the 1-year vs 2-year Prophet training threshold comparison (RQ2).

### `ProphetGB_Pipeline.ipynb` — Configuration 2 (Holt-Winters Removed)

Reuses the Gradient Boosting predictions from Configuration 1 and re-pairs them with Prophet (2-year, yearly seasonality enabled), excluding Holt-Winters entirely. This isolates whether Holt-Winters contributes independent signal — it does not; removing it improves RMSLE marginally (0.1879 → 0.1856), confirming Prophet structurally subsumes Holt-Winters.

### `LoadForecastingPipeline_R2improved.ipynb` — Configuration 3 (Enhanced, Final Model)

The enhanced pipeline. Expands the feature set from 58 to 93 features — most critically adding `kwh_lag_24`, `kwh_lag_168`, deeper rolling windows (24h/48h/168h), and weather × time interaction terms — and replaces scikit-learn's Gradient Boosting with **LightGBM**. This is the source of the thesis's headline result: **RMSLE 0.1473, R² 0.913**.

### `GB_Only_Ablation.ipynb` — Prophet Contribution Ablation

Loads predictions from the 1-year GB run, the 2-year Prophet run, and the hybrid ensemble, and computes the exact RMSLE delta attributable to Prophet — overall, per-zone, per-month, and per-hour. Produces the evidence behind the thesis's claim that feature engineering investment (21.6% RMSLE reduction) outweighs ensemble architecture tuning (1.2% RMSLE reduction) by a factor of ~18.



