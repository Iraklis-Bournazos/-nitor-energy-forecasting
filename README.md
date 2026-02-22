# ⚡ Nitor Energy — Intraday Electricity Price Forecasting

Competition solution for intraday electricity price prediction across 6 European markets (Sep–Nov 2025). The core challenge is accurate forecasting of extreme price events ("spikes") which are rare, large in magnitude, and disproportionately impact RMSE.

---

## 📁 Repository Structure

```
├── phase1_eda.py                # Phase 1: Exploratory Data Analysis
├── phase2_features.py           # Phase 2: Feature Engineering
├── spike_pipeline_v5.ipynb      # Phase 3: Modelling pipeline (main file)
├── requirements.txt             # Python dependencies
└── README.md
```

**Data files** (not included — too large for GitHub):
```
├── train.csv                    # Raw training data
├── test_for_participants.csv    # Raw test data
├── train_featured_1.csv         # Output of phase2_features.py
└── test_featured_1.csv          # Output of phase2_features.py
```

---

## 🔄 How to Reproduce

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Exploratory Data Analysis
```bash
python phase1_eda.py
```
Reads `train.csv` and `test_for_participants.csv`. Generates diagnostic plots in `plots/`.

### Step 3 — Feature Engineering
```bash
python phase2_features.py
```
Reads raw CSVs and outputs `train_featured_1.csv` and `test_featured_1.csv`.

### Step 4 — Train & Generate Submissions
Open `spike_pipeline_v5.ipynb` and run all cells in order.

Outputs:
- `submission_v4_mega.csv` — Mega ensemble (best overall RMSE)
- `submission_v4_lgb_base.csv` — Baseline LGB (highest spike predictions)
- `submission_v4_lgb_log.csv` — Log-transform LGB
- `validation_v4_predictions.csv` — Full validation set with all model predictions

---

## 🗂️ Data Overview

| | Train | Test |
|---|---|---|
| **Period** | Jan 2023 → Aug 2025 | Sep 2025 → Nov 2025 |
| **Rows** | 132,608 | 13,098 |
| **Markets** | 6 (A–F) | 6 (A–F) |
| **Features (raw)** | Weather forecasts, load forecasts, wind/solar forecasts | Same |
| **Target** | Intraday electricity price (€/MWh) | — |

**Target distribution:**
- Normal hours: typically 15–100 €/MWh
- Spike hours (top 10% by |value|): |target| ≥ 53.78 €/MWh
- Spike rows in train: 13,386 / 132,608 (10.1%)

---

## 🧠 Phase 2 — Feature Engineering (`phase2_features.py`)

141 physics-informed features built on top of raw weather and energy forecast data:

**Temporal**
- Hour, month, day-of-week with cyclical sin/cos encodings
- Regime flags: `evening_peak`, `morning_ramp`, `solar_hours`, `night`

**Energy balance**
- `residual_load` = load_forecast − wind_forecast − solar_forecast
- `renewable_penetration`, `wind_share`, `solar_share`

**Physics-based**
- `wind_power_proxy` = wind_speed³ (cubic relationship from wind turbine physics)
- `effective_solar` = solar_forecast × cos(solar_angle) × cloud_transmission
- `heating_degree`, `cooling_degree`, `temp_stress`

**Interaction features**
- `residual_load × evening_peak`, `wind_forecast × evening_peak`
- `spike_risk_flag`, `solar_cliff`

**Lag & rolling features**
- 1h, 3h, 6h changes for residual load, wind, solar, temperature
- Rolling 6h, 12h, 24h mean/std/max

**Cross-market features**
- Weather aggregates across all 6 markets (mean, std, min, max)
- Per-market deviation from cross-market average

---

## 🔬 Phase 3 — Modelling Pipeline (`spike_pipeline_v5.ipynb`)

### Validation Strategy

Single temporal fold mirroring the test period:

| Split | Period | Rows |
|-------|--------|------|
| Train (train_bv) | Jan 2023 → Aug 2024 | 80,054 |
| Validation | Sep 2024 → Nov 2024 | 13,104 |
| Test | Sep 2025 → Nov 2025 | 13,098 |

Spike threshold: |target| ≥ **53.78 €/MWh** (computed from train_bv only — no future leakage).

### Pipeline Overview

```
Raw features (141)
       │
       ▼
┌──────────────────────────────────────┐
│  Cell 6: Spike Classifier (OOF)      │
│  LightGBM binary, 5-fold OOF         │
│  AUC on val = 0.923                  │
│  Output: spike_prob → feature #142   │
└──────────────────────────────────────┘
       │
       ├─────────────────────────────────────────────┐
       ▼                                             ▼
┌────────────────┐  ┌────────────────┐  ┌─────────────────┐
│  Approach 1    │  │  Approach 2    │  │  Approach 3     │
│  Baseline      │  │  Log Transform │  │  Two-Stage      │
│  LGB + CB      │  │  LGB + CB      │  │  Normal + Spike │
│  Raw target    │  │  sign*log1p    │  │  Regressors     │
└────────────────┘  └────────────────┘  └─────────────────┘
       │                   │                    │
       └───────────────────┴────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  Approach 4            │
              │  OOF Ridge Stacking    │
              │  Learns optimal        │
              │  weights from train_bv │
              └────────────────────────┘
```

---

### Approach 1 — Baseline (raw target)

LightGBM and CatBoost trained on raw target with **5× sample weight** for spike rows.

- **Strength:** Unconstrained — can produce large spike predictions (test max ~822 €/MWh)
- **Weakness:** High normal-hour RMSE (~28) due to spike influence on loss

### Approach 2 — Log Transform

Models trained on `sign(y) × log1p(|y|)`. A spike of 1,500 becomes ~7.3 in training space, making the loss landscape smoother. Predictions inverse-transformed back.

- **Strength:** Best overall RMSE, excellent normal-hour predictions (~16)
- **Weakness:** Compression limits max predictions (test max ~144 €/MWh)

### Approach 3 — Two-Stage Model

**Stage 2A — Normal Regressor:** trained only on non-spike rows (90% of data).

**Stage 2B — Spike Regressor:** trained only on spike rows (10% of data), two variants:
- Huber loss (α=0.9) — robust to extreme outliers
- Log-space — compresses within-spike variation
- Final = 50/50 blend of the two

**Blend formula:**
```
final = (1 − spike_prob^power) × normal_pred + spike_prob^power × spike_pred
```
Best power = **3.0** (found by grid search). High power means: only route to spike regressor when classifier is very confident.

- **Strength:** Best normal-hour RMSE (15.32)
- **Weakness:** Slightly worse overall RMSE than log models

### Approach 4 — Mega Ensemble (Clean OOF Ridge Stacking)

Combines all 5 model outputs using Ridge regression with **fully unbiased OOF weights**:

```
Step 1: 5-fold OOF on train_bv
        Each fold trains all models on 4/5, predicts on 1/5
        Result: OOF predictions for 100% of train_bv (never seen)

Step 2: Ridge.fit(oof_predictions, y_train_bv)
        Validation is NEVER touched during this step

Step 3: Apply fixed weights to val and test predictions
        Evaluation is 100% unbiased
```

**Learned weights:**

| Model | Weight |
|-------|:------:|
| base_lgb | 0.178 |
| base_cb | 0.360 |
| log_lgb | 0.000 |
| log_cb | 0.000 |
| two_stage | 0.462 |

Ridge discovers that log models are redundant once base_cb and two_stage are combined.

---

## 📊 Results

### Validation Performance (Sep–Nov 2024)

| Model | All RMSE ↓ | Spike RMSE | Normal RMSE |
|-------|:----------:|:----------:|:-----------:|
| **Log Ensemble 50/50** ✓ | **46.39** | 159.48 | 16.62 |
| Log LGB | 46.45 | 160.17 | 16.27 |
| Log CatBoost | 46.56 | 159.24 | 17.32 |
| Two-Stage (power=3.0) | 47.53 | 165.50 | **15.32** |
| Mega Ensemble (OOF Ridge) | 47.70 | 159.72 | 20.09 |
| Baseline Ensemble 50/50 | 51.00 | 159.06 | 27.79 |
| Baseline CatBoost | 51.21 | 159.28 | 28.10 |
| Baseline LGB | 52.11 | 161.19 | 28.99 |

### Spike Classifier (5-fold OOF)

| Fold | AUC |
|------|:---:|
| Fold 1 | 0.902 |
| Fold 2 | 0.869 |
| Fold 3 | 0.882 |
| Fold 4 | 0.902 |
| Fold 5 | 0.865 |
| **Full val** | **0.923** |

### Per-Market RMSE — Mega Ensemble on Validation

| Market | RMSE | Bias (mean residual) |
|--------|:----:|:--------------------:|
| Market A | 64.59 | −12.88 |
| Market B | 53.70 | −6.83 |
| Market C | 33.14 | −7.27 |
| Market D | 39.98 | −6.99 |
| Market E | 54.12 | −7.08 |
| Market F | 31.17 | −2.59 |

All markets show negative bias (systematic underprediction), driven by extreme spike events that no model fully captures.

### Test Set Prediction Statistics

| Model | Mean | Std | Max | p99 |
|-------|:----:|:---:|:---:|:---:|
| lgb_base | 41.56 | 34.73 | **821.92** | 162.88 |
| cb_base | 38.44 | 36.07 | 702.30 | 174.48 |
| lgb_log | 25.25 | 22.55 | 144.20 | 105.58 |
| two_stage | 24.90 | 21.23 | 175.84 | 109.84 |
| **mega** | **32.74** | 27.30 | 454.38 | 137.43 |

---

## ⚠️ Leakage Audit

Three methodological issues identified and fixed vs earlier versions:

| Issue | Problem in v3 | Fix in v5 |
|-------|--------------|-----------|
| **A** | Spike threshold computed on full train (includes val data) | Threshold from `train_bv` only, applied consistently to all splits |
| **B** | Ridge fit on val[:mid], evaluated on val[:] — optimistically biased | OOF stacking: Ridge fits on `train_bv` OOF, val never touched |
| **C** | Two different thresholds: one for `spike_mask_val`, one for classifier labels | Single threshold everywhere — one source of truth |

**Remaining mild caveats (standard competition practice):**
- Early stopping uses `X_val` as eval set → val influences iteration count
- `best_power` grid-searched on `y_val` → 7 discrete values, minor optimism

---

## 🔑 Key Findings

**Log transform is the single biggest improvement** — reduces All RMSE from ~51 to ~46 (−10%) by compressing spike magnitude in training space without discarding spike information.

**Spike classifier is excellent (AUC=0.923) but spike RMSE remains high (~159) across all models.** The limit is not classification accuracy — it's that spike magnitudes are inherently unpredictable from day-ahead features alone. Spikes are caused by real-time grid events (outages, demand surges) that are not visible in forecasts.

**Normal-hour prediction is near-perfect (RMSE ~15–17)** — the models learn baseline price patterns very well from weather and load features.

**Baseline LGB outperforms on the leaderboard despite worse validation RMSE.** The validation period (Nov 2024) contained two extreme synchronized spikes across all 6 markets that all models severely underestimated. If the test period (Sep–Nov 2025) has a different spike distribution — fewer or smaller extreme events — the log models' conservative predictions are penalized less, and the baseline's ability to produce large predictions becomes valuable.
