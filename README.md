# ⚡ Nitor Energy — Intraday Electricity Price Forecasting

Competition solution for intraday electricity price prediction, with a focus on spike detection and accurate forecasting of extreme price events.

## 📁 Repository Structure

```
├── data/                        # Raw data (not uploaded — too large)
│   ├── train.csv
│   ├── test_for_participants.csv
│   ├── train_featured_1.csv     # Engineered features (output of phase2)
│   └── test_featured_1.csv
│
├── phase1_eda.py                # Exploratory Data Analysis
├── phase2_features.py           # Feature Engineering (physics-informed)
├── spike_pipeline_v4.ipynb      # Main modelling pipeline
│
├── plots/                       # Generated during EDA and feature engineering
├── requirements.txt
└── README.md
```

## 🔄 How to Reproduce

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Run EDA
```bash
python phase1_eda.py
```
Generates diagnostic plots in `plots/` directory.

### Step 3 — Feature Engineering
```bash
python phase2_features.py
```
Reads `train.csv` and `test_for_participants.csv`, outputs:
- `train_featured_1.csv`
- `test_featured_1.csv`

### Step 4 — Model Training & Submission
Open and run `spike_pipeline_v4.ipynb` cell by cell.

Outputs:
- `submission_v4_mega.csv`
- `submission_v4_lgb_base.csv`
- `validation_v4_predictions.csv`

---

## 🧠 Modelling Approach

### Feature Engineering (`phase2_features.py`)
Physics-informed features built on top of raw weather and energy forecast data:
- **Energy balance**: residual load, renewable penetration, wind/solar share
- **Non-linear physics**: wind power curve (speed³), effective solar irradiance
- **Temporal regimes**: evening peak flags, morning ramp, solar hours
- **Interactions**: residual load × evening peak, spike risk flag
- **Lag & rolling features**: 1h, 3h, 6h, 24h ramps and rolling windows
- **Cross-market aggregates**: wind speed deviation across markets

### Spike Pipeline (`spike_pipeline_v4.ipynb`)
Four approaches combined into a mega ensemble:

| Approach | Description |
|----------|-------------|
| **Baseline** | LightGBM + CatBoost on raw target, 5× spike sample weight |
| **Log Transform** | Models trained on `sign(y)*log1p(\|y\|)`, inverse-transformed for prediction |
| **Two-Stage** | Spike classifier → separate normal/spike regressors, blended by spike probability |
| **Mega Ensemble** | Ridge stacking of all models, weights learned on held-out validation half |

### Key Design Decisions
- **Spike definition**: top 10% by absolute value (threshold from train only — no future leakage)
- **Spike probability as feature**: OOF classifier output added as 142nd feature to all regressors
- **Validation**: Sep–Nov 2024 (mirrors test period Sep–Nov 2025)
- **Ridge stacking fix**: fit on 1st half of val, evaluated on 2nd half only (unbiased)

---

## 📊 Validation Results (Sep–Nov 2024)

| Model | All RMSE | Spike RMSE | Normal RMSE |
|-------|----------|------------|-------------|
| Log LGB | 46.47 | 138.65 | 16.19 |
| Log Ensemble | 46.48 | 138.94 | 15.96 |
| Mega Ensemble | 46.90 | 138.93 | 17.27 |
| Baseline LGB | 51.69 | 143.30 | 26.19 |

> **Note**: The Baseline LGB achieves a better public leaderboard score despite higher validation RMSE, because it produces larger spike predictions (max ~898 vs ~156 for log models). The validation period (Nov 2024) contained two extreme synchronized spikes across all markets that no model captures well.

---

## 🛠️ Dependencies

See `requirements.txt`. Main libraries:
- `lightgbm`
- `catboost`
- `scikit-learn`
- `pandas`, `numpy`
- `matplotlib`, `seaborn`
