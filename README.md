# ⚡ Nitor Energy — Intraday Electricity Price Forecasting

Competition solution for intraday electricity price prediction, with a strong focus on extreme spike modeling and robust validation methodology.

---

## 📁 Repository Structure

```
├── data/                        # Raw data (not uploaded — too large)
│   ├── train.csv
│   ├── test_for_participants.csv
│   ├── train_featured_1.csv
│   └── test_featured_1.csv
│
├── phase1_eda.py                # Exploratory Data Analysis
├── phase2_features.py           # Feature Engineering (physics-informed)
├── spike_pipeline_v4.ipynb      # Main modelling pipeline (final version)
│
├── plots/
├── requirements.txt
└── README.md
```

---

## 🔄 How to Reproduce

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Run EDA

```bash
python phase1_eda.py
```

Generates diagnostic plots in `plots/`.

### Step 3 — Feature Engineering

```bash
python phase2_features.py
```

Outputs:

- train_featured_1.csv  
- test_featured_1.csv  

### Step 4 — Model Training & Submission

Open and execute:

`spike_pipeline_v4.ipynb`

Outputs:

- submission_v4_mega.csv  
- submission_v4_lgb_base.csv  
- submission_v4_lgb_log.csv  
- validation_v4_predictions.csv  

---

# 🧠 Modelling Strategy

The pipeline consists of four modeling strategies, evaluated under strict leakage-free validation.

---

## 🔹 Feature Engineering (`phase2_features.py`)

Physics-informed transformations:

- Residual load & renewable penetration  
- Wind power curve (speed³)  
- Solar irradiance transformation  
- Evening peak regime flags  
- Ramps (1h, 3h, 6h, 24h)  
- Rolling statistics  
- Cross-market deviations  
- Interaction features  
- Spike risk heuristics  

Final feature count: **142 + spike_prob**

---

## 🔹 Spike Definition

- Spike = top 10% by absolute target value  
- Threshold computed only on training period  
- Applied consistently across train, validation, classifier and full retrain  

This avoids forward leakage (Fix A + Fix C).

---

## 🔹 Approaches

| Approach | Description |
|----------|------------|
| **Baseline** | LGB + CatBoost on raw target (5× spike weighting) |
| **Log Transform** | Train on `sign(y) * log1p(|y|)` |
| **Two-Stage** | Separate spike & normal regressors blended via spike probability |
| **Mega Ensemble** | Ridge stacking (trained on first half of val only) |

---

# 📊 Validation Results (Sep–Nov 2024)

| Model                              | All RMSE | Spike RMSE | Normal RMSE |
|------------------------------------|----------|------------|-------------|
| **Log Ensemble 50/50** ⭐          | **46.3899** | 159.4793 | 16.6242 |
| Log LGB                            | 46.4516 | 160.1688 | 16.2726 |
| Mega Ensemble (norm weights)       | 46.6576 | 158.9581 | 17.7945 |
| Two-Stage (power=3.0)              | 47.5316 | 165.4966 | 15.3196 |
| Baseline LGB                       | 52.1092 | 161.1936 | 28.9967 |
| Ridge (2nd half — honest eval)     | 61.2787 | 216.0733 | 16.8754 |

---

# 📈 Maximum Prediction Magnitudes (Test Set)

| Model | Approx Max Prediction |
|-------|----------------------|
| Baseline LGB | ~900 |
| Log LGB | ~150–170 |
| Log Ensemble | ~150–170 |
| Two-Stage | ~200–250 |
| Mega Ensemble | ~200 |

Interpretation:

- Raw baseline produces extreme amplitudes  
- Log models compress spikes significantly  
- Two-stage partially restores amplitude  
- Mega ensemble balances amplitude and stability  

---

# 🏁 Competition Insight

Although Log Ensemble achieves the best validation RMSE,  
Baseline LGB may achieve competitive public leaderboard scores due to higher spike amplitude.

Extreme synchronized spikes (Nov 2024) are not fully captured by any model.

Best public leaderboard score achieved:  
**22.732964 RMSE**

---

# 🛠️ Dependencies

Main libraries:

- lightgbm  
- catboost  
- scikit-learn  
- pandas  
- numpy  
- matplotlib  

See `requirements.txt`.

---

# 🚀 Recommended Submissions

- submission_v4_mega.csv — best overall validation balance  
- submission_v4_lgb_base.csv — highest spike amplitude  
- submission_v4_lgb_log.csv — stable alternative  
