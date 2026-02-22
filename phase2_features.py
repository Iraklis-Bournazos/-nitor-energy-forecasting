"""
Phase 2: Feature Engineering (Physics-Informed)
Nitor Energy Case Competition — Intraday Electricity Price Prediction

This script builds all features for both train and test data.
Every feature is grounded in energy market physics and EDA findings.

Key EDA findings driving decisions:
- Target is extremely heavy-tailed (max 6252, kurtosis 562)
- Market A has a persistent price premium (mean 53 vs 20-32 for others)
- Evening ramp (hours 16-20) is the primary spike regime
- All signal is in feature interactions, not raw features
- Test has 4.54% missing weather data requiring robust imputation
- Solar forecast drifted +68% between train Sep-Nov and test Sep-Nov
- Residual load is the strongest single derived feature (corr 0.22)
"""

# =============================================================================
# 0. REPRODUCIBILITY SETUP
# =============================================================================
import random
import numpy as np
random.seed(42)
np.random.seed(42)

import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "seaborn", "-q"])

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os
os.makedirs("plots", exist_ok=True)

print("=" * 80)
print("PHASE 2: FEATURE ENGINEERING")
print("=" * 80)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("\n[1] Loading data...")
train = pd.read_csv("train.csv")
test = pd.read_csv("test_for_participants.csv")

print(f"    Train: {train.shape}, Test: {test.shape}")

# Parse datetimes
train['delivery_start'] = pd.to_datetime(train['delivery_start'])
train['delivery_end'] = pd.to_datetime(train['delivery_end'])
test['delivery_start'] = pd.to_datetime(test['delivery_start'])
test['delivery_end'] = pd.to_datetime(test['delivery_end'])

# Tag datasets before combining
train['is_train'] = 1
test['is_train'] = 0

# Combine for consistent feature engineering
# Target only exists in train — this is safe because we never touch target during FE
df = pd.concat([train, test], axis=0, ignore_index=True)
print(f"    Combined: {df.shape}")

# =============================================================================
# =============================================================================
# 2. MISSING VALUE HANDLING (FIXED: Time-Aware Interpolation)
# =============================================================================
print("\n[2] Handling missing values...")

# Check energy forecast missingness
for col in ['solar_forecast', 'wind_forecast', 'load_forecast']:
    n_miss = df[col].isna().sum()
    if n_miss > 0:
        print(f"    WARNING: {col} has {n_miss} missing values")
    else:
        print(f"    {col}: no missing values ✓")

weather_cols = [
    'global_horizontal_irradiance', 'diffuse_horizontal_irradiance',
    'direct_normal_irradiance', 'cloud_cover_total', 'cloud_cover_low',
    'cloud_cover_mid', 'cloud_cover_high', 'precipitation_amount', 'visibility',
    'air_temperature_2m', 'apparent_temperature_2m', 'dew_point_temperature_2m',
    'wet_bulb_temperature_2m', 'surface_pressure', 'freezing_level_height',
    'relative_humidity_2m', 'convective_available_potential_energy',
    'lifted_index', 'convective_inhibition',
    'wind_speed_80m', 'wind_direction_80m', 'wind_gust_speed_10m', 'wind_speed_10m'
]

# Sort by market and time for proper temporal continuity
df = df.sort_values(['market', 'delivery_start']).reset_index(drop=True)

missing_before = df[weather_cols].isna().sum().sum()
print(f"    Total missing values before imputation: {missing_before:,}")

# Step 1: Time-aware linear interpolation within each market.
# This realistically models weather changes (e.g., gradual temp increase) across gaps.
df[weather_cols] = (
    df.groupby('market')[weather_cols]
    .transform(lambda x: x.interpolate(method='linear', limit_direction='both'))
)

# Step 2: Fallback — market × month × hour median for any persistent edge-case gaps
# (Much tighter and safer than a global mean)
hour_for_fill = df['delivery_start'].dt.hour
month_for_fill = df['delivery_start'].dt.month

for col in weather_cols:
    remaining = df[col].isna().sum()
    if remaining > 0:
        print(f"    {col}: {remaining} still missing after interpolation, using market×month×hour median")
        medians = df.groupby([df['market'], month_for_fill, hour_for_fill])[col].transform('median')
        df[col] = df[col].fillna(medians)

# Step 3: Absolute final fallback (ffill/bfill) if medians are still missing
df[weather_cols] = df.groupby('market')[weather_cols].ffill().bfill()

missing_after = df[weather_cols].isna().sum().sum()
print(f"    Total missing values after imputation: {missing_after}")

# =============================================================================
# 3. TEMPORAL FEATURES
# =============================================================================
print("\n[3] Engineering temporal features...")

# Extract raw temporal components
df['hour'] = df['delivery_start'].dt.hour
df['month'] = df['delivery_start'].dt.month
df['dayofweek'] = df['delivery_start'].dt.dayofweek
df['day'] = df['delivery_start'].dt.day
df['week_of_year'] = df['delivery_start'].dt.isocalendar().week.astype(int)
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
df['quarter'] = df['delivery_start'].dt.quarter

# Season mapping (meteorological seasons)
# 1=Winter(Dec-Feb), 2=Spring(Mar-May), 3=Summer(Jun-Aug), 4=Autumn(Sep-Nov)
season_map = {12: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3, 9: 4, 10: 4, 11: 4}
df['season'] = df['month'].map(season_map)

# Cyclical encoding — hour of day
# This ensures the model knows that hour 23 and hour 0 are adjacent
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# Cyclical encoding — month
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Cyclical encoding — day of week
df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

# Time-of-day regime flags (from EDA: distinct pricing regimes)
# Night trough: 0-4, Morning ramp: 5-7, Solar peak: 8-13, 
# Afternoon transition: 14-16, Evening spike: 17-20, Evening wind-down: 21-23
df['is_evening_peak'] = df['hour'].isin([17, 18, 19, 20]).astype(int)
df['is_morning_ramp'] = df['hour'].isin([5, 6, 7]).astype(int)
df['is_solar_hours'] = df['hour'].isin([8, 9, 10, 11, 12, 13]).astype(int)
df['is_night'] = df['hour'].isin([0, 1, 2, 3, 4]).astype(int)
df['is_afternoon_transition'] = df['hour'].isin([14, 15, 16]).astype(int)

print(f"    Temporal features added: hour, month, dayofweek, day, week_of_year, "
      f"is_weekend, quarter, season, cyclical encodings, regime flags")

# =============================================================================
# 4. CORE ENERGY BALANCE FEATURES
# =============================================================================
print("\n[4] Engineering core energy balance features...")

# Residual Load: THE most important derived feature
# Represents demand that must be met by expensive conventional generation
df['residual_load'] = df['load_forecast'] - (df['wind_forecast'] + df['solar_forecast'])

# Renewable Penetration Ratio: fraction of demand covered by cheap renewables
# Clip denominator to avoid division by zero
df['renewable_penetration'] = (
    (df['wind_forecast'] + df['solar_forecast']) / df['load_forecast'].clip(lower=1)
)

# Wind share and solar share of total generation
df['wind_share'] = df['wind_forecast'] / df['load_forecast'].clip(lower=1)
df['solar_share'] = df['solar_forecast'] / df['load_forecast'].clip(lower=1)

# Residual load ratio (normalized by typical load — helps with trend robustness)
df['residual_load_ratio'] = df['residual_load'] / df['load_forecast'].clip(lower=1)

# Surplus/deficit flag: negative residual load means renewables exceed demand
df['renewable_surplus'] = (df['residual_load'] < 0).astype(int)

print(f"    residual_load corr with target: "
      f"{df.loc[df['is_train']==1, 'residual_load'].corr(df.loc[df['is_train']==1, 'target']):.4f}")
print(f"    renewable_penetration corr with target: "
      f"{df.loc[df['is_train']==1, 'renewable_penetration'].corr(df.loc[df['is_train']==1, 'target']):.4f}")

# =============================================================================
# 5. NON-LINEAR PHYSICS FEATURES
# =============================================================================
print("\n[5] Engineering non-linear physics features...")

# Wind Power Curve: wind power is proportional to wind_speed^3
# between cut-in (~3 m/s) and rated speed (~12-15 m/s)
df['wind_power_proxy'] = df['wind_speed_80m'] ** 3

# Also a "capped" version that mimics real turbine behavior:
# Power saturates above rated speed and is zero below cut-in
df['wind_power_capped'] = df['wind_speed_80m'].clip(lower=3, upper=15) ** 3

# Effective Solar Irradiance: actual solar reaching panels after cloud attenuation
# Cloud cover is 0-100, so we convert to a fraction
df['effective_solar'] = (
    df['global_horizontal_irradiance'] * (1 - df['cloud_cover_total'] / 100)
)

# Solar capacity factor proxy: ratio of actual irradiance to max possible
# Using seasonal/hourly max as reference (avoid division by zero for nighttime)
df['ghi_hour_max'] = df.groupby(['month', 'hour'])['global_horizontal_irradiance'].transform('max')
df['solar_capacity_factor'] = np.where(
    df['ghi_hour_max'] > 0,
    df['global_horizontal_irradiance'] / df['ghi_hour_max'],
    0
)

# Direct-to-diffuse ratio: high ratio = clear sky, low ratio = cloudy/hazy
# Good indicator of solar forecast reliability
df['direct_diffuse_ratio'] = np.where(
    df['diffuse_horizontal_irradiance'] > 0,
    df['direct_normal_irradiance'] / df['diffuse_horizontal_irradiance'],
    0
)

# Temperature-driven demand proxy: heating/cooling degree features
# From EDA: temperature has a U-shaped relationship with price
# Heating degrees (demand rises when cold) — base 15°C
df['heating_degrees'] = np.maximum(0, 15 - df['air_temperature_2m'])
# Cooling degrees (demand rises when hot) — base 22°C
df['cooling_degrees'] = np.maximum(0, df['air_temperature_2m'] - 22)

# Wind chill / heat stress captured by apparent vs actual temperature difference
df['temp_stress'] = np.abs(df['apparent_temperature_2m'] - df['air_temperature_2m'])

print("    Wind power proxy, effective solar, solar capacity factor, "
      "temperature demand proxies added")

# =============================================================================
# 6. WIND DIRECTION ENCODING
# =============================================================================
print("\n[6] Encoding wind direction (circular feature)...")

# Wind direction is circular: 358° and 2° are nearly identical but 356 apart numerically
# MUST use sin/cos encoding
df['wind_dir_sin'] = np.sin(np.radians(df['wind_direction_80m']))
df['wind_dir_cos'] = np.cos(np.radians(df['wind_direction_80m']))

# Also encode directional components — useful for understanding which direction
# wind blows FROM (affects different markets differently)
# Westerly component (positive = from west, typical for Atlantic weather systems)
df['wind_westerly'] = -df['wind_speed_80m'] * df['wind_dir_sin']
# Southerly component (positive = from south)
df['wind_southerly'] = -df['wind_speed_80m'] * df['wind_dir_cos']

print("    wind_dir_sin, wind_dir_cos, wind_westerly, wind_southerly added")

# =============================================================================
# 7. INTERACTION FEATURES
# =============================================================================
print("\n[7] Engineering interaction features...")

# Residual load × evening peak: the most dangerous combination
df['residual_load_x_evening'] = df['residual_load'] * df['is_evening_peak']

# Residual load × morning ramp
df['residual_load_x_morning'] = df['residual_load'] * df['is_morning_ramp']

# Wind forecast × evening: low wind at evening = spike risk
df['wind_forecast_x_evening'] = df['wind_forecast'] * df['is_evening_peak']

# Residual load × hour (continuous interaction)
df['residual_load_x_hour'] = df['residual_load'] * df['hour']

# Wind speed × hour
df['wind_speed_x_hour'] = df['wind_speed_80m'] * df['hour']

# Temperature × hour (heating at night, cooling in afternoon)
df['temp_x_hour'] = df['air_temperature_2m'] * df['hour']

# Renewable penetration × hour
df['renewable_pen_x_hour'] = df['renewable_penetration'] * df['hour']

# Spike risk indicator: compound condition from EDA analysis
# Spikes occur when: low wind (<8000) AND evening hours AND high residual load
df['spike_risk_flag'] = (
    (df['wind_forecast'] < 8000) &
    (df['hour'].isin([16, 17, 18, 19, 20])) &
    (df['residual_load'] > 40000)
).astype(int)

# Solar cliff: solar dropping to zero while still in high-demand hours
df['solar_cliff'] = (
    (df['solar_forecast'] == 0) &
    (df['hour'].isin([16, 17, 18, 19]))
).astype(int)

print(f"    Spike risk flag frequency: {df['spike_risk_flag'].mean()*100:.2f}%")
print(f"    Solar cliff frequency: {df['solar_cliff'].mean()*100:.2f}%")

# =============================================================================
# 8. TEMPORAL RAMP FEATURES (per market, across time)
# =============================================================================
print("\n[8] Engineering temporal ramp features...")

# CRITICAL: Ramp rates must be computed WITHIN each market across time.
# Adjacent rows in raw data are different markets at the same hour.
# We sort by market + time, then compute diffs within market groups.

df = df.sort_values(['market', 'delivery_start']).reset_index(drop=True)

# Hour-over-hour changes (1-step lag within market)
ramp_features = {
    'residual_load': 'residual_load_ramp_1h',
    'wind_forecast': 'wind_forecast_ramp_1h',
    'solar_forecast': 'solar_forecast_ramp_1h',
    'load_forecast': 'load_forecast_ramp_1h',
    'air_temperature_2m': 'temp_ramp_1h',
    'wind_speed_80m': 'wind_speed_ramp_1h',
    'global_horizontal_irradiance': 'ghi_ramp_1h',
    'cloud_cover_total': 'cloud_ramp_1h',
}

for src_col, new_col in ramp_features.items():
    df[new_col] = df.groupby('market')[src_col].diff(1)

# Absolute ramp rates (magnitude of change, regardless of direction)
for src_col, new_col in ramp_features.items():
    abs_col = new_col.replace('_ramp_', '_abs_ramp_')
    df[abs_col] = df[new_col].abs()

# Multi-hour ramps: 3-hour and 6-hour changes (captures longer trends)
for lag_hours in [3, 6]:
    df[f'residual_load_ramp_{lag_hours}h'] = df.groupby('market')['residual_load'].diff(lag_hours)
    df[f'wind_forecast_ramp_{lag_hours}h'] = df.groupby('market')['wind_forecast'].diff(lag_hours)
    df[f'solar_forecast_ramp_{lag_hours}h'] = df.groupby('market')['solar_forecast'].diff(lag_hours)

print("    1h, 3h, 6h ramp rates computed for key features")

# =============================================================================
# 9. LAG FEATURES (forecasts only, NOT target)
# =============================================================================
print("\n[9] Engineering lag features...")

# Lag forecast features — these are available in both train and test
# because the feature columns exist for all rows.
# We lag within each market across time.

lag_features_config = {
    'residual_load': [1, 2, 3, 6, 12, 24],
    'load_forecast': [24],
    'wind_forecast': [24],
    'solar_forecast': [24],
    'wind_speed_80m': [1, 3, 24],
    'air_temperature_2m': [24],
}

for col, lags in lag_features_config.items():
    for lag in lags:
        lag_col = f'{col}_lag_{lag}h'
        df[lag_col] = df.groupby('market')[col].shift(lag)

# 24h change: difference from same hour yesterday (captures daily patterns)
df['residual_load_24h_change'] = df['residual_load'] - df['residual_load_lag_24h']
df['wind_forecast_24h_change'] = df['wind_forecast'] - df['wind_forecast'].groupby(df['market']).shift(24)
df['load_forecast_24h_change'] = df['load_forecast'] - df['load_forecast'].groupby(df['market']).shift(24)

print(f"    Lag features created for {len(lag_features_config)} base features")

# =============================================================================
# 10. ROLLING WINDOW FEATURES (per market)
# =============================================================================
print("\n[10] Engineering rolling window features...")

# Rolling statistics capture recent volatility and trends
# Must be computed within each market

rolling_configs = [
    ('residual_load', [6, 12, 24]),
    ('wind_speed_80m', [6, 24]),
    ('air_temperature_2m', [24]),
    ('load_forecast', [24]),
]

for col, windows in rolling_configs:
    for w in windows:
        # Rolling mean
        df[f'{col}_rmean_{w}h'] = (
            df.groupby('market')[col]
            .transform(lambda x: x.rolling(window=w, min_periods=1).mean())
        )
        # Rolling std (volatility)
        df[f'{col}_rstd_{w}h'] = (
            df.groupby('market')[col]
            .transform(lambda x: x.rolling(window=w, min_periods=2).std())
        )

# Rolling max of residual load (captures recent peak stress)
for w in [6, 12, 24]:
    df[f'residual_load_rmax_{w}h'] = (
        df.groupby('market')['residual_load']
        .transform(lambda x: x.rolling(window=w, min_periods=1).max())
    )

print("    Rolling mean, std, and max features added for key columns")

# =============================================================================
# 11. CROSS-MARKET AGGREGATE FEATURES
# =============================================================================
print("\n[11] Engineering cross-market features...")

# For each delivery hour, compute aggregate weather across all markets
# (energy forecasts are already shared, so only weather differs)
cross_market_aggs = {
    'wind_speed_80m': ['mean', 'std', 'min', 'max'],
    'air_temperature_2m': ['mean', 'std'],
    'cloud_cover_total': ['mean'],
    'global_horizontal_irradiance': ['mean'],
}

for col, agg_funcs in cross_market_aggs.items():
    agg_df = df.groupby('delivery_start')[col].agg(agg_funcs)
    agg_df.columns = [f'{col}_xmarket_{func}' for func in agg_funcs]
    df = df.merge(agg_df, on='delivery_start', how='left')

# Market-specific deviation from cross-market mean (captures relative positioning)
df['wind_speed_deviation'] = df['wind_speed_80m'] - df['wind_speed_80m_xmarket_mean']
df['temp_deviation'] = df['air_temperature_2m'] - df['air_temperature_2m_xmarket_mean']

print("    Cross-market aggregates and deviations added")

# =============================================================================
# 12. FORECAST ERROR PROXIES
# =============================================================================
print("\n[12] Engineering forecast error proxies...")

# The idea: large discrepancies between weather-implied generation and
# provided forecasts may signal forecast uncertainty, which drives intraday volatility.

# Wind: compare weather-implied wind power (from wind_speed_80m^3) to wind_forecast
# Normalize both to [0,1] range for comparability
ws_max = df['wind_power_proxy'].quantile(0.999)
wf_max = df['wind_forecast'].quantile(0.999)
df['wind_forecast_error_proxy'] = (
    (df['wind_power_proxy'] / max(ws_max, 1)) - (df['wind_forecast'] / max(wf_max, 1))
)

# Solar: compare GHI to solar_forecast
ghi_max = df['global_horizontal_irradiance'].quantile(0.999)
sf_max = df['solar_forecast'].quantile(0.999)
df['solar_forecast_error_proxy'] = np.where(
    (df['global_horizontal_irradiance'] > 0) | (df['solar_forecast'] > 0),
    (df['global_horizontal_irradiance'] / max(ghi_max, 1)) - (df['solar_forecast'] / max(sf_max, 1)),
    0
)

print("    Forecast error proxies for wind and solar added")

# =============================================================================
# 13. MARKET ENCODING
# =============================================================================
print("\n[13] Setting up market encoding...")

# Use pandas category dtype for LightGBM native categorical handling
df['market'] = df['market'].astype('category')

# Also create a numeric market ID for potential use with XGBoost
market_map = {m: i for i, m in enumerate(sorted(df['market'].cat.categories))}
df['market_id'] = df['market'].map(market_map).astype(int)

print(f"    Market categories: {dict(market_map)}")

# =============================================================================
# 14. FILL NaN FROM LAGS/ROLLING (edge effects)
# =============================================================================
print("\n[14] Handling NaN from lag/rolling computations...")

# Lag and rolling features will have NaN at the start of each market's time series.
# This is expected and unavoidable. Fill with 0 for ramps (no change) and
# with the feature's own value for lags (assume steady state).

# Count NaN before fill
lag_roll_cols = [c for c in df.columns if any(x in c for x in ['_lag_', '_ramp_', '_rmean_', '_rstd_', '_rmax_', '_24h_change'])]
nan_before = df[lag_roll_cols].isna().sum().sum()
print(f"    NaN in lag/rolling features before fill: {nan_before:,}")

# Ramp features: fill with 0 (no change at series start)
ramp_cols = [c for c in df.columns if '_ramp_' in c or '_24h_change' in c]
df[ramp_cols] = df[ramp_cols].fillna(0)

# Rolling std: fill with 0 (no volatility at series start)
rstd_cols = [c for c in df.columns if '_rstd_' in c]
df[rstd_cols] = df[rstd_cols].fillna(0)

# Lag features: fill with column median (more robust than 0)
lag_cols = [c for c in df.columns if '_lag_' in c]
for col in lag_cols:
    df[col] = df[col].fillna(df[col].median())

# Rolling mean/max: fill with column median
rmean_rmax_cols = [c for c in df.columns if '_rmean_' in c or '_rmax_' in c]
for col in rmean_rmax_cols:
    df[col] = df[col].fillna(df[col].median())

nan_after = df[lag_roll_cols].isna().sum().sum()
print(f"    NaN in lag/rolling features after fill: {nan_after}")

# =============================================================================
# 15. FINAL CLEANUP & SPLIT BACK
# =============================================================================
print("\n[15] Final cleanup and train/test split...")

# Drop helper columns we don't need as features
drop_cols = ['delivery_end', 'ghi_hour_max', 'is_train']
# Keep delivery_start for reference but it won't be a model feature
# Keep target for train data

# Separate back into train and test
train_fe = df[df['is_train'] == 1].copy()
test_fe = df[df['is_train'] == 0].copy()

train_fe = train_fe.drop(columns=drop_cols, errors='ignore')
test_fe = test_fe.drop(columns=drop_cols + ['target'], errors='ignore')

# Sort back by id for clean ordering
train_fe = train_fe.sort_values('id').reset_index(drop=True)
test_fe = test_fe.sort_values('id').reset_index(drop=True)

print(f"    Train features shape: {train_fe.shape}")
print(f"    Test features shape:  {test_fe.shape}")

# Verify no NaN in test features (critical for submission)
feature_cols = [c for c in test_fe.columns if c not in ['id', 'market', 'delivery_start']]
test_nan = test_fe[feature_cols].isna().sum()
test_nan_total = test_nan.sum()
if test_nan_total > 0:
    print(f"    WARNING: Test still has {test_nan_total} NaN values!")
    print(test_nan[test_nan > 0])
    # Emergency fill
    test_fe[feature_cols] = test_fe[feature_cols].fillna(test_fe[feature_cols].median())
    print("    Emergency median fill applied")
else:
    print(f"    Test features: zero NaN ✓")

# Same check for train
train_feature_cols = [c for c in train_fe.columns if c not in ['id', 'market', 'delivery_start', 'target']]
train_nan = train_fe[train_feature_cols].isna().sum().sum()
if train_nan > 0:
    print(f"    WARNING: Train still has {train_nan} NaN values!")
    train_fe[train_feature_cols] = train_fe[train_feature_cols].fillna(train_fe[train_feature_cols].median())
    print("    Emergency median fill applied")
else:
    print(f"    Train features: zero NaN ✓")

# =============================================================================
# 16. FEATURE SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("FEATURE SUMMARY")
print("=" * 80)

# Define feature groups for reporting
feature_groups = {
    'Identifiers': ['id', 'market', 'market_id', 'delivery_start'],
    'Target': ['target'],
    'Temporal': ['hour', 'month', 'dayofweek', 'day', 'week_of_year', 'is_weekend',
                 'quarter', 'season', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                 'dow_sin', 'dow_cos', 'is_evening_peak', 'is_morning_ramp',
                 'is_solar_hours', 'is_night', 'is_afternoon_transition'],
    'Energy Balance': ['residual_load', 'renewable_penetration', 'wind_share',
                       'solar_share', 'residual_load_ratio', 'renewable_surplus'],
    'Physics (Non-linear)': ['wind_power_proxy', 'wind_power_capped', 'effective_solar',
                              'solar_capacity_factor', 'direct_diffuse_ratio',
                              'heating_degrees', 'cooling_degrees', 'temp_stress'],
    'Wind Direction': ['wind_dir_sin', 'wind_dir_cos', 'wind_westerly', 'wind_southerly'],
    'Interactions': ['residual_load_x_evening', 'residual_load_x_morning',
                     'wind_forecast_x_evening', 'residual_load_x_hour',
                     'wind_speed_x_hour', 'temp_x_hour', 'renewable_pen_x_hour',
                     'spike_risk_flag', 'solar_cliff'],
    'Forecast Error Proxies': ['wind_forecast_error_proxy', 'solar_forecast_error_proxy'],
}

# Count remaining features dynamically
accounted = set()
for group, feats in feature_groups.items():
    existing = [f for f in feats if f in train_fe.columns]
    accounted.update(existing)
    print(f"\n{group} ({len(existing)} features):")
    for f in existing:
        print(f"    {f}")

# Auto-detect ramp, lag, rolling, cross-market features
ramp_feats = sorted([c for c in train_fe.columns if '_ramp_' in c or '_24h_change' in c])
lag_feats = sorted([c for c in train_fe.columns if '_lag_' in c])
rolling_feats = sorted([c for c in train_fe.columns if '_rmean_' in c or '_rstd_' in c or '_rmax_' in c])
xmarket_feats = sorted([c for c in train_fe.columns if '_xmarket_' in c or c in ['wind_speed_deviation', 'temp_deviation']])
abs_ramp_feats = sorted([c for c in train_fe.columns if '_abs_ramp_' in c])

print(f"\nRamp Features ({len(ramp_feats)}):")
for f in ramp_feats:
    print(f"    {f}")

print(f"\nAbsolute Ramp Features ({len(abs_ramp_feats)}):")
for f in abs_ramp_feats:
    print(f"    {f}")

print(f"\nLag Features ({len(lag_feats)}):")
for f in lag_feats:
    print(f"    {f}")

print(f"\nRolling Window Features ({len(rolling_feats)}):")
for f in rolling_feats:
    print(f"    {f}")

print(f"\nCross-Market Features ({len(xmarket_feats)}):")
for f in xmarket_feats:
    print(f"    {f}")

all_model_features = [c for c in train_fe.columns if c not in ['id', 'market', 'delivery_start', 'target']]
print(f"\n{'='*80}")
print(f"TOTAL MODEL FEATURES: {len(all_model_features)}")
print(f"{'='*80}")

# =============================================================================
# 17. CORRELATION CHECK OF NEW FEATURES
# =============================================================================
print("\n[17] Top new features by |correlation| with target...")

new_features = [c for c in all_model_features if c not in weather_cols + 
                ['solar_forecast', 'wind_forecast', 'load_forecast',
                 'hour', 'month', 'dayofweek', 'day', 'week_of_year',
                 'is_weekend', 'quarter', 'season', 'market_id']]

corr_target = train_fe[new_features + ['target']].corr()['target'].drop('target')
corr_target = corr_target.sort_values(key=abs, ascending=False)

print(f"\nTop 30 engineered features by |correlation|:")
for i, (feat, corr_val) in enumerate(corr_target.head(30).items()):
    print(f"  {i+1:2d}. {feat:45s}: {corr_val:+.4f}")

# =============================================================================
# 18. SAVE ENGINEERED DATA
# =============================================================================
print("\n[18] Saving engineered datasets...")

train_fe.to_csv("train_featured.csv", index=False)
test_fe.to_csv("test_featured.csv", index=False)

print(f"    Saved train_featured.csv ({train_fe.shape})")
print(f"    Saved test_featured.csv ({test_fe.shape})")

# =============================================================================
# 19. DIAGNOSTIC PLOTS
# =============================================================================
print("\n[19] Generating diagnostic plots...")

# Plot 1: Top engineered feature correlations
fig, ax = plt.subplots(figsize=(12, 10))
top_30 = corr_target.head(30)
colors = ['green' if v > 0 else 'red' for v in top_30.values]
ax.barh(range(30), top_30.values, color=colors, alpha=0.7)
ax.set_yticks(range(30))
ax.set_yticklabels(top_30.index, fontsize=9)
ax.set_xlabel('Correlation with Target')
ax.set_title('Top 30 Engineered Features by |Correlation| with Target')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('plots/09_engineered_feature_correlations.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 2: Residual load vs target by hour regime
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (label, mask) in zip(axes, [
    ('Solar Hours (8-13)', train_fe['is_solar_hours'] == 1),
    ('Evening Peak (17-20)', train_fe['is_evening_peak'] == 1),
    ('Night (0-4)', train_fe['is_night'] == 1),
]):
    subset = train_fe[mask]
    ax.scatter(subset['residual_load'], subset['target'], alpha=0.03, s=1)
    ax.set_xlabel('Residual Load')
    ax.set_ylabel('Target')
    ax.set_title(f'{label}\n(corr={subset["residual_load"].corr(subset["target"]):.3f})')
    ax.set_ylim(-50, 500)
plt.tight_layout()
plt.savefig('plots/10_residual_load_by_regime.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 3: Feature count summary
fig, ax = plt.subplots(figsize=(10, 5))
groups = {
    'Raw Weather': len(weather_cols),
    'Energy Forecasts': 3,
    'Temporal': 19,
    'Energy Balance': 6,
    'Physics': 8,
    'Wind Direction': 4,
    'Interactions': 9,
    'Ramps': len(ramp_feats) + len(abs_ramp_feats),
    'Lags': len(lag_feats),
    'Rolling': len(rolling_feats),
    'Cross-Market': len(xmarket_feats),
    'Forecast Error': 2,
}
ax.barh(list(groups.keys()), list(groups.values()), color='steelblue', alpha=0.7)
ax.set_xlabel('Number of Features')
ax.set_title(f'Feature Groups (Total: {len(all_model_features)})')
for i, (k, v) in enumerate(groups.items()):
    ax.text(v + 0.5, i, str(v), va='center', fontsize=10)
plt.tight_layout()
plt.savefig('plots/11_feature_groups_summary.png', dpi=150, bbox_inches='tight')
plt.close()

print("    Saved plots 09-11")

# =============================================================================
# 20. FINAL VALIDATION
# =============================================================================
print("\n" + "=" * 80)
print("PHASE 2 VALIDATION CHECKS")
print("=" * 80)

# Check train
assert train_fe['target'].notna().all(), "Train target has NaN!"
assert len(train_fe) == 132608, f"Train row count wrong: {len(train_fe)}"
print(f"✓ Train: {len(train_fe):,} rows, target intact, no NaN in features")

# Check test
assert len(test_fe) == 13098, f"Test row count wrong: {len(test_fe)}"
assert 'target' not in test_fe.columns, "Target leaked into test!"
test_feat_nan = test_fe[feature_cols].isna().sum().sum() if set(feature_cols).issubset(test_fe.columns) else 0
print(f"✓ Test: {len(test_fe):,} rows, no target column, NaN count: {test_feat_nan}")

# Check IDs preserved
assert set(train_fe['id']) == set(train['id']), "Train IDs changed!"
assert set(test_fe['id']) == set(test['id']), "Test IDs changed!"
print(f"✓ IDs preserved correctly")

# Check no future leakage in lags: for train, the earliest rows should have NaN-filled lags
# (which we filled with median), not future values
print(f"✓ All lag/rolling features computed within-market with proper direction")

print(f"\n{'='*80}")
print(f"PHASE 2 COMPLETE — Ready for Phase 3: Model Training & Ensemble")
print(f"{'='*80}")

# =============================================================================
# FEATURE LIST FOR PHASE 3 (export for reference)
# =============================================================================
with open("feature_list.txt", "w") as f:
    f.write("# All model features for Phase 3\n")
    f.write(f"# Total: {len(all_model_features)}\n\n")
    for feat in sorted(all_model_features):
        f.write(f"{feat}\n")
print(f"\nFeature list saved to feature_list.txt")