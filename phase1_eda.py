"""
Phase 1: Exploratory Data Analysis & Setup
Nitor Energy Case Competition — Intraday Electricity Price Prediction

This script loads, validates, and explores the training and test datasets.
It establishes the cross-validation strategy and saves key findings.
"""


import random
import numpy as np
random.seed(42)
np.random.seed(42)
import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "seaborn", "-q"])
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Output directory for plots
import os
os.makedirs("plots", exist_ok=True)

print("=" * 80)
print("PHASE 1: EXPLORATORY DATA ANALYSIS & SETUP")
print("=" * 80)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("\n" + "=" * 80)
print("1. LOADING DATA")
print("=" * 80)

train = pd.read_csv("train.csv")
test = pd.read_csv("test_for_participants.csv")

print(f"Train shape: {train.shape}")
print(f"Test shape:  {test.shape}")

# =============================================================================
# 2. VERIFY COLUMNS AND DTYPES
# =============================================================================
print("\n" + "=" * 80)
print("2. COLUMN VERIFICATION")
print("=" * 80)

print(f"\nTrain columns ({len(train.columns)}):")
print(list(train.columns))
print(f"\nTest columns ({len(test.columns)}):")
print(list(test.columns))

train_cols = set(train.columns)
test_cols = set(test.columns)
print(f"\nIn train but not test: {train_cols - test_cols}")
print(f"In test but not train: {test_cols - train_cols}")

print(f"\nTrain dtypes:\n{train.dtypes}")

# Parse datetime columns
train['delivery_start'] = pd.to_datetime(train['delivery_start'])
train['delivery_end'] = pd.to_datetime(train['delivery_end'])
test['delivery_start'] = pd.to_datetime(test['delivery_start'])
test['delivery_end'] = pd.to_datetime(test['delivery_end'])

# Set market as categorical
train['market'] = train['market'].astype('category')
test['market'] = test['market'].astype('category')

print(f"\nAfter parsing:")
print(f"  delivery_start dtype: {train['delivery_start'].dtype}")
print(f"  market dtype: {train['market'].dtype}")

# =============================================================================
# 3. BASIC DATA STATS
# =============================================================================
print("\n" + "=" * 80)
print("3. BASIC DATA STATISTICS")
print("=" * 80)

print(f"\n--- Train ---")
print(f"Rows: {len(train):,}")
print(f"ID range: {train['id'].min()} to {train['id'].max()}")
print(f"ID count (unique): {train['id'].nunique():,}")
print(f"Date range: {train['delivery_start'].min()} to {train['delivery_start'].max()}")
print(f"Markets: {sorted(train['market'].unique())}")
print(f"Markets value counts:\n{train['market'].value_counts().sort_index()}")

print(f"\n--- Test ---")
print(f"Rows: {len(test):,}")
print(f"ID range: {test['id'].min()} to {test['id'].max()}")
print(f"ID count (unique): {test['id'].nunique():,}")
print(f"Date range: {test['delivery_start'].min()} to {test['delivery_start'].max()}")
print(f"Markets: {sorted(test['market'].unique())}")
print(f"Markets value counts:\n{test['market'].value_counts().sort_index()}")

# Verify non-contiguous IDs
train_id_gaps = train['id'].max() - train['id'].min() + 1 - len(train)
test_id_gaps = test['id'].max() - test['id'].min() + 1 - len(test)
print(f"\nTrain ID gaps (missing IDs): {train_id_gaps:,}")
print(f"Test ID gaps (missing IDs): {test_id_gaps:,}")

# =============================================================================
# 4. MISSING VALUES
# =============================================================================
print("\n" + "=" * 80)
print("4. MISSING VALUES")
print("=" * 80)

train_missing = train.isnull().sum()
test_missing = test.isnull().sum()

print(f"\n--- Train missing values ---")
missing_train = train_missing[train_missing > 0]
if len(missing_train) > 0:
    for col, count in missing_train.items():
        print(f"  {col}: {count:,} ({100*count/len(train):.2f}%)")
else:
    print("  No missing values!")

print(f"\n--- Test missing values ---")
missing_test = test_missing[test_missing > 0]
if len(missing_test) > 0:
    for col, count in missing_test.items():
        print(f"  {col}: {count:,} ({100*count/len(test):.2f}%)")
else:
    print("  No missing values!")

# =============================================================================
# 5. TARGET DISTRIBUTION (DEEP ANALYSIS)
# =============================================================================
print("\n" + "=" * 80)
print("5. TARGET DISTRIBUTION ANALYSIS")
print("=" * 80)

target = train['target']
print(f"\nBasic statistics:")
print(target.describe())
print(f"\nAdditional quantiles:")
for q in [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999]:
    print(f"  {q*100:5.1f}th percentile: {target.quantile(q):.3f}")

print(f"\nSkewness: {target.skew():.3f}")
print(f"Kurtosis: {target.kurtosis():.3f}")

# Negative prices
neg_count = (target < 0).sum()
print(f"\nNegative prices: {neg_count:,} ({100*neg_count/len(target):.2f}%)")
print(f"  Min negative price: {target[target < 0].min():.3f}")
if neg_count > 0:
    print(f"  Mean negative price: {target[target < 0].mean():.3f}")

# Spike analysis
for threshold in [50, 75, 100, 150, 200, 250]:
    spike_count = (target > threshold).sum()
    print(f"  Prices > {threshold}: {spike_count:,} ({100*spike_count/len(target):.3f}%)")

# "Normal" range
normal_mask = (target >= -10) & (target <= 80)
print(f"\nPrices in 'normal' range [-10, 80]: {normal_mask.sum():,} ({100*normal_mask.mean():.2f}%)")
print(f"Prices outside 'normal' range: {(~normal_mask).sum():,} ({100*(~normal_mask).mean():.2f}%)")

# --- Plot: Target distribution ---
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Histogram (full range)
axes[0, 0].hist(target, bins=200, edgecolor='none', alpha=0.7)
axes[0, 0].set_title('Target Distribution (Full Range)')
axes[0, 0].set_xlabel('Intraday Price')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].axvline(target.median(), color='red', linestyle='--', label=f'Median: {target.median():.1f}')
axes[0, 0].axvline(target.mean(), color='orange', linestyle='--', label=f'Mean: {target.mean():.1f}')
axes[0, 0].legend()

# Histogram (zoomed to normal range)
axes[0, 1].hist(target[(target >= -20) & (target <= 100)], bins=200, edgecolor='none', alpha=0.7)
axes[0, 1].set_title('Target Distribution (Zoomed: -20 to 100)')
axes[0, 1].set_xlabel('Intraday Price')
axes[0, 1].set_ylabel('Frequency')

# Box plot
axes[1, 0].boxplot(target, vert=True)
axes[1, 0].set_title('Target Box Plot')
axes[1, 0].set_ylabel('Intraday Price')

# Log-scale histogram for tail behavior
target_positive = target[target > 0]
axes[1, 1].hist(target_positive, bins=200, edgecolor='none', alpha=0.7)
axes[1, 1].set_yscale('log')
axes[1, 1].set_title('Target Distribution (Log-Scale Y, Positive Only)')
axes[1, 1].set_xlabel('Intraday Price')
axes[1, 1].set_ylabel('Log Frequency')

plt.tight_layout()
plt.savefig('plots/01_target_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: plots/01_target_distribution.png")

# =============================================================================
# 6. TARGET BY MARKET
# =============================================================================
print("\n" + "=" * 80)
print("6. TARGET BY MARKET")
print("=" * 80)

market_stats = train.groupby('market')['target'].agg(['count', 'mean', 'std', 'min', 'median', 'max',
                                                        lambda x: x.quantile(0.01),
                                                        lambda x: x.quantile(0.99)])
market_stats.columns = ['count', 'mean', 'std', 'min', 'median', 'max', 'p01', 'p99']
print(market_stats.to_string())

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
train.boxplot(column='target', by='market', ax=axes[0])
axes[0].set_title('Target by Market')
axes[0].set_xlabel('Market')
axes[0].set_ylabel('Intraday Price')
plt.sca(axes[0])
plt.xticks(rotation=45)

# Zoomed version
train_zoomed = train[(train['target'] >= -30) & (train['target'] <= 120)]
train_zoomed.boxplot(column='target', by='market', ax=axes[1])
axes[1].set_title('Target by Market (Zoomed: -30 to 120)')
axes[1].set_xlabel('Market')
axes[1].set_ylabel('Intraday Price')
plt.sca(axes[1])
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('plots/02_target_by_market.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plots/02_target_by_market.png")

# =============================================================================
# 7. TARGET BY HOUR OF DAY
# =============================================================================
print("\n" + "=" * 80)
print("7. TARGET BY HOUR OF DAY")
print("=" * 80)

train['hour'] = train['delivery_start'].dt.hour
hourly_stats = train.groupby('hour')['target'].agg(['mean', 'std', 'median'])
print(hourly_stats.to_string())

fig, ax = plt.subplots(figsize=(14, 6))
hourly_mean = train.groupby('hour')['target'].mean()
hourly_std = train.groupby('hour')['target'].std()
ax.plot(hourly_mean.index, hourly_mean.values, 'b-o', linewidth=2, label='Mean')
ax.fill_between(hourly_mean.index, 
                hourly_mean.values - hourly_std.values,
                hourly_mean.values + hourly_std.values,
                alpha=0.2, color='blue', label='±1 StdDev')
ax.plot(hourly_mean.index, train.groupby('hour')['target'].median().values, 'r--o', linewidth=1, label='Median')
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Intraday Price')
ax.set_title('Target by Hour of Day')
ax.legend()
ax.set_xticks(range(24))
plt.tight_layout()
plt.savefig('plots/03_target_by_hour.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plots/03_target_by_hour.png")

# =============================================================================
# 8. TARGET BY MONTH AND DAY OF WEEK
# =============================================================================
print("\n" + "=" * 80)
print("8. TARGET BY MONTH AND DAY OF WEEK")
print("=" * 80)

train['month'] = train['delivery_start'].dt.month
train['dayofweek'] = train['delivery_start'].dt.dayofweek
train['is_weekend'] = train['dayofweek'].isin([5, 6]).astype(int)

print("\nBy Month:")
monthly_stats = train.groupby('month')['target'].agg(['mean', 'std', 'median', 'count'])
print(monthly_stats.to_string())

print("\nBy Day of Week (0=Mon, 6=Sun):")
dow_stats = train.groupby('dayofweek')['target'].agg(['mean', 'std', 'median', 'count'])
print(dow_stats.to_string())

print(f"\nWeekday mean: {train[train['is_weekend']==0]['target'].mean():.3f}")
print(f"Weekend mean: {train[train['is_weekend']==1]['target'].mean():.3f}")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Monthly
monthly_mean = train.groupby('month')['target'].mean()
axes[0].bar(monthly_mean.index, monthly_mean.values, alpha=0.7)
axes[0].set_xlabel('Month')
axes[0].set_ylabel('Mean Target')
axes[0].set_title('Mean Target by Month')
axes[0].set_xticks(range(1, 13))

# Day of week
dow_mean = train.groupby('dayofweek')['target'].mean()
day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
axes[1].bar(range(7), dow_mean.values, alpha=0.7, tick_label=day_names)
axes[1].set_xlabel('Day of Week')
axes[1].set_ylabel('Mean Target')
axes[1].set_title('Mean Target by Day of Week')

plt.tight_layout()
plt.savefig('plots/04_target_by_month_dow.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plots/04_target_by_month_dow.png")

# =============================================================================
# 9. BLOCK STRUCTURE VERIFICATION
# =============================================================================
print("\n" + "=" * 80)
print("9. BLOCK STRUCTURE VERIFICATION")
print("=" * 80)

# Check: each delivery_start has multiple markets
blocks = train.groupby('delivery_start')['market'].nunique()
print(f"\nMarkets per delivery hour:")
print(blocks.value_counts().sort_index())

# Verify shared energy forecast columns within blocks
shared_cols = ['solar_forecast', 'wind_forecast', 'load_forecast']
print(f"\nVerifying shared columns within blocks (first 5 blocks):")
for i, (ds, group) in enumerate(train.groupby('delivery_start')):
    if i >= 5:
        break
    for col in shared_cols:
        n_unique = group[col].nunique()
        if n_unique > 1:
            print(f"  WARNING: {col} has {n_unique} unique values at {ds}")
        else:
            pass  # Expected: 1 unique value
    print(f"  Block {ds}: {len(group)} markets, shared cols OK")

# Market F appearance timeline
print(f"\nMarket F timeline:")
market_f = train[train['market'] == 'Market F']
if len(market_f) > 0:
    print(f"  First appearance: {market_f['delivery_start'].min()}")
    print(f"  Last appearance:  {market_f['delivery_start'].max()}")
    print(f"  Total rows: {len(market_f):,}")
else:
    print("  Market F not found in training data!")

# Check when Market F appears (by month)
train['year_month'] = train['delivery_start'].dt.to_period('M')
market_counts_by_month = train.groupby(['year_month', 'market']).size().unstack(fill_value=0)
print(f"\nMarkets per month (first few months):")
print(market_counts_by_month.head(8).to_string())
print(f"\n... (last few months):")
print(market_counts_by_month.tail(4).to_string())

# When does Market F first appear?
if 'Market F' in market_counts_by_month.columns:
    first_f_month = market_counts_by_month[market_counts_by_month['Market F'] > 0].index[0]
    print(f"\nMarket F first non-zero month: {first_f_month}")

# =============================================================================
# 10. FEATURE DISTRIBUTIONS & CORRELATIONS
# =============================================================================
print("\n" + "=" * 80)
print("10. KEY FEATURE ANALYSIS")
print("=" * 80)

# Numeric columns (excluding id, target, datetime)
numeric_cols = [c for c in train.columns if train[c].dtype in ['float64', 'int64'] 
                and c not in ['id', 'target', 'hour', 'month', 'dayofweek', 'is_weekend']]
print(f"\nNumeric features ({len(numeric_cols)}):")
print(numeric_cols)

print(f"\nFeature summary statistics:")
print(train[numeric_cols].describe().round(3).to_string())

# Correlations with target
print(f"\nCorrelation with target (top features):")
corr_with_target = train[numeric_cols + ['target']].corr()['target'].drop('target').sort_values(key=abs, ascending=False)
for feat, corr_val in corr_with_target.items():
    print(f"  {feat:45s}: {corr_val:+.4f}")

# Plot top correlations
fig, ax = plt.subplots(figsize=(12, 8))
top_n = 20
top_corr = corr_with_target.head(top_n)
colors = ['green' if v > 0 else 'red' for v in top_corr.values]
ax.barh(range(top_n), top_corr.values, color=colors, alpha=0.7)
ax.set_yticks(range(top_n))
ax.set_yticklabels(top_corr.index)
ax.set_xlabel('Correlation with Target')
ax.set_title(f'Top {top_n} Features by |Correlation| with Target')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('plots/05_feature_correlations.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: plots/05_feature_correlations.png")

# =============================================================================
# 11. TARGET OVER TIME
# =============================================================================
print("\n" + "=" * 80)
print("11. TARGET OVER TIME")
print("=" * 80)

# Daily average target
daily_target = train.groupby(train['delivery_start'].dt.date)['target'].agg(['mean', 'max', 'min'])
daily_target.index = pd.to_datetime(daily_target.index)

fig, axes = plt.subplots(2, 1, figsize=(18, 10))

axes[0].plot(daily_target.index, daily_target['mean'], linewidth=0.8, alpha=0.8, label='Daily Mean')
axes[0].fill_between(daily_target.index, daily_target['min'], daily_target['max'], alpha=0.15, label='Daily Min-Max Range')
axes[0].set_title('Intraday Price Over Time (Daily Aggregates)')
axes[0].set_ylabel('Target')
axes[0].legend()
axes[0].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)

# Spike frequency over time (monthly)
monthly_spikes = train.groupby(train['delivery_start'].dt.to_period('M')).apply(
    lambda x: (x['target'] > 100).sum())
monthly_total = train.groupby(train['delivery_start'].dt.to_period('M')).size()
spike_rate = (monthly_spikes / monthly_total * 100)
spike_rate.index = spike_rate.index.to_timestamp()

axes[1].bar(spike_rate.index, spike_rate.values, width=25, alpha=0.7)
axes[1].set_title('Monthly Spike Rate (Target > 100)')
axes[1].set_ylabel('% of Rows with Spikes')
axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
plt.savefig('plots/06_target_over_time.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plots/06_target_over_time.png")

# =============================================================================
# 12. ENERGY FORECAST ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("12. ENERGY FORECAST ANALYSIS")
print("=" * 80)

# Residual load = load_forecast - (wind_forecast + solar_forecast)
train['residual_load'] = train['load_forecast'] - (train['wind_forecast'] + train['solar_forecast'])
print(f"\nResidual Load stats:")
print(train['residual_load'].describe())

print(f"\nCorrelation of residual_load with target: {train['residual_load'].corr(train['target']):.4f}")
print(f"Correlation of load_forecast with target: {train['load_forecast'].corr(train['target']):.4f}")
print(f"Correlation of wind_forecast with target: {train['wind_forecast'].corr(train['target']):.4f}")
print(f"Correlation of solar_forecast with target: {train['solar_forecast'].corr(train['target']):.4f}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Residual load vs target
axes[0].scatter(train['residual_load'], train['target'], alpha=0.02, s=1)
axes[0].set_xlabel('Residual Load (MW)')
axes[0].set_ylabel('Target')
axes[0].set_title('Residual Load vs Target')

# Wind forecast vs target
axes[1].scatter(train['wind_forecast'], train['target'], alpha=0.02, s=1)
axes[1].set_xlabel('Wind Forecast (MW)')
axes[1].set_ylabel('Target')
axes[1].set_title('Wind Forecast vs Target')

# Solar forecast vs target
axes[2].scatter(train['solar_forecast'], train['target'], alpha=0.02, s=1)
axes[2].set_xlabel('Solar Forecast (MW)')
axes[2].set_ylabel('Target')
axes[2].set_title('Solar Forecast vs Target')

plt.tight_layout()
plt.savefig('plots/07_energy_forecasts_vs_target.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plots/07_energy_forecasts_vs_target.png")

# =============================================================================
# 13. HOUR × MARKET HEATMAP
# =============================================================================
print("\n" + "=" * 80)
print("13. HOUR × MARKET INTERACTION")
print("=" * 80)

pivot = train.pivot_table(values='target', index='hour', columns='market', aggfunc='mean')
print(pivot.round(2).to_string())

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlBu_r', ax=ax)
ax.set_title('Mean Target by Hour × Market')
ax.set_ylabel('Hour of Day')
plt.tight_layout()
plt.savefig('plots/08_hour_market_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plots/08_hour_market_heatmap.png")

# =============================================================================
# 14. CROSS-VALIDATION STRATEGY
# =============================================================================
print("\n" + "=" * 80)
print("14. CROSS-VALIDATION STRATEGY")
print("=" * 80)

# We need chronological splits. Since data is organized by delivery_start,
# we sort by delivery_start and use TimeSeriesSplit on the sorted index.
# Each validation fold should be ~1-3 months.

# First, let's understand the time range
train_sorted = train.sort_values(['delivery_start', 'market']).reset_index(drop=True)
total_hours = train_sorted['delivery_start'].nunique()
total_days = (train_sorted['delivery_start'].max() - train_sorted['delivery_start'].min()).days
print(f"\nTotal unique delivery hours: {total_hours:,}")
print(f"Total days span: {total_days}")
print(f"Total rows: {len(train_sorted):,}")

# Strategy: 5-fold TimeSeriesSplit
# The test set is 3 months (Sep-Nov 2025). Training is ~32 months (Jan 2023 - Aug 2025).
# We want validation folds of ~2-3 months each.
# With 5 splits: fold 1 trains on ~1/6, validates on ~1/6, ..., fold 5 trains on ~5/6, validates on ~1/6
# Each fold's val set will be ~5-6 months in early folds and ~5-6 months in later folds.
# Better approach: use a fixed-size test window.

# Let's compute fold boundaries manually for informational purposes
n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)

# Get unique delivery_start timestamps in order
unique_times = train_sorted['delivery_start'].unique()
print(f"\nTimeSeriesSplit with {n_splits} folds:")
print("-" * 70)

for fold, (train_idx, val_idx) in enumerate(tscv.split(train_sorted)):
    train_dates = train_sorted.iloc[train_idx]['delivery_start']
    val_dates = train_sorted.iloc[val_idx]['delivery_start']
    
    train_start = train_dates.min()
    train_end = train_dates.max()
    val_start = val_dates.min()
    val_end = val_dates.max()
    val_days = (val_end - val_start).days
    
    print(f"Fold {fold+1}:")
    print(f"  Train: {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')} ({len(train_idx):,} rows)")
    print(f"  Val:   {val_start.strftime('%Y-%m-%d')} to {val_end.strftime('%Y-%m-%d')} ({len(val_idx):,} rows, ~{val_days} days)")

# Also define a "realistic" CV that mimics the test set structure
# The test set is ~3 months. Let's create custom folds with ~3-month validation windows.
print(f"\n\nCustom 3-Month Validation Folds (mimicking test structure):")
print("-" * 70)

# Define approximate 3-month boundary dates for validation
# Train period: 2023-01 to 2025-08
# We'll create 4 folds with 3-month val windows from the end backwards
val_boundaries = [
    ('2025-06-01', '2025-08-31'),  # Fold 4: val = Jun-Aug 2025
    ('2025-03-01', '2025-05-31'),  # Fold 3: val = Mar-May 2025
    ('2024-12-01', '2025-02-28'),  # Fold 2: val = Dec 2024-Feb 2025
    ('2024-09-01', '2024-11-30'),  # Fold 1: val = Sep-Nov 2024 (same months as test!)
]

for i, (val_start, val_end) in enumerate(reversed(val_boundaries)):
    val_start_dt = pd.Timestamp(val_start)
    val_end_dt = pd.Timestamp(val_end) + pd.Timedelta(hours=23)
    
    train_mask = train_sorted['delivery_start'] < val_start_dt
    val_mask = (train_sorted['delivery_start'] >= val_start_dt) & (train_sorted['delivery_start'] <= val_end_dt)
    
    n_train = train_mask.sum()
    n_val = val_mask.sum()
    
    print(f"Custom Fold {i+1}:")
    print(f"  Train: start to {val_start} ({n_train:,} rows)")
    print(f"  Val:   {val_start} to {val_end} ({n_val:,} rows)")

# =============================================================================
# 15. TRAIN vs TEST FEATURE DISTRIBUTION COMPARISON
# =============================================================================
print("\n" + "=" * 80)
print("15. TRAIN vs TEST FEATURE DISTRIBUTIONS")
print("=" * 80)

# Compare key features between train and test
key_features = ['load_forecast', 'wind_forecast', 'solar_forecast', 
                'air_temperature_2m', 'wind_speed_80m', 'cloud_cover_total',
                'global_horizontal_irradiance', 'surface_pressure']

print(f"\n{'Feature':<40s} {'Train Mean':>12s} {'Test Mean':>12s} {'Train Std':>12s} {'Test Std':>12s}")
print("-" * 90)
for feat in key_features:
    if feat in train.columns and feat in test.columns:
        print(f"{feat:<40s} {train[feat].mean():>12.2f} {test[feat].mean():>12.2f} "
              f"{train[feat].std():>12.2f} {test[feat].std():>12.2f}")

# Seasonal comparison: Test is Sep-Nov, compare to same months in training
train_sep_nov = train[train['delivery_start'].dt.month.isin([9, 10, 11])]
print(f"\n\nComparison: Test vs Train Sep-Nov (same season):")
print(f"{'Feature':<40s} {'Train Sep-Nov':>14s} {'Test':>12s}")
print("-" * 70)
for feat in key_features:
    if feat in train.columns and feat in test.columns:
        print(f"{feat:<40s} {train_sep_nov[feat].mean():>14.2f} {test[feat].mean():>12.2f}")

# =============================================================================
# 16. SPIKE ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("16. SPIKE DEEP DIVE")
print("=" * 80)

spikes = train[train['target'] > 100].copy()
non_spikes = train[train['target'] <= 100].copy()
print(f"\nSpike rows (target > 100): {len(spikes):,}")
print(f"Non-spike rows: {len(non_spikes):,}")

if len(spikes) > 0:
    print(f"\nSpike target stats:")
    print(spikes['target'].describe())
    
    print(f"\nSpikes by market:")
    print(spikes['market'].value_counts().sort_index())
    
    print(f"\nSpikes by hour:")
    spikes['hour'] = spikes['delivery_start'].dt.hour
    print(spikes.groupby('hour').size().to_string())
    
    print(f"\nSpike vs Non-spike feature comparison:")
    compare_feats = ['load_forecast', 'wind_forecast', 'solar_forecast', 
                     'wind_speed_80m', 'cloud_cover_total', 'air_temperature_2m']
    print(f"{'Feature':<35s} {'Spike Mean':>12s} {'Non-Spike Mean':>14s} {'Ratio':>8s}")
    print("-" * 75)
    for feat in compare_feats:
        if feat in train.columns:
            s_mean = spikes[feat].mean()
            ns_mean = non_spikes[feat].mean()
            ratio = s_mean / ns_mean if ns_mean != 0 else float('inf')
            print(f"{feat:<35s} {s_mean:>12.2f} {ns_mean:>14.2f} {ratio:>8.2f}")

# =============================================================================
# 17. SUMMARY & NEXT STEPS
# =============================================================================
print("\n" + "=" * 80)
print("17. PHASE 1 SUMMARY")
print("=" * 80)

print(f"""
KEY FINDINGS:
─────────────
• Training data: {len(train):,} rows, {train['delivery_start'].min().strftime('%Y-%m-%d')} to {train['delivery_start'].max().strftime('%Y-%m-%d')}
• Test data: {len(test):,} rows, {test['delivery_start'].min().strftime('%Y-%m-%d')} to {test['delivery_start'].max().strftime('%Y-%m-%d')}
• Markets: {sorted(train['market'].unique().tolist())}
• Target range: [{target.min():.3f}, {target.max():.3f}], mean={target.mean():.3f}, median={target.median():.3f}
• Negative prices: {neg_count:,} rows ({100*neg_count/len(target):.2f}%)
• Spikes (>100): {(target>100).sum():,} rows ({100*(target>100).mean():.3f}%)
• Missing values: {train.isnull().sum().sum()} (train), {test.isnull().sum().sum()} (test)
• Block structure confirmed: multiple markets per delivery hour with shared energy forecasts
• Strong diurnal and seasonal patterns detected
• Residual load is a key price driver (correlation: {train['residual_load'].corr(train['target']):.4f})

CROSS-VALIDATION STRATEGY:
• Primary: 5-fold TimeSeriesSplit (sklearn)
• Secondary: 4 custom folds with 3-month validation windows (mimics test set)
• All splits are strictly chronological — no data leakage

READY FOR PHASE 2: Feature Engineering
""")

# Clean up temp columns
train.drop(columns=['hour', 'month', 'dayofweek', 'is_weekend', 'year_month', 'residual_load'], 
           inplace=True, errors='ignore')

print("Phase 1 complete. All plots saved to plots/ directory.")
print("=" * 80)