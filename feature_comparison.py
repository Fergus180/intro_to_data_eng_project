"""
feature_comparison.py
Direct comparison of three temperature feature sets for the Random Forest.

All three configurations use identical settings:
  - Same target        : Mean_Energy
  - Same time controls : hour_sin, hour_cos, is_weekend
  - Same other weather : Humidity, Cloud_Cover, Precipitation, Wind_Speed
  - Same model         : RF 200 trees, min_samples_leaf=5, random_state=42
  - Same split         : chronological 80/20 from data_utils

Only the temperature representation changes:

  A. Raw temperature only
     Just Temperature. RF is non-parametric so it can learn the U-shape
     without any polynomial encoding — this is the simplest baseline.

  B. Raw temperature + quadratic term
     Temperature + Temperature_sq. The original approach before apparent
     temperature was introduced. Explicitly gives the model the U-shape.

  C. Apparent temperature  (current default in data_utils)
     Replaces Temperature, Temperature_sq, and partially Wind_Speed with
     the BOM perceived-heat formula. The hypothesis was that collapsing
     these into one physically meaningful variable would help. This test
     checks whether that hypothesis actually holds.

Results tell us whether abstracting to apparent temperature is justified,
or whether the model is equally capable of learning the relationship from
the raw measurements directly.

Figures saved to figures_feature_comparison/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from data_utils import (
    load_data, split_data, TARGET_COL,
    add_apparent_temperature, add_time_features,
)

os.makedirs('figures_feature_comparison', exist_ok=True)

COLOUR_MILD = '#2196F3'
COLOUR_WARM = '#F44336'

TIME_COLS    = ['hour_sin', 'hour_cos', 'is_weekend']
OTHER_WEATHER = ['Humidity', 'Cloud_Cover', 'Precipitation', 'Wind_Speed']

CONFIGS = {
    'A: Raw Temp':        ['Temperature'] + OTHER_WEATHER + TIME_COLS,
    'B: Temp + Temp²':    ['Temperature', 'Temperature_sq'] + OTHER_WEATHER + TIME_COLS,
    'C: Apparent Temp':   ['Apparent_Temp'] + OTHER_WEATHER + TIME_COLS,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def prepare_df(df):
    """Add all engineered columns regardless of which config is being tested."""
    df = add_apparent_temperature(df)
    df = add_time_features(df)
    df['Temperature_sq'] = df['Temperature'] ** 2
    return df


def chronological_split(df, feature_cols):
    """Chronological 80/20 split using an explicit feature list."""
    df_sorted = df.sort_values('Time').reset_index(drop=True)
    split_idx = int(len(df_sorted) * 0.80)
    train = df_sorted.iloc[:split_idx]
    test  = df_sorted.iloc[split_idx:]
    return (
        train[feature_cols], test[feature_cols],
        train[TARGET_COL],   test[TARGET_COL],
    )


def fit_and_score(df, feature_cols):
    """Fit RF with given features, return R², RMSE, and feature importances."""
    X_train, X_test, y_train, y_test = chronological_split(df, feature_cols)

    rf = RandomForestRegressor(
        n_estimators=200, min_samples_leaf=5, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    r2_val = r2_score(y_test, y_pred)
    rm     = rmse(y_test, y_pred)
    imp_df = pd.DataFrame({
        'Feature':    feature_cols,
        'Importance': rf.feature_importances_,
    }).sort_values('Importance', ascending=False)

    return r2_val, rm, imp_df


# ── Run comparisons ───────────────────────────────────────────────────────────

def run_comparison(mild_df, warm_df):
    rows = []
    importances = {}

    for config_label, feature_cols in CONFIGS.items():
        for zone_label, df in [('Mild (Sydney)', mild_df), ('Warm (Newcastle)', warm_df)]:
            r2_val, rm, imp_df = fit_and_score(df, feature_cols)
            rows.append({
                'Config':    config_label,
                'Zone':      zone_label,
                'R²':        round(r2_val, 4),
                'RMSE':      round(rm, 6),
            })
            importances[(config_label, zone_label)] = imp_df

    return pd.DataFrame(rows), importances


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_r2_comparison(results_df):
    """Grouped bar chart: R² for each config, both zones."""
    configs = list(CONFIGS.keys())
    x       = np.arange(len(configs))
    width   = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))

    mild_r2 = results_df[results_df['Zone'] == 'Mild (Sydney)']['R²'].tolist()
    warm_r2 = results_df[results_df['Zone'] == 'Warm (Newcastle)']['R²'].tolist()

    bars_mild = ax.bar(x - width / 2, mild_r2, width, label='Mild (Sydney)',
                       color=COLOUR_MILD, edgecolor='black', linewidth=0.5)
    bars_warm = ax.bar(x + width / 2, warm_r2, width, label='Warm (Newcastle)',
                       color=COLOUR_WARM, edgecolor='black', linewidth=0.5)

    for bars in [bars_mild, bars_warm]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylim(0, 1)
    ax.set_ylabel('R²  (chronological test set)')
    ax.set_title('Temperature Feature Comparison — Random Forest R²\n'
                 'All configs use identical time controls and other weather features')
    ax.legend()

    plt.tight_layout()
    path = os.path.join('figures_feature_comparison', 'r2_comparison.png')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_feature_importances(importances):
    """
    3 x 2 grid of feature importance bar charts — one panel per config per zone.
    Shows how importance is distributed across features in each approach.
    """
    configs    = list(CONFIGS.keys())
    zone_labels = ['Mild (Sydney)', 'Warm (Newcastle)']
    colours     = [COLOUR_MILD, COLOUR_WARM]

    fig, axes = plt.subplots(len(configs), 2, figsize=(14, 12))

    for row_i, config_label in enumerate(configs):
        for col_i, (zone_label, colour) in enumerate(zip(zone_labels, colours)):
            ax      = axes[row_i, col_i]
            imp_df  = importances[(config_label, zone_label)]

            ax.barh(imp_df['Feature'], imp_df['Importance'],
                    color=colour, edgecolor='black', linewidth=0.4)
            ax.set_xlabel('Mean Decrease in Impurity')
            ax.set_title(f'{config_label}  |  {zone_label}')
            ax.invert_yaxis()

    fig.suptitle('Feature Importance by Temperature Representation\n'
                 'Note how importance distributes across T vs T² vs Apparent_Temp',
                 fontsize=12)
    plt.tight_layout()
    path = os.path.join('figures_feature_comparison', 'feature_importances.png')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Loading data...")
    mild_df, warm_df = load_data()

    mild_df = prepare_df(mild_df)
    warm_df = prepare_df(warm_df)

    print("\nRunning comparison across three temperature representations...")
    results_df, importances = run_comparison(mild_df, warm_df)

    print("\nResults:")
    print(results_df.to_string(index=False))

    print("\nGenerating figures...")
    plot_r2_comparison(results_df)
    plot_feature_importances(importances)

    print("\nDone. Figures saved to figures_feature_comparison/")
