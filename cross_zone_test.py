"""
cross_zone_test.py
Cross-zone generalisation test for the Random Forest model.

Tests whether the weather-energy relationship learned in one climate zone
transfers to the other:
  - Model trained on Mild (Sydney) data, tested on Warm (Newcastle) data
  - Model trained on Warm (Newcastle) data, tested on Mild (Sydney) data

The training set is always the chronological first 80% of the training zone.
The test set is always the chronological last 20% of the OPPOSITE zone —
the same time window the within-zone tests use, keeping comparisons fair.

Reference: Section 2.3 of the report — "models will also be tested on data
from a different Australian region to assess generalisation beyond the
SGSC trial sites."

Figures saved to figures_cross_zone/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from data_utils import load_data, split_data, FEATURE_COLS, TARGET_COL

os.makedirs('figures_cross_zone', exist_ok=True)

COLOUR_MILD = '#2196F3'
COLOUR_WARM = '#F44336'


# ── Helper ────────────────────────────────────────────────────────────────────

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def fit_rf(X_train, y_train):
    rf = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    return rf


def chronological_test_set(df):
    """Return only the last 20 % of the zone by time (the held-out test rows)."""
    df_sorted = df.sort_values('Time').reset_index(drop=True)
    split_idx = int(len(df_sorted) * 0.80)
    test = df_sorted.iloc[split_idx:]
    return test[FEATURE_COLS], test[TARGET_COL]


# ── Cross-zone evaluation ─────────────────────────────────────────────────────

def run_cross_zone_tests(mild_df, warm_df):
    """
    Fit two models (one per zone) and evaluate them on both their own
    test set and the other zone's test set. Returns a results table.
    """
    # Training sets (first 80 % chronologically of each zone)
    X_mild_train, _, y_mild_train, _ = split_data(mild_df)
    X_warm_train, _, y_warm_train, _ = split_data(warm_df)

    # Test sets (last 20 % chronologically of each zone)
    X_mild_test, y_mild_test = chronological_test_set(mild_df)
    X_warm_test, y_warm_test = chronological_test_set(warm_df)

    # Train one model per zone
    rf_mild = fit_rf(X_mild_train, y_mild_train)
    rf_warm = fit_rf(X_warm_train, y_warm_train)

    # Evaluate every train/test combination
    results = []

    for model_label, model, train_zone in [
        ('Trained on Mild (Sydney)',     rf_mild, 'Mild'),
        ('Trained on Warm (Newcastle)', rf_warm, 'Warm'),
    ]:
        for test_label, X_test, y_test, test_zone in [
            ('Tested on Mild (Sydney)',     X_mild_test, y_mild_test, 'Mild'),
            ('Tested on Warm (Newcastle)', X_warm_test, y_warm_test, 'Warm'),
        ]:
            y_pred = model.predict(X_test)
            r2   = r2_score(y_test, y_pred)
            rm   = rmse(y_test, y_pred)
            same = (train_zone == test_zone)
            results.append({
                'Train zone': train_zone,
                'Test zone':  test_zone,
                'Type':       'Within-zone' if same else 'Cross-zone',
                'R²':         round(r2, 4),
                'RMSE':       round(rm, 6),
            })

    return pd.DataFrame(results), rf_mild, rf_warm, X_mild_test, X_warm_test, y_mild_test, y_warm_test


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_actual_vs_predicted(rf_mild, rf_warm, X_mild_test, X_warm_test, y_mild_test, y_warm_test):
    """
    2 x 2 grid: each model evaluated on its own zone and on the opposite zone.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    combos = [
        (axes[0, 0], rf_mild, X_mild_test, y_mild_test, 'Mild model on Mild test',  COLOUR_MILD, 'Within-zone'),
        (axes[0, 1], rf_mild, X_warm_test, y_warm_test, 'Mild model on Warm test',  COLOUR_WARM, 'Cross-zone'),
        (axes[1, 0], rf_warm, X_warm_test, y_warm_test, 'Warm model on Warm test', COLOUR_WARM, 'Within-zone'),
        (axes[1, 1], rf_warm, X_mild_test, y_mild_test, 'Warm model on Mild test', COLOUR_MILD, 'Cross-zone'),
    ]

    for ax, model, X_test, y_test, title, colour, zone_type in combos:
        y_pred = model.predict(X_test)
        r2_val = r2_score(y_test, y_pred)
        rm     = rmse(y_test, y_pred)

        ax.scatter(y_test, y_pred, alpha=0.3, s=7, color=colour)

        lo = min(float(y_test.min()), float(y_pred.min()))
        hi = max(float(y_test.max()), float(y_pred.max()))
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1.2, label='Perfect fit')

        ax.set_title(f'{title}  [{zone_type}]\nR² = {r2_val:.3f}   RMSE = {rm:.4f}')
        ax.set_xlabel('Actual Mean_Energy (kWh/household/hour)')
        ax.set_ylabel('Predicted')
        ax.legend(fontsize=8)

    fig.suptitle(
        'Cross-Zone Generalisation Test — Random Forest\n'
        'Top row: Sydney model   |   Bottom row: Newcastle model',
        fontsize=12,
    )
    plt.tight_layout()
    path = os.path.join('figures_cross_zone', 'cross_zone_actual_vs_predicted.png')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_r2_comparison(results_df):
    """
    Bar chart comparing within-zone vs cross-zone R² for both models.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    labels    = ['Mild model\non Mild test', 'Mild model\non Warm test',
                 'Warm model\non Warm test', 'Warm model\non Mild test']
    r2_values = results_df['R²'].tolist()
    colours   = [COLOUR_MILD, COLOUR_WARM, COLOUR_WARM, COLOUR_MILD]
    hatches   = ['', '///', '', '///']

    bars = ax.bar(labels, r2_values, color=colours, hatch=hatches,
                  edgecolor='black', linewidth=0.6)

    # Annotate each bar with its R² value
    for bar, val in zip(bars, r2_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    ax.set_ylim(0, 1)
    ax.set_ylabel('R²  (test set)')
    ax.set_title('Within-zone vs Cross-zone R²\nHatched bars = cross-zone (trained on opposite region)')
    ax.axhline(0, color='black', lw=0.5)

    plt.tight_layout()
    path = os.path.join('figures_cross_zone', 'cross_zone_r2_comparison.png')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_residuals_by_apparent_temp(rf_mild, rf_warm, mild_df, warm_df):
    """
    Plot prediction error vs apparent temperature for within- and cross-zone tests.
    Shows whether the model's errors are systematic at temperature extremes.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    test_sets = [
        (axes[0], rf_mild, mild_df, warm_df, 'Mild model (Sydney)', COLOUR_MILD, COLOUR_WARM),
        (axes[1], rf_warm, warm_df, mild_df, 'Warm model (Newcastle)', COLOUR_WARM, COLOUR_MILD),
    ]

    for ax, model, home_df, away_df, title, home_col, away_col in test_sets:
        for df, label, colour, ls in [
            (home_df, 'Within-zone', home_col, '-'),
            (away_df, 'Cross-zone',  away_col, '--'),
        ]:
            X_test, y_test = chronological_test_set(df)
            y_pred    = model.predict(X_test)
            error     = y_test.values - y_pred
            app_temp  = X_test['Apparent_Temp'].values

            ax.scatter(app_temp, error, alpha=0.2, s=6, color=colour, label=label)

        ax.axhline(0, color='black', lw=1)
        ax.set_xlabel('Apparent Temperature (°C)')
        ax.set_ylabel('Residual (actual − predicted)  kWh/household/hr')
        ax.set_title(title)
        ax.legend()

    fig.suptitle(
        'Prediction Residuals vs Apparent Temperature\n'
        'Systematic bias at temperature extremes suggests cross-zone limitations',
        fontsize=12,
    )
    plt.tight_layout()
    path = os.path.join('figures_cross_zone', 'cross_zone_residuals_vs_temp.png')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Loading data...")
    mild_df, warm_df = load_data()

    print("\nFitting models and running cross-zone tests...")
    results_df, rf_mild, rf_warm, X_mt, X_wt, y_mt, y_wt = run_cross_zone_tests(mild_df, warm_df)

    print("\nResults:")
    print(results_df.to_string(index=False))

    print("\nGenerating figures...")
    plot_actual_vs_predicted(rf_mild, rf_warm, X_mt, X_wt, y_mt, y_wt)
    plot_r2_comparison(results_df)
    plot_residuals_by_apparent_temp(rf_mild, rf_warm, mild_df, warm_df)

    print("\nDone. Figures saved to figures_cross_zone/")

    # ── Is this a good generalisation test? ──
    print()
    print("=" * 65)
    print("NOTE ON GENERALISATION")
    print("=" * 65)
    print("""
  This cross-zone test is a MODERATE generalisation test:

  Strengths:
    - Same time period in both train and test sets
    - Both zones use identical data collection methodology (SGSC)
    - Zones are geographically distinct (Sydney vs Newcastle)
    - Different mean temperatures, humidity profiles, and climate
      characteristics provide real distributional shift

  Limitations:
    - Both zones are NSW coastal temperate — weather ranges overlap
      heavily. A model trained on Sydney will have seen most of the
      temperature and humidity conditions that Newcastle experiences.
    - Same time period means no test of performance across different
      seasons or years not seen in training.
    - A stronger test would be an entirely different Australian region
      (e.g. Brisbane subtropical, or Melbourne cool temperate) where
      the weather distribution is more genuinely out-of-sample.
    """)
