"""
results_section.py
Generates all figures and tables for the results section of the report.

Self-contained: loads the CSVs directly and defines its own feature sets for
each phase. Does NOT import FEATURE_COLS or TARGET_COL from data_utils.py so
the current submission files are never touched.

Part 1 — Isolating weather effects
    Target  : normalized_nrg  (daily routine already removed)
    Features: Temperature, Temperature_sq, Humidity, Cloud_Cover,
              Precipitation, Wind_Speed
    Split   : random 80/20 (seed=42) — matches original submission

Part 2 — Full prediction model
    Target  : Mean_Energy  (raw consumption)
    Features: Temperature, Humidity, Cloud_Cover, Precipitation,
              Wind_Speed, hour_sin, hour_cos, is_weekend
    Split   : chronological 80/20 — more realistic forecasting test

All figures saved to figures_results/
All tables printed to console (copy into report).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

# ── Setup ─────────────────────────────────────────────────────────────────────

os.makedirs('figures_results', exist_ok=True)

_HERE     = os.path.dirname(os.path.abspath(__file__))
MILD_CSV  = os.path.join(_HERE, 'Final_mild_data.csv')
WARM_CSV  = os.path.join(_HERE, 'Final_warm_data.csv')

COLOUR_MILD = '#2196F3'
COLOUR_WARM = '#F44336'
ZONE_LABELS = ['Mild (Sydney)', 'Warm (Newcastle)']

plt.rcParams.update({'figure.dpi': 150, 'font.size': 10})

# ── Feature sets ──────────────────────────────────────────────────────────────

PH1_FEATURES = [
    'Temperature', 'Temperature_sq',
    'Humidity', 'Cloud_Cover', 'Precipitation', 'Wind_Speed',
]
PH1_TARGET = 'normalized_nrg'

PH2_FEATURES = [
    'Temperature', 'Humidity', 'Cloud_Cover', 'Precipitation', 'Wind_Speed',
    'hour_sin', 'hour_cos', 'is_weekend',
]
PH2_TARGET = 'Mean_Energy'

WEATHER_ONLY = ['Temperature', 'Humidity', 'Cloud_Cover', 'Precipitation', 'Wind_Speed']
TIME_ONLY    = ['hour_sin', 'hour_cos', 'is_weekend']


# ── Data loading ──────────────────────────────────────────────────────────────

def load_raw():
    mild = pd.read_csv(MILD_CSV, parse_dates=['Time'])
    warm = pd.read_csv(WARM_CSV, parse_dates=['Time'])
    mild['Zone'] = 'Mild (Sydney)'
    warm['Zone'] = 'Warm (Newcastle)'

    for df in [mild, warm]:
        df['Temperature_sq'] = df['Temperature'] ** 2
        df['hour_sin']       = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos']       = np.cos(2 * np.pi * df['hour'] / 24)
        df['is_weekend']     = df['day'].isin(['Saturday', 'Sunday']).astype(int)

    return mild, warm


# ── Split helpers ─────────────────────────────────────────────────────────────

def random_split(df, features, target):
    X = df[features]
    y = df[target]
    return train_test_split(X, y, test_size=0.20, random_state=42)


def chrono_split(df, features, target):
    df_s      = df.sort_values('Time').reset_index(drop=True)
    cut       = int(len(df_s) * 0.80)
    train, test = df_s.iloc[:cut], df_s.iloc[cut:]
    return train[features], test[features], train[target], test[target]


# ── Metric helpers ────────────────────────────────────────────────────────────

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def fit_rf(X_tr, y_tr):
    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=5,
                               random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    return rf


def fit_ols_sm(X_tr, y_tr):
    return sm.OLS(y_tr, sm.add_constant(X_tr, has_constant='add')).fit()


def fit_lasso(X_tr, y_tr):
    scaler = StandardScaler()
    Xs_tr  = scaler.fit_transform(X_tr)
    lasso  = LassoCV(cv=5, max_iter=10_000, random_state=42)
    lasso.fit(Xs_tr, y_tr)
    return lasso, scaler


def savefig(name):
    path = os.path.join('figures_results', f'{name}.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — WEATHER-ONLY ANALYSIS (normalized_nrg)
# ─────────────────────────────────────────────────────────────────────────────

def ph1_run_models(mild, warm):
    """Fit OLS, Lasso, RF on normalized_nrg for both zones. Return results table."""
    rows = []
    models_out = {}

    for zone_label, df in zip(ZONE_LABELS, [mild, warm]):
        Xtr, Xte, ytr, yte = random_split(df, PH1_FEATURES, PH1_TARGET)

        # OLS
        ols = fit_ols_sm(Xtr, ytr)
        yp  = ols.predict(sm.add_constant(Xte, has_constant='add'))
        rows.append({'Phase': 1, 'Model': 'OLS',   'Zone': zone_label,
                     'R²': round(r2_score(yte, yp), 4),
                     'RMSE': round(rmse(yte, yp), 5)})
        models_out[(zone_label, 'ols')] = (ols, Xtr, Xte, ytr, yte)

        # Lasso
        lasso, scaler = fit_lasso(Xtr, ytr)
        yp = lasso.predict(scaler.transform(Xte))
        rows.append({'Phase': 1, 'Model': 'Lasso', 'Zone': zone_label,
                     'R²': round(r2_score(yte, yp), 4),
                     'RMSE': round(rmse(yte, yp), 5)})

        # RF
        rf  = fit_rf(Xtr, ytr)
        yp  = rf.predict(Xte)
        rows.append({'Phase': 1, 'Model': 'RF',    'Zone': zone_label,
                     'R²': round(r2_score(yte, yp), 4),
                     'RMSE': round(rmse(yte, yp), 5)})
        models_out[(zone_label, 'rf')] = (rf, Xtr, Xte, ytr, yte)

    return pd.DataFrame(rows), models_out


# Figure 1 — Phase 1 model comparison
def fig_ph1_model_comparison(results_df):
    sub = results_df[results_df['Phase'] == 1]
    models   = ['OLS', 'Lasso', 'RF']
    x        = np.arange(len(models))
    width    = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    for offset, zone, colour in zip([-width/2, width/2], ZONE_LABELS, [COLOUR_MILD, COLOUR_WARM]):
        r2s = [sub[(sub['Zone'] == zone) & (sub['Model'] == m)]['R²'].values[0]
               for m in models]
        bars = ax.bar(x + offset, r2s, width, label=zone, color=colour,
                      edgecolor='black', linewidth=0.5)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 0.7)
    ax.set_ylabel('R²  (random 20 % test set)')
    ax.set_title('Phase 1 — Weather-Only Models\n'
                 'Target: normalized_nrg  (daily routine pre-removed)')
    ax.legend()
    plt.tight_layout()
    savefig('fig1_ph1_model_comparison')


# Figure 2 — Temperature vs normalized_nrg scatter with polynomial trend
def fig_ph1_temp_scatter(mild, warm):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, df, label, colour in zip(axes, [mild, warm], ZONE_LABELS,
                                      [COLOUR_MILD, COLOUR_WARM]):
        ax.scatter(df['Temperature'], df[PH1_TARGET],
                   alpha=0.12, s=5, color=colour)

        # Polynomial trend line (degree 2 captures U-shape)
        temp_range = np.linspace(df['Temperature'].min(), df['Temperature'].max(), 300)
        poly_fit   = np.polyfit(df['Temperature'], df[PH1_TARGET], 2)
        trend      = np.polyval(poly_fit, temp_range)
        ax.plot(temp_range, trend, color='black', lw=2, label='Quadratic trend')

        ax.axhline(0, color='grey', linestyle='--', lw=0.8)
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('normalized_nrg  (residual kWh/household/hr)')
        ax.set_title(label)
        ax.legend(fontsize=8)

    fig.suptitle(
        'Temperature vs Residual Energy Consumption\n'
        'U-shape confirms increased demand at both hot and cold extremes',
        fontsize=12,
    )
    plt.tight_layout()
    savefig('fig2_ph1_temp_scatter')


# Figure 3 — Phase 1 RF permutation importance
def fig_ph1_feature_importance(mild, warm):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, df, label, colour in zip(axes, [mild, warm], ZONE_LABELS,
                                      [COLOUR_MILD, COLOUR_WARM]):
        Xtr, Xte, ytr, yte = random_split(df, PH1_FEATURES, PH1_TARGET)
        rf  = fit_rf(Xtr, ytr)
        perm = permutation_importance(rf, Xte, yte, n_repeats=10,
                                      random_state=42, n_jobs=-1)
        imp_df = pd.DataFrame({
            'Feature': PH1_FEATURES,
            'Importance': perm.importances_mean,
            'Std': perm.importances_std,
        }).sort_values('Importance')

        ax.barh(imp_df['Feature'], imp_df['Importance'],
                xerr=imp_df['Std'], color=colour,
                edgecolor='black', linewidth=0.4, capsize=3)
        ax.axvline(0, color='black', lw=0.8)
        ax.set_xlabel('Permutation Importance (mean R² drop)')
        ax.set_title(label)

    fig.suptitle(
        'Phase 1 — RF Permutation Feature Importance  (normalized_nrg)\n'
        'Temperature is the dominant weather driver; precipitation contributes least',
        fontsize=12,
    )
    plt.tight_layout()
    savefig('fig3_ph1_feature_importance')


# ─────────────────────────────────────────────────────────────────────────────
# PART 2 — FULL PREDICTION MODEL (Mean_Energy)
# ─────────────────────────────────────────────────────────────────────────────

def ph2_run_models(mild, warm):
    """Fit OLS, Lasso, RF on Mean_Energy (chronological split) for both zones."""
    rows = []
    models_out = {}

    for zone_label, df in zip(ZONE_LABELS, [mild, warm]):
        Xtr, Xte, ytr, yte = chrono_split(df, PH2_FEATURES, PH2_TARGET)

        # OLS
        ols = fit_ols_sm(Xtr, ytr)
        yp  = ols.predict(sm.add_constant(Xte, has_constant='add'))
        rows.append({'Phase': 2, 'Model': 'OLS',   'Zone': zone_label,
                     'R²': round(r2_score(yte, yp), 4),
                     'RMSE': round(rmse(yte, yp), 5)})

        # Lasso
        lasso, scaler = fit_lasso(Xtr, ytr)
        yp = lasso.predict(scaler.transform(Xte))
        rows.append({'Phase': 2, 'Model': 'Lasso', 'Zone': zone_label,
                     'R²': round(r2_score(yte, yp), 4),
                     'RMSE': round(rmse(yte, yp), 5)})

        # RF
        rf  = fit_rf(Xtr, ytr)
        yp  = rf.predict(Xte)
        rows.append({'Phase': 2, 'Model': 'RF',    'Zone': zone_label,
                     'R²': round(r2_score(yte, yp), 4),
                     'RMSE': round(rmse(yte, yp), 5)})
        models_out[(zone_label, 'rf')] = (rf, Xtr, Xte, ytr, yte)

    return pd.DataFrame(rows), models_out


# Figure 4 — Phase 1 vs Phase 2 R² improvement (RF only)
def fig_phase_comparison(ph1_results, ph2_results):
    fig, ax = plt.subplots(figsize=(9, 5))

    x     = np.arange(len(ZONE_LABELS))
    width = 0.3

    ph1_r2 = [ph1_results[(ph1_results['Zone'] == z) &
                           (ph1_results['Model'] == 'RF')]['R²'].values[0]
              for z in ZONE_LABELS]
    ph2_r2 = [ph2_results[(ph2_results['Zone'] == z) &
                           (ph2_results['Model'] == 'RF')]['R²'].values[0]
              for z in ZONE_LABELS]

    b1 = ax.bar(x - width / 2, ph1_r2, width,
                label='Phase 1: weather-only (normalized_nrg, random split)',
                color='#90A4AE', edgecolor='black', linewidth=0.5)
    b2 = ax.bar(x + width / 2, ph2_r2, width,
                label='Phase 2: weather + time-of-day (Mean_Energy, chrono split)',
                color='#43A047', edgecolor='black', linewidth=0.5)

    for bars in [b1, b2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(ZONE_LABELS)
    ax.set_ylim(0, 1)
    ax.set_ylabel('R²')
    ax.set_title('Random Forest R²: Weather-Only vs Full Prediction Model\n'
                 'Phase 2 adds time-of-day controls and uses raw energy as target')
    ax.legend(fontsize=8)
    plt.tight_layout()
    savefig('fig4_phase_comparison')


# Figure 5 — Ablation: weather only / time only / combined
def fig_ablation(mild, warm):
    configs = {
        'Weather\nonly':      WEATHER_ONLY,
        'Time\nonly':         TIME_ONLY,
        'Weather\n+ Time':    PH2_FEATURES,
    }

    results = {label: {'OLS': [], 'RF': []} for label in configs}

    for df in [mild, warm]:
        for feat_label, cols in configs.items():
            Xtr, Xte, ytr, yte = chrono_split(df, cols, PH2_TARGET)

            # OLS
            ols = LinearRegression().fit(Xtr, ytr)
            results[feat_label]['OLS'].append(r2_score(yte, ols.predict(Xte)))

            # RF
            rf = fit_rf(Xtr, ytr)
            results[feat_label]['RF'].append(r2_score(yte, rf.predict(Xte)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x     = np.arange(len(configs))
    width = 0.35
    cfg_labels = list(configs.keys())

    for ax, model_key, title in zip(axes, ['OLS', 'RF'], ['OLS', 'Random Forest']):
        mild_r2 = [results[c][model_key][0] for c in configs]
        warm_r2 = [results[c][model_key][1] for c in configs]

        b1 = ax.bar(x - width / 2, mild_r2, width, label='Mild (Sydney)',
                    color=COLOUR_MILD, edgecolor='black', linewidth=0.5)
        b2 = ax.bar(x + width / 2, warm_r2, width, label='Warm (Newcastle)',
                    color=COLOUR_WARM, edgecolor='black', linewidth=0.5)

        for bars in [b1, b2]:
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.008,
                        f'{bar.get_height():.2f}',
                        ha='center', va='bottom', fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(cfg_labels)
        ax.set_ylim(0, 1)
        ax.set_ylabel('R²')
        ax.set_title(f'{title}')
        ax.legend(fontsize=8)

    fig.suptitle(
        'Ablation: Relative Contribution of Weather vs Time-of-Day Features\n'
        'Target: Mean_Energy, chronological split',
        fontsize=12,
    )
    plt.tight_layout()
    savefig('fig5_ablation')


# Figure 6 — Phase 2 RF permutation importance
def fig_ph2_feature_importance(mild, warm):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, df, label, colour in zip(axes, [mild, warm], ZONE_LABELS,
                                      [COLOUR_MILD, COLOUR_WARM]):
        Xtr, Xte, ytr, yte = chrono_split(df, PH2_FEATURES, PH2_TARGET)
        rf   = fit_rf(Xtr, ytr)
        perm = permutation_importance(rf, Xte, yte, n_repeats=10,
                                      random_state=42, n_jobs=-1)
        imp_df = pd.DataFrame({
            'Feature':    PH2_FEATURES,
            'Importance': perm.importances_mean,
            'Std':        perm.importances_std,
        }).sort_values('Importance')

        # Group weather vs time features by colour intensity
        bar_colours = ['#78909C' if f in TIME_ONLY else colour
                       for f in imp_df['Feature']]

        ax.barh(imp_df['Feature'], imp_df['Importance'],
                xerr=imp_df['Std'], color=bar_colours,
                edgecolor='black', linewidth=0.4, capsize=3)
        ax.axvline(0, color='black', lw=0.8)
        ax.set_xlabel('Permutation Importance (mean R² drop)')
        ax.set_title(label)

    fig.suptitle(
        'Phase 2 — RF Permutation Feature Importance  (Mean_Energy)\n'
        'Zone colour = weather features   |   Grey = time-of-day features',
        fontsize=12,
    )
    plt.tight_layout()
    savefig('fig6_ph2_feature_importance')


# Figure 7 — Phase 2 RF actual vs predicted (chronological test set)
def fig_ph2_actual_vs_predicted(mild, warm, ph2_models):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, df, label, colour in zip(axes, [mild, warm], ZONE_LABELS,
                                      [COLOUR_MILD, COLOUR_WARM]):
        rf, Xtr, Xte, ytr, yte = ph2_models[(label, 'rf')]
        y_pred = rf.predict(Xte)
        r2_val = r2_score(yte, y_pred)
        rm     = rmse(yte, y_pred)

        ax.scatter(yte, y_pred, alpha=0.3, s=7, color=colour)
        lo = min(float(yte.min()), float(y_pred.min()))
        hi = max(float(yte.max()), float(y_pred.max()))
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1.2, label='Perfect fit')

        ax.set_title(f'{label}\nR² = {r2_val:.3f}   RMSE = {rm:.4f} kWh/household/hr')
        ax.set_xlabel('Actual Mean_Energy  (kWh/household/hr)')
        ax.set_ylabel('Predicted')
        ax.legend(fontsize=8)

    fig.suptitle(
        'Phase 2 — Random Forest: Actual vs Predicted  (Chronological Test Set)\n'
        'Test period: November 2013 – March 2014',
        fontsize=12,
    )
    plt.tight_layout()
    savefig('fig7_ph2_actual_vs_predicted')


# ─────────────────────────────────────────────────────────────────────────────
# PART 3 — ROBUSTNESS CHECKS
# ─────────────────────────────────────────────────────────────────────────────

# Figure 8 — Cross-zone generalisation
def fig_cross_zone(mild, warm):
    def get_train_test(df):
        Xtr, Xte, ytr, yte = chrono_split(df, PH2_FEATURES, PH2_TARGET)
        return Xtr, Xte, ytr, yte

    Xm_tr, Xm_te, ym_tr, ym_te = get_train_test(mild)
    Xw_tr, Xw_te, yw_tr, yw_te = get_train_test(warm)

    rf_mild = fit_rf(Xm_tr, ym_tr)
    rf_warm = fit_rf(Xw_tr, yw_tr)

    combos = [
        ('Mild model\non Mild test',  rf_mild, Xm_te, ym_te, 'Within-zone', COLOUR_MILD),
        ('Mild model\non Warm test',  rf_mild, Xw_te, yw_te, 'Cross-zone',  COLOUR_WARM),
        ('Warm model\non Warm test', rf_warm, Xw_te, yw_te, 'Within-zone', COLOUR_WARM),
        ('Warm model\non Mild test', rf_warm, Xm_te, ym_te, 'Cross-zone',  COLOUR_MILD),
    ]

    labels  = [c[0] for c in combos]
    r2s     = [r2_score(c[3], c[1].predict(c[2])) for c in combos]
    colours = [c[5] for c in combos]
    hatches = ['', '///', '', '///']

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, r2s, color=colours, hatch=hatches,
                  edgecolor='black', linewidth=0.6)
    for bar, val in zip(bars, r2s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    ax.set_ylim(0, 1)
    ax.set_ylabel('R²  (chronological test period)')
    ax.set_title('Cross-Zone Generalisation — Random Forest\n'
                 'Hatched bars = cross-zone test  |  Solid bars = within-zone test')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOUR_MILD, label='Mild (Sydney) test data'),
        Patch(facecolor=COLOUR_WARM, label='Warm (Newcastle) test data'),
        Patch(facecolor='white', hatch='///', edgecolor='black', label='Cross-zone'),
    ]
    ax.legend(handles=legend_elements, fontsize=8)
    plt.tight_layout()
    savefig('fig8_cross_zone')


# ─────────────────────────────────────────────────────────────────────────────
# TABLES
# ─────────────────────────────────────────────────────────────────────────────

def print_table1(ph1_results):
    print("\n" + "=" * 65)
    print("TABLE 1 — Phase 1: Weather-Only Model Performance")
    print("Target: normalized_nrg  |  Split: random 80/20")
    print("=" * 65)
    pivot = ph1_results.pivot_table(
        index='Model', columns='Zone', values=['R²', 'RMSE']
    )
    print(pivot.to_string())


def print_table2(ph2_results):
    print("\n" + "=" * 65)
    print("TABLE 2 — Phase 2: Full Prediction Model Performance")
    print("Target: Mean_Energy  |  Split: chronological 80/20")
    print("=" * 65)
    pivot = ph2_results.pivot_table(
        index='Model', columns='Zone', values=['R²', 'RMSE']
    )
    print(pivot.to_string())


def print_table3_ols_coefficients(mild, warm):
    print("\n" + "=" * 65)
    print("TABLE 3 — Phase 1 OLS Coefficients (normalized_nrg)")
    print("=" * 65)
    for zone_label, df in zip(ZONE_LABELS, [mild, warm]):
        Xtr, Xte, ytr, yte = random_split(df, PH1_FEATURES, PH1_TARGET)
        ols = fit_ols_sm(Xtr, ytr)
        coef = pd.DataFrame({
            'Coefficient': ols.params[1:].round(5),
            'p-value':     ols.pvalues[1:].round(4),
            'Significant': ['Yes' if p < 0.05 else 'No' for p in ols.pvalues[1:]],
        })
        print(f"\n  {zone_label}:")
        print(coef.to_string())


def print_table4_cross_zone(mild, warm):
    print("\n" + "=" * 65)
    print("TABLE 4 — Cross-Zone Generalisation Results")
    print("=" * 65)
    Xm_tr, Xm_te, ym_tr, ym_te = chrono_split(mild, PH2_FEATURES, PH2_TARGET)
    Xw_tr, Xw_te, yw_tr, yw_te = chrono_split(warm, PH2_FEATURES, PH2_TARGET)
    rf_mild = fit_rf(Xm_tr, ym_tr)
    rf_warm = fit_rf(Xw_tr, yw_tr)

    rows = [
        {'Train': 'Mild (Sydney)',     'Test': 'Mild (Sydney)',     'Type': 'Within',
         'R²': round(r2_score(ym_te, rf_mild.predict(Xm_te)), 4),
         'RMSE': round(rmse(ym_te, rf_mild.predict(Xm_te)), 5)},
        {'Train': 'Mild (Sydney)',     'Test': 'Warm (Newcastle)', 'Type': 'Cross',
         'R²': round(r2_score(yw_te, rf_mild.predict(Xw_te)), 4),
         'RMSE': round(rmse(yw_te, rf_mild.predict(Xw_te)), 5)},
        {'Train': 'Warm (Newcastle)', 'Test': 'Warm (Newcastle)', 'Type': 'Within',
         'R²': round(r2_score(yw_te, rf_warm.predict(Xw_te)), 4),
         'RMSE': round(rmse(yw_te, rf_warm.predict(Xw_te)), 5)},
        {'Train': 'Warm (Newcastle)', 'Test': 'Mild (Sydney)',     'Type': 'Cross',
         'R²': round(r2_score(ym_te, rf_warm.predict(Xm_te)), 4),
         'RMSE': round(rmse(ym_te, rf_warm.predict(Xm_te)), 5)},
    ]
    print(pd.DataFrame(rows).to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Loading data...")
    mild, warm = load_raw()

    print("\n-- Part 1: Weather-only models (normalized_nrg) --")
    ph1_results, ph1_models = ph1_run_models(mild, warm)

    print("\n-- Part 2: Full prediction models (Mean_Energy) --")
    ph2_results, ph2_models = ph2_run_models(mild, warm)

    print("\n-- Generating figures --")
    fig_ph1_model_comparison(ph1_results)
    fig_ph1_temp_scatter(mild, warm)
    fig_ph1_feature_importance(mild, warm)
    fig_phase_comparison(ph1_results, ph2_results)
    fig_ablation(mild, warm)
    fig_ph2_feature_importance(mild, warm)
    fig_ph2_actual_vs_predicted(mild, warm, ph2_models)
    fig_cross_zone(mild, warm)

    print("\n-- Printing tables --")
    print_table1(ph1_results)
    print_table2(ph2_results)
    print_table3_ols_coefficients(mild, warm)
    print_table4_cross_zone(mild, warm)

    print("\n\nDone. All figures saved to figures_results/")
