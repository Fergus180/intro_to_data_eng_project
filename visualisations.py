"""
visualisations.py
All report figures — saved to the 'figures/' directory.

Figures produced:
  Exploratory (Section 4.1)
    1_correlation_heatmap.png     — Pearson correlation matrix
    2_scatter_weather_vs_nrg.png  — Scatter of each weather var vs normalized_nrg
    3_pca_loadings.png            — PCA loading plot (weather variables only)

  Hypothesis test (Section 4.2)
    4_zone_distributions.png      — KDE + mean lines for the t-test comparison

  OLS results (Section 4.3)
    5_ols_actual_vs_predicted.png — Test-set actual vs predicted
    5b_ols_residuals.png          — Residuals vs fitted + QQ plot

  Lasso results (Section 4.3)
    6_lasso_coefficients.png      — Coefficient bar chart (grey = zeroed)

  Random Forest results (Section 4.4)
    7_rf_feature_importance.png   — Impurity-based importance bar chart

  Cross-model comparison
    8_model_comparison.png        — R² and RMSE for all three models, both zones

Running this file refits all models so it is self-contained; the fixed
random seed in data_utils.py guarantees results match the model files.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import statsmodels.api as sm

from data_utils import load_data, split_data, FEATURE_COLS, TARGET_COL

# ── Setup ─────────────────────────────────────────────────────────────────────

os.makedirs('figures', exist_ok=True)

COLOUR_MILD = '#2196F3'   # blue  — Zone 6, Sydney
COLOUR_WARM = '#F44336'   # red   — Zone 5, Newcastle

ZONE_LABELS = [
    'Mild Zone 6 (Sydney)',
    'Warm Zone 5 (Newcastle)',
]

plt.rcParams.update({'figure.dpi': 150, 'font.size': 10})


# ── Tiny helpers ──────────────────────────────────────────────────────────────

def savefig(name):
    path = os.path.join('figures', f'{name}.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


# ── 1. Exploratory Analysis ───────────────────────────────────────────────────

def plot_correlation_heatmap(mild_df, warm_df):
    """
    Pearson correlation matrix for all weather predictors and normalized_nrg.
    Shows both zones side by side.
    """
    cols = ['Temperature', 'Humidity', 'Cloud_Cover', 'Precipitation', 'Wind_Speed', TARGET_COL]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, df, label in zip(axes, [mild_df, warm_df], ZONE_LABELS):
        corr = df[cols].corr()
        sns.heatmap(
            corr, ax=ax,
            annot=True, fmt='.2f',
            cmap='coolwarm', center=0, vmin=-1, vmax=1,
            linewidths=0.5, square=True,
        )
        ax.set_title(label, fontsize=11)

    fig.suptitle(
        'Pearson Correlation Matrix — Weather Variables and Residual Energy Consumption',
        fontsize=12,
    )
    plt.tight_layout()
    savefig('1_correlation_heatmap')


def plot_scatter_weather(mild_df, warm_df):
    """
    5 × 2 grid of scatter plots: each weather variable vs normalized_nrg,
    one row per zone.
    """
    weather_vars = ['Temperature', 'Humidity', 'Cloud_Cover', 'Precipitation', 'Wind_Speed']

    fig, axes = plt.subplots(2, 5, figsize=(18, 7))

    for col_i, var in enumerate(weather_vars):
        for row_i, (df, label, colour) in enumerate(zip(
            [mild_df, warm_df],
            ZONE_LABELS,
            [COLOUR_MILD, COLOUR_WARM],
        )):
            ax = axes[row_i, col_i]
            ax.scatter(df[var], df[TARGET_COL], alpha=0.15, s=5, color=colour)

            if col_i == 0:
                ax.set_ylabel(f'{label}\nnormalized_nrg', fontsize=8)
            if row_i == 0:
                ax.set_title(var)

            ax.set_xlabel(var, fontsize=8)

    fig.suptitle(
        'Weather Variables vs Residual Energy Consumption (normalized_nrg)',
        fontsize=12,
    )
    plt.tight_layout()
    savefig('2_scatter_weather_vs_nrg')


def plot_pca_loadings(mild_df, warm_df):
    """
    PCA loading plot showing how the five weather variables contribute to
    the first two principal components. Arrows pointing in a similar
    direction indicate correlated variables.
    """
    weather_vars = ['Temperature', 'Humidity', 'Cloud_Cover', 'Precipitation', 'Wind_Speed']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, df, label in zip(axes, [mild_df, warm_df], ZONE_LABELS):
        X = df[weather_vars].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=2)
        pca.fit(X_scaled)

        # loadings[i] = contribution of variable i to each PC
        loadings = pca.components_.T

        for i, var in enumerate(weather_vars):
            lx, ly = loadings[i, 0], loadings[i, 1]
            ax.annotate(
                '', xy=(lx, ly), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
            )
            ax.text(lx * 1.12, ly * 1.12, var, fontsize=9, ha='center')

        ax.axhline(0, color='grey', linestyle='--', lw=0.7)
        ax.axvline(0, color='grey', linestyle='--', lw=0.7)
        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-1.4, 1.4)

        var_exp = pca.explained_variance_ratio_ * 100
        ax.set_xlabel(f'PC 1  ({var_exp[0]:.1f} % variance explained)')
        ax.set_ylabel(f'PC 2  ({var_exp[1]:.1f} % variance explained)')
        ax.set_title(label)

    fig.suptitle(
        'PCA Loading Plot — Weather Variable Contributions to Principal Components',
        fontsize=12,
    )
    plt.tight_layout()
    savefig('3_pca_loadings')


# ── 2. Hypothesis Test ────────────────────────────────────────────────────────

def plot_zone_distributions(mild_df, warm_df):
    """
    Overlapping KDE curves + mean lines for both zones.
    Provides visual context for the two-sample t-test result.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    t_stat, p_value = stats.ttest_ind(
        warm_df[TARGET_COL].dropna(),
        mild_df[TARGET_COL].dropna(),
    )

    for df, label, colour in zip(
        [mild_df, warm_df],
        ZONE_LABELS,
        [COLOUR_MILD, COLOUR_WARM],
    ):
        vals = df[TARGET_COL].dropna()

        ax.hist(vals, bins=80, density=True, alpha=0.25, color=colour)

        kde_x = np.linspace(vals.min(), vals.max(), 400)
        kde   = stats.gaussian_kde(vals)
        ax.plot(kde_x, kde(kde_x), color=colour, lw=2, label=label)

        ax.axvline(vals.mean(), color=colour, linestyle='--', lw=1.5,
                   label=f'{label} mean = {vals.mean():.4f}')

    sig = 'p < 0.05  →  REJECT H0' if p_value < 0.05 else 'p ≥ 0.05  →  FAIL TO REJECT H0'
    ax.set_title(
        f'Zone Residual Consumption Distributions\n'
        f't = {t_stat:.3f},  p = {p_value:.5f}  |  {sig}',
        fontsize=11,
    )
    ax.set_xlabel('Residual Energy Consumption (normalized_nrg)')
    ax.set_ylabel('Density')
    ax.legend(fontsize=9)

    plt.tight_layout()
    savefig('4_zone_distributions')


# ── 3. OLS Results ────────────────────────────────────────────────────────────

def plot_ols_actual_vs_predicted(mild_df, warm_df):
    """Actual vs predicted scatter on the test set for each zone."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, df, label, colour in zip(
        axes,
        [mild_df, warm_df],
        ZONE_LABELS,
        [COLOUR_MILD, COLOUR_WARM],
    ):
        X_train, X_test, y_train, y_test = split_data(df)

        X_train_sm = sm.add_constant(X_train, has_constant='add')
        X_test_sm  = sm.add_constant(X_test,  has_constant='add')

        model  = sm.OLS(y_train, X_train_sm).fit()
        y_pred = model.predict(X_test_sm)

        r2_val = r2_score(y_test, y_pred)

        ax.scatter(y_test, y_pred, alpha=0.25, s=7, color=colour)

        # Perfect-prediction line
        lo = min(float(y_test.min()), float(y_pred.min()))
        hi = max(float(y_test.max()), float(y_pred.max()))
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1.2, label='Perfect fit')

        ax.set_title(f'{label}\nR² = {r2_val:.3f}')
        ax.set_xlabel('Actual normalized_nrg')
        ax.set_ylabel('Predicted normalized_nrg')
        ax.legend(fontsize=8)

    fig.suptitle('OLS Regression — Actual vs Predicted (Test Set)', fontsize=12)
    plt.tight_layout()
    savefig('5_ols_actual_vs_predicted')


def plot_ols_residuals(mild_df, warm_df):
    """
    Residuals-vs-fitted plot and QQ plot for each zone.
    Used to check homoscedasticity and normality assumptions.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for col_i, (df, label, colour) in enumerate(zip(
        [mild_df, warm_df],
        ZONE_LABELS,
        [COLOUR_MILD, COLOUR_WARM],
    )):
        X_train, X_test, y_train, y_test = split_data(df)

        X_train_sm = sm.add_constant(X_train, has_constant='add')
        X_test_sm  = sm.add_constant(X_test,  has_constant='add')

        model     = sm.OLS(y_train, X_train_sm).fit()
        y_pred    = model.predict(X_test_sm)
        residuals = y_test.values - y_pred.values

        # Residuals vs fitted
        ax_res = axes[0, col_i]
        ax_res.scatter(y_pred, residuals, alpha=0.2, s=6, color=colour)
        ax_res.axhline(0, color='black', lw=1)
        ax_res.set_xlabel('Fitted values')
        ax_res.set_ylabel('Residuals')
        ax_res.set_title(f'{label}\nResiduals vs Fitted')

        # QQ plot
        ax_qq = axes[1, col_i]
        sm.qqplot(residuals, line='s', ax=ax_qq, alpha=0.3, markersize=3)
        ax_qq.set_title(f'{label}\nQQ Plot of Residuals')

    fig.suptitle('OLS Regression Diagnostics (Test Set)', fontsize=12)
    plt.tight_layout()
    savefig('5b_ols_residuals')


# ── 4. Lasso Results ──────────────────────────────────────────────────────────

def plot_lasso_coefficients(mild_df, warm_df):
    """
    Horizontal bar chart of Lasso coefficients on standardised features.
    Bars in grey indicate a coefficient shrunk to zero (feature removed).
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, df, label, colour in zip(
        axes,
        [mild_df, warm_df],
        ZONE_LABELS,
        [COLOUR_MILD, COLOUR_WARM],
    ):
        X_train, X_test, y_train, y_test = split_data(df)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        lasso_cv = LassoCV(cv=5, max_iter=10_000, random_state=42)
        lasso_cv.fit(X_train_scaled, y_train)

        coef_series = pd.Series(lasso_cv.coef_, index=FEATURE_COLS)
        coef_series = coef_series.sort_values()

        bar_colours = [colour if c != 0.0 else '#BDBDBD' for c in coef_series.values]
        coef_series.plot(kind='barh', ax=ax, color=bar_colours, edgecolor='black', linewidth=0.4)

        ax.axvline(0, color='black', lw=0.8)
        ax.set_title(f'{label}\nλ = {lasso_cv.alpha_:.5f}')
        ax.set_xlabel('Coefficient (standardised features)')

    fig.suptitle(
        'Lasso Regression Coefficients  —  Grey = zeroed (feature removed by Lasso)',
        fontsize=12,
    )
    plt.tight_layout()
    savefig('6_lasso_coefficients')


# ── 5. Random Forest Results ──────────────────────────────────────────────────

def plot_rf_feature_importance(mild_df, warm_df):
    """
    Horizontal bar chart of impurity-based feature importances for each zone.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, df, label, colour in zip(
        axes,
        [mild_df, warm_df],
        ZONE_LABELS,
        [COLOUR_MILD, COLOUR_WARM],
    ):
        X_train, X_test, y_train, y_test = split_data(df)

        rf = RandomForestRegressor(
            n_estimators=200, min_samples_leaf=5, random_state=42, n_jobs=-1
        )
        rf.fit(X_train, y_train)

        imp_df = pd.DataFrame({
            'Feature':    FEATURE_COLS,
            'Importance': rf.feature_importances_,
        }).sort_values('Importance')

        ax.barh(
            imp_df['Feature'], imp_df['Importance'],
            color=colour, edgecolor='black', linewidth=0.4,
        )
        ax.set_title(label)
        ax.set_xlabel('Mean Decrease in Impurity')

    fig.suptitle('Random Forest Feature Importance (Impurity-Based)', fontsize=12)
    plt.tight_layout()
    savefig('7_rf_feature_importance')


# ── 6. Cross-Model Comparison ────────────────────────────────────────────────

def plot_model_comparison(mild_df, warm_df):
    """
    Side-by-side grouped bar charts comparing R² and RMSE for OLS, Lasso,
    and Random Forest on the test set, for both zones.

    All three models are refit here using the same split from data_utils,
    so the comparison is apples-to-apples.
    """
    model_names = ['OLS', 'Lasso', 'Random Forest']
    results = {
        'Mild': {'R2': [], 'RMSE': []},
        'Warm': {'R2': [], 'RMSE': []},
    }

    for zone_key, df in [('Mild', mild_df), ('Warm', warm_df)]:
        X_train, X_test, y_train, y_test = split_data(df)

        # OLS
        X_train_sm = sm.add_constant(X_train, has_constant='add')
        X_test_sm  = sm.add_constant(X_test,  has_constant='add')
        ols_model  = sm.OLS(y_train, X_train_sm).fit()
        y_pred_ols = ols_model.predict(X_test_sm)
        results[zone_key]['R2'].append(r2_score(y_test, y_pred_ols))
        results[zone_key]['RMSE'].append(rmse(y_test, y_pred_ols))

        # Lasso
        scaler         = StandardScaler()
        X_tr_scaled    = scaler.fit_transform(X_train)
        X_te_scaled    = scaler.transform(X_test)
        lasso_cv       = LassoCV(cv=5, max_iter=10_000, random_state=42)
        lasso_cv.fit(X_tr_scaled, y_train)
        y_pred_lasso   = lasso_cv.predict(X_te_scaled)
        results[zone_key]['R2'].append(r2_score(y_test, y_pred_lasso))
        results[zone_key]['RMSE'].append(rmse(y_test, y_pred_lasso))

        # Random Forest
        rf           = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred_rf    = rf.predict(X_test)
        results[zone_key]['R2'].append(r2_score(y_test, y_pred_rf))
        results[zone_key]['RMSE'].append(rmse(y_test, y_pred_rf))

    x     = np.arange(len(model_names))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # R² chart
    ax1.bar(x - width / 2, results['Mild']['R2'], width,
            label='Mild (Sydney)',     color=COLOUR_MILD)
    ax1.bar(x + width / 2, results['Warm']['R2'], width,
            label='Warm (Newcastle)',  color=COLOUR_WARM)
    ax1.set_title('R²  — Test Set')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names)
    ax1.set_ylabel('R²')
    ax1.set_ylim(0, 1)
    ax1.legend()

    # RMSE chart
    ax2.bar(x - width / 2, results['Mild']['RMSE'], width,
            label='Mild (Sydney)',     color=COLOUR_MILD)
    ax2.bar(x + width / 2, results['Warm']['RMSE'], width,
            label='Warm (Newcastle)',  color=COLOUR_WARM)
    ax2.set_title('RMSE  — Test Set')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names)
    ax2.set_ylabel('RMSE')
    ax2.legend()

    fig.suptitle('Cross-Model Comparison: OLS vs Lasso vs Random Forest', fontsize=12)
    plt.tight_layout()
    savefig('8_model_comparison')


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Loading data...")
    mild_df, warm_df = load_data()

    print("\n-- Exploratory analysis --")
    plot_correlation_heatmap(mild_df, warm_df)
    plot_scatter_weather(mild_df, warm_df)
    plot_pca_loadings(mild_df, warm_df)

    print("\n-- Hypothesis test --")
    plot_zone_distributions(mild_df, warm_df)

    print("\n-- OLS results --")
    plot_ols_actual_vs_predicted(mild_df, warm_df)
    plot_ols_residuals(mild_df, warm_df)

    print("\n-- Lasso results --")
    plot_lasso_coefficients(mild_df, warm_df)

    print("\n-- Random Forest results --")
    plot_rf_feature_importance(mild_df, warm_df)

    print("\n-- Cross-model comparison --")
    plot_model_comparison(mild_df, warm_df)

    print("\nDone. All figures saved to figures/")
