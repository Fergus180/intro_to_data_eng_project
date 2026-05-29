"""
model_ols.py
Ordinary Least Squares (OLS) Regression

Pipeline:
  1. Two-sample t-test comparing Zone 5 and Zone 6 (structural hypothesis 2.2.3)
  2. OLS with a quadratic temperature term for each zone
  3. Variance Inflation Factors (VIF) to check multicollinearity
  4. Test-set evaluation: R² and RMSE
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import r2_score

from data_utils import load_data, split_data, FEATURE_COLS, TARGET_COL


# ── T-test (Structural Hypothesis 2.2.3) ──────────────────────────────────────

def run_ttest(mild_df, warm_df):
    """
    Two-sample t-test on normalised residual consumption between zones.

    H0 : mean residual consumption is the same in both zones
    H1 : mean residual consumption differs between zones
    Alpha = 0.05
    """
    print("=" * 65)
    print("STRUCTURAL HYPOTHESIS TEST — Two-Sample t-Test")
    print("Zone 5 (Warm, Newcastle) vs Zone 6 (Mild, Sydney)")
    print("=" * 65)

    warm_vals = warm_df[TARGET_COL].dropna()
    mild_vals  = mild_df[TARGET_COL].dropna()

    t_stat, p_value = stats.ttest_ind(warm_vals, mild_vals)

    print(f"  Warm (Newcastle) — mean: {warm_vals.mean():.5f}   n = {len(warm_vals)}")
    print(f"  Mild (Sydney)    — mean: {mild_vals.mean():.5f}   n = {len(mild_vals)}")
    print(f"  t-statistic : {t_stat:.4f}")
    print(f"  p-value     : {p_value:.6f}")
    print()

    if p_value < 0.05:
        print("  Decision : REJECT H0  (p < 0.05)")
        print("  The two zones differ significantly in residual consumption.")
        print("  => Separate models are fitted for each zone.")
    else:
        print("  Decision : FAIL TO REJECT H0  (p >= 0.05)")
        print("  No significant difference detected between zones.")
        print("  => A pooled model could be considered.")
        print()
        print("  Note: normalized_nrg is constructed as a deviation from a zone-")
        print("  specific baseline, so both zone means are near zero by design.")
        print("  The t-test still validates whether the residual distributions")
        print("  overlap — use the variance and the zone distribution plot to")
        print("  assess whether separate modelling remains warranted.")

    print()
    return t_stat, p_value


# ── VIF Helper ────────────────────────────────────────────────────────────────

def compute_vif(X_train):
    """
    Compute Variance Inflation Factors for each predictor in X_train.

    VIF > 10 is conventionally considered problematic.
    Note: Temperature and Temperature_sq will naturally have high VIF
    because they are mathematically related — this is expected and
    does not invalidate the model.
    """
    X_with_const = sm.add_constant(X_train, has_constant='add')
    rows = []
    for i, col in enumerate(X_with_const.columns):
        vif_val = variance_inflation_factor(X_with_const.values, i)
        rows.append({'Feature': col, 'VIF': round(vif_val, 2)})
    return pd.DataFrame(rows)


# ── OLS Model ─────────────────────────────────────────────────────────────────

def fit_ols(zone_label, df):
    """
    Fit OLS for one zone, print the statsmodels summary, VIF table,
    and test-set R² / RMSE.

    Returns the fitted statsmodels model, test R², and test RMSE.
    """
    print("=" * 65)
    print(f"OLS REGRESSION : {zone_label}")
    print("=" * 65)

    X_train, X_test, y_train, y_test = split_data(df)

    # statsmodels requires an explicit intercept column
    X_train_sm = sm.add_constant(X_train, has_constant='add')
    X_test_sm  = sm.add_constant(X_test,  has_constant='add')

    model = sm.OLS(y_train, X_train_sm).fit()

    # Full regression summary (coefficients, p-values, R², F-stat, etc.)
    print(model.summary())
    print()

    # VIF table
    print("Variance Inflation Factors (on training set):")
    vif_df = compute_vif(X_train)
    print(vif_df.to_string(index=False))
    print("  Tip: VIF > 10 flags multicollinearity.")
    print("  Temperature and Temperature_sq will be high — this is expected.")
    print()

    # ── Test-set evaluation ──
    y_pred = model.predict(X_test_sm)

    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(np.mean((y_test.values - y_pred.values) ** 2))

    print("Test Set Performance (20 % hold-out):")
    print(f"  R²   : {r2:.4f}")
    print(f"  RMSE : {rmse:.6f}")
    print()

    return model, r2, rmse


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    mild_df, warm_df = load_data()

    # Step 1 — structural hypothesis test
    run_ttest(mild_df, warm_df)

    # Step 2 — OLS for each zone
    mild_model, mild_r2, mild_rmse = fit_ols('Zone 6 — Mild Temperate (Sydney)',     mild_df)
    warm_model, warm_r2, warm_rmse = fit_ols('Zone 5 — Warm Temperate (Newcastle)', warm_df)

    # Final side-by-side comparison
    print("=" * 65)
    print("SUMMARY COMPARISON")
    print("=" * 65)
    print(f"  Mild (Sydney)     R²: {mild_r2:.4f}   RMSE: {mild_rmse:.6f}")
    print(f"  Warm (Newcastle)  R²: {warm_r2:.4f}   RMSE: {warm_rmse:.6f}")
