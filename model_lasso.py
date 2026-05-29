"""
model_lasso.py
Lasso Regression with Cross-Validated Lambda (alpha)

Pipeline:
  1. Standardise features — Lasso penalises coefficients, so all
     predictors must be on the same scale before fitting.
  2. LassoCV selects the best regularisation strength (alpha/lambda)
     via 5-fold cross-validation.
  3. Coefficients shrunk to exactly zero are removed predictors;
     non-zero coefficients are retained predictors.
  4. Test-set evaluation: R² and RMSE using the same 20 % hold-out
     as the OLS and Random Forest models.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from data_utils import load_data, split_data, FEATURE_COLS, TARGET_COL


# ── Lasso Model ───────────────────────────────────────────────────────────────

def fit_lasso(zone_label, df):
    """
    Fit Lasso regression for one zone using 5-fold cross-validation
    to select the regularisation parameter.

    Prints the chosen alpha, the coefficient table (kept vs zeroed),
    and test-set R² / RMSE.

    Returns the fitted LassoCV model, scaler, test R², and test RMSE.
    """
    print("=" * 65)
    print(f"LASSO REGRESSION : {zone_label}")
    print("=" * 65)

    X_train, X_test, y_train, y_test = split_data(df)

    # Scale features so the L1 penalty treats all predictors equally
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # LassoCV searches over a log-spaced grid of alpha values and picks
    # the one with the lowest cross-validated mean squared error
    lasso_cv = LassoCV(cv=5, max_iter=10_000, random_state=42)
    lasso_cv.fit(X_train_scaled, y_train)

    print(f"  Best lambda (alpha) selected by CV : {lasso_cv.alpha_:.6f}")
    print()

    # Build a coefficient table showing which features were kept or removed
    coef_df = pd.DataFrame({
        'Feature':     FEATURE_COLS,
        'Coefficient': lasso_cv.coef_,
    })
    coef_df['Status'] = coef_df['Coefficient'].apply(
        lambda c: 'kept' if c != 0.0 else 'REMOVED (zeroed by Lasso)'
    )

    print("Coefficients on standardised features:")
    print(coef_df.to_string(index=False))
    print()
    print(f"  Intercept : {lasso_cv.intercept_:.6f}")
    print()

    # ── Test-set evaluation ──
    y_pred = lasso_cv.predict(X_test_scaled)

    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(np.mean((y_test.values - y_pred) ** 2))

    print("Test Set Performance (20 % hold-out):")
    print(f"  R²   : {r2:.4f}")
    print(f"  RMSE : {rmse:.6f}")
    print()

    return lasso_cv, scaler, r2, rmse


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    mild_df, warm_df = load_data()

    mild_model,  mild_scaler,  mild_r2,  mild_rmse  = fit_lasso(
        'Zone 6 — Mild Temperate (Sydney)',     mild_df
    )
    warm_model, warm_scaler, warm_r2, warm_rmse = fit_lasso(
        'Zone 5 — Warm Temperate (Newcastle)', warm_df
    )

    print("=" * 65)
    print("SUMMARY COMPARISON")
    print("=" * 65)
    print(f"  Mild (Sydney)     R²: {mild_r2:.4f}   RMSE: {mild_rmse:.6f}")
    print(f"  Warm (Newcastle)  R²: {warm_r2:.4f}   RMSE: {warm_rmse:.6f}")
