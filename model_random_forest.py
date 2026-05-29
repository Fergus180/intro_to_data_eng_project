"""
model_random_forest.py
Random Forest Regression

Pipeline:
  1. Fit RandomForestRegressor for each zone (200 trees).
  2. Print impurity-based feature importances (fast, built-in).
  3. Print permutation importances on the test set (slower but more
     reliable — not influenced by high-cardinality features).
  4. Test-set evaluation: R² and RMSE using the same 20 % hold-out
     as the OLS and Lasso models.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score

from data_utils import load_data, split_data, FEATURE_COLS, TARGET_COL


# ── Random Forest Model ───────────────────────────────────────────────────────

def fit_random_forest(zone_label, df):
    """
    Fit a Random Forest for one zone, print feature importances,
    and evaluate on the held-out test set.

    Two importance measures are reported:
      - Impurity-based (MDI): fast, computed from training data.
      - Permutation: shuffles each feature on the test set; the drop
        in R² is the importance score. More reliable for correlated features.

    Returns the fitted model, test R², and test RMSE.
    """
    print("=" * 65)
    print(f"RANDOM FOREST : {zone_label}")
    print("=" * 65)

    X_train, X_test, y_train, y_test = split_data(df)

    rf = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=5,   # avoids overfitting on tiny leaf nodes
        random_state=42,
        n_jobs=-1,            # use all available CPU cores
    )
    rf.fit(X_train, y_train)

    # ── Impurity-based importances ──
    importance_df = pd.DataFrame({
        'Feature':            FEATURE_COLS,
        'Importance (MDI)':   rf.feature_importances_,
    }).sort_values('Importance (MDI)', ascending=False)

    print("Feature Importances — Mean Decrease in Impurity (training set):")
    print(importance_df.to_string(index=False))
    print()

    # ── Permutation importances (test set) ──
    perm = permutation_importance(
        rf, X_test, y_test,
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )
    perm_df = pd.DataFrame({
        'Feature':    FEATURE_COLS,
        'Perm Imp (mean)': perm.importances_mean,
        'Perm Imp (std)':  perm.importances_std,
    }).sort_values('Perm Imp (mean)', ascending=False)

    print("Permutation Feature Importances (test set, 10 repeats):")
    print(perm_df.to_string(index=False))
    print()

    # ── Test-set evaluation ──
    y_pred = rf.predict(X_test)

    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(np.mean((y_test.values - y_pred) ** 2))

    print("Test Set Performance (20 % hold-out):")
    print(f"  R²   : {r2:.4f}")
    print(f"  RMSE : {rmse:.6f}")
    print()

    return rf, r2, rmse


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    mild_df, warm_df = load_data()

    mild_rf,  mild_r2,  mild_rmse  = fit_random_forest(
        'Zone 6 — Mild Temperate (Sydney)',     mild_df
    )
    warm_rf, warm_r2, warm_rmse = fit_random_forest(
        'Zone 5 — Warm Temperate (Newcastle)', warm_df
    )

    print("=" * 65)
    print("SUMMARY COMPARISON")
    print("=" * 65)
    print(f"  Mild (Sydney)     R²: {mild_r2:.4f}   RMSE: {mild_rmse:.6f}")
    print(f"  Warm (Newcastle)  R²: {warm_r2:.4f}   RMSE: {warm_rmse:.6f}")
