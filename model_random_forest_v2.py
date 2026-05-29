"""
model_random_forest_v2.py
Random Forest Regression — Apparent Temperature variant

Only change from v1: Temperature and Temperature_sq are replaced by a single
Apparent Temperature feature using the Australian Bureau of Meteorology formula:

    AT = T + 0.33 * e - 0.70 * ws - 4.00

where:
    e  = vapour pressure (hPa) = (RH/100) * 6.105 * exp(17.27*T / (237.7+T))
    ws = wind speed in m/s  (Wind_Speed column is km/h, so divide by 3.6)
    T  = air temperature in °C

Rationale: apparent temperature is what actually drives heating/cooling decisions
— people respond to how hot or cold it FEELS, not the raw thermometer reading.
Combining temperature, humidity, and wind into one perceived-heat value may give
the model a cleaner signal than supplying them as separate linear terms.

All other settings (200 trees, min_samples_leaf=5, random seed, 80/20 split)
are identical to v1 so results are directly comparable.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score

from data_utils import load_data, split_data, TARGET_COL

os.makedirs('figuresv2', exist_ok=True)

# Features for this variant — Temperature and Temperature_sq removed,
# Apparent_Temp added in their place. Wind_Speed is kept as a separate
# predictor because the BOM formula uses wind to compute AT and the model
# may still find additional signal in raw wind speed.
FEATURE_COLS_V2 = [
    'Apparent_Temp',
    'Humidity',
    'Cloud_Cover',
    'Precipitation',
    'Wind_Speed',
]


# ── Apparent Temperature ──────────────────────────────────────────────────────

def add_apparent_temperature(df):
    """
    Add an Apparent_Temp column using the Australian BOM formula.
    Wind_Speed in the dataset is km/h; the formula requires m/s.
    """
    wind_ms    = df['Wind_Speed'] / 3.6
    vapour_prs = (df['Humidity'] / 100) * 6.105 * np.exp(
        17.27 * df['Temperature'] / (237.7 + df['Temperature'])
    )
    df = df.copy()
    df['Apparent_Temp'] = df['Temperature'] + 0.33 * vapour_prs - 0.70 * wind_ms - 4.0
    return df


# ── Random Forest Model ───────────────────────────────────────────────────────

def fit_random_forest_v2(zone_label, df):
    """
    Fit Random Forest using Apparent_Temp in place of Temperature + Temperature_sq.
    Prints both impurity-based and permutation importances, then test R² / RMSE.
    Returns the fitted model, test R², and test RMSE.
    """
    print("=" * 65)
    print(f"RANDOM FOREST v2 (Apparent Temp) : {zone_label}")
    print("=" * 65)

    df = add_apparent_temperature(df)

    # Use the same split logic from data_utils but override the feature columns
    from data_utils import FEATURE_COLS
    import sklearn.model_selection as ms

    X = df[FEATURE_COLS_V2].copy()
    y = df[TARGET_COL].copy()
    X_train, X_test, y_train, y_test = ms.train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    rf = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    # Apparent_Temp range for context
    print(f"  Apparent_Temp range : {df['Apparent_Temp'].min():.1f} to {df['Apparent_Temp'].max():.1f} °C")
    print()

    # ── Impurity-based importances ──
    imp_df = pd.DataFrame({
        'Feature':           FEATURE_COLS_V2,
        'Importance (MDI)':  rf.feature_importances_,
    }).sort_values('Importance (MDI)', ascending=False)

    print("Feature Importances — Mean Decrease in Impurity (training set):")
    print(imp_df.to_string(index=False))
    print()

    # ── Permutation importances (test set) ──
    perm = permutation_importance(
        rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )
    perm_df = pd.DataFrame({
        'Feature':               FEATURE_COLS_V2,
        'Perm Imp (mean)':       perm.importances_mean,
        'Perm Imp (std)':        perm.importances_std,
    }).sort_values('Perm Imp (mean)', ascending=False)

    print("Permutation Feature Importances (test set, 10 repeats):")
    print(perm_df.to_string(index=False))
    print()

    # ── Test-set evaluation ──
    y_pred = rf.predict(X_test)
    r2_val = r2_score(y_test, y_pred)
    rmse   = np.sqrt(np.mean((y_test.values - y_pred) ** 2))

    print("Test Set Performance (20 % hold-out):")
    print(f"  R²   : {r2_val:.4f}")
    print(f"  RMSE : {rmse:.6f}")
    print()

    return rf, r2_val, rmse


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    mild_df, warm_df = load_data()

    mild_rf,  mild_r2,  mild_rmse  = fit_random_forest_v2(
        'Zone 6 — Mild Temperate (Sydney)',     mild_df
    )
    warm_rf, warm_r2, warm_rmse = fit_random_forest_v2(
        'Zone 5 — Warm Temperate (Newcastle)', warm_df
    )

    print("=" * 65)
    print("COMPARISON vs v1 baseline")
    print("=" * 65)
    print("  v1 used Temperature + Temperature_sq")
    print("  v2 uses Apparent_Temp (BOM formula)")
    print()
    print(f"  Mild (Sydney)     v2 R²: {mild_r2:.4f}   RMSE: {mild_rmse:.6f}")
    print(f"  Warm (Newcastle)  v2 R²: {warm_r2:.4f}   RMSE: {warm_rmse:.6f}")
    print()
    print("  v1 baseline:  Mild R²=0.3567  Warm R²=0.4391")
