"""
data_utils.py
Shared data loading and train/test split utilities.

All model files import from here so every model uses exactly the same
80/20 train/test split (fixed random seed = 42).

--- v3 changes ---
Previously: target was normalized_nrg (residual after subtracting a
stratified hourly/seasonal baseline). Problem: that baseline was computed
from data that itself contained weather effects, so subtracting it
partially cancelled the temperature signal before modelling began.

Now: target is raw Mean_Energy. Time-of-day patterns (hour, weekend) are
controlled for *inside* the model as features rather than pre-subtracted.
This lets the models disentangle routine daily patterns from weather-driven
deviations without accidentally removing weather signal in the process.

Temperature and Temperature_sq are also replaced by Apparent_Temp using
the Australian Bureau of Meteorology formula, which combines temperature,
humidity, and wind into a single perceived-heat value — the quantity that
actually drives heating/cooling decisions.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Constants ─────────────────────────────────────────────────────────────────

RANDOM_SEED = 42
TEST_SIZE   = 0.20

# Apparent temperature replaces the raw Temperature + Temperature_sq pair.
# Hour sin/cos encode time-of-day as a smooth cycle (so hour 23 and hour 0
# are treated as close together, unlike dummy variables).
# is_weekend captures the weekday vs weekend consumption difference.
FEATURE_COLS = [
    'Apparent_Temp',
    'Humidity',
    'Cloud_Cover',
    'Precipitation',
    'Wind_Speed',
    'hour_sin',
    'hour_cos',
    'is_weekend',
]

TARGET_COL = 'Mean_Energy'

# Resolve file paths relative to this script so the code runs from any directory
_HERE = os.path.dirname(os.path.abspath(__file__))
MILD_CSV = os.path.join(_HERE, 'Final_mild_data.csv')
WARM_CSV = os.path.join(_HERE, 'Final_warm_data.csv')


# ── Feature engineering helpers ───────────────────────────────────────────────

def add_apparent_temperature(df):
    """
    Add Apparent_Temp using the Australian Bureau of Meteorology formula:

        AT = T + 0.33 * e - 0.70 * ws - 4.00

    where:
        e  = vapour pressure (hPa) = (RH/100) * 6.105 * exp(17.27*T / (237.7+T))
        ws = wind speed in m/s  (Wind_Speed column is km/h, divide by 3.6)
        T  = air temperature in °C
    """
    wind_ms    = df['Wind_Speed'] / 3.6
    vapour_prs = (df['Humidity'] / 100) * 6.105 * np.exp(
        17.27 * df['Temperature'] / (237.7 + df['Temperature'])
    )
    df = df.copy()
    df['Apparent_Temp'] = df['Temperature'] + 0.33 * vapour_prs - 0.70 * wind_ms - 4.0
    return df


def add_time_features(df):
    """
    Add cyclical hour encoding and a weekend flag.

    Sin/cos encoding preserves the circular nature of the clock — the model
    sees hour 23 and hour 0 as adjacent, which integer or dummy encoding
    does not achieve.
    """
    df = df.copy()
    df['hour_sin']   = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos']   = np.cos(2 * np.pi * df['hour'] / 24)
    df['is_weekend'] = df['day'].isin(['Saturday', 'Sunday']).astype(int)
    return df


# ── Functions ─────────────────────────────────────────────────────────────────

def load_data():
    """
    Load the mild (Zone 6, Sydney) and warm (Zone 5, Newcastle) datasets
    and apply all feature engineering steps.

    Returns
    -------
    mild_df : pd.DataFrame
    warm_df : pd.DataFrame
    """
    mild_df = pd.read_csv(MILD_CSV, parse_dates=['Time'])
    warm_df = pd.read_csv(WARM_CSV, parse_dates=['Time'])

    mild_df['Zone'] = 'Mild - Zone 6 (Sydney)'
    warm_df['Zone'] = 'Warm - Zone 5 (Newcastle)'

    mild_df = add_apparent_temperature(mild_df)
    mild_df = add_time_features(mild_df)

    warm_df = add_apparent_temperature(warm_df)
    warm_df = add_time_features(warm_df)

    return mild_df, warm_df


def split_data(df):
    """
    Split a DataFrame into 80 % training and 20 % test sets.

    The fixed random seed means every model file receives the same split,
    making test-set comparisons between models fair.

    Returns
    -------
    X_train, X_test : pd.DataFrame
    y_train, y_test : pd.Series
    """
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
    )

    return X_train, X_test, y_train, y_test
