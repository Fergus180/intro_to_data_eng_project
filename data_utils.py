"""
data_utils.py
Shared data loading and train/test split utilities.

All model files import from here so every model uses exactly the same
80/20 train/test split (fixed random seed = 42).
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Constants ─────────────────────────────────────────────────────────────────

RANDOM_SEED = 42
TEST_SIZE   = 0.20

# Weather predictors used across all models.
# Temperature_sq is added during loading to capture the U-shaped demand curve
# (high consumption at both hot and cold extremes).
FEATURE_COLS = [
    'Temperature',
    'Temperature_sq',
    'Humidity',
    'Cloud_Cover',
    'Precipitation',
    'Wind_Speed',
]

TARGET_COL = 'normalized_nrg'

# Resolve file paths relative to this script so the code runs from any directory
_HERE = os.path.dirname(os.path.abspath(__file__))
MILD_CSV = os.path.join(_HERE, 'Final_mild_data.csv')
WARM_CSV = os.path.join(_HERE, 'Final_warm_data.csv')


# ── Functions ─────────────────────────────────────────────────────────────────

def load_data():
    """
    Load the mild (Zone 6, Sydney) and warm (Zone 5, Newcastle) datasets.

    Adds:
      - 'Zone' label column
      - 'Temperature_sq' for capturing the non-linear heating/cooling effect

    Returns
    -------
    mild_df : pd.DataFrame
    warm_df : pd.DataFrame
    """
    mild_df = pd.read_csv(MILD_CSV, parse_dates=['Time'])
    warm_df = pd.read_csv(WARM_CSV, parse_dates=['Time'])

    mild_df['Zone'] = 'Mild - Zone 6 (Sydney)'
    warm_df['Zone'] = 'Warm - Zone 5 (Newcastle)'

    mild_df['Temperature_sq'] = mild_df['Temperature'] ** 2
    warm_df['Temperature_sq'] = warm_df['Temperature'] ** 2

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
