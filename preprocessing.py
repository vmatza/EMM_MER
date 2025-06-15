import pandas as pd

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic preprocessing:
    - Drop rows with missing key columns
    - Convert time points to numeric values
    """
    df = df.dropna(subset=['respondentid', 'mm'])
    time_map = {
        'Intake ronde': 0,
        'T3': 3,
        'T6': 6,
        'T9': 9,
        'T12': 12
    }
    df = df[df['mm'].isin(time_map.keys())].copy()
    df['time_numeric'] = df['mm'].map(time_map).astype(float)
    return df