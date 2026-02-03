import pandas as pd
from typing import Tuple

def time_split(df: pd.DataFrame, train_frac: float, val_frac: float, test_frac: float
              ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if abs((train_frac + val_frac + test_frac) - 1.0) > 1e-9:
        raise ValueError("Split fractions must sum to 1.0")

    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train:n_train + n_val].copy()
    test = df.iloc[n_train + n_val:].copy()

    return train, val, test
