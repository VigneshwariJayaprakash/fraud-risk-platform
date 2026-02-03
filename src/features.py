import numpy as np
import pandas as pd
from typing import Iterable

def _safe_log1p(x: pd.Series) -> pd.Series:
    return np.log1p(pd.to_numeric(x, errors="coerce").fillna(0.0).clip(lower=0.0))

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Amount features
    if "TransactionAmt" in out.columns:
        out["log_amt"] = _safe_log1p(out["TransactionAmt"])
    else:
        out["log_amt"] = 0.0

    # Missingness count 
    out["missing_count"] = out.isna().sum(axis=1).astype(np.int16)

    return out

def add_entity_temporal_features(
    df: pd.DataFrame,
    entity_cols: Iterable[str],
    rolling_k: int = 5
) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values("TransactionDT").reset_index(drop=True)

    # Fill entity cols (missing becomes a category)
    for c in entity_cols:
        if c not in out.columns:
            out[c] = "MISSING_COL"
        out[c] = out[c].astype("object").fillna("MISSING")

    # Helper: create a composite entity key
    # We keep it simple: single + pair keys to capture behavior.
    out["entity_key_1"] = out[entity_cols[0]].astype(str)
    if len(entity_cols) >= 2:
        out["entity_key_2"] = out[entity_cols[0]].astype(str) + "|" + out[entity_cols[1]].astype(str)
    else:
        out["entity_key_2"] = out["entity_key_1"]

    keys = ["entity_key_1", "entity_key_2"]

    # Temporal: time since last transaction per key (past-only)
    for k in keys:
        out[f"dt_since_prev_{k}"] = (
            out.groupby(k)["TransactionDT"]
               .diff()
               .fillna(-1.0)
               .astype(np.float32)
        )

    # Behavioral: rolling stats of amount per key (past-only)
    # Use shifted rolling so the current row doesn't see itself.
    if "TransactionAmt" in out.columns:
        amt = pd.to_numeric(out["TransactionAmt"], errors="coerce").fillna(0.0).astype(np.float32)
    else:
        amt = pd.Series(np.zeros(len(out), dtype=np.float32))

    out["amt"] = amt

    for k in keys:
        g = out.groupby(k)["amt"]
        out[f"amt_mean_prev{rolling_k}_{k}"] = (
            g.apply(lambda s: s.shift(1).rolling(rolling_k, min_periods=1).mean())
             .reset_index(level=0, drop=True)
             .astype(np.float32)
        )
        out[f"amt_std_prev{rolling_k}_{k}"] = (
            g.apply(lambda s: s.shift(1).rolling(rolling_k, min_periods=1).std(ddof=0))
             .reset_index(level=0, drop=True)
             .fillna(0.0)
             .astype(np.float32)
        )
        out[f"amt_z_prev{rolling_k}_{k}"] = (
            (out["amt"] - out[f"amt_mean_prev{rolling_k}_{k}"]) /
            (out[f"amt_std_prev{rolling_k}_{k}"] + 1e-6)
        ).astype(np.float32)

        # Simple velocity feature: count of prior txns in last K (proxy)
        out[f"txn_count_prev{rolling_k}_{k}"] = (
            g.apply(lambda s: s.shift(1).rolling(rolling_k, min_periods=1).count())
             .reset_index(level=0, drop=True)
             .astype(np.float32)
        )

    # Cleanup helper columns
    out.drop(columns=["entity_key_1", "entity_key_2"], inplace=True, errors="ignore")

    return out

def make_model_matrix(df: pd.DataFrame, label_col: str = "isFraud"):
    y = df[label_col].astype(int).values
    X = df.drop(columns=[label_col], errors="ignore").copy()

    # Drop high-leak risk IDs / non-feature columns
    drop_cols = [c for c in ["TransactionID"] if c in X.columns]
    X.drop(columns=drop_cols, inplace=True, errors="ignore")

    # Convert objects to category codes (simple baseline)
    for c in X.select_dtypes(include=["object"]).columns:
        X[c] = X[c].astype("category").cat.codes.astype(np.int32)

    # Fill remaining missing
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(-999.0)

    return X, y
