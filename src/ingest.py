import pandas as pd

def load_and_merge(tx_path: str, id_path: str) -> pd.DataFrame:
    # Read only training data from Kaggle
    tx = pd.read_csv(tx_path, low_memory=False)
    ident = pd.read_csv(id_path, low_memory=False)

    # Left join: identity is missing for many transactions
    df = tx.merge(ident, on="TransactionID", how="left")

    # Basic sanity
    required = ["TransactionID", "TransactionDT", "isFraud"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Time is numeric
    df["TransactionDT"] = pd.to_numeric(df["TransactionDT"], errors="coerce")
    df = df.dropna(subset=["TransactionDT", "isFraud"])
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    return df
