from dataclasses import dataclass

@dataclass(frozen=True)
class Paths:
    tx_path: str = "data/raw/train_transaction.csv"
    id_path: str = "data/raw/train_identity.csv"
    model_dir: str = "models"
    report_dir: str = "reports"

@dataclass(frozen=True)
class SplitConfig:
    # fractions on time-sorted data
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15

@dataclass(frozen=True)
class FeatureConfig:
    # entity proxies for behavioral features (chosen to be common + available)
    entity_cols: tuple = ("card1", "card2", "addr1", "P_emaildomain", "DeviceType")
    # rolling window in number of transactions per entity (simple + leakage-safe)
    rolling_k: int = 5
