import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    matthews_corrcoef,
    average_precision_score,
    precision_recall_curve,
    classification_report
)
from xgboost import XGBClassifier

from config import Paths, SplitConfig, FeatureConfig
from ingest import load_and_merge
from split import time_split
from features import add_basic_features, add_entity_temporal_features, make_model_matrix

def best_threshold_by_mcc(y_true: np.ndarray, prob: np.ndarray):
    # Evaluate thresholds from PR curve (unique candidate thresholds)
    prec, rec, thr = precision_recall_curve(y_true, prob)
    # thr has len = len(prec)-1
    best_t, best_mcc = 0.5, -1.0
    for t in thr:
        pred = (prob >= t).astype(int)
        mcc = matthews_corrcoef(y_true, pred)
        if mcc > best_mcc:
            best_mcc, best_t = mcc, float(t)
    return best_t, best_mcc

def main():
    paths = Paths()
    split_cfg = SplitConfig()
    feat_cfg = FeatureConfig()

    os.makedirs(paths.model_dir, exist_ok=True)
    os.makedirs(paths.report_dir, exist_ok=True)

    df = load_and_merge(paths.tx_path, paths.id_path)

    # Optional: lock to 280k+ transactions for resume alignment (document this!)
    # Use earliest 300k by time so it’s realistic.
    if len(df) > 300_000:
        df = df.iloc[:300_000].copy()

    df = add_basic_features(df)
    df = add_entity_temporal_features(df, entity_cols=feat_cfg.entity_cols, rolling_k=feat_cfg.rolling_k)

    train_df, val_df, test_df = time_split(df, split_cfg.train_frac, split_cfg.val_frac, split_cfg.test_frac)

    X_train, y_train = make_model_matrix(train_df)
    X_val, y_val = make_model_matrix(val_df)
    X_test, y_test = make_model_matrix(test_df)

    # Handle imbalance
    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_pos_weight = (neg / max(pos, 1))

    model = XGBClassifier(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_weight=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        random_state=42,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    val_prob = model.predict_proba(X_val)[:, 1]
    best_t, best_mcc = best_threshold_by_mcc(y_val, val_prob)
    val_ap = average_precision_score(y_val, val_prob)

    test_prob = model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= best_t).astype(int)
    test_mcc = matthews_corrcoef(y_test, test_pred)
    test_ap = average_precision_score(y_test, test_prob)

    report = classification_report(y_test, test_pred, digits=4, output_dict=True)

    metrics = {
        "rows_total": int(len(df)),
        "rows_train": int(len(train_df)),
        "rows_val": int(len(val_df)),
        "rows_test": int(len(test_df)),
        "fraud_rate_train": float(y_train.mean()),
        "best_threshold_by_mcc_val": float(best_t),
        "mcc_val": float(best_mcc),
        "ap_val": float(val_ap),
        "mcc_test": float(test_mcc),
        "ap_test": float(test_ap),
        "classification_report_test": report,
        "scale_pos_weight": float(scale_pos_weight),
        "entity_cols": list(feat_cfg.entity_cols),
        "rolling_k": int(feat_cfg.rolling_k),
    }

    # Save model + feature columns used
    joblib.dump({"model": model, "columns": list(X_train.columns), "threshold": best_t},
                os.path.join(paths.model_dir, "xgb_risk_model.joblib"))

    with open(os.path.join(paths.report_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved:")
    print(f"- {paths.model_dir}/xgb_risk_model.joblib")
    print(f"- {paths.report_dir}/metrics.json")
    print(f"Test MCC={test_mcc:.4f} | Test AP(PR-AUC)={test_ap:.4f} | Threshold={best_t:.4f}")

if __name__ == "__main__":
    main()
