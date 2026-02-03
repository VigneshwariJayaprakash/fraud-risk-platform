import os
import json
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

from config import Paths
from ingest import load_and_merge
from features import add_basic_features, add_entity_temporal_features, make_model_matrix
from split import time_split
from config import SplitConfig, FeatureConfig


def main():
    paths = Paths()
    split_cfg = SplitConfig()
    feat_cfg = FeatureConfig()

    os.makedirs(paths.report_dir, exist_ok=True)

    # 1) Load the saved model bundle
    bundle = joblib.load(os.path.join(paths.model_dir, "xgb_risk_model.joblib"))
    model = bundle["model"]
    trained_cols = bundle["columns"]

    # 2) Rebuild the same features (must match training)
    df = load_and_merge(paths.tx_path, paths.id_path)
    df = add_basic_features(df)
    df = add_entity_temporal_features(df, entity_cols=feat_cfg.entity_cols, rolling_k=feat_cfg.rolling_k)

    # Use the same time split so we explain on "future" (test) data
    _, _, test_df = time_split(df, split_cfg.train_frac, split_cfg.val_frac, split_cfg.test_frac)

    X_test, y_test = make_model_matrix(test_df)

    # 3) Align columns to exactly what the model was trained on
    # If any columns are missing, add them; if extra columns exist, drop them.
    for c in trained_cols:
        if c not in X_test.columns:
            X_test[c] = -999.0
    X_test = X_test[trained_cols]

    # 4) Sample a subset for speed (SHAP can be heavy on full test)
    # You can increase this later.
    n = min(5000, len(X_test))
    X_sample = X_test.iloc[:n].copy()

    # 5) Explain using TreeExplainer (fast for XGBoost trees)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # 6) Global importance plot (bar)
    plt.figure()
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    out_bar = os.path.join(paths.report_dir, "shap_summary_bar.png")
    plt.tight_layout()
    plt.savefig(out_bar, dpi=200)
    plt.close()

    # 7) Global impact plot (beeswarm)
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    out_swarm = os.path.join(paths.report_dir, "shap_summary_beeswarm.png")
    plt.tight_layout()
    plt.savefig(out_swarm, dpi=200)
    plt.close()

    # 8) Save a tiny JSON with top features (optional, useful for README)
    # Compute mean absolute SHAP per feature
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:20]
    top_features = [{"feature": trained_cols[i], "mean_abs_shap": float(mean_abs[i])} for i in top_idx]

    with open(os.path.join(paths.report_dir, "shap_top_features.json"), "w") as f:
        json.dump(top_features, f, indent=2)

    print("Saved SHAP reports:")
    print("-", out_bar)
    print("-", out_swarm)
    print("-", os.path.join(paths.report_dir, "shap_top_features.json"))


if __name__ == "__main__":
    main()
