# Transaction Risk Scoring & Fraud Detection Platform

This project implements a **time-aware transaction fraud risk scoring pipeline** built on the **IEEE-CIS Fraud Detection dataset**.  
The focus is on **leakage-safe modeling**, **behavioral feature engineering**, and **interpretable evaluation** under **severe class imbalance**, reflecting real-world fraud detection constraints.

---

## Project Overview

Fraud detection is a highly imbalanced, time-dependent classification problem where improper validation or data leakage can lead to misleading results.  
This project addresses these challenges by:

- Applying **leakage-safe, time-based train/validation/test splits**
- Engineering **temporal and behavioral features** from transaction history
- Training a **cost-sensitive XGBoost model** to handle extreme imbalance
- Evaluating performance using **MCC and PR-AUC**, which are robust to skewed class distributions
- Explaining model behavior using **SHAP** to surface key fraud drivers

The pipeline is designed to be **reproducible, interpretable, and portfolio-ready**.

---

## Dataset

- **Source:** IEEE-CIS Fraud Detection (Kaggle)
- **Subset Used:** 300,000 earliest transactions (by time)
- **Fraud Rate (train):** ~3%
- **Files (not committed):**
  - `train_transaction.csv`
  - `train_identity.csv`

Raw datasets are intentionally excluded from version control in accordance with Kaggle’s terms.

---

## Modeling Approach

### 1. Time-Aware Data Splitting

Transactions are split chronologically to reflect real deployment conditions:

- **70%** Train  
- **15%** Validation  
- **15%** Test  

This prevents future information from leaking into model training.

---

### 2. Feature Engineering

The following leakage-safe features are constructed:

- **Basic features**
  - Log-transformed transaction amount
  - Missing-value count per transaction

- **Temporal & behavioral features**
  - Time since previous transaction (per entity)
  - Rolling mean, standard deviation, and z-score of transaction amounts
  - Recent transaction counts over rolling windows

Entity signals include:
- Card identifiers
- Address information
- Email domains
- Device type

---

### 3. Handling Class Imbalance

Instead of synthetic oversampling, the model uses:

- **Cost-sensitive learning** via `scale_pos_weight`
- **Threshold optimization** using Matthews Correlation Coefficient (MCC)

This mirrors common industry practices in fraud systems.

---

### 4. Model

- **Algorithm:** XGBoost (gradient-boosted decision trees)
- **Objective:** Binary logistic classification
- **Tree method:** Histogram-based (CPU-efficient)

The model outputs **fraud probabilities**, which are converted to decisions using an MCC-optimized threshold.

---

## Evaluation Metrics

Because fraud data is highly imbalanced, the following metrics are used:

- **Matthews Correlation Coefficient (MCC)**
- **Precision–Recall AUC (PR-AUC)**

These metrics provide a more truthful assessment than accuracy or ROC-AUC alone.

---

## Final Model Performance (Test Set)

- **MCC:** 0.50  
- **PR-AUC:** 0.50  
- **Fraud Precision:** ~0.70  
- **Fraud Recall:** ~0.38  

The selected threshold prioritizes **false-positive control**, resulting in a conservative but interpretable fraud detection system.

Detailed metrics are available in:
```
reports/metrics.json
```

---

## Model Interpretability (SHAP)

To ensure transparency, **SHAP TreeExplainer** is used to analyze model predictions.

Generated outputs:
- **Global feature importance (bar plot)**
- **Feature impact distribution (beeswarm plot)**

Key fraud drivers include:
- Card-related attributes
- Address signals
- Email domains
- Transaction amount and timing patterns

Artifacts:
```
reports/shap_summary_bar.png
reports/shap_summary_beeswarm.png
```

---

## Project Structure

```
.
├── src/
│   ├── train_supervised.py
│   ├── ingest.py
│   ├── features.py
│   ├── split.py
│   └── config.py
├── reports/
│   ├── metrics.json
│   ├── shap_summary_bar.png
│   └── shap_summary_beeswarm.png
├── requirements.txt
├── .gitignore
└── README.md
```

---

## How to Run

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add the dataset
Download the IEEE-CIS dataset from Kaggle and place the following files in:
```
data/raw/
```
- `train_transaction.csv`
- `train_identity.csv`

(These files are ignored by Git and not committed.)

---

### 5. Train the model
```bash
python src/train_supervised.py
```

This will:
- Train the model
- Save evaluation metrics
- Generate SHAP explainability plots

---

## Dependencies

Key libraries used:
- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `matplotlib`
- `joblib`
- `shap`

Full list available in `requirements.txt`.

---

## Notes

- This project intentionally avoids synthetic oversampling (e.g., SMOTE) to preserve temporal realism.
- All evaluation is performed on a strictly held-out test set.
- The emphasis is on **methodology correctness and interpretability**, not leaderboard optimization.
