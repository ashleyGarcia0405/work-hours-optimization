Here’s a clean, copy-paste **README.md** you can use. Tweak the org name / links as needed.

---

# Work-Hours Optimization (UCI Adult / Census)

Predict whether an individual earns **> \$50K** from hours worked **plus** demographics, then surface insights about **who is over/under-compensated** relative to workload.

* **Stack:** Python, scikit-learn, Pandas, NumPy, Matplotlib/Seaborn
* **Highlights:** Leakage-safe Pipelines, class-imbalance handling, CV + hyperparameter tuning, threshold optimization, fairness slice metrics
* **Test results:** **ROC AUC 0.919**, **PR AUC 0.800**, **F1 0.710**, **Balanced Acc 0.808** (threshold **0.661**)

---

## 1) Problem Overview

* **Goal:** Supervised **binary classification** of `income_binary` (`>50K` vs `<=50K`).
* **Motivation:** Identify workload-pay mismatches to inform **pay-equity reviews**, **retention risk**, and **workforce planning**.
* **Key question:** How does **hours\_per\_week** interact with human-capital and job attributes in predicting higher income?

---

## 2) Data

* **Source:** UCI Adult / Census 1994 (provided as `data/censusData.csv`).
* **Label:** `income_binary` (`>50K`, `<=50K`) → converted to `label` (1/0).
* **Selected features:**

  * **Numeric:** `age`, `education_num`, `hours_per_week`, `capital_gain_log`, `capital_loss_log`
  * **Categorical:** `marital_status`, `occupation` (rare→“Other”), `workclass`, `relationship`, `native_country` (US/Other/Unknown)
* **Fairness audit (report-only):** `race`, `sex_selfID` (excluded from training)

---

## 3) Repo Structure

```
.
├── data/
│   └── censusData.csv
├── notebooks/
│   └── work_hours_optimization.ipynb
├── artifacts/
│   ├── final_model.pkl
│   ├── tuned_threshold.txt
│   ├── test_metrics.json
│   ├── perm_importance_original.csv
│   ├── rf_importance_aggregated.csv
│   ├── hours_effect.csv
│   ├── hours_effect.png
│   ├── confusion_matrix_test.png
│   ├── fairness_sex.csv
│   └── fairness_race.csv
└── README.md
```

---

## 4) Environment & Setup

```bash
# Python 3.9+ recommended
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
# (or minimal)
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

Open the notebook:

```bash
jupyter lab  # or jupyter notebook
```

---

## 5) Methodology (what the code does)

**EDA & Cleaning**

* Standardize column names; trim string whitespace.
* Handle missingness: **median** (numeric), **most\_frequent** + explicit **“Unknown”** (categorical).
* Outliers: clip `hours_per_week` to **\[1, 99]**.
* Heavy tails: `capital_gain`, `capital_loss` → **log1p** features.
* Category reduction: `native_country` → **US/Other/Unknown**; merge rare `occupation` values.

**Preprocessing & Leakage Safety**

* `ColumnTransformer` with **One-Hot Encoding** (categoricals) + **StandardScaler** (numerics).
* Everything wrapped in a **Pipeline**.

**Modeling & Selection**

* Baseline: `DummyClassifier (stratified)`.
* **Logistic Regression (L1/L2)** and **Random Forest** with **Stratified 5-fold CV** via `RandomizedSearchCV`.
* **Class imbalance:** `class_weight='balanced'`.
* **Threshold tuning:** choose probability cutoff that maximizes **F1** on validation.

**Evaluation**

* Primary: **ROC AUC**; Secondary: **PR AUC**, **F1**, **Balanced Accuracy**.
* **Permutation importance** (original features) + aggregated RF importances (sum over one-hot features).
* Hours effect: Avg predicted **P(>50K)** by hours bins (≤30, 31–40, 41–50, >50).

**Fairness (report-only)**

* Slice metrics by **sex\_selfID** and **race** on the test set (sensitive attributes *not* used to train).

---

## 6) Results (Test Set)

* **Chosen model:** Random Forest (class-weighted)
* **Tuned threshold:** **0.661**
* **Metrics:**

  * ROC AUC: **0.9187**
  * PR AUC: **0.7999**
  * F1 (tuned): **0.7103**
  * Balanced Accuracy: **0.8083**

**Top drivers (examples)**

* Permutation importance (original features): `marital_status`, `capital_gain_log`, `education_num`, `age`, `occupation`, `relationship`, `hours_per_week` …
* Aggregated RF importances (summed across one-hot): `marital_status`, `relationship`, `education_num`, `capital_gain_log`, `occupation`, `age`, `hours_per_week` …

**Hours vs Predicted P(>50K) (Test)**

* ≤30h: **0.14**
* 31–40h: **0.33**
* 41–50h: **0.53**
* > 50h: **0.56**

**Fairness slice metrics (report-only)**

* Positive prediction rate by **sex\_selfID**: **Non-Female 0.300** vs **Female 0.107**
* By **race**: range **0.038 (Amer-Indian-Inuit; small n)** to **0.295 (Asian-Pac-Islander)**
* *Note:* These are descriptive; consider calibration checks, subgroup thresholding (with care), or post-processing if deploying.

---

## 7) Reproduce Locally

1. Place `censusData.csv` in `data/`.
2. Open `notebooks/work_hours_optimization.ipynb`.
3. Run cells sequentially. Artifacts will appear in `artifacts/`.

To export artifacts programmatically (without the notebook), adapt the `artifacts` save cell into a script under `src/` and run:

```bash
python src/train_and_save.py
```

---

## 8) Key Commands / Snippets

**Install**

```bash
pip install -r requirements.txt
```

**Run notebook**

```bash
jupyter lab
```

**Load final model and score new rows (example)**

```python
import joblib, pandas as pd
model = joblib.load("artifacts/final_model.pkl")
X_new = pd.DataFrame([...])  # must match training schema
proba = model.predict_proba(X_new)[:, 1]
```

---

## 9) Limitations & Next Steps

* This uses 1994 Census data; relationships may shift over time.
* Threshold is tuned for F1 on a held-out split; different business goals may require different cutoffs (precision-oriented vs recall-oriented).
* Consider **probability calibration** (e.g., `CalibratedClassifierCV`) and **subgroup monitoring** in deployment.
* Try **Gradient Boosting** (e.g., HistGB) for potential performance gains; explore **interaction features** (e.g., `hours × education`).

---

## 10) Acknowledgments

* **UCI Adult / Census Income** dataset.
* scikit-learn, Pandas, NumPy, Matplotlib, Seaborn.

---

### requirements.txt (example)

```
pandas>=2.0
numpy>=1.23
scikit-learn>=1.2
matplotlib>=3.7
seaborn>=0.12
joblib>=1.3
jupyterlab>=4.0
```

---
