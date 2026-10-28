
from pathlib import Path

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
    roc_auc_score
)

PROJECT_DIR = Path(
    r"C:\Users\ghaka\Documents\credit-risk-intelligence"
)

DATA_PATH = PROJECT_DIR / "data" / "credit_default.csv"
MODEL_PATH = PROJECT_DIR / "models" / "credit_risk_model.joblib"

# Load and prepare the dataset
df = pd.read_csv(DATA_PATH)

feature_names = [
    "credit_limit", "gender", "education", "marital_status", "age",
    "pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6",
    "bill_amt1", "bill_amt2", "bill_amt3", "bill_amt4",
    "bill_amt5", "bill_amt6",
    "pay_amt1", "pay_amt2", "pay_amt3", "pay_amt4",
    "pay_amt5", "pay_amt6"
]

df.columns = feature_names + ["default_next_month"]
df = df.drop_duplicates().copy()

X = df.drop(columns=["default_next_month"])
y = df["default_next_month"].astype(int)

# Recreate the same stratified split used in training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = joblib.load(MODEL_PATH)

# Get predicted probabilities, not just class labels
probabilities = model.predict_proba(X_test)[:, 1]

print("=" * 65)
print("CREDIT RISK — THRESHOLD ANALYSIS")
print("=" * 65)

print(
    "\nROC-AUC:",
    round(roc_auc_score(y_test, probabilities), 4)
)
print(
    "PR-AUC:",
    round(average_precision_score(y_test, probabilities), 4)
)

results = []

thresholds = [0.20, 0.30, 0.40, 0.50, 0.60]

for threshold in thresholds:

    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    results.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn
    })

results_df = pd.DataFrame(results)

print("\nThreshold comparison:")
print(results_df.round(4).to_string(index=False))

results_df.to_csv(
    PROJECT_DIR / "threshold_comparison.csv",
    index=False
)

print("\nSaved results to:")
print(PROJECT_DIR / "threshold_comparison.csv")
