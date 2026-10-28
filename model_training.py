
from pathlib import Path

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix
)

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_DIR = Path(
    r"C:\Users\ghaka\Documents\credit-risk-intelligence"
)

DATA_PATH = PROJECT_DIR / "data" / "credit_default.csv"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. LOAD AND PREPARE DATA
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

feature_names = [
    "credit_limit", "gender", "education", "marital_status", "age",
    "pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6",
    "bill_amt1", "bill_amt2", "bill_amt3", "bill_amt4",
    "bill_amt5", "bill_amt6",
    "pay_amt1", "pay_amt2", "pay_amt3", "pay_amt4",
    "pay_amt5", "pay_amt6"
]

if df.shape[1] != len(feature_names) + 1:
    raise ValueError(
        f"Unexpected number of columns: {df.shape[1]}"
    )

df.columns = feature_names + ["default_next_month"]

print("=" * 60)
print("CREDIT RISK MODEL TRAINING")
print("=" * 60)

print("\nOriginal dataset shape:", df.shape)

# Remove exact duplicate records before splitting.
duplicate_count = int(df.duplicated().sum())
df = df.drop_duplicates().copy()

print("Exact duplicate rows removed:", duplicate_count)
print("Dataset after cleaning:", df.shape)

X = df.drop(columns=["default_next_month"])
y = df["default_next_month"].astype(int)

# Treat these coded categories as categories, not continuous numbers.
categorical_features = [
    "gender", "education", "marital_status"
]

numeric_features = [
    col for col in X.columns
    if col not in categorical_features
]

# --------------------------------------------------
# 3. TRAIN / TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))
print("Default rate in test set:", round(y_test.mean() * 100, 2), "%")

# --------------------------------------------------
# 4. PREPROCESSING
# --------------------------------------------------

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# --------------------------------------------------
# 5. DEFINE MODELS
# --------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
}

results = []
trained_models = {}

# --------------------------------------------------
# 6. TRAIN AND EVALUATE
# --------------------------------------------------

for model_name, classifier in models.items():

    print("\n" + "=" * 60)
    print("MODEL:", model_name)
    print("=" * 60)

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1_score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "pr_auc": average_precision_score(y_test, probabilities)
    }

    results.append(metrics)
    trained_models[model_name] = pipeline

    print("\nEvaluation metrics:")
    for metric_name, value in metrics.items():
        if metric_name != "model":
            print(f"{metric_name}: {value:.4f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["No default", "Default"],
        zero_division=0
    ))

# --------------------------------------------------
# 7. COMPARE MODELS
# --------------------------------------------------

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="pr_auc",
    ascending=False
)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
print(results_df.round(4).to_string(index=False))

results_df.to_csv(
    PROJECT_DIR / "model_comparison.csv",
    index=False
)

# --------------------------------------------------
# 8. SAVE THE BEST MODEL
# --------------------------------------------------

best_model_name = results_df.iloc[0]["model"]
best_model = trained_models[best_model_name]

joblib.dump(
    best_model,
    MODEL_DIR / "credit_risk_model.joblib"
)

print("\nBest model by PR-AUC:", best_model_name)
print("Saved model to:", MODEL_DIR / "credit_risk_model.joblib")
print("Saved comparison to:", PROJECT_DIR / "model_comparison.csv")
print("\nModel training completed successfully.")
