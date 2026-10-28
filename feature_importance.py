
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import joblib

# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

PROJECT_DIR = Path(
    r"C:\Users\ghaka\Documents\credit-risk-intelligence"
)

DATA_PATH = PROJECT_DIR / "data" / "credit_default.csv"
MODEL_PATH = PROJECT_DIR / "models" / "credit_risk_model.joblib"

# --------------------------------------------------
# 2. LOAD MODEL AND DATA
# --------------------------------------------------

model = joblib.load(MODEL_PATH)
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

# --------------------------------------------------
# 3. EXTRACT RANDOM FOREST FEATURE IMPORTANCE
# --------------------------------------------------

preprocessor = model.named_steps["preprocessing"]
classifier = model.named_steps["classifier"]

transformed_names = preprocessor.get_feature_names_out()
importance_values = classifier.feature_importances_

importance_df = pd.DataFrame({
    "transformed_feature": transformed_names,
    "importance": importance_values
})

# Aggregate one-hot encoded categories back to their
# original feature, such as gender or education.
def original_feature(transformed_name):
    name = transformed_name.split("__", 1)[-1]

    for feature in feature_names:
        if name == feature or name.startswith(feature + "_"):
            return feature

    return name

importance_df["feature"] = (
    importance_df["transformed_feature"].apply(original_feature)
)

summary = (
    importance_df.groupby("feature", as_index=False)["importance"]
    .sum()
    .sort_values("importance", ascending=False)
)

summary["importance_percent"] = summary["importance"] * 100

# --------------------------------------------------
# 4. DISPLAY TOP FEATURES
# --------------------------------------------------

print("=" * 60)
print("CREDIT RISK — FEATURE IMPORTANCE")
print("=" * 60)

print("\nTop 15 features:")
print(
    summary.head(15)[["feature", "importance_percent"]]
    .round(2)
    .to_string(index=False)
)

summary.to_csv(
    PROJECT_DIR / "feature_importance.csv",
    index=False
)

# --------------------------------------------------
# 5. VISUALISE TOP 15 FEATURES
# --------------------------------------------------

top_features = summary.head(15).sort_values("importance")

plt.figure(figsize=(10, 7))
plt.barh(
    top_features["feature"],
    top_features["importance_percent"]
)

plt.xlabel("Feature importance (%)")
plt.ylabel("Feature")
plt.title("Top 15 Features in the Credit Risk Model")
plt.tight_layout()

plt.savefig(
    PROJECT_DIR / "feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nSaved feature importance table to:")
print(PROJECT_DIR / "feature_importance.csv")

print("\nSaved chart to:")
print(PROJECT_DIR / "feature_importance.png")
