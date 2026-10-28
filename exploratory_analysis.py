
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Locate the downloaded dataset
PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "credit_default.csv"


df = pd.read_csv(DATA_PATH)

# Restore the documented names of the UCI dataset features.
feature_names = [
    "credit_limit", "gender", "education", "marital_status", "age",
    "pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6",
    "bill_amt1", "bill_amt2", "bill_amt3", "bill_amt4",
    "bill_amt5", "bill_amt6",
    "pay_amt1", "pay_amt2", "pay_amt3", "pay_amt4",
    "pay_amt5", "pay_amt6"
]

expected_columns = len(feature_names) + 1

if df.shape[1] != expected_columns:
    raise ValueError(
        f"Expected {expected_columns} columns, but found {df.shape[1]}."
    )

df.columns = feature_names + ["default_next_month"]

print("Original feature names restored successfully.")


TARGET = "default_next_month"

print("=" * 60)
print("CREDIT RISK — EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# 1. Dataset overview
print("\n1. DATASET OVERVIEW")
print("Rows and columns:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

# 2. Data quality
print("\n2. DATA QUALITY")
print("Duplicate rows:", df.duplicated().sum())
print("Missing cells:", df.isna().sum().sum())
print("\nMissing values by column:")
print(df.isna().sum().sort_values(ascending=False).head(10))

# 3. Target distribution
print("\n3. DEFAULT DISTRIBUTION")
target_counts = df[TARGET].value_counts().sort_index()
print(target_counts)

default_rate = df[TARGET].mean() * 100
print(f"\nOverall default rate: {default_rate:.2f}%")

# 4. Default rates by repayment status
repayment_columns = [
    col for col in df.columns
    if col.startswith("pay_") and col != TARGET
]

if repayment_columns:
    repayment_col = repayment_columns[0]

    print(f"\n4. DEFAULT RATE BY {repayment_col.upper()}")
    repayment_analysis = (
        df.groupby(repayment_col)[TARGET]
        .agg(["count", "mean"])
        .rename(columns={
            "count": "customer_count",
            "mean": "default_rate"
        })
    )

    repayment_analysis["default_rate"] *= 100
    print(repayment_analysis.round(2))

    repayment_analysis["default_rate"].plot(
        kind="bar",
        figsize=(10, 5),
        title=f"Default rate by {repayment_col}"
    )
    plt.ylabel("Default rate (%)")
    plt.xlabel(repayment_col)
    plt.tight_layout()
    plt.show()
else:
    print("No repayment-status columns found. Check column names.")

# 5. Default rates by credit limit
credit_columns = [
    col for col in df.columns
    if "limit_bal" in col or "credit_limit" in col
]

if credit_columns:
    credit_col = credit_columns[0]

    df["credit_limit_group"] = pd.qcut(
        df[credit_col],
        q=5,
        duplicates="drop"
    )

    credit_analysis = (
        df.groupby("credit_limit_group", observed=False)[TARGET]
        .agg(["count", "mean"])
        .rename(columns={
            "count": "customer_count",
            "mean": "default_rate"
        })
    )

    credit_analysis["default_rate"] *= 100

    print(f"\n5. DEFAULT RATE BY {credit_col.upper()} GROUP")
    print(credit_analysis.round(2))

    credit_analysis["default_rate"].plot(
        kind="bar",
        figsize=(10, 5),
        title="Default rate by credit-limit group"
    )
    plt.ylabel("Default rate (%)")
    plt.xlabel("Credit-limit group")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()
else:
    print("No credit-limit column found. Check column names.")

# 6. Numeric summary
print("\n6. NUMERIC SUMMARY")
print(df.describe().round(2).to_string())

print("\nEDA completed successfully.")
