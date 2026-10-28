

from pathlib import Path
from ucimlrepo import fetch_ucirepo

# Project folders
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Download the official UCI dataset
print("Downloading credit default dataset...")

dataset = fetch_ucirepo(id=350)

# Separate the input features and target
X = dataset.data.features.copy()
y = dataset.data.targets.copy()

# Combine them into one DataFrame
df = X.copy()
df["default_next_month"] = y.iloc[:, 0].to_numpy()

# Standardise column names
df.columns = [
    str(col).strip().lower().replace(" ", "_").replace("/", "_")
    for col in df.columns
]

# Save locally
output_path = DATA_DIR / "credit_default.csv"
df.to_csv(output_path, index=False)

print("\nDownload complete!")
print("Saved to:", output_path)
print("Dataset shape:", df.shape)

print("\nFirst five rows:")
print(df.head())

print("\nTarget distribution:")
print(df["default_next_month"].value_counts(dropna=False))

print("\nMissing values:")
print(df.isnull().sum().sort_values(ascending=False).head(10))

