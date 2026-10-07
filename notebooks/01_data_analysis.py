import pandas as pd

# ==========================================
# STUDENT PERFORMANCE ANALYSIS
# ==========================================

# Load dataset
df = pd.read_excel("data/student-mat.xlsx")

print("=" * 50)
print("STUDENT PERFORMANCE DATASET")
print("=" * 50)

# Dataset dimensions
print("\nDataset Shape:")
print(df.shape)

# Column names
print("\nColumn Names:")
print(df.columns.tolist())

# Data types
print("\nData Types:")
print(df.dtypes)

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Basic statistics
print("\nBasic Statistics:")
print(df.describe())

# Target variable
print("\nFinal Grade (G3) Statistics:")
print(df["G3"].describe())

print("\nFinal Grade Distribution:")
print(df["G3"].value_counts().sort_index())

print("\nAnalysis completed successfully!")

# ==========================================
# CORRELATION ANALYSIS
# ==========================================

print("\n" + "=" * 50)
print("CORRELATION WITH FINAL GRADE (G3)")
print("=" * 50)

# Select numeric columns
numeric_df = df.select_dtypes(include=["int64", "float64"])

# Calculate correlations with G3
correlation = numeric_df.corr()["G3"].sort_values(ascending=False)

print("\nCorrelation of Numeric Variables with G3:")
print(correlation)

print("\nTop 10 Positive/Negative Relationships:")
print(correlation.drop("G3").sort_values(key=abs, ascending=False).head(10))

# ==========================================
# PREPARE DATA FOR MACHINE LEARNING
# ==========================================

print("\n" + "=" * 50)
print("PREPARING DATA FOR MACHINE LEARNING")
print("=" * 50)

# Remove G1 and G2 to avoid target leakage
ml_df = df.drop(columns=["G1", "G2"])

print("\nRemoved G1 and G2 to prevent target leakage.")

print("\nML Dataset Shape:")
print(ml_df.shape)

print("\nML Dataset Columns:")
print(ml_df.columns.tolist())

print("\nML Dataset Preview:")
print(ml_df.head())

print("\nMachine learning dataset prepared successfully!")