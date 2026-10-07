
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# STUDENT PERFORMANCE ML MODEL
# ==========================================

print("=" * 60)
print("STUDENT PERFORMANCE PREDICTION MODEL")
print("=" * 60)


# ------------------------------------------
# 1. LOAD DATA
# ------------------------------------------

df = pd.read_excel("data/student-mat.xlsx")

print("\nOriginal Dataset Shape:", df.shape)


# ------------------------------------------
# 2. REMOVE G1 AND G2
# ------------------------------------------
# G1 and G2 are previous-period grades.
# They are excluded from the main model because
# they contain information very closely related
# to the final grade G3 and could cause target leakage.

df = df.drop(columns=["G1", "G2"], errors="ignore")

print("After removing G1 and G2:", df.shape)


# ------------------------------------------
# 3. KEEP STUDENT_ID ONLY FOR IDENTIFICATION
# ------------------------------------------
# Student_ID is an identifier, not a meaningful
# academic predictor.
#
# It can be used by the website to find a student,
# but it should NOT be given to the ML model.

if "Student_ID" in df.columns:
    df = df.drop(columns=["Student_ID"])

    print("Student_ID removed from ML features.")
else:
    print("Student_ID not found in dataset. Continuing.")


# ------------------------------------------
# 4. SEPARATE FEATURES AND TARGET
# ------------------------------------------

X = df.drop(columns=["G3"])
y = df["G3"]

print("\nFeatures:", X.shape)
print("Target:", y.shape)


# ------------------------------------------
# 5. IDENTIFY COLUMN TYPES
# ------------------------------------------

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numeric_features = X.select_dtypes(
    exclude=["object"]
).columns.tolist()

print("\nCategorical Features:")
print(categorical_features)

print("\nNumeric Features:")
print(numeric_features)


# ------------------------------------------
# 6. PREPROCESSING
# ------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)


# ------------------------------------------
# 7. TRAIN-TEST SPLIT
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# ==========================================
# 8. MODEL 1 — RIDGE REGRESSION
# ==========================================

ridge_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", Ridge(alpha=1.0))
    ]
)

ridge_pipeline.fit(X_train, y_train)

ridge_predictions = ridge_pipeline.predict(X_test)

ridge_mae = mean_absolute_error(
    y_test,
    ridge_predictions
)

ridge_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        ridge_predictions
    )
)

ridge_r2 = r2_score(
    y_test,
    ridge_predictions
)


# ==========================================
# 9. MODEL 2 — RANDOM FOREST
# ==========================================

rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestRegressor(
                n_estimators=300,
                random_state=42,
                max_depth=8,
                min_samples_leaf=2
            )
        )
    ]
)

rf_pipeline.fit(X_train, y_train)

rf_predictions = rf_pipeline.predict(X_test)

rf_mae = mean_absolute_error(
    y_test,
    rf_predictions
)

rf_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        rf_predictions
    )
)

rf_r2 = r2_score(
    y_test,
    rf_predictions
)


# ==========================================
# 10. MODEL 3 — GRADIENT BOOSTING
# ==========================================

gb_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=2,
                random_state=42
            )
        )
    ]
)

gb_pipeline.fit(X_train, y_train)

gb_predictions = gb_pipeline.predict(X_test)

gb_mae = mean_absolute_error(
    y_test,
    gb_predictions
)

gb_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        gb_predictions
    )
)

gb_r2 = r2_score(
    y_test,
    gb_predictions
)


# ==========================================
# 11. MODEL COMPARISON
# ==========================================

results = pd.DataFrame({
    "Model": [
        "Ridge Regression",
        "Random Forest",
        "Gradient Boosting"
    ],
    "MAE": [
        ridge_mae,
        rf_mae,
        gb_mae
    ],
    "RMSE": [
        ridge_rmse,
        rf_rmse,
        gb_rmse
    ],
    "R2 Score": [
        ridge_r2,
        rf_r2,
        gb_r2
    ]
})


print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results.to_string(
        index=False
    )
)


# ==========================================
# 12. BEST MODEL
# ==========================================

best_model_name = results.loc[
    results["R2 Score"].idxmax(),
    "Model"
]

print("\nBest Model:", best_model_name)


# ==========================================
# 13. SAVE MODEL RESULTS
# ==========================================

os.makedirs("models", exist_ok=True)

# Save the complete model pipeline.
# The pipeline includes preprocessing + Random Forest.

joblib.dump(
    rf_pipeline,
    "models/student_performance_rf.pkl"
)


# Save model comparison for use in the report/website later.

results.to_csv(
    "models/model_comparison.csv",
    index=False
)


# Save the exact ML feature names.

feature_info = pd.DataFrame({
    "Feature": X.columns
})

feature_info.to_csv(
    "models/ml_features.csv",
    index=False
)


# ==========================================
# 14. FINAL OUTPUT
# ==========================================

print("\n" + "=" * 60)
print("FINAL MODEL INFORMATION")
print("=" * 60)

print("Selected Model:", best_model_name)

print("\nRandom Forest Performance:")
print(f"MAE  : {rf_mae:.4f}")
print(f"RMSE : {rf_rmse:.4f}")
print(f"R2   : {rf_r2:.4f}")

print("\nNumber of ML Features:", len(X.columns))

print("\nStudent_ID: EXCLUDED FROM ML")
print("G1: EXCLUDED FROM ML")
print("G2: EXCLUDED FROM ML")
print("G3: TARGET VARIABLE")

print("\nML MODELING COMPLETED SUCCESSFULLY!")

print("\nFinal Random Forest model saved successfully!")
print("Location: models/student_performance_rf.pkl")

print("\nModel comparison saved:")
print("models/model_comparison.csv")

print("\nML feature list saved:")
print("models/ml_features.csv")

