import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt


# ==========================================
# SHAP EXPLAINABILITY ANALYSIS
# ==========================================

print("=" * 60)
print("SHAP EXPLAINABILITY ANALYSIS")
print("=" * 60)


# ------------------------------------------
# 1. LOAD DATA
# ------------------------------------------

df = pd.read_excel("data/student-mat.xlsx")

print("\nOriginal Dataset Shape:", df.shape)


# ------------------------------------------
# 2. REMOVE G1 AND G2
# ------------------------------------------
# G1 and G2 are excluded because they contain
# previous-period grade information and could
# cause target leakage when predicting G3.

df = df.drop(
    columns=["G1", "G2"],
    errors="ignore"
)

print("After removing G1 and G2:", df.shape)


# ------------------------------------------
# 3. REMOVE STUDENT_ID
# ------------------------------------------
# Student_ID is an identifier only.
# It is used by the website for student lookup,
# but it is NOT a meaningful ML predictor.
#
# SHAP must explain the same features used by
# the Random Forest model.

if "Student_ID" in df.columns:

    df = df.drop(
        columns=["Student_ID"]
    )

    print("Student_ID removed from SHAP features.")

else:

    print("Student_ID not found. Continuing.")


# ------------------------------------------
# 4. SEPARATE FEATURES AND TARGET
# ------------------------------------------

X = df.drop(
    columns=["G3"]
)

y = df["G3"]

print("\nSHAP Input Features:", X.shape)
print("Target:", y.shape)


# ------------------------------------------
# 5. LOAD SAVED RANDOM FOREST PIPELINE
# ------------------------------------------

model = joblib.load(
    "models/student_performance_rf.pkl"
)

print("\nRandom Forest model loaded successfully!")


# ------------------------------------------
# 6. GET PREPROCESSOR AND RANDOM FOREST
# ------------------------------------------

preprocessor = model.named_steps["preprocessor"]

rf_model = model.named_steps["model"]


# ------------------------------------------
# 7. TRANSFORM DATA
# ------------------------------------------

X_transformed = preprocessor.transform(X)

feature_names = preprocessor.get_feature_names_out()

print("\nOriginal Features:", X.shape[1])

print(
    "Transformed Features:",
    X_transformed.shape[1]
)


# ------------------------------------------
# 8. CREATE SHAP EXPLAINER
# ------------------------------------------

explainer = shap.TreeExplainer(
    rf_model
)


# ------------------------------------------
# 9. CALCULATE SHAP VALUES
# ------------------------------------------

shap_values = explainer.shap_values(
    X_transformed
)


# ------------------------------------------
# 10. GLOBAL FEATURE IMPORTANCE
# ------------------------------------------

feature_importance = pd.DataFrame({

    "Feature": feature_names,

    "Importance": abs(shap_values).mean(
        axis=0
    )

})


# Sort from highest to lowest importance

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)


# ------------------------------------------
# 11. DISPLAY TOP FEATURES
# ------------------------------------------

print("\n" + "=" * 60)

print(
    "TOP 15 FEATURES INFLUENCING PREDICTIONS"
)

print("=" * 60)

print(
    feature_importance
    .head(15)
    .to_string(index=False)
)


# ------------------------------------------
# 12. SAVE FEATURE IMPORTANCE
# ------------------------------------------

feature_importance.to_csv(
    "models/shap_feature_importance.csv",
    index=False
)

print(
    "\nSHAP feature importance saved:"
)

print(
    "models/shap_feature_importance.csv"
)


# ------------------------------------------
# 13. CREATE SHAP SUMMARY PLOT
# ------------------------------------------
# The Flask website reads the SHAP image
# from the static folder.
#
# Therefore, save the final image directly
# inside static/.

plt.figure()

shap.summary_plot(
    shap_values,
    X_transformed,
    feature_names=feature_names,
    show=False
)

plt.tight_layout()


# ------------------------------------------
# 14. SAVE SHAP SUMMARY IMAGE
# ------------------------------------------

plt.savefig(
    "static/shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "\nSHAP summary plot saved:"
)

print(
    "static/shap_summary.png"
)


# ------------------------------------------
# 15. FINAL OUTPUT
# ------------------------------------------

print("\n" + "=" * 60)

print(
    "SHAP ANALYSIS COMPLETED SUCCESSFULLY!"
)

print("=" * 60)


print(
    "\nSHAP and ML feature count should now match:"
)

print(
    "ML Features : 30"
)

print(
    "SHAP Features:",
    X.shape[1]
)


print(
    "\nStudent_ID: EXCLUDED"
)

print(
    "G1: EXCLUDED"
)

print(
    "G2: EXCLUDED"
)

print(
    "G3: TARGET"
)


# ------------------------------------------
# 16. FINAL PROJECT INTERPRETATION
# ------------------------------------------

print("\n" + "=" * 60)

print(
    "PROJECT INTERPRETATION"
)

print("=" * 60)

print(
    "SHAP identifies which features influenced"
)

print(
    "the Random Forest model's predictions."
)

print(
    "SHAP importance indicates model influence,"
)

print(
    "not direct causation."
)

print(
    "G1 and G2 were excluded to prevent"
)

print(
    "target leakage."
)

print(
    "Student_ID was excluded because it is"
)

print(
    "only an identification field."
)

print("=" * 60)