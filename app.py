from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# ============================================================
# PATHS
# Deployment-safe absolute paths
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "student-mat.xlsx"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "student_performance_rf.pkl"
)

SHAP_IMAGE = "shap_summary.png"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_excel(DATA_PATH)

# Create Student_ID if it does not already exist
if "Student_ID" not in df.columns:
    df.insert(
        0,
        "Student_ID",
        [f"STU{i:03d}" for i in range(1, len(df) + 1)]
    )


# ============================================================
# LOAD MODEL
# ============================================================

model_pipeline = joblib.load(MODEL_PATH)


# ============================================================
# MODEL FEATURES
# IMPORTANT:
# Student_ID = identifier only
# G1/G2 = excluded to prevent target leakage
# ============================================================

MODEL_FEATURES = [
    "school",
    "sex",
    "age",
    "address",
    "famsize",
    "Pstatus",
    "Medu",
    "Fedu",
    "Mjob",
    "Fjob",
    "reason",
    "guardian",
    "traveltime",
    "studytime",
    "failures",
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "nursery",
    "higher",
    "internet",
    "romantic",
    "famrel",
    "freetime",
    "goout",
    "Dalc",
    "Walc",
    "health",
    "absences"
]


# ============================================================
# PERFORMANCE CATEGORY
# ============================================================

def performance_category(grade):

    if grade < 8:
        return "High Risk"

    elif grade < 10:
        return "Moderate Risk"

    elif grade < 12:
        return "Average"

    elif grade < 15:
        return "Good"

    else:
        return "Excellent"


# ============================================================
# ABSENCE GROUP
# ============================================================

def absence_group(absences):

    if absences <= 5:
        return "Very Low"

    elif absences <= 10:
        return "Low"

    elif absences <= 20:
        return "Moderate"

    else:
        return "High"


# ============================================================
# STUDY TIME LABEL
# UCI encoding:
# 1 = <2 hours
# 2 = 2–5 hours
# 3 = 5–10 hours
# 4 = >10 hours
# ============================================================

def study_time_label(value):

    labels = {
        1: "< 2 hours",
        2: "2–5 hours",
        3: "5–10 hours",
        4: "> 10 hours"
    }

    return labels.get(int(value), str(value))


# ============================================================
# PREVIOUS FAILURE LABEL
# ============================================================

def failure_label(value):

    value = int(value)

    if value == 0:
        return "No Failures"

    elif value == 1:
        return "1 Failure"

    else:
        return f"{value} Failures"


# ============================================================
# PREPARE MODEL INPUT
# ============================================================

def prepare_model_input(row):

    input_data = {}

    for feature in MODEL_FEATURES:
        input_data[feature] = row[feature]

    return pd.DataFrame([input_data])


# ============================================================
# PREDICTION
# ============================================================

def predict_student(row):

    model_input = prepare_model_input(row)

    prediction = float(
        model_pipeline.predict(model_input)[0]
    )

    # Keep prediction inside valid grade range
    prediction = max(0, min(20, prediction))

    category = performance_category(prediction)

    return prediction, category


# ============================================================
# DASHBOARD DATA
# ============================================================

def build_dashboard_data(data):

    # --------------------------------------------------------
    # KPIs
    # --------------------------------------------------------

    total_students = len(data)

    average_grade = (
        float(data["G3"].mean())
        if total_students > 0
        else 0
    )

    high_performers = (
        int((data["G3"] >= 15).sum())
        if total_students > 0
        else 0
    )

    at_risk = (
        int((data["G3"] < 8).sum())
        if total_students > 0
        else 0
    )

    # --------------------------------------------------------
    # Grade Distribution
    # --------------------------------------------------------

    grade_distribution = (
        data["G3"]
        .value_counts()
        .sort_index()
    )

    grade_labels = [
        str(int(x))
        for x in grade_distribution.index
    ]

    grade_values = [
        int(x)
        for x in grade_distribution.values
    ]

    # --------------------------------------------------------
    # Study Time Analysis
    # --------------------------------------------------------

    study = (
        data.groupby("studytime")["G3"]
        .mean()
        .reindex([1, 2, 3, 4])
    )

    study_labels = [
        "< 2 hrs",
        "2–5 hrs",
        "5–10 hrs",
        "> 10 hrs"
    ]

    study_values = [
        round(float(x), 2) if pd.notna(x) else 0
        for x in study.values
    ]

    # --------------------------------------------------------
    # Previous Failures
    # --------------------------------------------------------

    failures = (
        data.groupby("failures")["G3"]
        .mean()
        .reindex([0, 1, 2, 3])
    )

    failure_labels = [
        "0",
        "1",
        "2",
        "3"
    ]

    failure_values = [
        round(float(x), 2) if pd.notna(x) else 0
        for x in failures.values
    ]

    # --------------------------------------------------------
    # Absence Analysis
    # --------------------------------------------------------

    temp = data.copy()

    temp["absence_group"] = temp["absences"].apply(
        absence_group
    )

    absence_order = [
        "Very Low",
        "Low",
        "Moderate",
        "High"
    ]

    absence_analysis = (
        temp.groupby("absence_group")["G3"]
        .mean()
        .reindex(absence_order)
    )

    absence_values = [
        round(float(x), 2) if pd.notna(x) else 0
        for x in absence_analysis.values
    ]

    # --------------------------------------------------------
    # Gender Analysis
    # --------------------------------------------------------

    gender = (
        data.groupby("sex")["G3"]
        .mean()
    )

    gender_labels = []
    gender_values = []

    for key in ["F", "M"]:

        if key in gender.index:

            gender_labels.append(
                "Female" if key == "F" else "Male"
            )

            gender_values.append(
                round(float(gender[key]), 2)
            )

    # --------------------------------------------------------
    # School Analysis
    # --------------------------------------------------------

    school = (
        data.groupby("school")["G3"]
        .mean()
    )

    school_labels = []
    school_values = []

    for key in school.index:

        school_labels.append(str(key))

        school_values.append(
            round(float(school[key]), 2)
        )

    # --------------------------------------------------------
    # Performance Categories
    # --------------------------------------------------------

    categories = data["G3"].apply(
        performance_category
    )

    category_order = [
        "High Risk",
        "Moderate Risk",
        "Average",
        "Good",
        "Excellent"
    ]

    category_counts = [
        int((categories == category).sum())
        for category in category_order
    ]

    return {

        "total_students":
            total_students,

        "average_grade":
            round(average_grade, 2),

        "high_performers":
            high_performers,

        "at_risk":
            at_risk,

        "grade_labels":
            grade_labels,

        "grade_values":
            grade_values,

        "study_labels":
            study_labels,

        "study_values":
            study_values,

        "failure_labels":
            failure_labels,

        "failure_values":
            failure_values,

        "absence_labels":
            absence_order,

        "absence_values":
            absence_values,

        "gender_labels":
            gender_labels,

        "gender_values":
            gender_values,

        "school_labels":
            school_labels,

        "school_values":
            school_values,

        "category_labels":
            category_order,

        "category_values":
            category_counts
    }


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    dashboard_data = build_dashboard_data(df)

    return render_template(
        "dashboard.html",
        dashboard_data=dashboard_data
    )


# ============================================================
# DASHBOARD API
# 4 FILTERS:
# School
# Gender
# Study Time
# Performance Level
# ============================================================

@app.route("/api/dashboard")
def dashboard_api():

    data = df.copy()

    school = request.args.get(
        "school",
        "All"
    )

    gender = request.args.get(
        "gender",
        "All"
    )

    studytime = request.args.get(
        "studytime",
        "All"
    )

    performance = request.args.get(
        "performance",
        "All"
    )

    # --------------------------------------------------------
    # SCHOOL FILTER
    # --------------------------------------------------------

    if school != "All" and school:

        data = data[
            data["school"].astype(str) == school
        ]

    # --------------------------------------------------------
    # GENDER FILTER
    # --------------------------------------------------------

    if gender != "All" and gender:

        data = data[
            data["sex"].astype(str) == gender
        ]

    # --------------------------------------------------------
    # STUDY TIME FILTER
    # --------------------------------------------------------

    if studytime != "All" and studytime:

        try:

            study_value = int(studytime)

            data = data[
                data["studytime"] == study_value
            ]

        except ValueError:

            pass

    # --------------------------------------------------------
    # PERFORMANCE FILTER
    # --------------------------------------------------------

    if performance != "All" and performance:

        data = data[
            data["G3"]
            .apply(performance_category)
            == performance
        ]

    result = build_dashboard_data(data)

    return jsonify(result)


# ============================================================
# STUDENT LOOKUP
# ============================================================

@app.route("/api/student/<student_id>")
def student_lookup(student_id):

    matches = df[
        df["Student_ID"].astype(str).str.upper()
        == student_id.upper()
    ]

    if matches.empty:

        return jsonify({
            "success": False,
            "message": "Student ID not found."
        }), 404

    student = matches.iloc[0].copy()

    prediction, category = predict_student(student)

    return jsonify({

        "success":
            True,

        "student_id":
            student["Student_ID"],

        "school":
            student["school"],

        "sex":
            (
                "Female"
                if student["sex"] == "F"
                else "Male"
            ),

        "age":
            int(student["age"]),

        "studytime":
            int(student["studytime"]),

        "studytime_label":
            study_time_label(
                student["studytime"]
            ),

        "failures":
            int(student["failures"]),

        "absences":
            int(student["absences"]),

        "actual_grade":
            round(
                float(student["G3"]),
                2
            ),

        "predicted_grade":
            round(
                prediction,
                2
            ),

        "performance":
            category
    })


# ============================================================
# PREDICTION API
# Student ID based prediction
# ============================================================

@app.route("/api/predict", methods=["POST"])
def predict_api():

    data = request.get_json()

    if not data:

        return jsonify({
            "success": False,
            "message": "No input received."
        }), 400

    student_id = str(
        data.get("student_id", "")
    ).strip()

    if not student_id:

        return jsonify({
            "success": False,
            "message": "Please enter a Student ID."
        }), 400

    matches = df[
        df["Student_ID"].astype(str).str.upper()
        == student_id.upper()
    ]

    if matches.empty:

        return jsonify({
            "success": False,
            "message": "Student ID not found."
        }), 404

    student = matches.iloc[0].copy()

    prediction, category = predict_student(student)

    return jsonify({

        "success":
            True,

        "student_id":
            student["Student_ID"],

        "predicted_grade":
            round(
                prediction,
                2
            ),

        "performance":
            category,

        "actual_grade":
            round(
                float(student["G3"]),
                2
            ),

        "model":
            "Random Forest Regression",

        "r2":
            0.29,

        "mae":
            3.06,

        "rmse":
            3.82,

        "features":
            30,

        "leakage_note":
            "G1 and G2 were excluded to prevent target leakage."
    })


# ============================================================
# WHAT-IF SIMULATOR
# ============================================================

@app.route("/api/whatif", methods=["POST"])
def whatif_api():

    data = request.get_json()

    if not data:

        return jsonify({
            "success": False,
            "message": "No input received."
        }), 400

    student_id = str(
        data.get("student_id", "")
    ).strip()

    matches = df[
        df["Student_ID"].astype(str).str.upper()
        == student_id.upper()
    ]

    if matches.empty:

        return jsonify({
            "success": False,
            "message": "Student ID not found."
        }), 404

    original_student = matches.iloc[0].copy()

    # --------------------------------------------------------
    # Original prediction
    # --------------------------------------------------------

    original_prediction, original_category = (
        predict_student(original_student)
    )

    # --------------------------------------------------------
    # Create modified student
    # --------------------------------------------------------

    modified_student = original_student.copy()

    # --------------------------------------------------------
    # Study time
    # --------------------------------------------------------

    if data.get("studytime") not in [
        None,
        "",
        "All"
    ]:

        try:

            modified_student["studytime"] = int(
                data["studytime"]
            )

        except (ValueError, TypeError):

            pass

    # --------------------------------------------------------
    # Previous failures
    # --------------------------------------------------------

    if data.get("failures") not in [
        None,
        "",
        "All"
    ]:

        try:

            modified_student["failures"] = int(
                data["failures"]
            )

        except (ValueError, TypeError):

            pass

    # --------------------------------------------------------
    # Absences
    # --------------------------------------------------------

    if data.get("absences") not in [
        None,
        ""
    ]:

        try:

            modified_student["absences"] = max(
                0,
                int(data["absences"])
            )

        except (ValueError, TypeError):

            pass

    # --------------------------------------------------------
    # New prediction
    # --------------------------------------------------------

    new_prediction, new_category = (
        predict_student(modified_student)
    )

    difference = (
        new_prediction
        - original_prediction
    )

    return jsonify({

        "success":
            True,

        "student_id":
            student_id,

        "original_prediction":
            round(
                original_prediction,
                2
            ),

        "new_prediction":
            round(
                new_prediction,
                2
            ),

        "difference":
            round(
                difference,
                2
            ),

        "original_category":
            original_category,

        "new_category":
            new_category,

        "studytime":
            int(
                modified_student["studytime"]
            ),

        "failures":
            int(
                modified_student["failures"]
            ),

        "absences":
            int(
                modified_student["absences"]
            )
    })


# ============================================================
# SHAP
# ============================================================

@app.route("/api/shap")
def shap_api():

    shap_path = os.path.join(
        BASE_DIR,
        "static",
        SHAP_IMAGE
    )

    exists = os.path.exists(shap_path)

    return jsonify({

        "success":
            exists,

        "image":
            f"/static/{SHAP_IMAGE}"
            if exists
            else None,

        "message":
            (
                "SHAP explanation available."
                if exists
                else
                "SHAP image not found."
            )
    })


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/api/status")
def status():

    return jsonify({

        "status":
            "online",

        "model":
            "Random Forest Regression",

        "features":
            30,

        "target":
            "G3",

        "student_id":
            "Identification only",

        "leakage":
            "G1/G2 excluded"
    })


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("StudentIQ | Student Performance Analysis")
    print("=" * 60)
    print("Model: Random Forest Regression")
    print("Model features: 30")
    print("Student_ID: Identification only")
    print("G1/G2: Excluded to prevent target leakage")
    print(f"Students loaded: {len(df)}")
    print()
    print("Website:")
    print("http://127.0.0.1:5000/")
    print("=" * 60)

    app.run(
        debug=True,
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                5000
            )
        )
    )