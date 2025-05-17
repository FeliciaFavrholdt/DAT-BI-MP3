import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Dynamically resolve base path
BASE_DIR = Path(__file__).resolve().parent.parent  # from /app/pages → /app
MODELS_DIR = BASE_DIR / "models"

st.set_page_config(page_title="Employee Behavior Prediction", layout="centered")
st.title("🔮 Predict Employee Behavior")

# Sidebar: Choose model
model_type = st.selectbox("Select Model Type", [
    "Regression (Predict Income)",
    "Classification (Predict Attrition)",
    "Clustering (Segment Employee)"
])

# Input form (including all clustering features)
st.subheader("Enter Employee Information")
age = st.slider("Age", 18, 60, 30)
distance = st.slider("Distance from Home (km)", 1, 30, 5)
income = st.number_input("Monthly Income", min_value=1000, value=5000)
satisfaction = st.slider("Job Satisfaction", 1, 4, 3)
performance = st.slider("Performance Rating", 1, 4, 3)
years_at_company = st.slider("Years at Company", 0, 40, 5)

# This matches the trained pipeline feature order
input_data = {
    "Age": age,
    "DistanceFromHome": distance,
    "MonthlyIncome": income,
    "JobSatisfaction": satisfaction,
    "PerformanceRating": performance,
    "YearsAtCompany": years_at_company
}
input_df = pd.DataFrame([input_data])

# Predict button
if st.button("Run Prediction"):
    try:
        if "Regression" in model_type:
            model = joblib.load(MODELS_DIR / "regression_pipeline.pkl")
            reg_input = input_df.drop(columns=["MonthlyIncome"])
            prediction = model.predict(reg_input)[0]
            st.success(f"💰 Predicted Monthly Income: ${prediction:,.2f}")

        elif "Classification" in model_type:
            model = joblib.load(MODELS_DIR / "classifier_pipeline.pkl")
            clf_input = input_df.drop(columns=["MonthlyIncome"])
            prediction = model.predict(clf_input)[0]
            probability = model.predict_proba(clf_input)[0][1]
            label = "Yes" if prediction == 1 else "No"
            st.success(f"⚠️ Attrition Prediction: {label} ({probability:.1%} probability)")

        elif "Clustering" in model_type:
            model = joblib.load(MODELS_DIR / "cluster_pipeline.pkl")
            cluster = model.predict(input_df)[0]
            st.success(f"👥 Employee belongs to Cluster: {cluster}")

            # Add cluster label explanation here, properly indented
            cluster_labels = {
                0: "Young, high-satisfaction, low-income group",
                1: "Experienced employees with stable performance",
                2: "At-risk group: long commute, low satisfaction"
            }
            description = cluster_labels.get(cluster, "Unknown segment")
            st.info(f"🧠 Cluster Insight: {description}")

        else:
            st.error("Unknown model type selected.")

    except FileNotFoundError:
        st.error("🔍 Model file not found. Please ensure it is saved in `app/models/`.")
    except Exception as e:
        st.error(f"❌ Something went wrong: {e}")
