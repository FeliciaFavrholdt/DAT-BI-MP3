import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Employee Behavior Prediction", layout="centered")
st.title("🔮 Predict Employee Behavior")

# Sidebar: Choose model
model_type = st.selectbox("Select Model Type", [
    "Regression (Predict Income)",
    "Classification (Predict Attrition)",
    "Clustering (Segment Employee)"
])

# Input form
st.subheader("Enter Employee Information")
age = st.slider("Age", 18, 60, 30)
distance = st.slider("Distance from Home (km)", 1, 30, 5)
satisfaction = st.slider("Job Satisfaction", 1, 4, 3)
performance = st.slider("Performance Rating", 1, 4, 3)
years_at_company = st.slider("Years at Company", 0, 40, 5)
income = st.number_input("Monthly Income", min_value=1000, value=5000)

# Build input DataFrame
input_data = {
    "Age": age,
    "DistanceFromHome": distance,
    "JobSatisfaction": satisfaction,
    "PerformanceRating": performance,
    "YearsAtCompany": years_at_company,
    "MonthlyIncome": income
}
input_df = pd.DataFrame([input_data])

# Predict button
if st.button("Run Prediction"):
    try:
        if "Regression" in model_type:
            model = joblib.load("models/regression_pipeline.pkl")
            reg_input = input_df.drop(columns=["MonthlyIncome"])
            prediction = model.predict(reg_input)[0]
            st.success(f"💰 Predicted Monthly Income: ${prediction:,.2f}")

        elif "Classification" in model_type:
            model = joblib.load("models/classifier_pipeline.pkl")
            clf_input = input_df.drop(columns=["MonthlyIncome"])
            prediction = model.predict(clf_input)[0]
            probability = model.predict_proba(clf_input)[0][1]
            label = "Yes" if prediction == 1 else "No"
            st.success(f"⚠️ Attrition Prediction: {label} ({probability:.1%} probability)")

        elif "Clustering" in model_type:
            model = joblib.load("models/cluster_pipeline.pkl")
            cluster = model.predict(input_df)[0]
            st.success(f"👥 Employee belongs to Cluster: {cluster}")

        else:
            st.error("Unknown model type selected.")

    except FileNotFoundError:
        st.error("Model file not found. Please ensure the model is trained and saved.")
    except Exception as e:
        st.error(f"Something went wrong: {e}")
