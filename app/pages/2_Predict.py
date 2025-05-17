import streamlit as st
import pandas as pd
import joblib

st.title("🔮 Predict Employee Behavior")

# Model selection
model_type = st.selectbox("Select a Model", ["Regression (Income)", "Classification (Attrition)", "Clustering (Segments)"])

# Input features common to models
st.subheader("Enter Employee Data")

age = st.slider("Age", 18, 60, 30)
distance = st.slider("Distance from Home", 1, 30, 5)
satisfaction = st.slider("Job Satisfaction", 1, 4, 3)
performance = st.slider("Performance Rating", 1, 4, 3)
years_at_company = st.slider("Years at Company", 0, 40, 5)
income = st.number_input("Monthly Income (Required for Clustering)", min_value=1000, value=5000)

# Prepare input for prediction
input_data = {
    "Age": age,
    "DistanceFromHome": distance,
    "JobSatisfaction": satisfaction,
    "PerformanceRating": performance,
    "YearsAtCompany": years_at_company,
    "MonthlyIncome": income
}

input_df = pd.DataFrame([input_data])

# Prediction logic
if st.button("Run Prediction"):
    try:
        if "Regression" in model_type:
            # Use regression pipeline
            model = joblib.load("app/models/regression_pipeline.pkl")
            reg_input = input_df.drop(columns=["MonthlyIncome"])  # Regression doesn't use MonthlyIncome as input
            prediction = model.predict(reg_input)[0]
            st.success(f"💰 Predicted Monthly Income: ${prediction:,.2f}")

        elif "Classification" in model_type:
            model = joblib.load("app/models/classifier_pipeline.pkl")  # Make sure this is saved correctly
            clf_input = input_df.drop(columns=["MonthlyIncome"])  # Assume income wasn't used for attrition
            prediction = model.predict(clf_input)[0]
            probability = model.predict_proba(clf_input)[0][1]
            st.success(f"⚠️ Attrition Prediction: {'Yes' if prediction else 'No'} ({probability:.1%} probability)")

        elif "Clustering" in model_type:
            model = joblib.load("app/models/cluster_pipeline.pkl")  # Assume you saved KMeans pipeline
            cluster = model.predict(input_df)[0]
            st.success(f"👥 Employee belongs to Cluster: {cluster}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
