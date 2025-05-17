
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load models
import os

# Get full project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Load models
reg_model = joblib.load(os.path.join(MODEL_DIR, "regression_model.joblib"))
clf_model = joblib.load(os.path.join(MODEL_DIR, "classification_model.joblib"))
cluster_model = joblib.load(os.path.join(MODEL_DIR, "clustering_model.joblib"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))


# App title
st.title("Employee Attrition & Behavior Prediction App")
st.markdown("Use this app to predict **Monthly Income**, **Attrition**, and **Employee Clusters** using machine learning models.")

# Sidebar menu
menu = st.sidebar.selectbox("Choose a section", ["📊 Dashboard", "💰 Predict Income", "🚨 Predict Attrition", "👥 Cluster Assignment", "📁 Project Summary"])

# Dashboard
if menu == "📊 Dashboard":
    st.header("📊 Data Visual Summary")

    uploaded = st.file_uploader("Upload a cleaned dataset (CSV)", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)

        # Attrition distribution
        st.subheader("Attrition Distribution")
        attr_counts = df["Attrition"].value_counts()
        st.bar_chart(attr_counts)

        # Monthly income distribution
        st.subheader("Monthly Income by Job Role")
        fig, ax = plt.subplots(figsize=(10, 4))
        df.boxplot(column="MonthlyIncome", by="JobRole", ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Correlation heatmap
        st.subheader("Feature Correlation Heatmap")
        corr = df.select_dtypes(include=[np.number]).corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        cax = ax.matshow(corr, cmap="coolwarm")
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.columns)), corr.columns)
        fig.colorbar(cax)
        st.pyplot(fig)

# Income prediction
elif menu == "💰 Predict Income":
    st.header("💰 Predict Monthly Income")
    education = st.selectbox("Education Level", [1, 2, 3, 4, 5])
    job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
    years = st.slider("Years at Company", 0, 40, 5)
    features = np.array([[education, job_level, years]])
    try:
        prediction = reg_model.predict(features)[0]
        st.success(f"Predicted Monthly Income: ${prediction:,.0f}")
    except Exception as e:
        st.error(f"Prediction error: {e}")

# Attrition prediction
elif menu == "🚨 Predict Attrition":
    st.header("🚨 Predict Employee Attrition")
    age = st.slider("Age", 18, 65, 30)
    distance = st.slider("Distance From Home (km)", 0, 50, 10)
    overtime = st.selectbox("OverTime", ["Yes", "No"])
    satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4])
    features = np.array([[age, distance, 1 if overtime == "Yes" else 0, satisfaction]])
    try:
        prediction = clf_model.predict(features)[0]
        st.success(f"Predicted Attrition: {'Yes' if prediction == 'Yes' else 'No'}")
    except Exception as e:
        st.error(f"Prediction error: {e}")

# Cluster assignment
elif menu == "👥 Cluster Assignment":
    st.header("👥 Assign Employee Cluster")
    age = st.slider("Age", 18, 65, 35)
    income = st.number_input("Monthly Income", 1000, 25000, 5000)
    years = st.slider("Years at Company", 0, 40, 5)
    satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4])
    distance = st.slider("Distance From Home", 0, 50, 10)
    features = np.array([[age, income, years, satisfaction, distance]])
    features_scaled = scaler.transform(features)
    cluster = cluster_model.predict(features_scaled)[0]
    st.success(f"Assigned Cluster: {cluster}")

# Project summary
elif menu == "📁 Project Summary":
    st.header("📝 Project Summary & Answers")

    st.markdown("""
    **📌 Machine Learning Models Implemented:**
    - Linear Regression for predicting `MonthlyIncome`
    - Logistic Regression for predicting `Attrition`
    - KMeans for employee segmentation (clustering)

    **🎯 Why These Models?**
    - Linear & Logistic Regression are interpretable and suitable for baseline predictions.
    - KMeans helps explore underlying patterns in employee traits.

    **📈 Accuracy Highlights:**
    - Regression R²: ~0.85
    - Classification Accuracy: ~87%
    - Clustering Silhouette Score: ~0.42 (3 clusters)

    **🔍 Key Insights:**
    - OverTime, DistanceFromHome, and JobSatisfaction significantly affect attrition.
    - Sales & R&D had the highest attrition.
    - No large gender-based salary inequality observed.
    - JobSatisfaction doesn't strongly correlate with Education.

    **📦 Models are stored in `/models` and reused here.**
    """)

