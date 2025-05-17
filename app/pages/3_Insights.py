import streamlit as st

st.title("📘 Project Insights & Documentation")

st.markdown("""
### Overview
This application analyzes employee attrition using IBM's HR dataset and applies machine learning for prediction and segmentation.

### Models Used
- **Regression**: Predicts Monthly Income
- **Classification**: Predicts Attrition (Yes/No)
- **Clustering**: Segments employees into similar groups

### Key Findings
- **Top attrition factors**: Distance from home, job satisfaction, performance rating
- **Departments at risk**: Sales, Human Resources
- **Satisfaction vs Performance**: Positive correlation
- **Family status** and **commute distance** impact work-life balance

### Model Metrics
- **R² (Regression)**: ~0.82
- **Classification Accuracy**: ~88%
- **Best Silhouette Score**: ~0.61 (3 clusters)

---

_This is part of Mini Project 3: Machine Learning for Analysis & Prediction_
""")
