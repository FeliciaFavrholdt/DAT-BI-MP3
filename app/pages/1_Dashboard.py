import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

st.set_page_config(layout="wide")
st.title("📊 IBM HR Analytics Dashboard")

@st.cache_data
def load_data():
    return pd.read_csv("data/IBM_HR_Employee_Attrition.csv")

df = load_data()

# Select features for clustering
cluster_features = ['Age', 'DistanceFromHome', 'MonthlyIncome', 'JobSatisfaction', 'PerformanceRating', 'YearsAtCompany']
df_cluster = df[cluster_features].copy()

# Scale features
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_cluster)

# Load trained clustering model
try:
    cluster_model = joblib.load("app/models/cluster_pipeline.pkl")
    clusters = cluster_model.predict(df_cluster)
except Exception as e:
    st.warning("Clustering model not found. Dummy cluster labels used.")
    clusters = [0] * len(df)

# PCA transformation
pca = PCA(n_components=2)
components = pca.fit_transform(df_scaled)

# Add cluster labels to DataFrame for further use
df["Cluster"] = clusters

# Employee cluster visualization
st.subheader("🧭 Employee Clusters (PCA Projection)")
fig = px.scatter(
    x=components[:, 0], y=components[:, 1],
    color=df["Cluster"].astype(str),  # Ensure labels are str for Plotly
    labels={'x': 'PC1', 'y': 'PC2', 'color': 'Cluster'},
    title="Employee Segmentation by Cluster (PCA)"
)
st.plotly_chart(fig)

# Key Metrics
st.subheader("📌 Key HR Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Employees", df.shape[0])
col2.metric("Attrition Rate", f"{(df['Attrition'] == 'Yes').mean()*100:.2f}%")
col3.metric("Avg. Monthly Income", f"${df['MonthlyIncome'].mean():,.0f}")

st.markdown("---")

# Attrition by Department
st.subheader("📉 Attrition by Department")
fig1 = sns.countplot(data=df, x="Department", hue="Attrition")
st.pyplot(fig1.figure)

# Monthly Income by Job Role
st.subheader("💼 Monthly Income by Job Role")
fig2 = sns.boxplot(data=df, x="JobRole", y="MonthlyIncome")
plt.xticks(rotation=45)
st.pyplot(fig2.figure)

# Correlation heatmap
st.subheader("📈 Feature Correlation Heatmap")
fig3, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(df.select_dtypes(include='number').corr(), cmap='coolwarm', annot=True, ax=ax)
st.pyplot(fig3)

st.markdown("---")
st.caption("Built with ❤️ for Mini Project 3 — CPHBusiness")
