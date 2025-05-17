import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA
import joblib
from pathlib import Path

st.set_page_config(layout="wide")
st.title("IBM HR Analytics Dashboard")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("data/IBM_HR_Employee_Attrition.csv")

df = load_data()

# Define clustering features
cluster_features = ['Age', 'DistanceFromHome', 'MonthlyIncome', 'JobSatisfaction', 'PerformanceRating', 'YearsAtCompany']
df_cluster = df[cluster_features].copy()

# Load trained clustering pipeline
try:
    cluster_model = joblib.load(Path(__file__).resolve().parent.parent / "models" / "cluster_pipeline.pkl")
    clusters = cluster_model.predict(df_cluster)
    df["Cluster"] = clusters
except Exception as e:
    st.warning("Clustering model not found or failed to load.")
    df["Cluster"] = [0] * len(df)

# PCA for visualization
pca = PCA(n_components=2)
pca_components = pca.fit_transform(df_cluster)

# Descriptive cluster names
cluster_labels = {
    0: "Cluster 0 — Young & Low Income",
    1: "Cluster 1 — Experienced & Stable",
    2: "Cluster 2 — At-Risk & Dissatisfied"
}
df["Cluster Name"] = df["Cluster"].map(cluster_labels)

# Cluster scatter plot
st.subheader("Employee Clusters (PCA Projection)")
fig = px.scatter(
    x=pca_components[:, 0],
    y=pca_components[:, 1],
    color=df["Cluster Name"],
    labels={'x': 'PC1', 'y': 'PC2', 'color': 'Cluster'},
    title="Employee Segmentation by Cluster (PCA)"
)
st.plotly_chart(fig)

# Moved here: Cluster summary table
st.subheader("Cluster Summary Statistics")
st.dataframe(df.groupby("Cluster")[cluster_features].mean().round(1))

# HR Metrics
st.subheader("Key HR Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Employees", df.shape[0])
col2.metric("Attrition Rate", f"{(df['Attrition'] == 'Yes').mean()*100:.2f}%")
col3.metric("Average Monthly Income", f"${df['MonthlyIncome'].mean():,.0f}")

st.markdown("---")

# Attrition by Department
st.subheader("Attrition by Department")
fig1 = sns.countplot(data=df, x="Department", hue="Attrition")
st.pyplot(fig1.figure)

# Monthly Income by Job Role
st.subheader("Monthly Income by Job Role")
fig2 = sns.boxplot(data=df, x="JobRole", y="MonthlyIncome")
plt.xticks(rotation=45)
st.pyplot(fig2.figure)

# Correlation heatmap
st.subheader("Feature Correlation Heatmap")
fig3, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(df.select_dtypes(include='number').corr(), cmap='coolwarm', annot=True, ax=ax)
st.pyplot(fig3)

st.markdown("---")

# Attrition by cluster
st.subheader("Attrition Rate by Cluster")
attrition_by_cluster = df.groupby("Cluster")["Attrition"].apply(lambda x: (x == "Yes").mean() * 100).round(1)
st.bar_chart(attrition_by_cluster)

# Cluster descriptions
for cluster_id, desc in cluster_labels.items():
    if cluster_id in df["Cluster"].unique():
        st.markdown(f"**{desc}**")

st.caption("Built for Mini Project 3 — CPHBusiness")
