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

@st.cache_data
def load_data():
    return pd.read_csv("data/IBM_HR_Employee_Attrition.csv")

df = load_data()

cluster_features = ['Age', 'DistanceFromHome', 'MonthlyIncome', 'JobSatisfaction', 'PerformanceRating', 'YearsAtCompany']
df_cluster = df[cluster_features].copy()

# Load clustering model
try:
    cluster_model = joblib.load(Path(__file__).resolve().parent.parent / "models" / "cluster_pipeline.pkl")
    clusters = cluster_model.predict(df_cluster)
    df["Cluster"] = clusters
except Exception as e:
    st.warning("Clustering model not found or failed to load.")
    df["Cluster"] = [0] * len(df)

# PCA transformation
pca = PCA(n_components=2)
pca_components = pca.fit_transform(df_cluster)
df["PC1"] = pca_components[:, 0]
df["PC2"] = pca_components[:, 1]

cluster_labels = {
    0: "Cluster 0 — Young & Low Income",
    1: "Cluster 1 — Experienced & Stable",
    2: "Cluster 2 — At-Risk & Dissatisfied"
}
df["Cluster Name"] = df["Cluster"].map(cluster_labels)

# Filters
st.sidebar.header("Filter Data")
selected_cluster = st.sidebar.multiselect("Cluster", sorted(df["Cluster"].unique()), default=sorted(df["Cluster"].unique()))
selected_department = st.sidebar.multiselect("Department", sorted(df["Department"].unique()), default=sorted(df["Department"].unique()))
selected_gender = st.sidebar.multiselect("Gender", sorted(df["Gender"].unique()), default=sorted(df["Gender"].unique()))

df_filtered = df[
    df["Cluster"].isin(selected_cluster) &
    df["Department"].isin(selected_department) &
    df["Gender"].isin(selected_gender)
]

# PCA plot
st.subheader("Employee Clusters (PCA Projection)")
fig = px.scatter(
    df_filtered,
    x="PC1", y="PC2",
    color="Cluster Name",
    hover_data=["Age", "MonthlyIncome", "JobSatisfaction", "Attrition"],
    labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'},
    title="Employee Segmentation by Cluster (PCA Projection)"
)
st.plotly_chart(fig)

# Cluster summary
st.subheader("Cluster Summary Statistics")
st.dataframe(df_filtered.groupby("Cluster")[cluster_features].mean().round(1))

# HR Metrics
st.subheader("Key HR Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Employees", df_filtered.shape[0])
col2.metric("Attrition Rate", f"{(df_filtered['Attrition'] == 'Yes').mean()*100:.2f}%")
col3.metric("Average Monthly Income", f"${df_filtered['MonthlyIncome'].mean():,.0f}")

st.markdown("---")

# Attrition by department
st.subheader("Attrition by Department")
fig1 = sns.countplot(data=df_filtered, x="Department", hue="Attrition")
st.pyplot(fig1.figure)

# Monthly income by role
st.subheader("Monthly Income by Job Role")
fig2 = sns.boxplot(data=df_filtered, x="JobRole", y="MonthlyIncome")
plt.xticks(rotation=45)
st.pyplot(fig2.figure)

# Correlation heatmap
st.subheader("Feature Correlation Heatmap")
fig3, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(df_filtered.select_dtypes(include='number').corr(), cmap='coolwarm', annot=True, ax=ax)
st.pyplot(fig3)

st.markdown("---")

# Attrition by cluster
st.subheader("Attrition Rate by Cluster")
attrition_by_cluster = df_filtered.groupby("Cluster")["Attrition"].apply(lambda x: (x == "Yes").mean() * 100).round(1)
st.bar_chart(attrition_by_cluster)

# Cluster descriptions
for cluster_id, desc in cluster_labels.items():
    if cluster_id in df_filtered["Cluster"].unique():
        st.markdown(f"**{desc}**")

# Download CSV
st.markdown("### Download Filtered Data")
csv = df_filtered.to_csv(index=False).encode('utf-8')
st.download_button("Download as CSV", data=csv, file_name="filtered_hr_clusters.csv", mime="text/csv")

st.caption("Built for Mini Project 3 — CPHBusiness")
