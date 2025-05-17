import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("📊 IBM HR Analytics Dashboard")

@st.cache_data
def load_data():
    return pd.read_csv("data/IBM_HR_Employee_Attrition.csv")

df = load_data()

st.subheader("Key Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Employees", df.shape[0])
col2.metric("Attrition Rate", f"{(df['Attrition'] == 'Yes').mean()*100:.2f}%")
col3.metric("Avg. Monthly Income", f"${df['MonthlyIncome'].mean():,.0f}")

st.markdown("---")

st.subheader("Attrition by Department")
fig1 = sns.countplot(data=df, x="Department", hue="Attrition")
st.pyplot(fig1.figure)

st.subheader("Monthly Income by Job Role")
fig2 = sns.boxplot(data=df, x="JobRole", y="MonthlyIncome")
plt.xticks(rotation=45)
st.pyplot(fig2.figure)

st.subheader("Correlation Heatmap")
fig3, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(df.select_dtypes(include='number').corr(), cmap='coolwarm', annot=True, ax=ax)
st.pyplot(fig3)
