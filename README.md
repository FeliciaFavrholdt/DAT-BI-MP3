# IBM HR Attrition Prediction Project

### Created by Group 4 – Alberte & Felicia

#### We used pair programming to create this project. Felicia wrote the code while Alberte contributed through discussion and review.

## Objective

The objective of this mini project is to gain practical experience in data analysis and prediction using regression, classification, and clustering algorithms.

## Problem Statement

Attrition is the rate at which employees leave their jobs. High attrition rates can disrupt company operations and increase recruitment costs. Therefore, it's important to understand why employees leave and identify the most influential factors behind those decisions.

## Dataset

This project is based on the IBM HR Employee Attrition dataset, available on Kaggle:

https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset/data

## Tasks Completed

1. Data Wrangling and Exploration  
2. Supervised Learning – Linear Regression (Predicting Monthly Income)  
3. Supervised Learning – Classification (Predicting Attrition)  
4. Unsupervised Learning – Clustering (Employee Segmentation)  
5. Streamlit Web App (deployment-ready interface)  

## Technologies Used

- Python, Pandas, NumPy  
- Scikit-learn  
- Matplotlib, Seaborn  
- Streamlit (for UI deployment)

## Directory Structure

- `notebook/` – Jupyter notebooks for each task  
- `models/` – Saved ML models (for app use)  
- `data/` – Raw and cleaned datasets  
- `app/` – Streamlit front-end files  

## Project Summary

This project applied both supervised and unsupervised machine learning techniques to analyze and predict employee behavior.

We began with data wrangling and exploration, visualizing trends in job roles, compensation, and attrition. For regression, we built a model to predict monthly income using features like job level, experience, and role. The model achieved an R² score of 0.94.

For classification, we trained logistic regression (with and without class weighting) and a random forest to predict whether an employee would leave. We evaluated models using accuracy, recall, F1-score, and ROC-AUC. Balanced logistic regression improved recall to 66%, while random forest achieved strong AUC performance (0.78).

In the final task, we used K-Means clustering to segment employees into 8 groups based on similarities. Using PCA and silhouette scores, we confirmed well-separated clusters. We labeled each group (e.g., "Young Entry-Level", "Experienced Managers") to help interpret the segments. These insights can support HR decisions around engagement, compensation, and retention.

## Key Findings

- Overtime, job role, and total working years are strong predictors of both salary and attrition.
- Balanced logistic regression improved recall significantly in detecting attrition risk.
- Random Forest captured complex patterns and provided interpretable feature importance.
- Employees naturally segmented into 8 groups with distinct profiles in terms of age, experience, and salary.
- Visual tools like ROC curves and PCA helped validate model performance and clustering quality.

## Questions Answered

- Which machine learning methods did you choose to apply and why?  
  We used linear regression for salary prediction, logistic regression and random forest for attrition classification, and K-Means for clustering. These methods were chosen for their balance of interpretability, performance, and suitability for structured data.

- How accurate is your prediction solution?  
  The regression model achieved an R² score of 0.94. The balanced logistic regression model reached 87% accuracy and 66% recall, offering reliable predictions of attrition.

- Which are the most decisive factors for quitting a job?  
  OverTime, JobRole, JobLevel, and TotalWorkingYears were key factors. Overtime in particular was highly correlated with attrition.

- What could be done to further improve model accuracy?  
  Techniques like hyperparameter tuning, SMOTE resampling, ensemble methods, or using additional features (like performance reviews or survey results) could further improve accuracy and recall.

- Which work positions and departments are at higher risk of losing employees?  
  Sales Representatives and Research Scientists showed higher attrition rates. Sales and Human Resources departments also had above-average turnover.

- Are employees of different genders paid equally in all departments?  
  On average, salaries between genders were comparable, but some roles showed variation, likely due to differing role distributions.

- Do the family status and distance from work influence work-life balance?  
  Single employees and those with longer commutes reported lower work-life balance, and these factors aligned with higher attrition.

- Does education make people more satisfied at work?  
  Slight increases in satisfaction were noted with higher education, but the effect was modest and not strongly correlated with attrition.

- What were the main challenges during the project?  
  Handling class imbalance for classification, selecting meaningful features, and ensuring proper data preprocessing for clustering were key challenges. Iterative testing, visualization, and collaboration helped resolve these issues.