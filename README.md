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

### Project Questions and Reflections

#### Which machine learning methods did you choose to apply in the application and why?

We selected different machine learning methods depending on the task. For the regression task, we used linear regression to predict monthly income, as it is simple, interpretable, and well-suited for continuous target variables. It helped us understand how factors like experience, education, and job level influence salary. For the classification task, where we predicted whether an employee would leave the company (attrition), we used Random Forest, as it can handle both categorical and numerical features and model complex non-linear relationships. Logistic regression was used as a baseline due to its simplicity and interpretability. For clustering, we applied K-Means, a widely used method for grouping employees into segments based on shared characteristics. The number of clusters was chosen using the highest silhouette score, which indicates how well-separated and cohesive the clusters are.

#### How accurate is your prediction solution? Explain the meaning of the quality measures.

Our classification model for predicting attrition achieved an accuracy between 85% and 90% on the test set. We used several quality measures to evaluate model performance: accuracy shows the proportion of correct predictions, precision indicates how many predicted leavers actually left, recall shows how many actual leavers were identified by the model, and F1-score balances precision and recall, which is especially important in imbalanced datasets. For the regression model predicting monthly income, we used R² and RMSE to assess performance. R² shows how much of the variance in income the model can explain, and RMSE (root mean squared error) shows the average deviation between predicted and actual values – the lower the RMSE, the better.

#### Which are the most decisive factors for quitting a job? Why do people quit their job?

The most important factors for leaving a job were low job satisfaction, limited career opportunities, low pay, long commute distances, and working overtime. Many employees tend to leave when they experience poor working conditions, lack of work-life balance, or no prospects for personal and professional development. Our model showed that variables like job satisfaction, distance from home, workload, and overtime were often closely related to attrition.

#### What could be done for further improvement of the accuracy of the models?

To improve model accuracy, we could collect more detailed and recent data, including qualitative inputs like employee feedback or satisfaction survey results. We could also include more relevant features, such as leadership quality, team dynamics, and psychological safety. More advanced models like neural networks could be explored, especially with a larger dataset. For the classification task, we could use techniques like SMOTE to handle class imbalance, and improve performance through hyperparameter tuning (e.g., using grid search or Bayesian optimization).

#### Which work positions and departments are in higher risk of losing employees?

Our analysis showed that employees in sales and customer service roles were more likely to leave, likely due to high demands, stress, and lower job satisfaction. Technical support and certain administrative roles also had higher turnover, possibly because these positions are routine-based, lower-paid, and offer fewer growth opportunities. Positions with high overtime and low internal mobility also appeared to carry higher attrition risk.

#### Are employees of different gender paid equally in all departments?

Our analysis showed some gender pay differences in certain departments, where men earned more on average than women in the same role. These differences were less visible in administrative positions but more pronounced in sales and leadership roles. While some of this may be explained by seniority or experience, potential gender bias in hiring or promotion cannot be ruled out. A deeper analysis is recommended to identify and address structural inequalities.

#### Do the family status and the distance from work influence the work-life balance?

Yes, the data clearly showed that employees with families, especially those with young children, experienced greater work-life balance challenges – particularly when combined with long commuting distances. This combination was linked to lower job satisfaction, higher stress, and an increased risk of considering leaving the company. Flexible schedules and remote work options could help support this group of employees.

#### Does education make people happy (satisfied with their work)?

In general, there was a slight trend showing that employees with higher education levels reported greater job satisfaction, likely due to having roles with more responsibility, better pay, and professional development opportunities. However, overqualification sometimes led to lower satisfaction, especially if the job did not match the employee’s skills or offered limited advancement. So while education can increase satisfaction, it depends heavily on role alignment.

#### Which were the challenges in the project development?

We encountered several challenges during the project. A major issue was the class imbalance in the classification task, since far more employees stayed than left. This required careful model evaluation and strategies to balance the classes. Another challenge was selecting meaningful and predictive features, which required domain understanding and experimentation with feature engineering. For clustering, determining the right number of clusters and interpreting them meaningfully also took time. Iterative testing, visualization, and team discussion helped us address these issues effectively.
