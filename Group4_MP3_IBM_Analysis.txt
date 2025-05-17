# IBM HR Attrition Prediction Project

### Github Link: https://github.com/FeliciaFavrholdt/DAT-BI-MP3/tree/main 

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

### Project Questions and Reflections

#### Which machine learning methods did you choose to apply in the application and why?
We used different machine learning methods depending on the task type. For regression, we applied linear regression to predict monthly income. This method was chosen because it is interpretable and well-suited to continuous target variables. It helped us understand how features such as total working years, job level, and department influence salary. For the classification task, where we predicted whether an employee would leave the company (attrition), we used both logistic regression and Random Forest. Logistic regression served as a simple and interpretable baseline, while Random Forest was chosen for its ability to handle complex, non-linear patterns and provide feature importance insights. We also trained a balanced version of logistic regression using `class_weight='balanced'` to handle class imbalance. For clustering, we used K-Means to segment employees based on similarity in their profiles. The number of clusters was selected based on the highest silhouette score.

#### How accurate is your prediction solution? Explain the meaning of the quality measures.
Our regression model for predicting monthly income achieved an R² score of 0.94, indicating that it explains 94% of the variation in salary. For classification, the logistic regression model achieved an accuracy of 87%, but recall was initially low. After balancing the class weights, recall increased to 66%, which means the model became better at identifying employees who actually left. We also used precision, recall, and F1-score to assess performance. Precision tells us how many predicted leavers were correct, recall tells us how many actual leavers were captured, and the F1-score balances both. For model comparison, we also used ROC curves and AUC scores. The Random Forest and balanced logistic regression models both showed strong AUC scores around 0.80.

#### Which are the most decisive factors for quitting a job? Why do people quit their job?
The most decisive factors for leaving the company were overtime, job role, total working years, and marital status. Employees who regularly worked overtime had a significantly higher risk of attrition. Single employees and those in certain roles like Sales Representative and Research Scientist were also more likely to leave. These results suggest that both workload and personal circumstances contribute to attrition risk. Our classification models consistently highlighted overtime and job role as key predictors.

#### What could be done for further improvement of the accuracy of the models?
To further improve model accuracy, we could include more features such as satisfaction survey responses, performance ratings over time, or qualitative data like employee feedback. We could also experiment with more advanced models such as Gradient Boosting or XGBoost, and use hyperparameter tuning (e.g. GridSearchCV) to optimize performance. In the classification task, oversampling techniques like SMOTE could be applied to balance the dataset and improve the model’s ability to detect true attrition cases.

#### Which work positions and departments are in higher risk of losing employees?

Our analysis showed that employees in sales and research roles, particularly Sales Representatives and Research Scientists, had the highest attrition rates. These roles often involve high demands, overtime, or unclear growth paths. Department-wise, Sales and Human Resources showed more attrition compared to Research & Development. These trends suggest that job stress, role clarity, and work-life balance are important factors influencing turnover.

#### Are employees of different gender paid equally in all departments?
In general, we observed that male and female employees earned similar salaries on average. However, some small differences appeared in specific roles and departments. These differences could be influenced by job role distributions, experience levels, or promotions. To fully assess pay equity, a more detailed analysis would be required that controls for these variables.

#### Do the family status and the distance from work influence the work-life balance?
Yes. Our data exploration showed that single employees and those with longer commutes tended to have lower work-life balance ratings. These factors also correlated with higher attrition. This suggests that personal life circumstances, combined with logistical challenges like distance from home, influence job satisfaction and turnover. Addressing these challenges through flexible work arrangements could help reduce attrition.

#### Does education make people happy (satisfied with their work)?
We observed a mild trend where employees with higher education levels had slightly better job satisfaction scores. This may be due to these employees holding roles with more responsibility or higher income. However, the effect was not very strong, and overqualification might still lead to dissatisfaction if the job does not match the person’s skills or expectations. So while education may contribute to satisfaction, job fit plays a critical role.

#### Which were the challenges in the project development?
One major challenge was handling the class imbalance in the attrition classification task. Because only a small portion of employees had actually left the company, we had to use techniques like class weighting and careful evaluation metrics to get a realistic model. Another challenge was interpreting clustering results and selecting a meaningful number of clusters. We addressed this by using silhouette scores and PCA to evaluate and visualize the clusters. Overall, combining technical modeling with business understanding required a lot of iteration, testing, and collaboration.