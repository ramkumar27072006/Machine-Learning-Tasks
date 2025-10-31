Machine Learning Projects Collection

This repository contains two applied Machine Learning projects developed using Python. Both projects focus on building intelligent systems that perform predictive analysis using datasets and multiple supervised learning algorithms.

Task 1 — Credit Scoring Model

Objective  
To develop a Credit Scoring System that predicts an individual’s creditworthiness based on demographic, employment, and financial data.  
The model classifies applicants as good or bad credit risk, supporting data-driven lending decisions.

Dataset Overview  
Dataset: Credit Scoring Dataset (Kaggle) — 1,000 samples  
Key features include: Age, Gender, Marital Status, Education Level, Employment Status, Credit Utilization Ratio, Payment History, Number of Credit Accounts, Loan Amount, Interest Rate, Loan Term, and Type of Loan.

Methods Used  
Multiple ML algorithms were compared:  
- Logistic Regression  
- Decision Tree  
- Random Forest  
- Gradient Boosting  
- AdaBoost  
- SVM (RBF Kernel)  
- XGBoost  
Performance metrics: Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

Results Summary

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---------------------|----------|-----------|--------|----------|---------|
| Decision Tree       | 1.00     | 1.00      | 1.00   | 1.00     | 1.00    |
| Random Forest       | 1.00     | 1.00      | 1.00   | 1.00     | 1.00    |
| Gradient Boosting   | 1.00     | 1.00      | 1.00   | 1.00     | 1.00    |
| XGBoost             | 1.00     | 1.00      | 1.00   | 1.00     | 1.00    |
| AdaBoost            | 1.00     | 1.00      | 1.00   | 1.00     | 1.00    |
| Logistic Regression | 0.98     | 0.98      | 1.00   | 0.99     | 0.99    |
| SVM (RBF)           | 0.98     | 0.98      | 1.00   | 0.99     | 0.96    |

Best Model: Decision Tree (Accuracy = 100%)

Insights  
Tree-based ensemble models achieved perfect classification.  
Key predictive attributes: Payment History, Credit Utilization Ratio, and Loan Amount.  
The framework accurately distinguishes reliable vs risky borrowers.

Outputs  
- Confusion Matrices for each model  
- Combined Model Comparison Chart  
- Feature Importance Plot  
- ROC Curve Comparison

Task 2 — Disease Prediction from Medical Data

Objective  
To design a predictive system that determines whether a person is disease-positive (diabetic) or disease-negative based on medical parameters.  
The goal is to support early diagnosis and preventive healthcare using machine learning.

Dataset Overview  
Dataset: Pima Indians Diabetes Database (Kaggle) — 768 patient records  
Features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, and target variable Outcome.

Models Implemented  
- Logistic Regression  
- Random Forest  
- Gradient Boosting  
- AdaBoost  
- SVM (RBF Kernel)  
- XGBoost  
Models were trained and evaluated on accuracy, precision, recall, F1-score, and ROC-AUC.

Results

| Model             | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------------------|----------|-----------|--------|----------|---------|
| AdaBoost          | 0.766    | 0.688     | 0.611  | 0.647    | 0.824   |
| Random Forest     | 0.753    | 0.660     | 0.611  | 0.635    | 0.815   |
| Gradient Boosting | 0.747    | 0.647     | 0.611  | 0.629    | 0.832   |
| SVM (RBF)         | 0.747    | 0.647     | 0.611  | 0.629    | 0.793   |
| XGBoost           | 0.740    | 0.630     | 0.630  | 0.630    | 0.811   |
| Logistic Regression| 0.714    | 0.609     | 0.519  | 0.560    | 0.823   |

Best Model: AdaBoost Classifier (Accuracy = 0.766, ROC-AUC = 0.824)

Top Contributing Features

| Rank | Feature                | Importance |
|-------|------------------------|------------|
| 1     | Glucose                | 0.272      |
| 2     | DiabetesPedigreeFunction| 0.234     |
| 3     | BMI                    | 0.152      |
| 4     | Insulin                | 0.098      |
| 5     | Age                    | 0.097      |

Insights  
AdaBoost gave the best predictive performance.  
Glucose and genetic predisposition were the strongest indicators of disease.  
Ensemble models (Boosting & Random Forest) generalized better than linear methods.  
The trained model can accurately predict whether a patient is diabetic or healthy based on medical data.

Outputs  
- Target Distribution Plot  
- Model Comparison Graph  
- Confusion Matrix (AdaBoost)  
- Feature Importance Plot  
- ROC Curve Comparison

Technologies Used  
- Python (Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, XGBoost)  
- Jupyter / VS Code for experimentation  
- Machine Learning Algorithms: Decision Trees, Boosting, SVM, Logistic Regression  
- Visualization: ROC Curves, Feature Importance, Confusion Matrices

Overall Outcomes  
Built and evaluated multiple machine learning pipelines on datasets.  
Achieved high prediction accuracy across both financial and medical domains.  
Demonstrated proficiency in data preprocessing, model comparison, and result visualization.  
The frameworks are extendable for real-time prediction systems in healthcare and finance.
