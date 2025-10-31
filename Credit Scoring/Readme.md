Task 1 — Credit Scoring Model
Objective
To build a machine learning model capable of evaluating an individual’s creditworthiness based on personal, financial, and loan-related parameters.
The model aims to predict whether a person is a good or bad credit risk, supporting data-driven lending decisions.

Dataset Overview
The dataset contains 1,000 customer records with the following key attributes:
Feature / Description
Age — Age of the applicant
Gender — Male or Female
Marital Status — Single / Married
Education Level — Level of education attained
Employment Status — Type of employment
Credit Utilization Ratio — Percentage of used credit limit
Payment History — Previous repayment behavior
Number of Credit Accounts — Total active credit accounts
Loan Amount — Principal amount borrowed
Interest Rate — Applicable interest rate
Loan Term — Loan duration in months
Type of Loan — Personal, Auto, or other loan types

Methods Used
A comparative machine learning approach was implemented using multiple classification algorithms to evaluate credit scoring performance.
Models Trained:

Logistic Regression

Decision Tree Classifier

Random Forest Classifier

Gradient Boosting Classifier

AdaBoost Classifier

Support Vector Machine (RBF Kernel)

XGBoost Classifier
Each model was evaluated using:
Accuracy
Precision
Recall
F1-Score
ROC-AUC Score

Results Summary
Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC
Decision Tree | 1.00 | 1.00 | 1.00 | 1.00 | 1.00
Random Forest | 1.00 | 1.00 | 1.00 | 1.00 | 1.00
Gradient Boosting | 1.00 | 1.00 | 1.00 | 1.00 | 1.00
XGBoost | 1.00 | 1.00 | 1.00 | 1.00 | 1.00
AdaBoost | 1.00 | 1.00 | 1.00 | 1.00 | 1.00
Logistic Regression | 0.98 | 0.98 | 1.00 | 0.9899 | 0.9872
SVM (RBF) | 0.98 | 0.98 | 1.00 | 0.9899 | 0.9630
Best Model: Decision Tree Classifier (Accuracy = 1.00)

Visual Outputs
Confusion Matrices for each model
Combined Bar Chart showing metric comparisons
Feature Importance plot (highlighting top financial indicators)
ROC Curves for all models

Insights
All tree-based ensemble models performed exceptionally well, achieving near-perfect results.
The most influential features in predicting creditworthiness were Payment History, Credit Utilization Ratio, and Loan Amount.
Logistic Regression and SVM models also performed strongly, confirming the data’s clean linear-separable patterns.
Gradient-boosted trees (XGBoost, GBM) demonstrated consistent predictive stability with minimal overfitting.
