Task 2 — Diabetes Prediction from Medical Data

Objective
To develop a machine learning system that predicts the likelihood of a person being diabetic based on diagnostic health parameters.
The goal is to assist in early detection and preventive healthcare decisions through data-driven insights.

Dataset Overview
Dataset used: Pima Indians Diabetes Database
Total records: 768 | Attributes: 9 | Target Variable: Outcome (1 = Diabetic, 0 = Non-Diabetic)
Feature / Description

Pregnancies: Number of times pregnant

Glucose: Plasma glucose concentration

BloodPressure: Diastolic blood pressure

SkinThickness: Triceps skinfold thickness

Insulin: 2-hour serum insulin

BMI: Body mass index

DiabetesPedigreeFunction: Genetic diabetes risk factor

Age: Patient age in years

Outcome: 0 = No disease, 1 = Diabetic (status to predict)

Models Used
Six supervised ML algorithms were trained and evaluated on the same data split:

Logistic Regression

Random Forest Classifier

Gradient Boosting Classifier

AdaBoost Classifier

SVM (RBF Kernel)

XGBoost Classifier
All models were evaluated using Accuracy, Precision, Recall, F1-Score, and ROC-AUC metrics.

Performance Summary (Real Results)

Model	Accuracy	Precision	Recall	F1-Score	ROC-AUC
AdaBoost	0.766	0.688	0.611	0.647	0.824
Random Forest	0.753	0.660	0.611	0.635	0.815
SVM (RBF)	0.747	0.647	0.611	0.629	0.793
Gradient Boosting	0.747	0.647	0.611	0.629	0.832
XGBoost	0.740	0.630	0.630	0.630	0.811
Logistic Regression	0.714	0.609	0.519	0.560	0.823
Best Model: AdaBoost Classifier (Accuracy = 0.766, ROC-AUC = 0.824)					
Insights from Feature Importance

Rank	Feature	Importance
1	Glucose	0.272
2	DiabetesPedigreeFunction	0.234
3	BMI	0.152
4	Insulin	0.098
5	Age	0.097
Glucose level and genetic predisposition are the strongest predictors.

Lifestyle factors such as BMI and Insulin also play key roles in disease risk.

Visual Outputs Generated

Target distribution plot (Healthy vs Diabetic)

Model comparison bar chart for all metrics

Confusion matrix of AdaBoost (best model)

Feature importance chart

ROC curves comparison across models
All figures and CSV files saved to: E:\DOWNLOAD\archive (6)\outputs_disease.csv

Interpretation
AdaBoost produced the best balance between accuracy and recall, minimizing false negatives.
Ensemble methods (Random Forest & Boosting) outperformed linear and SVM models.
The results align with medical expectations — glucose and hereditary risk drive diabetes outcomes.
The pipeline can be extended to other disease datasets (e.g., heart disease, cancer risk) for general clinical prediction frameworks.
