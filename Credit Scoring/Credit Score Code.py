import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


DATA_PATH = r"E:\DOWNLOAD\archive (5)\credit_scoring.csv"   # <-- update this
df = pd.read_csv(DATA_PATH)
print("Dataset Loaded Successfully\n")
print(df.head())
print("\nShape:", df.shape)


df.dropna(inplace=True)

if "Credit Status" not in df.columns:
    df["Credit Status"] = np.where(df["Payment History"] >= 80, 1, 0)

cat_cols = ["Gender", "Marital Status", "Education Level", "Employment Status", "Type of Loan"]
encoder = LabelEncoder()
for col in cat_cols:
    df[col] = encoder.fit_transform(df[col])

X = df[[
    "Age", "Gender", "Marital Status", "Education Level", "Employment Status",
    "Credit Utilization Ratio", "Payment History", "Number of Credit Accounts",
    "Loan Amount", "Interest Rate", "Loan Term", "Type of Loan"
]]
y = df["Credit Status"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=42),
    "SVM (RBF Kernel)": SVC(kernel="rbf", probability=True, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=250, learning_rate=0.1, max_depth=5, eval_metric="logloss")
}

metrics = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else np.zeros_like(y_pred)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    metrics.append([name, acc, prec, rec, f1, auc])

   
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
    plt.title(f"Confusion Matrix — {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

results = pd.DataFrame(metrics, columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC AUC"])
print("\nModel Comparison:\n")
print(results.sort_values(by="Accuracy", ascending=False))

plt.figure(figsize=(10, 5))
sns.barplot(data=results.melt(id_vars="Model", var_name="Metric", value_name="Score"),
            x="Model", y="Score", hue="Metric", palette="magma")
plt.title("Model Performance Comparison")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


best_model_name = results.sort_values(by="F1-Score", ascending=False).iloc[0, 0]
best_model = models[best_model_name]
print(f"\nBest Performing Model: {best_model_name}")

if hasattr(best_model, "feature_importances_"):
    fi = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=fi.values, y=fi.index, palette="viridis")
    plt.title(f"Feature Importance — {best_model_name}")
    plt.tight_layout()
    plt.show()


plt.figure(figsize=(7, 6))
for name, model in models.items():
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name}")

plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves Comparison")
plt.legend()
plt.tight_layout()
plt.show()
