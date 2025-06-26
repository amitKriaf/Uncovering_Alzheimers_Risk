import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from xgboost import XGBClassifier, plot_importance
import shap

# Load and prepare data
df = pd.read_csv("/home/tzlilse@mta.ac.il/combined_alzheimer_dataset.csv")
X = df.drop(columns=["has alzheimer"])
y = df["has alzheimer"]

# Split: Train (80%) / Validation (10%) / Test (10%)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

# Compute class imbalance
scale = (y_train == 0).sum() / (y_train == 1).sum()

# Model & Grid Search
model = XGBClassifier(use_label_encoder=False)
'''
param_grid = {

    'n_estimators': [800],
    'max_depth': [7],
    'learning_rate': [0.05],
    'min_child_weight': [3],
    'gamma': [0.1],
    'subsample': [0.8],
    'colsample_bytree': [0.7]
}

grid_search = GridSearchCV(model, param_grid, scoring='f1', cv=3, n_jobs=1, verbose=1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
'''
# Final training on training + validation
X_combined = pd.concat([X_train, X_val])
y_combined = pd.concat([y_train, y_val])
model.fit(X_combined, y_combined)

# Predict on test set
y_proba = model.predict_proba(X_test)[:, 1]
threshold = 0.5
y_pred = (y_proba > threshold).astype(int)

# Evaluation
print(f"\nFinal threshold used: {threshold}")
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=["Healthy", "Alzheimer"]))

# Confusion Matrix
conf_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=["Healthy", "Alzheimer"], yticklabels=["Healthy", "Alzheimer"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("final_confusion_matrix_fixed_threshold.png")
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc_score = roc_auc_score(y_test, y_proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.close()
print(f"ROC AUC Score: {auc_score:.2f}")

# SHAP Summary Plot
explainer = shap.Explainer(model, X_combined)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("shap_summary_plot.png")
plt.close()

# SHAP Force Plot for a Sample
shap.initjs()
force_plot = shap.plots.force(shap_values[0], matplotlib=True, show=False)
plt.savefig("shap_force_plot_sample0.png")
plt.close()

# Feature Importance Plot
plt.figure(figsize=(10, 6))
plot_importance(model, max_num_features=10)
plt.title("Top 10 Feature Importances")
plt.tight_layout()
plt.savefig("feature_importance_plot.png")
plt.close()

print("All plots saved successfully.")
