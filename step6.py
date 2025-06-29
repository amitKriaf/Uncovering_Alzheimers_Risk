import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from xgboost import XGBClassifier, plot_importance
import shap

# Load and prepare data
df = pd.read_csv("/home/tzlilse@mta.ac.il/combined_alzheimer_dataset.csv")
X = df.drop(columns=["has alzheimer"])
y = df["has alzheimer"]

# Split: Train (80%) / Test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Compute class imbalance: 0 - Healthy, 1 - Sick
scale = (y_train == 0).sum() / (y_train == 1).sum()

# Final model with fixed hyperparameters
model = XGBClassifier(
    n_estimators=800,         # Number of boosting rounds (trees)
    max_depth=7,              # Maximum depth of each tree
    learning_rate=0.05,
    min_child_weight=3,       # Minimum number of samples required to make a split
    gamma=0.1,                # Minimum improvement in performance needed to split
    subsample=0.8,            # Percent of training data used for each tree (adds randomness) - rows
    colsample_bytree=0.8,     # Percent of features used per tree (adds randomness) - columns
    use_label_encoder=False,  # Disable deprecated label encoder
    scale_pos_weight=scale,   # Adjust for class imbalance
    eval_metric='logloss'     # Metric used during training
)

# Train the model on training set only
model.fit(X_train, y_train)

# Predict on test set
y_proba = model.predict_proba(X_test)[:, 1]
threshold = 0.4 #lower means more AD diagnosis,higher means the model is more strict
y_pred = (y_proba > threshold).astype(int)

# Evaluation
print(f"\nFinal threshold used: {threshold}")
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=["Healthy", "Alzheimer"]))

# Confusion Matrix
conf_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Healthy", "Alzheimer"], yticklabels=["Healthy", "Alzheimer"])
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
explainer = shap.Explainer(model, X_train)
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
