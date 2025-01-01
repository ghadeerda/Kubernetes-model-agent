import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import pickle

# Ensure directories exist
figures_dir = "./figures"
results_dir = "./results"
os.makedirs(figures_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# Load dataset
data = pd.read_csv("./data/final_data_combined.csv")
data = data.drop(columns=["id"], errors="ignore")
X = data.drop("attack", axis=1)
y = data["attack"]

# Normalize features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Define classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=500),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Support Vector Machine": SVC(random_state=42, probability=True),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Naive Bayes": GaussianNB(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss"),
    "LightGBM": LGBMClassifier(random_state=42),
}

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Check if results already exist
results_file = os.path.join(results_dir, "model_results.pkl")
if os.path.exists(results_file):
    with open(results_file, "rb") as f:
        results = pickle.load(f)
    print("Loaded saved results.")
else:
    results = {}
    for name, clf in classifiers.items():
        print(f"Training {name}...")
        cv_scores = cross_val_score(clf, X_normalized, y, cv=cv, scoring="accuracy")
        clf.fit(X_normalized, y)
        results[name] = {
            "cv_scores": cv_scores,
            "model": clf,
            "feature_importances": None,  # Placeholder for feature importances
        }
        print(f"{name} CV Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    with open(results_file, "wb") as f:
        pickle.dump(results, f)
    print("Saved results.")

# Compute feature importance (native or permutation)
for name, res in results.items():
    clf = res["model"]
    if hasattr(clf, "feature_importances_"):  # Tree-based models
        res["feature_importances"] = clf.feature_importances_
    elif hasattr(clf, "coef_"):  # Linear models
        res["feature_importances"] = np.abs(clf.coef_).flatten()
    else:  # Use permutation importance
        perm_importance = permutation_importance(clf, X_normalized, y, n_repeats=5, random_state=42)
        res["feature_importances"] = perm_importance.importances_mean

# Plot cross-validation results
plt.figure(figsize=(10, 6))
plt.boxplot([res["cv_scores"] for res in results.values()], labels=results.keys(), showmeans=True)
plt.title("Classifier Cross-Validation Accuracy")
plt.ylabel("Accuracy")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "classifier_cross_validation.jpeg"), dpi=300, format="jpeg")
plt.close()

# Separate feature importance plots for all classifiers
for name, res in results.items():
    importances = res["feature_importances"]
    if importances is not None:
        sorted_idx = np.argsort(importances)[::-1]
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(importances)), importances[sorted_idx], color="gray", edgecolor="black")
        plt.xticks(range(len(importances)), X.columns[sorted_idx], rotation=45, ha="right")
        plt.title(f"Feature Importances - {name}")
        plt.ylabel("Importance")
        plt.xlabel("Features")
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f"feature_importances_{name.lower().replace(' ', '_')}.jpeg"), dpi=300, format="jpeg")
        plt.close()

# Combined feature importance plot (3x3 grid)
fig, axes = plt.subplots(3, 3, figsize=(15, 15))  # Create a grid for all classifiers
axes = axes.flatten()  # Flatten the 2D array for easier iteration

for i, (name, res) in enumerate(results.items()):
    importances = res["feature_importances"]
    if importances is not None:
        sorted_idx = np.argsort(importances)[::-1]
        axes[i].bar(range(len(importances)), importances[sorted_idx], color="gray", edgecolor="black")
        axes[i].set_xticks(range(len(importances)))
        axes[i].set_xticklabels(X.columns[sorted_idx], rotation=45, ha="right", fontsize=6)
        axes[i].set_title(name, fontsize=10)
        axes[i].set_ylabel("Importance", fontsize=8)
        axes[i].set_xlabel("Features", fontsize=8)

# Hide unused subplots (if classifiers are fewer than 9)
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "combined_feature_importances.jpeg"), dpi=300, format="jpeg")
plt.close()

# Improved ROC curve with distinct line styles
line_styles = ["solid", "dashed", "dotted", "dashdot"] * 3  # Cycle styles for up to 12 classifiers
plt.figure(figsize=(12, 8))
for (name, res), style in zip(results.items(), line_styles):
    model = res["model"]
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_normalized)[:, 1]
    else:
        y_proba = model.decision_function(X_normalized)
    fpr, tpr, _ = roc_curve(y, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})", linestyle=style, color="black")

plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.title("ROC Curve Comparison - All Classifiers")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "roc_curve_comparison_all.jpeg"), dpi=300, format="jpeg")
plt.close()
