"""
Iris Flower Classifier - Training Pipeline
==========================================
Covers: data loading, EDA, preprocessing, training, evaluation, saving model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
import joblib
import os

# ── 0. Setup ────────────────────────────────────────────────────────────────
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("  🌸  IRIS FLOWER CLASSIFIER")
print("=" * 60)

# ── 1. Load Data ─────────────────────────────────────────────────────────────
print("\n[1] Loading dataset...")
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="species")
species_names = iris.target_names

print(f"    Samples : {X.shape[0]}")
print(f"    Features: {X.shape[1]}  →  {list(X.columns)}")
print(f"    Classes : {list(species_names)}")

# ── 2. Exploratory Data Analysis ─────────────────────────────────────────────
print("\n[2] Exploratory Data Analysis...")
df = X.copy()
df["species"] = y.map(dict(enumerate(species_names)))

print("\n    Basic Stats:")
print(df.describe().round(2).to_string())

print("\n    Class distribution:")
print(df["species"].value_counts().to_string())

# Pairplot
print("\n    Saving pairplot...")
sns.set_theme(style="whitegrid", font_scale=0.9)
pair = sns.pairplot(df, hue="species", palette="Set2", corner=True,
                    plot_kws={"alpha": 0.7, "s": 40})
pair.fig.suptitle("Iris Feature Pairplot", y=1.02, fontsize=13, fontweight="bold")
pair.savefig(f"{OUTPUT_DIR}/pairplot.png", dpi=150, bbox_inches="tight")
plt.close()

# Correlation heatmap
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(X.corr(), annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, ax=ax)
ax.set_title("Feature Correlation Heatmap", fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/correlation_heatmap.png", dpi=150)
plt.close()
print("    Plots saved to outputs/")

# ── 3. Preprocessing ─────────────────────────────────────────────────────────
print("\n[3] Preprocessing...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
print(f"    Train: {X_train_sc.shape[0]} samples  |  Test: {X_test_sc.shape[0]} samples")

# ── 4. Train Multiple Models ──────────────────────────────────────────────────
print("\n[4] Training models...")
models = {
    "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree":        DecisionTreeClassifier(max_depth=4, random_state=42),
    "Random Forest":        RandomForestClassifier(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train_sc, y_train)
    cv_scores = cross_val_score(model, X_train_sc, y_train, cv=5)
    test_acc  = accuracy_score(y_test, model.predict(X_test_sc))
    results[name] = {"cv_mean": cv_scores.mean(), "cv_std": cv_scores.std(),
                     "test_acc": test_acc, "model": model}
    print(f"    {name:<26}  CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}  |  Test: {test_acc:.3f}")

# ── 5. Best Model Evaluation ─────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["test_acc"])
best      = results[best_name]
best_model = best["model"]
print(f"\n[5] Best model: {best_name}  (test accuracy = {best['test_acc']:.3f})")

y_pred = best_model.predict(X_test_sc)
print("\n    Classification Report:")
print(classification_report(y_test, y_pred, target_names=species_names))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=species_names, yticklabels=species_names, ax=ax)
ax.set_xlabel("Predicted", fontweight="bold")
ax.set_ylabel("Actual", fontweight="bold")
ax.set_title(f"Confusion Matrix — {best_name}", fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=150)
plt.close()
print("    Confusion matrix saved.")

# Model accuracy bar chart
fig, ax = plt.subplots(figsize=(7, 4))
names   = list(results.keys())
accs    = [results[n]["test_acc"] for n in names]
colors  = ["#4CAF50" if n == best_name else "#90CAF9" for n in names]
bars = ax.barh(names, accs, color=colors, edgecolor="white", height=0.5)
ax.set_xlim(0.85, 1.02)
ax.set_xlabel("Test Accuracy")
ax.set_title("Model Comparison", fontweight="bold")
for bar, acc in zip(bars, accs):
    ax.text(acc + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{acc:.3f}", va="center", fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/model_comparison.png", dpi=150)
plt.close()

# ── 6. Feature Importance (Random Forest) ────────────────────────────────────
if "Random Forest" in results:
    rf = results["Random Forest"]["model"]
    fi = pd.Series(rf.feature_importances_, index=iris.feature_names).sort_values()
    fig, ax = plt.subplots(figsize=(6, 4))
    fi.plot(kind="barh", ax=ax, color="#4CAF50", edgecolor="white")
    ax.set_title("Feature Importance — Random Forest", fontweight="bold")
    ax.set_xlabel("Importance Score")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/feature_importance.png", dpi=150)
    plt.close()
    print("    Feature importance chart saved.")

# ── 7. Save Artifacts ─────────────────────────────────────────────────────────
print("\n[6] Saving model and scaler...")
joblib.dump(best_model, f"{OUTPUT_DIR}/best_model.pkl")
joblib.dump(scaler,     f"{OUTPUT_DIR}/scaler.pkl")
print(f"    Saved: outputs/best_model.pkl  ('{best_name}')")
print(f"    Saved: outputs/scaler.pkl")

print("\n" + "=" * 60)
print("  ✅  Training complete!  Run predict.py to classify new flowers.")
print("=" * 60)
