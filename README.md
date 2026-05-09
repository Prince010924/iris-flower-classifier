# 🌸 Iris Flower Classifier

A beginner-friendly Machine Learning project covering the full ML pipeline:
data loading → EDA → preprocessing → training → evaluation → prediction.

---

## 📁 Project Structure

```
iris_classifier/
├── train.py            ← Full training pipeline
├── predict.py          ← Classify new flower measurements
├── requirements.txt    ← Python dependencies
├── outputs/            ← Auto-created after training
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── pairplot.png
│   ├── correlation_heatmap.png
│   ├── confusion_matrix.png
│   ├── model_comparison.png
│   └── feature_importance.png
└── .vscode/
    ├── launch.json     ← Run configs (F5 to use)
    └── settings.json
```

---

## 🚀 Quick Start

### 1. Open project in VS Code
```bash
code iris_classifier
```

### 2. Create & activate a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the model
```bash
python train.py
```
This saves the trained model + 5 visualisation charts to the `outputs/` folder.

### 5. Make predictions
```bash
# Interactive mode
python predict.py

# Inline mode (sepal_len sepal_wid petal_len petal_wid)
python predict.py 5.1 3.5 1.4 0.2     # → setosa
python predict.py 6.3 3.3 6.0 2.5     # → virginica
```

---

## 📊 What I learned

| Step | Concept | Code |
|------|---------|------|
| Load data | `sklearn.datasets.load_iris` | `load_iris()` |
| Explore | EDA with pandas + seaborn | `df.describe()`, pairplot |
| Split | Train/test split | `train_test_split(..., stratify=y)` |
| Scale | Standardisation | `StandardScaler` |
| Train | 3 classifiers | KNN, Decision Tree, Random Forest |
| Evaluate | CV + metrics | `cross_val_score`, `classification_report` |
| Save | Model persistence | `joblib.dump / .load` |
| Predict | Inference on new data | `model.predict(X_sc)` |

---

## 🌿 The Dataset

- **150 samples** of iris flowers
- **4 features**: sepal length, sepal width, petal length, petal width (all in cm)
- **3 classes**: setosa 🌸 · versicolor 🌺 · virginica 🌼
- Perfectly balanced: 50 samples per class

---

## 💡 Next Steps (after this project)

1. **Tune hyperparameters** — try `GridSearchCV` for the best KNN `k` value
2. **Add more models** — try `SVC` (Support Vector Classifier) or `LogisticRegression`
3. **Build a web UI** — wrap `predict.py` with Streamlit: `pip install streamlit`
4. **Try a new dataset** — Titanic (classification) or Boston Housing (regression)
5. **Learn scikit-learn pipelines** — chain `StandardScaler + model` into one `Pipeline`

---

## 🧠 Key ML Concepts Glossary

| Term | Meaning |
|------|---------|
| **Train/test split** | Separate data so the model never "cheats" by training on test data |
| **Standardisation** | Rescale features to mean=0, std=1 so no feature dominates |
| **Cross-validation** | Train/evaluate on 5 different splits to get a robust accuracy estimate |
| **Confusion matrix** | Table showing correct vs wrong predictions per class |
| **Feature importance** | How much each feature contributed to the Random Forest decisions |

---

*Built as part of a Data Science learning journey. Happy experimenting!*
