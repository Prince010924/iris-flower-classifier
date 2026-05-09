"""
Iris Flower Classifier — Streamlit Web App
==========================================
Run with:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import streamlit as st
from sklearn.datasets import load_iris

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'DM Serif Display', serif !important;
    }
    .main { background-color: #fafaf7; }

    .predict-card {
        background: linear-gradient(135deg, #1a472a 0%, #2d6a4f 100%);
        border-radius: 20px;
        padding: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 20px 60px rgba(26,71,42,0.3);
        margin: 1rem 0;
    }
    .predict-card h2 { color: white !important; font-size: 2rem; margin-bottom: 0.3rem; }
    .predict-card p  { font-size: 1.1rem; opacity: 0.85; margin: 0; }
    .confidence-bar  { background: rgba(255,255,255,0.15); border-radius: 50px;
                        height: 12px; margin: 0.5rem 0; }
    .confidence-fill { background: #95d5b2; border-radius: 50px; height: 12px; }

    .metric-box {
        background: white;
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        border-left: 4px solid #2d6a4f;
        margin-bottom: 0.8rem;
    }
    .metric-box .label { font-size: 0.78rem; color: #888; text-transform: uppercase;
                          letter-spacing: 0.08em; margin-bottom: 0.2rem; }
    .metric-box .value { font-size: 1.5rem; font-weight: 500; color: #1a1a1a; }

    .stSlider > div > div > div > div { background: #2d6a4f !important; }
    div[data-testid="stSidebar"] { background: #f0f4f0; }
</style>
""", unsafe_allow_html=True)

# ── Load model & data ────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path  = "outputs/best_model.pkl"
    scaler_path = "outputs/scaler.pkl"
    if not os.path.exists(model_path):
        return None, None
    return joblib.load(model_path), joblib.load(scaler_path)

@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = pd.Series(iris.target).map(dict(enumerate(iris.target_names)))
    return df, iris

model, scaler = load_model()
df, iris      = load_data()

SPECIES_EMOJI = {"setosa": "🌸", "versicolor": "🌺", "virginica": "🌼"}
SPECIES_DESC  = {
    "setosa":     "Small, compact flower with short petals. Found in Arctic and alpine regions.",
    "versicolor": "Medium-sized flower with mixed colouring. Common in North America.",
    "virginica":  "Large, vibrant flower with long petals. Native to eastern North America.",
}

# ── Sidebar — inputs ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 Flower Measurements")
    st.markdown("Adjust the sliders to match your flower's measurements.")
    st.divider()

    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.4, 0.1)
    sepal_width  = st.slider("Sepal Width (cm)",  2.0, 4.5, 3.0, 0.1)
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 3.7, 0.1)
    petal_width  = st.slider("Petal Width (cm)",  0.1, 2.5, 1.2, 0.1)

    st.divider()
    st.markdown("**Quick examples:**")
    col1, col2, col3 = st.columns(3)
    if col1.button("🌸 Setosa"):
        sepal_length, sepal_width, petal_length, petal_width = 5.1, 3.5, 1.4, 0.2
    if col2.button("🌺 Versic."):
        sepal_length, sepal_width, petal_length, petal_width = 6.0, 2.9, 4.5, 1.5
    if col3.button("🌼 Virgin."):
        sepal_length, sepal_width, petal_length, petal_width = 6.3, 3.3, 6.0, 2.5

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("# 🌸 Iris Flower Classifier")
st.markdown("An interactive machine learning demo — adjust measurements in the sidebar to classify a flower in real time.")
st.divider()

# ── Prediction ───────────────────────────────────────────────────────────────
if model is None:
    st.error("⚠️  Model not found. Please run `train.py` first, then restart the app.")
else:
    measurements = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    X_sc         = scaler.transform(measurements)
    pred_idx     = model.predict(X_sc)[0]
    pred_species = iris.target_names[pred_idx]
    proba        = model.predict_proba(X_sc)[0]
    confidence   = proba[pred_idx]
    emoji        = SPECIES_EMOJI[pred_species]

    # Prediction card
    st.markdown(f"""
    <div class="predict-card">
        <h2>{emoji} Iris {pred_species.capitalize()}</h2>
        <p>{SPECIES_DESC[pred_species]}</p>
        <br>
        <p style="font-size:0.85rem; opacity:0.7; margin-bottom:0.3rem">
            CONFIDENCE — {confidence:.1%}
        </p>
        <div class="confidence-bar">
            <div class="confidence-fill" style="width:{confidence*100:.0f}%"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Probability breakdown
    st.markdown("### Probability Breakdown")
    prob_cols = st.columns(3)
    for i, (sp, prob) in enumerate(zip(iris.target_names, proba)):
        with prob_cols[i]:
            st.markdown(f"""
            <div class="metric-box">
                <div class="label">{SPECIES_EMOJI[sp]} {sp}</div>
                <div class="value">{prob:.1%}</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ── Charts ───────────────────────────────────────────────────────────────
    st.markdown("### 📊 Explore the Dataset")
    tab1, tab2, tab3 = st.tabs(["Feature Distributions", "Petal vs Sepal", "Model Charts"])

    with tab1:
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        features  = iris.feature_names
        palette   = {"setosa": "#52b788", "versicolor": "#f4a261", "virginica": "#e76f51"}
        for ax, feat in zip(axes.flatten(), features):
            for sp in df["species"].unique():
                vals = df[df["species"] == sp][feat]
                ax.hist(vals, alpha=0.6, label=sp, color=palette[sp], bins=15, edgecolor="white")
            ax.set_title(feat, fontweight="bold", fontsize=10)
            ax.set_xlabel("cm"); ax.legend(fontsize=8)
            ax.spines[["top","right"]].set_visible(False)
        # Mark user's values
        user_vals = [sepal_length, sepal_width, petal_length, petal_width]
        for ax, val in zip(axes.flatten(), user_vals):
            ax.axvline(val, color="#1a472a", linewidth=2, linestyle="--", label="Your flower")
        fig.suptitle("Feature Distributions by Species", fontweight="bold", fontsize=13)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab2:
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        palette   = {"setosa": "#52b788", "versicolor": "#f4a261", "virginica": "#e76f51"}
        # Sepal scatter
        for sp in df["species"].unique():
            sub = df[df["species"] == sp]
            axes[0].scatter(sub["sepal length (cm)"], sub["sepal width (cm)"],
                            label=sp, alpha=0.7, s=50, color=palette[sp], edgecolors="white", linewidth=0.5)
        axes[0].scatter(sepal_length, sepal_width, color="#1a472a", s=200,
                        zorder=5, marker="*", label="Your flower")
        axes[0].set_xlabel("Sepal Length (cm)"); axes[0].set_ylabel("Sepal Width (cm)")
        axes[0].set_title("Sepal Measurements", fontweight="bold")
        axes[0].legend(); axes[0].spines[["top","right"]].set_visible(False)
        # Petal scatter
        for sp in df["species"].unique():
            sub = df[df["species"] == sp]
            axes[1].scatter(sub["petal length (cm)"], sub["petal width (cm)"],
                            label=sp, alpha=0.7, s=50, color=palette[sp], edgecolors="white", linewidth=0.5)
        axes[1].scatter(petal_length, petal_width, color="#1a472a", s=200,
                        zorder=5, marker="*", label="Your flower")
        axes[1].set_xlabel("Petal Length (cm)"); axes[1].set_ylabel("Petal Width (cm)")
        axes[1].set_title("Petal Measurements", fontweight="bold")
        axes[1].legend(); axes[1].spines[["top","right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.caption("⭐ Star = your flower's position in the dataset")

    with tab3:
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            if os.path.exists("outputs/confusion_matrix.png"):
                st.image("outputs/confusion_matrix.png", caption="Confusion Matrix", use_container_width=True)
        with chart_col2:
            if os.path.exists("outputs/model_comparison.png"):
                st.image("outputs/model_comparison.png", caption="Model Comparison", use_container_width=True)
        if os.path.exists("outputs/feature_importance.png"):
            st.image("outputs/feature_importance.png", caption="Feature Importance (Random Forest)", use_container_width=True)

    st.divider()
    st.markdown(
        "<p style='text-align:center; color:#aaa; font-size:0.85rem'>"
        "Built with Python · scikit-learn · Streamlit &nbsp;|&nbsp; "
        "<a href='https://github.com/Prince010924/iris-flower-classifier' "
        "style='color:#2d6a4f'>View on GitHub ↗</a></p>",
        unsafe_allow_html=True
    )
