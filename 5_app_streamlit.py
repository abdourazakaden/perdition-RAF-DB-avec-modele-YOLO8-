"""
Application Web Streamlit - Emotion Recognition
YOLOv8 × RAF-DB | Tout en un seul fichier
Lancer : streamlit run 5_app_streamlit.py
"""

import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
from ultralytics import YOLO

# ─────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────
EMOTION_LABELS = [
    "surprise", "fear", "disgust",
    "happiness", "sadness", "anger", "neutral"
]

EMOTION_EMOJIS = {
    "surprise"  : "😲",
    "fear"      : "😨",
    "disgust"   : "🤢",
    "happiness" : "😊",
    "sadness"   : "😢",
    "anger"     : "😠",
    "neutral"   : "😐"
}

EMOTION_COLORS = {
    "surprise"  : (255, 165, 0),
    "fear"      : (148, 0, 211),
    "disgust"   : (0, 128, 0),
    "happiness" : (255, 215, 0),
    "sadness"   : (70, 130, 180),
    "anger"     : (220, 20, 60),
    "neutral"   : (128, 128, 128)
}

MODEL_PATH = "runs/emotion/rafdb_yolov8/weights/best.pt"
IMG_SIZE   = 224

# ─────────────────────────────────────────
# CONFIGURATION STREAMLIT
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Emotion Recognition - YOLOv8 + RAF-DB",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .emotion-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
    }
    .emotion-big { font-size: 4rem; margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# CHARGEMENT MODÈLE (CACHE)
# ─────────────────────────────────────────
@st.cache_resource
def load_model(path: str):
    return YOLO(path)


# ─────────────────────────────────────────
# FONCTION DE PRÉDICTION
# ─────────────────────────────────────────
def predict(model, image: Image.Image, top_k: int = 3) -> dict:
    t0     = time.time()
    result = model.predict(image, imgsz=IMG_SIZE, verbose=False)[0]
    probs  = result.probs.data.cpu().numpy()

    top_indices = np.argsort(probs)[::-1][:top_k]
    top_results = [
        {
            "emotion"   : EMOTION_LABELS[i],
            "confidence": float(probs[i]),
            "emoji"     : EMOTION_EMOJIS.get(EMOTION_LABELS[i], ""),
            "percent"   : f"{probs[i]*100:.1f}%"
        }
        for i in top_indices
    ]

    best = top_results[0]
    return {
        "emotion"          : best["emotion"],
        "confidence"       : best["confidence"],
        "emoji"            : best["emoji"],
        "percent"          : best["percent"],
        "top_k"            : top_results,
        "all_probs"        : {EMOTION_LABELS[i]: float(probs[i])
                              for i in range(len(EMOTION_LABELS))},
        "inference_time_ms": round((time.time() - t0) * 1000, 1)
    }


# ─────────────────────────────────────────
# GRAPHIQUES
# ─────────────────────────────────────────
def chart_probs(result: dict) -> go.Figure:
    probs  = result["all_probs"]
    labels = list(probs.keys())
    values = [probs[l] * 100 for l in labels]
    colors = [f"rgb{EMOTION_COLORS.get(l, (100,100,100))}" for l in labels]
    emojis = [EMOTION_EMOJIS.get(l, "") for l in labels]

    fig = go.Figure(go.Bar(
        x=[f"{e} {l}" for e, l in zip(emojis, labels)],
        y=values,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside"
    ))
    fig.update_layout(
        title="Distribution des probabilités",
        yaxis_title="Probabilité (%)",
        yaxis=dict(range=[0, 115]),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=350,
        showlegend=False,
        margin=dict(t=40, b=20, l=20, r=20)
    )
    return fig


def chart_gauge(confidence: float, emotion: str) -> go.Figure:
    color = f"rgb{EMOTION_COLORS.get(emotion, (100,100,100))}"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        title={"text": "Confiance (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 40],   "color": "#ffcccc"},
                {"range": [40, 70],  "color": "#fff3cc"},
                {"range": [70, 100], "color": "#ccffcc"},
            ]
        },
        number={"suffix": "%", "font": {"size": 28}}
    ))
    fig.update_layout(height=250, margin=dict(t=30, b=10, l=20, r=20))
    return fig


# ─────────────────────────────────────────
# APP PRINCIPALE
# ─────────────────────────────────────────
def main():
    st.markdown("""
    <div class="main-header">
        <h1>🎭 Facial Emotion Recognition</h1>
        <p>YOLOv8 × RAF-DB Dataset | 7 Émotions</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Paramètres")
        top_k      = st.slider("Top prédictions", 1, 7, 3)
        show_probs = st.checkbox("Graphique des probabilités", value=True)
        show_gauge = st.checkbox("Jauge de confiance", value=True)

        st.markdown("---")
        st.markdown("### 📊 Modèle")
        st.info("**YOLOv8m-cls**\n\nRAF-DB | 7 classes\nSplit 50/50")

        st.markdown("---")
        st.markdown("### 🎭 Classes")
        for emotion, emoji in EMOTION_EMOJIS.items():
            st.markdown(f"{emoji} {emotion.capitalize()}")

    # ── Layout ───────────────────────────────────────────────────────
    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown("### 📤 Image d'entrée")
        mode = st.radio("Mode", ["📁 Fichier", "📸 Webcam"], horizontal=True)

        image = None
        if "Fichier" in mode:
            uploaded = st.file_uploader(
                "Choisir une image",
                type=["jpg", "jpeg", "png", "webp"]
            )
            if uploaded:
                image = Image.open(uploaded).convert("RGB")
                st.image(image, caption="Image uploadée", use_column_width=True)
        else:
            snapshot = st.camera_input("Prendre une photo")
            if snapshot:
                image = Image.open(snapshot).convert("RGB")

        predict_btn = st.button(
            "🔍 Analyser l'émotion",
            type="primary",
            use_container_width=True,
            disabled=(image is None)
        )

    # ── Résultats ─────────────────────────────────────────────────────
    with col_result:
        st.markdown("### 🎯 Résultat")

        if predict_btn and image is not None:
            if not os.path.exists(MODEL_PATH):
                st.error(f"❌ Modèle introuvable : `{MODEL_PATH}`")
                st.info("Lancez d'abord l'entraînement avec `2_train_model.py`")
                return

            with st.spinner("Analyse en cours..."):
                try:
                    model  = load_model(MODEL_PATH)
                    result = predict(model, image, top_k=top_k)

                    emotion    = result["emotion"]
                    confidence = result["confidence"]
                    emoji      = result["emoji"]
                    color      = EMOTION_COLORS.get(emotion, (0, 0, 0))

                    st.markdown(f"""
                    <div class="emotion-card">
                        <div class="emotion-big">{emoji}</div>
                        <h2 style="color: rgb{color}; text-transform: uppercase;">
                            {emotion}
                        </h2>
                        <h3>{confidence*100:.1f}% de confiance</h3>
                        <p style="color: #888;">⏱️ {result['inference_time_ms']} ms</p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("")
                    st.markdown(f"**Top-{top_k} prédictions :**")
                    for r in result["top_k"]:
                        c1, c2 = st.columns([2, 3])
                        with c1:
                            st.markdown(f"{r['emoji']} **{r['emotion'].capitalize()}**")
                        with c2:
                            st.progress(r["confidence"])
                            st.caption(r["percent"])

                    if show_probs:
                        st.plotly_chart(chart_probs(result), use_container_width=True)
                    if show_gauge:
                        st.plotly_chart(chart_gauge(confidence, emotion), use_container_width=True)

                    if "history" not in st.session_state:
                        st.session_state.history = []
                    st.session_state.history.append({
                        "time": len(st.session_state.history) + 1,
                        "emotion": emotion,
                        "confidence": confidence * 100
                    })

                except Exception as e:
                    st.error(f"❌ Erreur : {str(e)}")

        else:
            st.info("⬆️ Uploadez une image et cliquez sur **Analyser**")

    # ── Historique ────────────────────────────────────────────────────
    if "history" in st.session_state and len(st.session_state.history) > 1:
        st.markdown("---")
        st.markdown("### 📈 Historique")
        df  = pd.DataFrame(st.session_state.history)
        fig = px.line(df, x="time", y="confidence", color="emotion",
                      markers=True, height=300)
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Images analysées", len(st.session_state.history))
        top_emo = df["emotion"].mode()[0]
        c2.metric("Émotion dominante", f"{EMOTION_EMOJIS.get(top_emo,'')} {top_emo}")
        c3.metric("Confiance moyenne", f"{df['confidence'].mean():.1f}%")

        if st.button("🗑️ Effacer l'historique"):
            st.session_state.history = []
            st.rerun()

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#888;'>"
        "🎭 Emotion Recognition | YOLOv8 × RAF-DB | Streamlit"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
