"""
ÉTAPE 5 - Application Web Streamlit
=====================================
Interface utilisateur pour la reconnaissance d'émotions.

Lancer avec :
    streamlit run 5_app_streamlit.py
"""

import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

# Import du module de prédiction
from predict import EmotionPredictor, EMOTION_EMOJIS, EMOTION_COLORS

# ─────────────────────────────────────────
# CONFIGURATION STREAMLIT
# ─────────────────────────────────────────
st.set_page_config(
    page_title  = "Emotion Recognition - YOLOv8 + RAF-DB",
    page_icon   = "🎭",
    layout      = "wide",
    initial_sidebar_state = "expanded"
)

# CSS personnalisé
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
    .emotion-big {
        font-size: 4rem;
        margin-bottom: 0.5rem;
    }
    .metric-box {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        border-left: 4px solid #667eea;
    }
    .stProgress > div > div > div > div {
        background-color: #667eea;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# CHARGEMENT DU MODÈLE (CACHE)
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "runs/emotion/rafdb_yolov8/weights/best.pt"
    return EmotionPredictor(model_path)


# ─────────────────────────────────────────
# COMPOSANTS UI
# ─────────────────────────────────────────
def render_header():
    st.markdown("""
    <div class="main-header">
        <h1>🎭 Facial Emotion Recognition</h1>
        <p>YOLOv8 × RAF-DB Dataset | 7 Émotions</p>
    </div>
    """, unsafe_allow_html=True)


def render_probability_chart(result: dict) -> go.Figure:
    """Graphique en barres des probabilités par émotion."""
    probs   = result["all_probs"]
    labels  = list(probs.keys())
    values  = [probs[l] * 100 for l in labels]
    emojis  = [EMOTION_EMOJIS.get(l, "") for l in labels]
    colors  = [f"rgb{EMOTION_COLORS.get(l, (100,100,100))}" for l in labels]

    fig = go.Figure(go.Bar(
        x           = [f"{e} {l}" for e, l in zip(emojis, labels)],
        y           = values,
        marker_color= colors,
        text        = [f"{v:.1f}%" for v in values],
        textposition= "outside"
    ))
    fig.update_layout(
        title      = "Distribution des probabilités",
        yaxis_title= "Probabilité (%)",
        yaxis      = dict(range=[0, 110]),
        plot_bgcolor= "white",
        paper_bgcolor= "white",
        height     = 350,
        showlegend = False,
        margin     = dict(t=40, b=20, l=20, r=20)
    )
    return fig


def render_gauge(confidence: float, emotion: str) -> go.Figure:
    """Jauge de confiance."""
    color = f"rgb{EMOTION_COLORS.get(emotion, (100,100,100))}"
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = confidence * 100,
        title = {"text": "Confiance (%)"},
        gauge = {
            "axis"     : {"range": [0, 100]},
            "bar"      : {"color": color},
            "steps"    : [
                {"range": [0, 40],  "color": "#ffcccc"},
                {"range": [40, 70], "color": "#fff3cc"},
                {"range": [70, 100],"color": "#ccffcc"},
            ],
            "threshold": {
                "line" : {"color": "red", "width": 2},
                "value": 70
            }
        },
        number = {"suffix": "%", "font": {"size": 28}}
    ))
    fig.update_layout(height=250, margin=dict(t=30, b=10, l=20, r=20))
    return fig


def render_history_chart(history: list) -> go.Figure:
    """Évolution des émotions au fil du temps."""
    if len(history) < 2:
        return None
    df = pd.DataFrame(history)
    fig = px.line(
        df, x="time", y="confidence",
        color="emotion",
        markers=True,
        title="Historique des prédictions",
        labels={"confidence": "Confiance", "time": "Image #"}
    )
    fig.update_layout(height=300, margin=dict(t=40, b=10))
    return fig


# ─────────────────────────────────────────
# PAGE PRINCIPALE
# ─────────────────────────────────────────
def main():
    render_header()

    # ── Sidebar ──────────────────────────────────────────────────────
    with st.sidebar:
        st.image("https://ultralytics.com/assets/logo.png", width=150)
        st.markdown("### ⚙️ Paramètres")

        top_k = st.slider("Nombre de top prédictions", 1, 7, 3)
        show_probs = st.checkbox("Afficher toutes les probabilités", value=True)
        show_gauge = st.checkbox("Afficher la jauge de confiance", value=True)

        st.markdown("---")
        st.markdown("### 📊 Infos Modèle")
        st.info("**Modèle**: YOLOv8m-cls\n\n**Dataset**: RAF-DB\n\n**Classes**: 7 émotions\n\n**Split**: 50/50 train/val")

        st.markdown("---")
        st.markdown("### 🎭 Classes")
        for emotion, emoji in EMOTION_EMOJIS.items():
            st.markdown(f"{emoji} {emotion.capitalize()}")

    # ── Contenu principal ─────────────────────────────────────────────
    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown("### 📤 Entrée")
        mode = st.radio("Mode", ["📁 Fichier image", "🎥 Webcam (snapshot)"],
                         horizontal=True)

        image = None

        if "Fichier" in mode:
            uploaded = st.file_uploader(
                "Choisir une image",
                type=["jpg", "jpeg", "png", "webp"],
                help="Uploadez une photo de visage"
            )
            if uploaded:
                image = Image.open(uploaded).convert("RGB")
                st.image(image, caption="Image uploadée", use_column_width=True)

        else:  # Webcam
            snapshot = st.camera_input("📸 Prendre une photo")
            if snapshot:
                image = Image.open(snapshot).convert("RGB")

        predict_btn = st.button("🔍 Analyser l'émotion", type="primary",
                                 use_container_width=True,
                                 disabled=(image is None))

    # ── Résultats ─────────────────────────────────────────────────────
    with col_result:
        st.markdown("### 🎯 Résultat")

        if predict_btn and image is not None:
            with st.spinner("Analyse en cours..."):
                try:
                    predictor = load_model()
                    result    = predictor.predict(image, top_k=top_k)

                    # Émotion principale
                    emotion    = result["emotion"]
                    confidence = result["confidence"]
                    emoji      = result["emoji"]

                    # Carte principale
                    st.markdown(f"""
                    <div class="emotion-card">
                        <div class="emotion-big">{emoji}</div>
                        <h2 style="color: rgb{EMOTION_COLORS.get(emotion, (0,0,0))};
                                   text-transform: uppercase;">
                            {emotion}
                        </h2>
                        <h3>{confidence*100:.1f}% de confiance</h3>
                        <p style="color: #888;">⏱️ {result['inference_time_ms']} ms</p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("")

                    # Top-K résultats
                    st.markdown(f"**Top-{top_k} prédictions :**")
                    for i, r in enumerate(result["top_k"]):
                        col_e, col_p = st.columns([2, 3])
                        with col_e:
                            st.markdown(f"{r['emoji']} **{r['emotion'].capitalize()}**")
                        with col_p:
                            st.progress(r["confidence"])
                            st.caption(r["percent"])

                    # Graphique probabilités
                    if show_probs:
                        st.plotly_chart(
                            render_probability_chart(result),
                            use_container_width=True
                        )

                    # Jauge
                    if show_gauge:
                        st.plotly_chart(
                            render_gauge(confidence, emotion),
                            use_container_width=True
                        )

                    # Sauvegarder dans l'historique
                    if "history" not in st.session_state:
                        st.session_state.history = []
                    st.session_state.history.append({
                        "time"      : len(st.session_state.history) + 1,
                        "emotion"   : emotion,
                        "confidence": confidence * 100
                    })

                except Exception as e:
                    st.error(f"❌ Erreur : {str(e)}")
                    st.info("Vérifiez que le modèle est bien entraîné (2_train_model.py)")

        else:
            st.info("⬆️ Uploadez une image et cliquez sur **Analyser**")

            # Exemple de visualisation vide
            placeholder_data = {emotion: 1/7 for emotion in EMOTION_EMOJIS.keys()}
            st.caption("Distribution uniforme (avant prédiction)")

    # ── Historique ────────────────────────────────────────────────────
    if "history" in st.session_state and len(st.session_state.history) > 1:
        st.markdown("---")
        st.markdown("### 📈 Historique des analyses")
        fig = render_history_chart(st.session_state.history)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        history_df = pd.DataFrame(st.session_state.history)
        with col1:
            st.metric("Images analysées", len(st.session_state.history))
        with col2:
            top_emotion = history_df["emotion"].mode()[0]
            st.metric("Émotion dominante", f"{EMOTION_EMOJIS.get(top_emotion,'')} {top_emotion}")
        with col3:
            avg_conf = history_df["confidence"].mean()
            st.metric("Confiance moyenne", f"{avg_conf:.1f}%")

        if st.button("🗑️ Effacer l'historique"):
            st.session_state.history = []
            st.rerun()

    # ── Footer ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#888;'>"
        "🎭 Emotion Recognition App | YOLOv8 × RAF-DB | Built with Streamlit"
        "</div>",
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────
if __name__ == "__main__":
    main()
