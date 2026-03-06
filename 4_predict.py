"""
ÉTAPE 4 - Fonction de Prédiction d'Émotion
===========================================
Module réutilisable pour prédire les émotions.
Utilisé par l'application Streamlit.
"""

import os
import time
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
MODEL_PATH = "runs/emotion/rafdb_yolov8/weights/best.pt"
IMG_SIZE   = 224

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


# ─────────────────────────────────────────
# CLASSE PRINCIPALE
# ─────────────────────────────────────────
class EmotionPredictor:
    """
    Prédicateur d'émotions basé sur YOLOv8.

    Usage :
        predictor = EmotionPredictor()
        result = predictor.predict("photo.jpg")
        print(result["emotion"], result["confidence"])
    """

    def __init__(self, model_path: str = MODEL_PATH):
        print(f"⏳ Chargement du modèle : {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Modèle introuvable : {model_path}\n"
                "Lancez d'abord : python 2_train_model.py"
            )
        self.model  = YOLO(model_path)
        self.labels = EMOTION_LABELS
        print("✅ Modèle chargé !")

    # ── Prédiction sur une image ──────────────────────────────────────
    def predict(self, image: Union[str, np.ndarray, Image.Image],
                top_k: int = 3) -> dict:
        """
        Prédit l'émotion d'une image.

        Args:
            image  : chemin fichier, array numpy ou PIL Image
            top_k  : nombre de top prédictions à retourner

        Returns:
            dict avec keys: emotion, confidence, emoji, top_k, inference_time_ms
        """
        t0 = time.time()

        # Normaliser l'entrée
        if isinstance(image, str):
            pil_img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            raise TypeError(f"Type d'image non supporté : {type(image)}")

        # Inférence
        result = self.model.predict(pil_img, imgsz=IMG_SIZE, verbose=False)[0]
        probs  = result.probs.data.cpu().numpy()

        # Top-K résultats
        top_indices = np.argsort(probs)[::-1][:top_k]
        top_results = [
            {
                "emotion"   : self.labels[i],
                "confidence": float(probs[i]),
                "emoji"     : EMOTION_EMOJIS.get(self.labels[i], ""),
                "percent"   : f"{probs[i]*100:.1f}%"
            }
            for i in top_indices
        ]

        best = top_results[0]
        ms   = (time.time() - t0) * 1000

        return {
            "emotion"          : best["emotion"],
            "confidence"       : best["confidence"],
            "emoji"            : best["emoji"],
            "percent"          : best["percent"],
            "top_k"            : top_results,
            "all_probs"        : {self.labels[i]: float(probs[i])
                                  for i in range(len(self.labels))},
            "inference_time_ms": round(ms, 1)
        }

    # ── Prédiction batch ─────────────────────────────────────────────
    def predict_batch(self, images: list, top_k: int = 1) -> list:
        """Prédit les émotions pour une liste d'images."""
        return [self.predict(img, top_k=top_k) for img in images]

    # ── Annotation d'une image ────────────────────────────────────────
    def annotate_image(self, image: Union[np.ndarray, Image.Image],
                       result: dict) -> np.ndarray:
        """
        Dessine l'émotion prédite sur l'image (OpenCV).

        Returns:
            Image annotée (numpy BGR)
        """
        if isinstance(image, Image.Image):
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            img_cv = image.copy()

        h, w = img_cv.shape[:2]
        emotion    = result["emotion"]
        confidence = result["confidence"]
        color      = EMOTION_COLORS.get(emotion, (255, 255, 255))

        # Fond semi-transparent
        overlay = img_cv.copy()
        cv2.rectangle(overlay, (0, h - 50), (w, h), color, -1)
        cv2.addWeighted(overlay, 0.6, img_cv, 0.4, 0, img_cv)

        # Texte
        label = f"{EMOTION_EMOJIS.get(emotion, '')} {emotion.upper()} {confidence*100:.1f}%"
        cv2.putText(img_cv, label, (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Barre de confiance
        bar_w = int(w * confidence)
        cv2.rectangle(img_cv, (0, h - 55), (bar_w, h - 52), color, -1)

        return img_cv

    # ── Webcam temps réel ─────────────────────────────────────────────
    def run_webcam(self, camera_id: int = 0):
        """Lance la prédiction en temps réel via webcam."""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("[ERREUR] Impossible d'ouvrir la caméra")
            return

        print("🎥 Webcam démarrée — Appuyez sur 'q' pour quitter")
        fps_counter = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t_start = time.time()
            result  = self.predict(frame)
            fps_counter.append(1 / (time.time() - t_start + 1e-6))

            # Annotation
            annotated = self.annotate_image(frame, result)

            # FPS
            fps = np.mean(fps_counter[-10:])
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Emotion Recognition - YOLOv8", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("👋 Webcam fermée")


# ─────────────────────────────────────────
# TEST RAPIDE
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 ÉTAPE 4 : Test de la fonction de prédiction\n")

    predictor = EmotionPredictor()

    # Test sur une image
    test_img = "test_face.jpg"
    if os.path.exists(test_img):
        result = predictor.predict(test_img)
        print(f"\n🎭 Résultat :")
        print(f"   Émotion    : {result['emoji']} {result['emotion']}")
        print(f"   Confiance  : {result['percent']}")
        print(f"   Temps      : {result['inference_time_ms']} ms")
        print(f"\n   Top-3 :")
        for r in result["top_k"]:
            print(f"     {r['emoji']} {r['emotion']:<12} {r['percent']}")
    else:
        print(f"[!] Aucune image de test trouvée ({test_img})")
        print("    Placez une image 'test_face.jpg' pour tester")

    # Lancer webcam (décommenter si nécessaire)
    # predictor.run_webcam()

    print("\n✅ Module de prédiction prêt !")
    print("   Prochaine étape : python 5_app_streamlit.py")
