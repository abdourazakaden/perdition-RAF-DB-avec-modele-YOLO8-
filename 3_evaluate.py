"""
ÉTAPE 3 - Évaluation Finale sur l'Ensemble de Test
===================================================
Ce script :
- Évalue le modèle entraîné sur le set de test (RAF-DB)
- Génère la matrice de confusion
- Calcule Precision / Recall / F1 par classe
- Sauvegarde un rapport complet
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    top_k_accuracy_score
)

from ultralytics import YOLO

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
MODEL_PATH   = "runs/emotion/rafdb_yolov8/weights/best.pt"
TEST_DIR     = "./dataset_5050/test"
OUTPUT_DIR   = "./evaluation_results"
IMG_SIZE     = 224

EMOTION_LABELS = [
    "surprise", "fear", "disgust",
    "happiness", "sadness", "anger", "neutral"
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────
# 1. PRÉDICTIONS SUR LE SET DE TEST
# ─────────────────────────────────────────
def predict_test_set(model_path: str, test_dir: str):
    """Lance les prédictions sur toutes les images de test."""
    model = YOLO(model_path)

    y_true, y_pred, y_scores = [], [], []

    print("🔍 Prédiction sur le set de test...")
    for emotion in EMOTION_LABELS:
        cls_dir = Path(test_dir) / emotion
        if not cls_dir.exists():
            continue
        images = list(cls_dir.glob("*.jpg"))
        true_label = EMOTION_LABELS.index(emotion)

        for img_path in tqdm(images, desc=f"  {emotion}"):
            result = model.predict(str(img_path), imgsz=IMG_SIZE,
                                   verbose=False)[0]
            probs      = result.probs.data.cpu().numpy()
            pred_label = int(np.argmax(probs))

            y_true.append(true_label)
            y_pred.append(pred_label)
            y_scores.append(probs)

    return np.array(y_true), np.array(y_pred), np.array(y_scores)


# ─────────────────────────────────────────
# 2. MÉTRIQUES GLOBALES
# ─────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_scores):
    """Calcule et affiche les métriques de performance."""
    top1 = accuracy_score(y_true, y_pred)
    top5 = top_k_accuracy_score(y_true, y_scores, k=5) if y_scores.shape[1] >= 5 else top1

    print("\n" + "=" * 55)
    print("📊 RÉSULTATS FINAUX - SET DE TEST")
    print("=" * 55)
    print(f"  Top-1 Accuracy  : {top1 * 100:.2f}%")
    print(f"  Top-5 Accuracy  : {top5 * 100:.2f}%")
    print("=" * 55)
    print("\nRapport par classe :")
    print(classification_report(
        y_true, y_pred,
        target_names=EMOTION_LABELS,
        digits=4
    ))

    # Sauvegarder le rapport
    report = classification_report(
        y_true, y_pred,
        target_names=EMOTION_LABELS,
        digits=4,
        output_dict=True
    )
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(os.path.join(OUTPUT_DIR, "classification_report.csv"))
    print(f"✅ Rapport CSV sauvegardé : {OUTPUT_DIR}/classification_report.csv")

    return top1, top5


# ─────────────────────────────────────────
# 3. MATRICE DE CONFUSION
# ─────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred):
    """Génère et sauvegarde la matrice de confusion."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Matrice de Confusion - RAF-DB Test Set", fontsize=14, fontweight="bold")

    # Matrice absolue
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS,
                ax=axes[0])
    axes[0].set_title("Valeurs absolues")
    axes[0].set_ylabel("Vrai label")
    axes[0].set_xlabel("Label prédit")
    plt.setp(axes[0].get_xticklabels(), rotation=45, ha="right")

    # Matrice normalisée
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS,
                ax=axes[1])
    axes[1].set_title("Normalisée (par classe réelle)")
    axes[1].set_ylabel("Vrai label")
    axes[1].set_xlabel("Label prédit")
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"📊 Matrice de confusion sauvegardée : {out}")
    plt.show()


# ─────────────────────────────────────────
# 4. ACCURACY PAR CLASSE (BAR CHART)
# ─────────────────────────────────────────
def plot_per_class_accuracy(y_true, y_pred):
    """Barre d'accuracy par émotion."""
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A",
              "#98D8C8", "#F7DC6F", "#BB8FCE"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(EMOTION_LABELS, per_class_acc * 100, color=colors, edgecolor="black")
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy par émotion - Set de Test")
    ax.axhline(y=np.mean(per_class_acc) * 100, color="red",
               linestyle="--", label=f"Moyenne : {np.mean(per_class_acc)*100:.1f}%")

    for bar, acc in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{acc*100:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.legend()
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "per_class_accuracy.png")
    plt.savefig(out, dpi=150)
    print(f"📊 Accuracy par classe sauvegardée : {out}")
    plt.show()


# ─────────────────────────────────────────
# 5. EXEMPLES D'ERREURS
# ─────────────────────────────────────────
def show_error_examples(y_true, y_pred, test_dir: str, n: int = 12):
    """Affiche des exemples mal classifiés."""
    errors = [(i, y_true[i], y_pred[i]) for i in range(len(y_true))
              if y_true[i] != y_pred[i]]

    if not errors:
        print("✅ Aucune erreur trouvée !")
        return

    # Collecter les images
    all_test_imgs = []
    for emotion in EMOTION_LABELS:
        cls_dir = Path(test_dir) / emotion
        if cls_dir.exists():
            all_test_imgs.extend(list(cls_dir.glob("*.jpg")))

    n_show = min(n, len(errors))
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    fig.suptitle("Exemples d'erreurs de classification", fontsize=13)

    for k, (idx, true, pred) in enumerate(errors[:n_show]):
        row, col = divmod(k, 4)
        if idx < len(all_test_imgs):
            img = Image.open(all_test_imgs[idx])
            axes[row][col].imshow(img)
        axes[row][col].set_title(
            f"Vrai: {EMOTION_LABELS[true]}\nPrédit: {EMOTION_LABELS[pred]}",
            fontsize=8, color="red"
        )
        axes[row][col].axis("off")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "error_examples.png")
    plt.savefig(out, dpi=120)
    print(f"❌ Exemples d'erreurs sauvegardés : {out}")
    plt.show()


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 ÉTAPE 3 : Évaluation Finale\n")

    # 1. Prédictions
    y_true, y_pred, y_scores = predict_test_set(MODEL_PATH, TEST_DIR)

    # 2. Métriques
    top1, top5 = compute_metrics(y_true, y_pred, y_scores)

    # 3. Visualisations
    plot_confusion_matrix(y_true, y_pred)
    plot_per_class_accuracy(y_true, y_pred)
    show_error_examples(y_true, y_pred, TEST_DIR)

    print("\n✅ ÉTAPE 3 TERMINÉE")
    print(f"   Top-1 Accuracy : {top1*100:.2f}%")
    print(f"   Résultats dans : {OUTPUT_DIR}/")
    print("   Prochaine étape : python 4_predict.py")
