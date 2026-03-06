"""
ÉTAPE 2 - Entraînement et Validation YOLOv8 (Split 50/50)
==========================================================
Ce script :
- Configure YOLOv8 pour la classification d'émotions
- Entraîne sur RAF-DB (split train/val 50/50)
- Sauvegarde le meilleur modèle
- Affiche les courbes d'entraînement
"""

import os
import yaml
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm

from ultralytics import YOLO

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
DATASET_DIR   = "./dataset_yolo"      # ← Créé par 1_prepare_data.py
OUTPUT_DIR    = "./dataset_5050"      # ← Dataset avec split 50/50
MODEL_NAME    = "yolov8m-cls.pt"      # nano/small/medium/large/xlarge
EPOCHS        = 50
IMG_SIZE      = 224
BATCH_SIZE    = 32
LEARNING_RATE = 0.001
WORKERS       = 4
DEVICE        = "0"                   # "0" pour GPU, "cpu" pour CPU
RANDOM_SEED   = 42

EMOTION_LABELS = [
    "surprise", "fear", "disgust",
    "happiness", "sadness", "anger", "neutral"
]


# ─────────────────────────────────────────
# 1. PRÉPARER LE SPLIT 50/50
# ─────────────────────────────────────────
def prepare_5050_split(dataset_dir: str, output_dir: str):
    """
    Crée un split 50% train / 50% val à partir du dossier train existant.
    (Le test reste séparé)
    """
    print("📂 Préparation du split 50/50 train/val...")

    for emotion in EMOTION_LABELS:
        src_train = Path(dataset_dir) / "train" / emotion
        if not src_train.exists():
            print(f"  [!] Dossier manquant : {src_train}")
            continue

        all_images = list(src_train.glob("*.jpg"))
        train_imgs, val_imgs = train_test_split(
            all_images, test_size=0.5, random_state=RANDOM_SEED
        )

        for split_name, imgs in [("train", train_imgs), ("val", val_imgs)]:
            dst = Path(output_dir) / split_name / emotion
            dst.mkdir(parents=True, exist_ok=True)
            for img_path in imgs:
                shutil.copy2(img_path, dst / img_path.name)

        # Copier le test tel quel
        src_test = Path(dataset_dir) / "test" / emotion
        dst_test = Path(output_dir) / "test" / emotion
        dst_test.mkdir(parents=True, exist_ok=True)
        if src_test.exists():
            for img_path in src_test.glob("*.jpg"):
                shutil.copy2(img_path, dst_test / img_path.name)

    # Résumé
    print("\n📊 Résumé split 50/50 :")
    for split in ["train", "val", "test"]:
        total = sum(
            len(list((Path(output_dir) / split / e).glob("*.jpg")))
            for e in EMOTION_LABELS
            if (Path(output_dir) / split / e).exists()
        )
        print(f"  {split:<6}: {total} images")

    return output_dir


# ─────────────────────────────────────────
# 2. CONFIGURATION YAML
# ─────────────────────────────────────────
def create_dataset_yaml(output_dir: str) -> str:
    """Crée le fichier de configuration dataset pour YOLOv8."""
    config = {
        "path"  : os.path.abspath(output_dir),
        "train" : "train",
        "val"   : "val",
        "test"  : "test",
        "nc"    : len(EMOTION_LABELS),
        "names" : EMOTION_LABELS
    }
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"\n✅ Config YAML créée : {yaml_path}")
    return yaml_path


# ─────────────────────────────────────────
# 3. ENTRAÎNEMENT
# ─────────────────────────────────────────
def train_model(dataset_dir: str):
    """Lance l'entraînement YOLOv8 Classification."""

    print(f"\n🏋️  Démarrage entraînement YOLOv8")
    print(f"   Modèle    : {MODEL_NAME}")
    print(f"   Epochs    : {EPOCHS}")
    print(f"   Batch     : {BATCH_SIZE}")
    print(f"   Img size  : {IMG_SIZE}")
    print(f"   Device    : {DEVICE}")
    print(f"   Dataset   : {dataset_dir}")

    model = YOLO(MODEL_NAME)

    results = model.train(
        data        = dataset_dir,
        epochs      = EPOCHS,
        imgsz       = IMG_SIZE,
        batch       = BATCH_SIZE,
        lr0         = LEARNING_RATE,
        workers     = WORKERS,
        device      = DEVICE,
        project     = "runs/emotion",
        name        = "rafdb_yolov8",
        exist_ok    = True,
        pretrained  = True,
        patience    = 15,           # Early stopping
        save        = True,
        save_period = 10,
        plots       = True,
        verbose     = True,
        # Augmentations
        hsv_h       = 0.015,
        hsv_s       = 0.5,
        hsv_v       = 0.3,
        flipud      = 0.0,
        fliplr      = 0.5,
        degrees     = 10.0,
        translate   = 0.1,
        scale       = 0.3,
        mosaic      = 0.0,
    )

    best_model = "runs/emotion/rafdb_yolov8/weights/best.pt"
    print(f"\n✅ Entraînement terminé !")
    print(f"   Meilleur modèle : {best_model}")
    return best_model, results


# ─────────────────────────────────────────
# 4. COURBES D'ENTRAÎNEMENT
# ─────────────────────────────────────────
def plot_training_curves(run_dir: str = "runs/emotion/rafdb_yolov8"):
    """Affiche les courbes loss/accuracy depuis les résultats."""
    results_csv = os.path.join(run_dir, "results.csv")
    if not os.path.exists(results_csv):
        print(f"[!] Fichier results.csv introuvable : {results_csv}")
        return

    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Courbes d'Entraînement YOLOv8 - RAF-DB", fontsize=14)

    # Loss
    if "train/loss" in df.columns:
        axes[0].plot(df["epoch"], df["train/loss"], label="Train Loss", color="blue")
    if "val/loss" in df.columns:
        axes[0].plot(df["epoch"], df["val/loss"],   label="Val Loss",   color="orange")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    acc_cols = [c for c in df.columns if "acc" in c.lower()]
    colors   = ["blue", "orange", "green", "red"]
    for idx, col in enumerate(acc_cols[:4]):
        axes[1].plot(df["epoch"], df[col], label=col, color=colors[idx])
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(run_dir, "training_curves.png")
    plt.savefig(out, dpi=150)
    print(f"📈 Courbes sauvegardées : {out}")
    plt.show()


# ─────────────────────────────────────────
# 5. VALIDATION RAPIDE
# ─────────────────────────────────────────
def validate_model(model_path: str, dataset_dir: str):
    """Évalue le modèle sur le set de validation."""
    print(f"\n🔍 Validation du modèle : {model_path}")
    model = YOLO(model_path)
    metrics = model.val(data=dataset_dir, split="val", imgsz=IMG_SIZE)
    print(f"\n📊 Résultats Validation :")
    print(f"   Top-1 Accuracy : {metrics.top1:.4f}")
    print(f"   Top-5 Accuracy : {metrics.top5:.4f}")
    return metrics


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 ÉTAPE 2 : Entraînement YOLOv8 sur RAF-DB\n")

    # 1. Préparer split 50/50
    dataset_5050 = prepare_5050_split(DATASET_DIR, OUTPUT_DIR)

    # 2. (Optionnel) créer YAML si besoin
    # create_dataset_yaml(OUTPUT_DIR)

    # 3. Entraîner
    best_model, results = train_model(dataset_5050)

    # 4. Courbes
    plot_training_curves()

    # 5. Validation
    validate_model(best_model, dataset_5050)

    print("\n✅ ÉTAPE 2 TERMINÉE")
    print(f"   Modèle sauvegardé  : {best_model}")
    print("   Prochaine étape    : python 3_evaluate.py")
