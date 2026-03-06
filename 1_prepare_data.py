"""
ÉTAPE 1 - Préparation des Données RAF-DB pour YOLOv8
=====================================================
Ce script :
- Charge et explore le dataset RAF-DB
- Convertit la structure en format YOLOv8 Classification
- Split train/val/test (50/25/25 ou selon RAF-DB officiel)
- Applique les augmentations de base
"""

import os
import shutil
import random
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
RAF_DB_ROOT   = "./RAF-DB"          # ← Chemin vers votre dossier RAF-DB
OUTPUT_DIR    = "./dataset_yolo"    # ← Où le dataset YOLO sera créé
IMG_SIZE      = 224                 # Taille des images (224x224)
RANDOM_SEED   = 42

# 7 classes de base RAF-DB
EMOTION_LABELS = {
    1: "surprise",
    2: "fear",
    3: "disgust",
    4: "happiness",
    5: "sadness",
    6: "anger",
    7: "neutral"
}

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ─────────────────────────────────────────
# 1. EXPLORATION DU DATASET
# ─────────────────────────────────────────
def explore_rafdb(raf_root: str):
    """Affiche les statistiques du dataset RAF-DB."""
    label_file = os.path.join(raf_root, "basic", "EmoLabel", "list_patition_label.txt")

    if not os.path.exists(label_file):
        print(f"[ERREUR] Fichier label introuvable : {label_file}")
        print("Structure attendue :")
        print("  RAF-DB/")
        print("  ├── basic/")
        print("  │   ├── EmoLabel/list_patition_label.txt")
        print("  │   └── Image/aligned/  (images *_aligned.jpg)")
        return None

    df = pd.read_csv(label_file, sep=" ", header=None, names=["filename", "label"])
    df["split"] = df["filename"].apply(lambda x: "train" if x.startswith("train") else "test")
    df["emotion"] = df["label"].map(EMOTION_LABELS)

    print("=" * 50)
    print("📊 STATISTIQUES RAF-DB")
    print("=" * 50)
    print(f"Total images      : {len(df)}")
    print(f"Images train      : {len(df[df['split']=='train'])}")
    print(f"Images test       : {len(df[df['split']=='test'])}")
    print()
    print("Distribution par émotion :")
    for emo, count in df["emotion"].value_counts().items():
        bar = "█" * (count // 100)
        print(f"  {emo:<12} : {count:>5}  {bar}")
    print("=" * 50)

    return df


# ─────────────────────────────────────────
# 2. CONVERSION FORMAT YOLOv8
# ─────────────────────────────────────────
def build_yolo_dataset(df: pd.DataFrame, raf_root: str, output_dir: str):
    """
    Crée la structure de dossiers YOLOv8 Classification :
    dataset_yolo/
    ├── train/
    │   ├── happiness/  *.jpg
    │   ├── sadness/    *.jpg
    │   └── ...
    ├── val/
    └── test/
    """
    img_dir = os.path.join(raf_root, "basic", "Image", "aligned")

    # Séparer train en train+val (80/20)
    train_df = df[df["split"] == "train"].copy()
    test_df  = df[df["split"] == "test"].copy()

    train_df = train_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    val_size  = int(len(train_df) * 0.2)
    val_df    = train_df[:val_size]
    train_df  = train_df[val_size:]

    splits = {"train": train_df, "val": val_df, "test": test_df}

    print("\n📁 Création du dataset YOLOv8...")

    for split_name, split_df in splits.items():
        print(f"\n  → Split '{split_name}' ({len(split_df)} images)")
        for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"  {split_name}"):
            emotion = row["emotion"]
            fname   = row["filename"]

            # Nom de l'image alignée
            base    = fname.replace(".jpg", "") + "_aligned.jpg"
            src     = os.path.join(img_dir, base)

            dst_dir = os.path.join(output_dir, split_name, emotion)
            os.makedirs(dst_dir, exist_ok=True)
            dst     = os.path.join(dst_dir, base)

            if not os.path.exists(src):
                continue  # image manquante, on passe

            # Redimensionner et sauvegarder
            img = Image.open(src).convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
            img.save(dst, quality=95)

    print("\n✅ Dataset créé avec succès !")
    _print_dataset_summary(output_dir)
    return output_dir


# ─────────────────────────────────────────
# 3. VÉRIFICATION VISUELLE
# ─────────────────────────────────────────
def visualize_samples(output_dir: str, n_per_class: int = 4):
    """Affiche une grille d'exemples par classe."""
    fig, axes = plt.subplots(7, n_per_class, figsize=(n_per_class * 2, 16))
    fig.suptitle("Exemples RAF-DB par émotion", fontsize=14, fontweight="bold")

    for i, emotion in enumerate(EMOTION_LABELS.values()):
        cls_dir = os.path.join(output_dir, "train", emotion)
        if not os.path.exists(cls_dir):
            continue
        images = list(Path(cls_dir).glob("*.jpg"))[:n_per_class]
        for j, img_path in enumerate(images):
            img = Image.open(img_path)
            axes[i][j].imshow(img)
            axes[i][j].axis("off")
            if j == 0:
                axes[i][j].set_ylabel(emotion, fontsize=10, rotation=0,
                                       labelpad=60, va="center")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dataset_preview.png"), dpi=100)
    print(f"\n🖼️  Aperçu sauvegardé : {output_dir}/dataset_preview.png")
    plt.show()


def _print_dataset_summary(output_dir: str):
    print("\n📊 Résumé du dataset créé :")
    for split in ["train", "val", "test"]:
        split_path = os.path.join(output_dir, split)
        if not os.path.exists(split_path):
            continue
        total = sum(len(list(Path(os.path.join(split_path, c)).glob("*.jpg")))
                    for c in os.listdir(split_path)
                    if os.path.isdir(os.path.join(split_path, c)))
        print(f"  {split:<8}: {total} images")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 ÉTAPE 1 : Préparation des données RAF-DB\n")

    # 1. Explorer
    df = explore_rafdb(RAF_DB_ROOT)

    if df is not None:
        # 2. Construire le dataset YOLO
        build_yolo_dataset(df, RAF_DB_ROOT, OUTPUT_DIR)

        # 3. Visualiser
        visualize_samples(OUTPUT_DIR)

        print("\n✅ ÉTAPE 1 TERMINÉE")
        print(f"   Dataset prêt dans : {OUTPUT_DIR}/")
        print("   Prochaine étape   : python 2_train_model.py")
