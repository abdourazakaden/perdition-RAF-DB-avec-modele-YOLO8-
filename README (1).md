# 🎭 Emotion Recognition — YOLOv8 × RAF-DB

## 📁 Structure du Projet

```
projet/
├── RAF-DB/                    ← Dataset RAF-DB (à télécharger)
│   └── basic/
│       ├── EmoLabel/list_patition_label.txt
│       └── Image/aligned/     ← Images *_aligned.jpg
│
├── 1_prepare_data.py          ← Préparation données
├── 2_train_model.py           ← Entraînement YOLOv8 (50/50)
├── 3_evaluate.py              ← Évaluation sur le test set
├── 4_predict.py               ← Fonction de prédiction
├── 5_app_streamlit.py         ← Application Web
├── requirements.txt
└── README.md
```

## 🚀 Installation

```bash
pip install -r requirements.txt
```

## 📊 7 Classes d'Émotions

| Classe     | Emoji | Label RAF-DB |
|------------|-------|--------------|
| Surprise   | 😲    | 1            |
| Fear       | 😨    | 2            |
| Disgust    | 🤢    | 3            |
| Happiness  | 😊    | 4            |
| Sadness    | 😢    | 5            |
| Anger      | 😠    | 6            |
| Neutral    | 😐    | 7            |

## ▶️ Exécution Étape par Étape

### Étape 1 — Préparer les données
```bash
python 1_prepare_data.py
```
- Lit RAF-DB depuis `./RAF-DB/`
- Crée `./dataset_yolo/` avec structure train/val/test
- Redimensionne toutes les images à 224×224

### Étape 2 — Entraîner le modèle (Split 50/50)
```bash
python 2_train_model.py
```
- Split 50% train / 50% val
- Modèle : YOLOv8m-cls (pré-entraîné ImageNet)
- 50 epochs avec early stopping (patience=15)
- Sauvegarde dans `runs/emotion/rafdb_yolov8/weights/best.pt`

### Étape 3 — Évaluer sur le test set
```bash
python 3_evaluate.py
```
- Calcule Top-1 / Top-5 accuracy
- Génère la matrice de confusion
- Rapport CSV par classe (Precision/Recall/F1)

### Étape 4 — Tester la prédiction
```bash
python 4_predict.py
```
- Test sur une image fixe
- Décommenter `predictor.run_webcam()` pour temps réel

### Étape 5 — Lancer l'application Web
```bash
streamlit run 5_app_streamlit.py
```
- Ouvrez http://localhost:8501
- Uploadez une photo ou utilisez la webcam
- Résultats avec graphiques interactifs

## ⚙️ Configuration

Modifiez les variables en haut de chaque script :

| Paramètre   | Fichier          | Valeur par défaut |
|-------------|------------------|-------------------|
| RAF_DB_ROOT | 1_prepare_data   | `./RAF-DB`        |
| MODEL_NAME  | 2_train_model    | `yolov8m-cls.pt`  |
| EPOCHS      | 2_train_model    | `50`              |
| BATCH_SIZE  | 2_train_model    | `32`              |
| DEVICE      | 2_train_model    | `"0"` (GPU)       |

## 📈 Résultats Attendus

| Modèle      | Top-1 Acc (test) |
|-------------|-----------------|
| YOLOv8n-cls | ~82%            |
| YOLOv8s-cls | ~85%            |
| YOLOv8m-cls | ~87%            |
| YOLOv8l-cls | ~88%            |

## 🔍 Structure RAF-DB Attendue

```
RAF-DB/
└── basic/
    ├── EmoLabel/
    │   └── list_patition_label.txt   ← "train_00001.jpg 4"
    └── Image/
        └── aligned/
            ├── train_00001_aligned.jpg
            ├── train_00002_aligned.jpg
            └── test_0001_aligned.jpg
```
