# GAN avec Attention - Génération de Visages CelebA

## Installation
```bash
# Cloner le repo
git clone <url>
cd GAN_CelebA_Project

# Créer environnement
python -m venv venv
venv\Scripts\activate  # Windows

# download dépendances
pip install -r requirements.txt
```

## Télécharger le dataset
1. Aller sur https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
2. Télécharger et extraire dans `data/celeba/`

## Lancer l'entraînement
```bash
# V1 : GAN classique
python training/train_gan.py

# V2 : GAN + Attention
python training/train_gan_attention.py
```

## Structure du projet
Voir `docs/architecture.md`
```

---

## 5. Architecture des Réseaux (Vue d'ensemble)

### Générateur V1 (baseline)
```
Input: Noise (100,)
    ↓
Dense(8*8*256)
    ↓
Reshape(8, 8, 256)
    ↓
Conv2DTranspose(128) → 16x16x128
BatchNorm + ReLU
    ↓
Conv2DTranspose(64) → 32x32x64
BatchNorm + ReLU
    ↓
Conv2DTranspose(3) → 64x64x3
Tanh activation
    ↓
Output: Image (64, 64, 3)
```

### Générateur V2 (avec attention)
```
[Même architecture que V1]
MAIS avec blocs CBAM après chaque Conv2DTranspose:
    ↓
Conv2DTranspose(128) → CBAM → BatchNorm → ReLU
    ↓
Conv2DTranspose(64) → CBAM → BatchNorm → ReLU
    ↓
Conv2DTranspose(3) → Tanh
```

### Discriminateur
```
Input: Image (64, 64, 3)
    ↓
Conv2D(64, stride=2) → 32x32x64
LeakyReLU + Dropout
    ↓
Conv2D(128, stride=2) → 16x16x128
LeakyReLU + Dropout
    ↓
Conv2D(256, stride=2) → 8x8x256
LeakyReLU + Dropout
    ↓
Flatten
    ↓
Dense(1) + Sigmoid
    ↓
Output: Probabilité [0, 1]