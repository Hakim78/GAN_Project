"""
Configuration centralisée pour tous les hyperparamètres.
Version : entraînement sur 1000 epochs, sauvegarde d’images toutes les 50.
"""

# -----------------------------
# Chemins
# -----------------------------
DATA_DIR = "data/celeba/img_align_celeba"
OUTPUT_DIR = "outputs/"
CHECKPOINT_DIR = "outputs/checkpoints/"
GENERATED_DIR = "outputs/generated_images/"
PLOTS_DIR = "outputs/plots/"

# -----------------------------
# Hyperparamètres généraux
# -----------------------------
IMG_SIZE = 64          # Taille des images
IMG_CHANNELS = 3       # RGB
NOISE_DIM = 100        # Dimension du bruit
BATCH_SIZE = 128       # Baisse à 64/32 si mémoire limite
EPOCHS = 1000          # 🔹 Entraînement sur 1000 epochs
SAVE_INTERVAL = 50     # 🔹 Sauvegarder des images toutes les 50 epochs
CHECKPOINT_INTERVAL = 200  # 🔹 Sauvegarder les modèles tous les 200 epochs

# -----------------------------
# Optimiseurs
# -----------------------------
LEARNING_RATE_G = 1e-4
LEARNING_RATE_D = 2e-4
BETA_1 = 0.5
BETA_2 = 0.999

# -----------------------------
# Attention (V2)
# -----------------------------
USE_ATTENTION = True
ATTENTION_RATIO = 8

# -----------------------------
# Diffusion (V3 bonus)
# -----------------------------
DIFFUSION_STEPS = 1000
BETA_START = 1e-4
BETA_END = 0.02
