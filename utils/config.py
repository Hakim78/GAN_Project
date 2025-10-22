"""
Configuration centralisée pour tous les hyperparamètres.
Je modifie ce fichier pour ajuster mes hyperparamètres.
"""

# Chemins
DATA_DIR = "data/celeba/img_align_celeba"
OUTPUT_DIR = 'outputs/'
CHECKPOINT_DIR = 'outputs/checkpoints/'
GENERATED_DIR = 'outputs/generated_images/'
PLOTS_DIR = 'outputs/plots/'

# Hyperparamètres généraux
IMG_SIZE = 64  # ou 128
IMG_CHANNELS = 3
NOISE_DIM = 100
BATCH_SIZE = 128
EPOCHS = 10000

# Optimiseurs
LEARNING_RATE_G = 1e-4
LEARNING_RATE_D = 2e-4
BETA_1 = 0.5
BETA_2 = 0.999

# Sauvegarde
SAVE_INTERVAL = 500  # Sauvegarder images tous les N epochs
CHECKPOINT_INTERVAL = 1000  # Sauvegarder modèle tous les N epochs

# Attention (V2)
USE_ATTENTION = True  # Activer/désactiver CBAM
ATTENTION_RATIO = 8

# Diffusion (V3 bonus)
DIFFUSION_STEPS = 1000
BETA_START = 1e-4
BETA_END = 0.02