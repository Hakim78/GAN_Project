"""
Fonctions de visualisation pour suivre l'entraînement du GAN.
Je crée des graphiques clairs pour analyser la progression.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from utils.config import *

# ---------- Helpers ----------

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _denorm(x: np.ndarray) -> np.ndarray:
    """
    Dénormalise des images de [-1, 1] -> [0, 1] et clippe dans [0,1].
    """
    x = (x + 1.0) / 2.0
    return np.clip(x, 0.0, 1.0)

# ---------- API demandée ----------

def plot_losses(g_losses, d_losses, save_path=None):
    """
    Trace les courbes de perte du générateur et discriminateur.

    Args:
        g_losses (list): Liste des pertes du générateur
        d_losses (list): Liste des pertes du discriminateur
        save_path (str): Chemin pour sauvegarder (optionnel)
    """
    _ensure_dir(PLOTS_DIR)
    if save_path is None:
        save_path = os.path.join(PLOTS_DIR, "loss_curves.png")

    plt.figure(figsize=(10, 5))
    # G en bleu
    plt.plot(g_losses, label='Generator Loss', color='tab:blue')
    # D en rouge
    plt.plot(d_losses, label='Discriminator Loss', color='tab:red')
    plt.title("GAN Training Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()

def save_generated_images(epoch, generator, noise_dim=NOISE_DIM, examples=16, save_dir=GENERATED_DIR):
    """
    Génère et sauvegarde une grille d'images.

    Args:
        epoch (int): Numéro de l'epoch actuel
        generator (tf.keras.Model): Modèle générateur
        noise_dim (int): Dimension du bruit
        examples (int): Nombre d'images à générer (16 = grille 4x4)
        save_dir (str): Dossier de sauvegarde
    """
    _ensure_dir(save_dir)

    # Générer du bruit puis des images
    noise = np.random.randn(examples, noise_dim).astype("float32")
    preds = generator.predict(noise, verbose=0)

    # Dénormaliser [-1,1] → [0,1]
    imgs = _denorm(preds)

    # Grille 4x4
    rows, cols = 4, 4
    fig, axes = plt.subplots(rows, cols, figsize=(8, 8))
    axes = axes.flatten()
    for i in range(rows * cols):
        axes[i].imshow(imgs[i])
        axes[i].axis('off')

    plt.suptitle(f"Generated @ epoch {epoch}")
    plt.tight_layout()

    # Sauvegarde
    out_path = os.path.join(save_dir, f"epoch_{epoch}.png")
    plt.savefig(out_path, dpi=120)
    plt.close()

def compare_real_fake(real_images, generator, noise_dim=NOISE_DIM, save_path=None):
    """
    Compare 8 images réelles vs 8 générées côte à côte.

    Args:
        real_images (np.array): Batch d'images réelles ([-1,1])
        generator (tf.keras.Model): Modèle générateur
        noise_dim (int): Dimension du bruit
        save_path (str): Chemin pour sauvegarder
    """
    _ensure_dir(GENERATED_DIR)
    if save_path is None:
        save_path = os.path.join(GENERATED_DIR, "real_vs_fake.png")

    # 8 images réelles
    real = real_images[:8]
    # tf.Tensor -> np (si besoin)
    if hasattr(real, "numpy"):
        real = real.numpy()
    real = _denorm(real)

    # 8 images fausses
    noise = np.random.randn(8, noise_dim).astype("float32")
    fake = generator.predict(noise, verbose=0)
    fake = _denorm(fake)

    # Subplot (2, 8) : ligne 1 = real, ligne 2 = fake
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    for i in range(8):
        axes[0, i].imshow(real[i]); axes[0, i].axis('off')
        axes[1, i].imshow(fake[i]); axes[1, i].axis('off')
    plt.suptitle("Top: Real | Bottom: Fake")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
