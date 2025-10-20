"""
Fonctions de visualisation pour suivre l'entraînement du GAN.
Je crée des graphiques clairs pour analyser la progression.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from utils.config import *

def plot_losses(g_losses, d_losses, save_path=None):
    """
    Trace les courbes de perte du générateur et discriminateur.
    
    Args:
        g_losses (list): Liste des pertes du générateur
        d_losses (list): Liste des pertes du discriminateur
        save_path (str): Chemin pour sauvegarder (optionnel)
    """
    # TODO: Créer figure avec plt.figure(figsize=(10, 5))
    # TODO: Tracer g_losses en bleu (label='Generator Loss')
    # TODO: Tracer d_losses en rouge (label='Discriminator Loss')
    # TODO: Ajouter légende, titre, labels
    # TODO: Sauvegarder si save_path fourni
    # TODO: plt.show()
    pass

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
    # TODO: Générer noise aléatoire (examples, noise_dim)
    # TODO: Générer images avec generator.predict()
    # TODO: Dénormaliser [-1,1] → [0,1] : (images + 1) / 2
    # TODO: Créer grille 4x4 avec subplot
    # TODO: Sauvegarder f'{save_dir}/epoch_{epoch}.png'
    pass

def compare_real_fake(real_images, generator, noise_dim=NOISE_DIM, save_path=None):
    """
    Compare 8 images réelles vs 8 générées côte à côte.
    
    Args:
        real_images (np.array): Batch d'images réelles
        generator (tf.keras.Model): Modèle générateur
        noise_dim (int): Dimension du bruit
        save_path (str): Chemin pour sauvegarder
    """
    # TODO: Générer 8 images fake
    # TODO: Créer subplot (2, 8) : ligne 1 = real, ligne 2 = fake
    # TODO: Dénormaliser les images
    # TODO: Afficher avec imshow
    # TODO: Sauvegarder si save_path fourni
    pass