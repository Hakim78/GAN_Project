"""
Fonctions de visualisation pour suivre l'entraînement du GAN.
Je crée des graphiques clairs pour analyser la progression.

tools :
- Matplotlib : bibliothèque standard pour visualisation (prof l'utilise)
- Grilles d'images : permet de voir l'évolution de la qualité
- Courbes de perte : essentiel pour détecter problèmes (mode collapse, divergence)
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from utils.config import *

def plot_losses(g_losses, d_losses, save_path=None):
    """
    Trace les courbes de perte du générateur et discriminateur
    
    - Deux courbes sur le même graphique pour comparaison facile
    - Bleu pour générateur, rouge pour discriminateur (convention)
    - Grille pour faciliter la lecture des valeurs
    - Sauvegarde automatique pour garder l'historique
    
    Args:
        g_losses (list): Liste des pertes du générateur
        d_losses (list): Liste des pertes du discriminateur
        save_path (str): Chemin pour sauvegarder (optionnel)
    """
    # figure avec plt.figure(figsize=(10, 5))
    plt.figure(figsize=(10,5))

    # TODO: Tracer g_losses en bleu (label='Generator Loss')
    plt.plot(g_losses, label='Generator Loss', color='blue', linewidth=1.5)

    # TODO: Tracer d_losses en rouge (label='Discriminator Loss')
    plt.plot(d_losses, label='Discriminator Loss', color='red', linewidth=1.5)
    
    # J'ajoute les labels et le titre
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Losses (Generator vs Discriminator)', fontsize=14, fontweight='bold')
    
    # J'affiche la légende
    plt.legend(loc='best', fontsize=10)

    # grille 
    plt.grid(True, alpha=0.3)

    #  save if save_path fourni
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Graphique des pertes sauvegardé dans {save_path}")
    # plt.show()
    plt.show()
    plt.close()

    pass

def save_generated_images(epoch, generator, noise_dim=NOISE_DIM, examples=16, save_dir=GENERATED_DIR):
    """
    Génère et sauvegarde une grille d'images.
    
    Justifications :
    - examples=16 : grille 4x4, bon compromis entre diversité et lisibilité
    - Dénormalisation [-1,1]→[0,1] : nécessaire pour affichage correct avec matplotlib
    - Format PNG : sans perte, idéal pour images générées
    - Nom de fichier avec epoch : permet de suivre la progression
    
    Args:
        epoch (int): Numéro de l'epoch actuel
        generator (tf.keras.Model): Modèle générateur
        noise_dim (int): Dimension du bruit
        examples (int): Nombre d'images à générer (16 = grille 4x4)
        save_dir (str): Dossier de sauvegarde
    """
    # Je crée le dossier de sauvegarde s'il n'existe pas
    os.makedirs(save_dir, exist_ok=True)
    
    # Je génère du bruit aléatoire
    # Shape : (16, 100) pour générer 16 images
    # Justification : distribution normale standard (mean=0, std=1) comme pendant l'entraînement
    noise = tf.random.normal([examples, noise_dim])
    
    # Je génère les images avec le générateur
    # training=False : mode inférence (pas de dropout, BatchNorm en mode test)
    generated_images = generator(noise, training=False)
    
    # Je dénormalise les images de [-1, 1] à [0, 1]
    # Justification : matplotlib attend des valeurs [0, 1] pour afficher correctement
    # Formule : (x + 1) / 2 transforme [-1, 1] en [0, 1]
    generated_images = (generated_images + 1) / 2.0
    
    # Je crée une grille 4x4
    # Justification : 4x4 = 16 images, format carré, facile à visualiser
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    
    # J'aplatit le tableau 2D d'axes pour itérer facilement
    axes = axes.flatten()
    
    # Je remplis la grille avec les images générées
    for i in range(examples):
        # J'affiche l'image dans la sous-figure correspondante
        axes[i].imshow(generated_images[i])
        
        # Je désactive les axes pour un affichage propre
        # Justification : les axes (x, y) ne sont pas pertinents pour des images
        axes[i].axis('off')
    
    # J'ajoute un titre global avec le numéro d'epoch
    plt.suptitle(f'Generated Images - Epoch {epoch}', fontsize=14, fontweight='bold')
    
    # J'ajuste l'espacement entre les sous-figures
    # Justification : évite que les images se chevauchent
    plt.tight_layout()
    
    # Je sauvegarde la grille
    # Format : epoch_00001000.png, epoch_00002000.png, etc.
    # Justification : zfill(8) assure un tri alphabétique correct
    save_path = os.path.join(save_dir, f'epoch_{str(epoch).zfill(8)}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    # Je ferme la figure pour libérer la mémoire
    # Justification : évite l'accumulation de figures en mémoire pendant l'entraînement
    plt.close()


def compare_real_fake(real_images, generator, noise_dim=NOISE_DIM, save_path=None):
    """
    Compare 8 images réelles vs 8 générées côte à côte.
    
    Justifications :
    - 8 images : suffisant pour voir la diversité sans surcharger
    - Disposition verticale : ligne du haut = réelles, ligne du bas = fausses
    - Permet de comparer directement la qualité du générateur
    
    Args:
        real_images (np.array): Batch d'images réelles (normalisées [-1, 1])
        generator (tf.keras.Model): Modèle générateur
        noise_dim (int): Dimension du bruit
        save_path (str): Chemin pour sauvegarder (optionnel)
    """
    # Je génère 8 images fake
    noise = tf.random.normal([8, noise_dim])
    fake_images = generator(noise, training=False)
    
    # Je crée une grille 2 lignes × 8 colonnes
    # Ligne 1 : images réelles
    # Ligne 2 : images générées
    # Justification : permet la comparaison directe colonne par colonne
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    
    # Je remplis la première ligne avec les images réelles
    for i in range(8):
        # Je dénormalise de [-1, 1] à [0, 1]
        real_img = (real_images[i] + 1) / 2.0
        axes[0, i].imshow(real_img)
        axes[0, i].axis('off')
        
        # J'ajoute un titre seulement à la première image de chaque ligne
        if i == 0:
            axes[0, i].set_title('Real Images', fontsize=12, fontweight='bold')
    
    # Je remplis la deuxième ligne avec les images générées
    for i in range(8):
        # Je dénormalise de [-1, 1] à [0, 1]
        fake_img = (fake_images[i] + 1) / 2.0
        axes[1, i].imshow(fake_img)
        axes[1, i].axis('off')
        
        # J'ajoute un titre seulement à la première image
        if i == 0:
            axes[1, i].set_title('Generated Images', fontsize=12, fontweight='bold')
    
    # J'ajuste l'espacement
    plt.tight_layout()
    
    # Je sauvegarde si un chemin est fourni
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparaison réel/fake sauvegardée dans {save_path}")
    
    # J'affiche le graphique
    plt.show()
    plt.close()


if __name__ == '__main__':
    # Test des fonctions de visualisation
    print("=== Test des fonctions de visualisation ===\n")
    
    # Test 1 : plot_losses
    print("Test 1 : Courbes de perte")
    fake_g_losses = [2.0 - 0.001 * i for i in range(1000)]
    fake_d_losses = [0.8 - 0.0003 * i for i in range(1000)]
    plot_losses(fake_g_losses, fake_d_losses, save_path='outputs/plots/test_losses.png')
    
    # Test 2 : save_generated_images
    print("\nTest 2 : Grille d'images générées")
    from models.generator import build_generator
    generator = build_generator()
    save_generated_images(epoch=1000, generator=generator, save_dir='outputs/test_images/')
    print("Grille sauvegardée dans outputs/test_images/")
    
    # Test 3 : compare_real_fake
    print("\nTest 3 : Comparaison réel vs fake")
    from utils.data_loader import load_celeba_dataset
    dataset = load_celeba_dataset()
    
    # Je récupère un batch de vraies images
    for real_batch in dataset.take(1):
        compare_real_fake(real_batch, generator, save_path='outputs/plots/test_comparison.png')
    
    print("\nTest terminé avec succès !")
    print("\nFichiers créés :")
    print("  - outputs/plots/test_losses.png")
    print("  - outputs/test_images/epoch_00001000.png")
    print("  - outputs/plots/test_comparison.png")