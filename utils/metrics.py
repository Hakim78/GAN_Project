"""
Métriques d'évaluation pour les GANs.
Pour l'instant, je crée juste la structure de base.
Le FID score sera ajouté en bonus (semaine 3).
"""

import numpy as np
import tensorflow as tf

def calculate_inception_score(images, num_splits=10):
    """
    Calcule l'Inception Score (IS) - BONUS.
    Pour l'instant, je laisse en TODO.
    
    Args:
        images (np.array): Images générées
        num_splits (int): Nombre de splits pour le calcul
    
    Returns:
        float: IS score
    """
    # TODO BONUS SEMAINE 3
    pass

def calculate_fid(real_images, fake_images):
    """
    Calcule le Fréchet Inception Distance (FID) - BONUS.
    Pour l'instant, je laisse en TODO.
    
    Args:
        real_images (np.array): Images réelles
        fake_images (np.array): Images générées
    
    Returns:
        float: FID score (plus bas = meilleur)
    """
    # TODO BONUS SEMAINE 3
    pass

def log_training_metrics(epoch, g_loss, d_loss):
    """
    Affiche les métriques pendant l'entraînement.
    
    Args:
        epoch (int): Numéro d'epoch
        g_loss (float): Perte du générateur
        d_loss (float): Perte du discriminateur
    """
    # TODO: print(f'Epoch {epoch} | G Loss: {g_loss:.4f} | D Loss: {d_loss:.4f}')
    pass