"""
Chargement et prétraitement du dataset CelebA.
Je m'assure que les images sont normalisées entre [-1, 1] pour fonctionner avec tanh.
"""

import tensorflow as tf
from utils.config import *

def load_celeba_dataset(data_dir=DATA_DIR, batch_size=BATCH_SIZE, img_size=IMG_SIZE):
    """
    Charge le dataset CelebA depuis le dossier local.
    
    Args:
        data_dir (str): Chemin vers le dossier contenant les images
        batch_size (int): Taille des batches
        img_size (int): Taille cible des images (64 ou 128)
    
    Returns:
        tf.data.Dataset: Dataset prêt pour l'entraînement
    """
    # TODO: Je dois implémenter le chargement des images
    # TODO: Je dois les redimensionner à (img_size, img_size)
    # TODO: Je dois les normaliser entre [-1, 1]
    # TODO: Je dois créer des batches et shuffle
    pass

def preprocess_image(image_path):
    """
    Prétraite une image individuelle.
    
    Args:
        image_path (str): Chemin vers l'image
    
    Returns:
        tf.Tensor: Image normalisée (img_size, img_size, 3)
    """
    # TODO: Charger l'image
    # TODO: Redimensionner
    # TODO: Normaliser [-1, 1] = (pixel - 127.5) / 127.5
    pass