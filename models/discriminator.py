"""
Discriminateur pour le GAN.
Je crée un CNN qui prend une image (64, 64, 3) et retourne une probabilité [0, 1].
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from utils.config import *

def build_discriminator(img_size=IMG_SIZE, img_channels=IMG_CHANNELS):
    """
    Construit le discriminateur.
    
    Architecture inspirée du code du prof (exemple MNIST adapté).
    
    Args:
        img_size (int): Taille des images d'entrée
        img_channels (int): Nombre de canaux (3 pour RGB)
    
    Returns:
        tf.keras.Model: Discriminateur compilé
    """
    # TODO: Input layer (img_size, img_size, img_channels)
    # TODO: Conv2D(64, kernel_size=5, strides=2, padding='same')
    # TODO: LeakyReLU(0.2)
    # TODO: Dropout(0.3)
    # 
    # TODO: Conv2D(128, kernel_size=5, strides=2, padding='same')
    # TODO: LeakyReLU(0.2)
    # TODO: Dropout(0.3)
    #
    # TODO: Conv2D(256, kernel_size=5, strides=2, padding='same')
    # TODO: LeakyReLU(0.2)
    # TODO: Dropout(0.3)
    #
    # TODO: Flatten()
    # TODO: Dense(1, activation='sigmoid')
    
    pass