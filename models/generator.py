"""
Générateur pour le GAN.
Je crée un réseau qui transforme un vecteur aléatoire (100,) en image (64, 64, 3).
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from utils.config import *

def build_generator(noise_dim=NOISE_DIM, img_size=IMG_SIZE, img_channels=IMG_CHANNELS):
    """
    Construit le générateur.
    
    Architecture inspirée du code du prof (exemple MNIST adapté).
    
    Args:
        noise_dim (int): Dimension du vecteur latent (100)
        img_size (int): Taille de sortie (64 ou 128)
        img_channels (int): Nombre de canaux (3 pour RGB)
    
    Returns:
        tf.keras.Model: Générateur compilé
    """
    # TODO: Input layer (noise_dim,)
    # TODO: Dense(8*8*256, use_bias=False)
    # TODO: BatchNormalization()
    # TODO: ReLU()
    # TODO: Reshape((8, 8, 256))
    #
    # TODO: Conv2DTranspose(128, kernel_size=5, strides=2, padding='same', use_bias=False)
    # TODO: BatchNormalization()
    # TODO: ReLU()
    #
    # TODO: Conv2DTranspose(64, kernel_size=5, strides=2, padding='same', use_bias=False)
    # TODO: BatchNormalization()
    # TODO: ReLU()
    #
    # TODO: Conv2DTranspose(3, kernel_size=5, strides=2, padding='same', use_bias=False, activation='tanh')
    
    pass