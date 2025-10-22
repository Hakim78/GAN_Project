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

    Architecture inspirée du code du prof (exemple MNIST adapté) et du README:
    - Conv2D(64, k=5, s=2)  → LeakyReLU(0.2) → Dropout(0.3)
    - Conv2D(128, k=5, s=2) → LeakyReLU(0.2) → Dropout(0.3)
    - Conv2D(256, k=5, s=2) → LeakyReLU(0.2) → Dropout(0.3)
    - Flatten → Dense(1, sigmoid)

    Args:
        img_size (int): Taille des images d'entrée (64)
        img_channels (int): Nombre de canaux (3 pour RGB)

    Returns:
        tf.keras.Model: Discriminateur (non compilé, pour entraînement custom)
    """
    # Input layer (img_size, img_size, img_channels)
    inp = layers.Input(shape=(img_size, img_size, img_channels), name="disc_input")

    # Bloc 1
    x = layers.Conv2D(64, kernel_size=5, strides=2, padding='same',
                      kernel_initializer='he_normal', name='conv1')(inp)
    x = layers.LeakyReLU(alpha=0.2, name='lrelu1')(x)
    x = layers.Dropout(0.3, name='drop1')(x)

    # Bloc 2
    x = layers.Conv2D(128, kernel_size=5, strides=2, padding='same',
                      kernel_initializer='he_normal', name='conv2')(x)
    x = layers.LeakyReLU(alpha=0.2, name='lrelu2')(x)
    x = layers.Dropout(0.3, name='drop2')(x)

    # Bloc 3
    x = layers.Conv2D(256, kernel_size=5, strides=2, padding='same',
                      kernel_initializer='he_normal', name='conv3')(x)
    x = layers.LeakyReLU(alpha=0.2, name='lrelu3')(x)
    x = layers.Dropout(0.3, name='drop3')(x)

    # Tête binaire
    x = layers.Flatten(name='flatten')(x)
    out = layers.Dense(1, activation='sigmoid', name='prob')(x)

    model = Model(inputs=inp, outputs=out, name='Discriminator')

    # On NE compile PAS ici pour laisser la boucle d'entraînement custom gérer
    # les pertes/optimiseurs (train_gan.py).
    return model


if __name__ == '__main__':
    # Petit test rapide : shape + passage avant entraînement
    print("=== Test du discriminateur ===\n")
    disc = build_discriminator()
    disc.summary()

    # Batch factice en [-1, 1] (comme les vraies images après normalisation)
    dummy = tf.random.uniform([4, IMG_SIZE, IMG_SIZE, IMG_CHANNELS], minval=-1.0, maxval=1.0)
    y = disc(dummy, training=False)
    print("\nSortie du discriminateur (shape, min, max):",
          y.shape, float(tf.reduce_min(y)), float(tf.reduce_max(y)))
    print("\nTest terminé avec succès !")
