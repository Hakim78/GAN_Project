"""
Discriminateur pour le GAN.
Je crée un CNN qui prend une image (64, 64, 3) et retourne une probabilité [0, 1].
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from utils.config import *

def build_discriminator(img_size=IMG_SIZE, img_channels=IMG_CHANNELS):
    """
    Construit le discriminateur du GAN
    
    Architecture inspirée du code du prof (exemple MNIST adapté).
    
    Args:
        img_size (int): Taille des images d'entrée
        img_channels (int): Nombre de canaux (3 pour RGB)
    
    Returns:
        tf.keras.Model: Discriminateur compilé
    """

    # je def l'entrée des images
    image_input = Input(shape=(img_size, img_size, img_channels), name='image_input')

    # bloc 1 : Conv2D 64x64x3 -> 32x32x64
    x = layers.Conv2D(
        64,
        kernel_size=5,
        strides=2,
        padding='same',
        name='conv_1'
    )(image_input)

    # shape after 

    # j'applique LeakyReLU au lieu de relu afin de permettre au gradient négatif de passer
    x = layers.LeakyReLU(alpha=0.2, name='leaky_relu_1')(x)

    # j'évite l'overfitting avec du dropout
    x = layers.Dropout(0.3, name='dropout_1')(x)

    # bloc 2 : Conv2D 32x32x64 -> 16x16x128
    x = layers.Conv2D(
        128,
        kernel_size=5,
        strides=2,
        padding='same',
        name='conv_2'
    )(x)

    # shape after

    x = layers.LeakyReLU(alpha=0.2, name='leaky_relu_2')(x)
    x = layers.Dropout(0.3, name='dropout_2')(x)

    # bloc 3 : Conv2D 16x16x128 -> 8x8x256
    x = layers.Conv2D(
        256,
        kernel_size=5,
        strides=2,
        padding='same',
        name='conv_3'
    )(x)

    # shape after

    x = layers.LeakyReLU(alpha=0.2, name='leaky_relu_3')(x)
    x = layers.Dropout(0.3, name='dropout_3')(x)

    # flatten pour passer en fully connected
    x = layers.Flatten(name='flatten')(x)

    # couche de sortie : 1 neurone avec activation sigmoid pour probabilité [0, 1]
    output = layers.Dense(1, activation='sigmoid', name='output_dense')(x)

    # je crée le modèle Keras
    discriminator = Model(inputs=image_input, outputs=output, name='discriminator')

    return discriminator

#####  partie test ##### supp apres


if __name__ == '__main__':
    # Test du discriminateur
    print("=== Test du discriminateur ===\n")
    
    # Je crée le discriminateur
    discriminator = build_discriminator()
    
    # J'affiche l'architecture
    print("Architecture du discriminateur :")
    discriminator.summary()
    
    # Je teste la discrimination sur une image factice
    print("\n=== Test de discrimination ===")
    
    # Je crée une fausse image (bruit aléatoire normalisé [-1, 1])
    fake_image = tf.random.normal([1, IMG_SIZE, IMG_SIZE, IMG_CHANNELS])
    
    # Je passe l'image dans le discriminateur
    prediction = discriminator(fake_image, training=False)
    
    print(f"Shape de la prédiction : {prediction.shape}")
    print(f"Probabilité que l'image soit vraie : {prediction.numpy()[0][0]:.4f}")
    print(f"Probabilité que l'image soit fausse : {1 - prediction.numpy()[0][0]:.4f}")
    
    # Note : avant entraînement, la prédiction est aléatoire (environ 0.5)
    # C'est normal, le discriminateur n'a pas encore appris
    
    print("\n=== Test avec un batch ===")
    
    # Je teste avec un batch de 5 images
    batch_images = tf.random.normal([5, IMG_SIZE, IMG_SIZE, IMG_CHANNELS])
    batch_predictions = discriminator(batch_images, training=False)
    
    print(f"Shape des prédictions : {batch_predictions.shape}")
    print("Prédictions pour chaque image :")
    for i, pred in enumerate(batch_predictions.numpy()):
        print(f"  Image {i+1}: {pred[0]:.4f} (probabilité d'être vraie)")
    
    print("\nTest terminé avec succès !")