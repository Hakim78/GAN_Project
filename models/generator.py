"""
Générateur pour le GAN.
Je crée un réseau qui transforme un vecteur aléatoire (100,) en image (64, 64, 3).
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from utils.config import *

def build_generator(noise_dim=NOISE_DIM, img_size=IMG_SIZE, img_channels=IMG_CHANNELS):
    """
     Construit le générateur du GAN.
    
    Architecture progressive : 
    Vecteur latent (100,) → Dense → Reshape 8x8x256 → Upsampling × 3 → Image 64x64x3
    
    Justifications des choix architecturaux :
    
    1. Dense(8*8*256) :
       - 8x8 = taille de départ (sera upsampleé 3 fois : 8→16→32→64)
       - 256 canaux = profondeur initiale (prof utilise 256 pour MNIST)
       - use_bias=False : BatchNorm compense le biais, économise des paramètres
    
    2. BatchNormalization :
       - Stabilise l'entraînement du GAN (essentiel selon le prof)
       - Normalise les activations entre couches
       - Réduit le mode collapse
    
    3. ReLU :
       - Activation standard pour les couches cachées (prof utilise ReLU)
       - Plus simple que LeakyReLU pour le générateur
    
    4. Conv2DTranspose (upsampling) :
       - kernel_size=5 : standard pour GANs (prof utilise 5x5)
       - strides=2 : double la résolution à chaque couche (8→16→32→64)
       - padding='same' : garde les dimensions prévisibles
       - use_bias=False : BatchNorm compense
    
    5. Progression des canaux : 256 → 128 → 64 → 3
       - Diminution progressive (pyramide inversée)
       - 3 canaux finaux pour RGB
    
    6. Activation finale tanh :
       - Produit des valeurs dans [-1, 1]
       - Correspond à la normalisation des vraies images
       - Standard pour les GANs (prof utilise tanh dans tous ses exemples)
    
    Args:
        noise_dim (int): Dimension du vecteur latent (100 par défaut)
        img_size (int): Taille de sortie (64 ou 128)
        img_channels (int): Nombre de canaux (3 pour RGB)
    
    Returns:
        tf.keras.Model: Générateur compilé
    """
    
    # je def l'entrée du veteur de bruit
    noise_input = Input(shape=(noise_dim,), name='noise_input')

    # couche 1 dense layer pour passer de 100 à 8*8*256 = chp cmb de neurones
    x = layers.Dense(8 * 8 * 256, use_bias=False, name='dense_projection')(noise_input)

    # normalisat° des activations pour stabiliser l'entraînement
    x = layers.BatchNormalization(name='bn_dense')(x)

    # applicat° de Relu pour la non-linéarité
    x = layers.ReLU(name='relu_dense')(x)

    # reshape pour obtenir 8*8 avec 256 canaux
    x = layers.Reshape((8, 8, 256), name='reshape_8x8')(x)

    # bloc 1 d'upsampling : 8x8x256 -> 16x16x128, grâce à strides=2 je double la résolution
    x = layers.Conv2DTranspose(
        128,
        kernel_size=5,
        strides=2,
        padding='same',
        use_bias=False,
        name='conv_transpose_1'
    )(x)

    #  shape after 
    
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.ReLU(name='relu_1')(x)

    # bloc 2 d'upsampling : 16x16x128 -> 32x32x64
    x = layers.Conv2DTranspose(
        64,
        kernel_size=5,
        strides=2,
        padding='same',
        use_bias=False,
        name='conv_transpose_2'
    )(x)

    #  shape after 
    
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.ReLU(name='relu_2')(x)

    # bloc 3 je passe à 3 canaux pour RGB + use Tanh en sortie 
    output = layers.Conv2DTranspose(
        img_channels,
        kernel_size=5,
        strides=2,
        padding='same',
        activation='tanh',
        name='output_conv'
    )(x)

    # shape final 

    # création du modèle Keras
    generator = Model(inputs=noise_input, outputs=output, name='Generator')

    return generator

if __name__ == '__main__':
    # Test du générateur
    print("=== Test du générateur ===\n")
    
    # Je crée le générateur
    generator = build_generator()
    
    # J'affiche l'architecture
    print("Architecture du générateur :")
    generator.summary()
    
    # Je teste la génération d'une image
    print("\n=== Test de génération ===")
    noise = tf.random.normal([1, NOISE_DIM])
    fake_image = generator(noise, training=False)
    
    print(f"Shape de l'image générée : {fake_image.shape}")
    print(f"Min : {tf.reduce_min(fake_image).numpy():.4f}")
    print(f"Max : {tf.reduce_max(fake_image).numpy():.4f}")
    print(f"Moyenne : {tf.reduce_mean(fake_image).numpy():.4f}")
    
    # Je visualise l'image (bruit aléatoire avant entraînement)
    import matplotlib.pyplot as plt
    
    image_display = (fake_image[0] + 1) / 2  # Dénormalisation [-1,1] → [0,1]
    
    plt.figure(figsize=(4, 4))
    plt.imshow(image_display)
    plt.axis('off')
    plt.title('Image générée (avant entraînement)')
    plt.tight_layout()
    plt.savefig('outputs/test_generator.png')
    print("\nImage sauvegardée dans outputs/test_generator.png")
    plt.show()
    
    print("\nTest terminé avec succès !")