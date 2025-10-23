### générateur avec attention CBAM , Architecture V1 + blocs CBAM after chaque Conv2DTranspose.

import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from models.attention import CBAM
from utils.config import *


def build_generator_attention(noise_dim=NOISE_DIM, img_size=IMG_SIZE, img_channels=IMG_CHANNELS):
    """
    modif par rapport à V1 ajout de CBAM after chaque bloc Conv2DTranspose permet au générateur de se concentrer sur zones importantes

    - Dense(8*8*256) : taille initiale
    - Strides=2 : doublement résolution
    - Kernel_size=5 : standard GANs

    """

    noise_input = Input(shape=(noise_dim,), name='noise_input')

    # Dense projection 
    x = layers.Dense(8 * 8 * 256, use_bias=False, name='dense_projection')(noise_input)
    x = layers.BatchNormalization(name='bn_dense')(x)
    x = layers.ReLU(name='relu_dense')(x)
    x = layers.Reshape((8, 8, 256), name='reshape_8x8')(x)

    # Bloc 1 : 8x8x256 -> 16x16x128 + CBAM
    x = layers.Conv2DTranspose(128, kernel_size=5, strides=2, padding='same',
                                use_bias=False, name='conv_transpose_1')(x)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.ReLU(name='relu_1')(x)
    x = CBAM(ratio=8, kernel_size=7, name='cbam_1')(x)

    # Bloc 2 : 16x16x128 -> 32x32x64 + CBAM
    x = layers.Dense(64, kernel_size=5, strides=2, padding='same',
                     use_bias=False, name='conv_transpose_2')(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.ReLU(name='relu_2')(x)
    x = CBAM(ratio=8, kernel_size=7, name='cbam_2')

    # BLOC 3: 32x32x64 -> 64x64x3 (pas de CBAM sur sortie)
    output = layers.Conv2DTranspose(img_channels, kernel_size=5, strides=2,
                                    padding='same', activation='tanh', name='output_conv')(x)
    
    generator = Model(inputs=noise_input, outputs=output, name='Generator_Attention')
    return generator


    