"""
Discriminateur avec attention CBAM Architecture V1 + CBAM après chaque Conv2D
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from models.attention import CBAM
from utils.config import *


def build_discriminator_attention(img_size=IMG_SIZE, img_channels=IMG_CHANNELS):
    """
    Discriminateur avec CBAM
    Architecture :
    [Conv2D → LeakyReLU → CBAM → Dropout] x3 → Flatten → Dense
    
    value prof :
    - LeakyReLU(0.2) : évite dying ReLU
    - Dropout(0.3) : régularisation
    - Strides=2 : downsampling
    """
    
    image_input = Input(shape=(img_size, img_size, img_channels), name='image_input')
    
    # Bloc 1 : 64x64x3 -> 32x32x64
    x = layers.Conv2D(64, kernel_size=5, strides=2, padding='same',
                     kernel_initializer='he_normal', name='conv_1')(image_input)
    x = layers.LeakyReLU(alpha=0.2, name='leaky_relu_1')(x)
    x = CBAM(ratio=8, kernel_size=7, name='cbam_1')(x)
    x = layers.Dropout(0.3, name='dropout_1')(x)
    
    # Bloc 2 : 32x32x64 -> 16x16x128
    x = layers.Conv2D(128, kernel_size=5, strides=2, padding='same',
                     kernel_initializer='he_normal', name='conv_2')(x)
    x = layers.LeakyReLU(alpha=0.2, name='leaky_relu_2')(x)
    x = CBAM(ratio=8, kernel_size=7, name='cbam_2')(x)
    x = layers.Dropout(0.3, name='dropout_2')(x)
    
    # Bloc 3 : 16x16x128 -> 8x8x256
    x = layers.Conv2D(256, kernel_size=5, strides=2, padding='same',
                     kernel_initializer='he_normal', name='conv_3')(x)
    x = layers.LeakyReLU(alpha=0.2, name='leaky_relu_3')(x)
    x = CBAM(ratio=8, kernel_size=7, name='cbam_3')(x)
    x = layers.Dropout(0.3, name='dropout_3')(x)
    
    # Classification
    x = layers.Flatten(name='flatten')(x)
    output = layers.Dense(1, activation='sigmoid', name='output_dense')(x)
    
    discriminator = Model(inputs=image_input, outputs=output, name='Discriminator_Attention')
    return discriminator


if __name__ == '__main__':
    print("Test Discriminateur Attention\n")
    
    disc = build_discriminator_attention()
    disc.summary()
    
    fake_image = tf.random.normal([1, IMG_SIZE, IMG_SIZE, IMG_CHANNELS])
    prediction = disc(fake_image, training=False)
    
    print(f"\nPrédiction shape : {prediction.shape}")
    print(f"Probabilité : {prediction.numpy()[0][0]:.4f}")