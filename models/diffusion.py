"""
DDPM Conditionnel avec U-Net.
Architecture basée sur le cours DDPM du professeur + conditionnement attributs.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from utils.config import *


# Paramètres DDPM
T = 1000  # Nombre de timesteps
BETA_START = 1e-4
BETA_END = 0.02


def get_beta_schedule(T=T):
    """
    Schedule linéaire de bruit (prof utilise linéaire pour DDPM).
    
    Formule :
    beta_t croît linéairement de BETA_START à BETA_END
    """
    return np.linspace(BETA_START, BETA_END, T, dtype=np.float32)


def compute_alpha_schedule(betas):
    """
    Calcul des alphas pour forward process.
    
    Formules du prof :
    alpha_t = 1 - beta_t
    alpha_bar_t = produit des alpha_i pour i=1..t
    """
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas)
    return alphas, alphas_cumprod


# Précalcul des schedules
BETAS = get_beta_schedule()
ALPHAS, ALPHAS_CUMPROD = compute_alpha_schedule(BETAS)


class TimeEmbedding(layers.Layer):
    """
    Embedding du timestep t (positional encoding sinusoïdal).
    Valeur standard : dim=256 (papier DDPM).
    """
    
    def __init__(self, dim=256, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.dense1 = layers.Dense(dim, activation='relu')
        self.dense2 = layers.Dense(dim)
    
    def call(self, t):
        # t shape: (batch_size,)
        # Sinusoidal encoding
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = tf.exp(tf.range(half_dim, dtype=tf.float32) * -emb)
        emb = tf.cast(t, tf.float32)[:, None] * emb[None, :]
        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
        
        # MLP
        emb = self.dense1(emb)
        emb = self.dense2(emb)
        return emb


class ConditionEmbedding(layers.Layer):
    """
    Embedding des attributs CelebA (40 dimensions).
    Projection vers dim=256 pour fusion avec time embedding.
    """
    
    def __init__(self, num_attributes=40, dim=256, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(dim)
    
    def call(self, attributes):
        # attributes shape: (batch_size, 40)
        x = self.dense1(attributes)
        x = self.dense2(x)
        return x


class ResidualBlock(layers.Layer):
    """
    Bloc résiduel avec injection time + condition.
    """
    
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.conv1 = layers.Conv2D(channels, 3, padding='same')
        self.conv2 = layers.Conv2D(channels, 3, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        
        self.time_proj = layers.Dense(channels)
        self.cond_proj = layers.Dense(channels)
        
        # projection pour skip si nécessaire
        self.skip_conv = None
    
    def build(self, input_shape):
        # vérif if projection is  nécessaire
        if input_shape[-1] != self.channels:
            self.skip_conv = layers.Conv2D(self.channels, 1, padding='same')
        super().build(input_shape)
    
    def call(self, x, time_emb, cond_emb):
        h = self.conv1(x)
        h = self.bn1(h)
        
        time_proj = self.time_proj(time_emb)[:, None, None, :]
        cond_proj = self.cond_proj(cond_emb)[:, None, None, :]
        h = h + time_proj + cond_proj
        
        h = tf.nn.relu(h)
        h = self.conv2(h)
        h = self.bn2(h)
        
        # skip connection avec projection si nécessaire
        if self.skip_conv is not None:
            x = self.skip_conv(x)
        
        return x + h


def build_unet_conditional(img_size=IMG_SIZE, img_channels=IMG_CHANNELS, num_attributes=40):
    """
    U-Net conditionnel pour DDPM.
    
    Architecture :
    - Encodeur : downsample 64 -> 32 -> 16 -> 8
    - Décodeur : upsample 8 -> 16 -> 32 -> 64
    - Skip connections
    - Time embedding injecté à chaque bloc
    - Condition embedding injecté à chaque bloc
    
    Input : (image_noisy, timestep, attributes)
    Output : noise prédit
    """
    
    # Inputs
    img_input = layers.Input(shape=(img_size, img_size, img_channels), name='noisy_image')
    time_input = layers.Input(shape=(), dtype=tf.int32, name='timestep')
    cond_input = layers.Input(shape=(num_attributes,), name='attributes')
    
    # Embeddings
    time_emb = TimeEmbedding()(time_input)
    cond_emb = ConditionEmbedding(num_attributes)(cond_input)
    
    # Encodeur
    # 64x64x3 -> 64x64x64
    x = layers.Conv2D(64, 3, padding='same')(img_input)
    x = ResidualBlock(64)(x, time_emb, cond_emb)
    skip1 = x
    
    # 64x64x64 -> 32x32x128
    x = layers.Conv2D(128, 3, strides=2, padding='same')(x)
    x = ResidualBlock(128)(x, time_emb, cond_emb)
    skip2 = x
    
    # 32x32x128 -> 16x16x256
    x = layers.Conv2D(256, 3, strides=2, padding='same')(x)
    x = ResidualBlock(256)(x, time_emb, cond_emb)
    skip3 = x
    
    # 16x16x256 -> 8x8x512 (bottleneck)
    x = layers.Conv2D(512, 3, strides=2, padding='same')(x)
    x = ResidualBlock(512)(x, time_emb, cond_emb)
    
    # Décodeur avec skip connections
    # 8x8x512 -> 16x16x256
    x = layers.Conv2DTranspose(256, 3, strides=2, padding='same')(x)
    x = layers.Concatenate()([x, skip3])
    x = ResidualBlock(256)(x, time_emb, cond_emb)
    
    # 16x16x256 -> 32x32x128
    x = layers.Conv2DTranspose(128, 3, strides=2, padding='same')(x)
    x = layers.Concatenate()([x, skip2])
    x = ResidualBlock(128)(x, time_emb, cond_emb)
    
    # 32x32x128 -> 64x64x64
    x = layers.Conv2DTranspose(64, 3, strides=2, padding='same')(x)
    x = layers.Concatenate()([x, skip1])
    x = ResidualBlock(64)(x, time_emb, cond_emb)
    
    # Output : prédiction du bruit
    noise_pred = layers.Conv2D(img_channels, 3, padding='same', name='noise_output')(x)
    
    model = Model(inputs=[img_input, time_input, cond_input], outputs=noise_pred, name='UNet_Conditional')
    return model


def forward_diffusion(x_0, t, noise=None):
    """
    Forward process : ajouter bruit à l'image.
    
    Formule du prof :
    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
    
    Args:
        x_0 : image originale (batch, 64, 64, 3)
        t : timesteps (batch,)
        noise : bruit gaussien optionnel
    
    Returns:
        x_t : image bruitée
        noise : bruit utilisé
    """
    if noise is None:
        noise = tf.random.normal(tf.shape(x_0))
    
    # Récupérer alpha_bar_t pour chaque timestep
    alpha_bar_t = tf.gather(ALPHAS_CUMPROD, t)
    alpha_bar_t = tf.cast(alpha_bar_t, tf.float32)
    
    # Reshape pour broadcast (batch, 1, 1, 1)
    alpha_bar_t = tf.reshape(alpha_bar_t, [-1, 1, 1, 1])
    
    # Formule forward
    sqrt_alpha_bar = tf.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar = tf.sqrt(1.0 - alpha_bar_t)
    
    x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
    
    return x_t, noise


def reverse_diffusion_step(model, x_t, t, attributes):
    """
    Reverse process : une étape de débruitage.
    
    Formule du prof :
    x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-alpha_bar_t) * epsilon_theta(x_t, t))
    
    Args:
        model : U-Net conditionnel
        x_t : image bruitée au temps t
        t : timestep actuel
        attributes : conditions (batch, 40)
    
    Returns:
        x_{t-1} : image débruitée au temps t-1
    """
    # Prédire le bruit
    noise_pred = model([x_t, t, attributes], training=False)
    
    # Récupérer paramètres
    alpha_t = ALPHAS[t[0]]
    alpha_bar_t = ALPHAS_CUMPROD[t[0]]
    beta_t = BETAS[t[0]]
    
    # Formule reverse
    coef1 = 1.0 / tf.sqrt(alpha_t)
    coef2 = beta_t / tf.sqrt(1.0 - alpha_bar_t)
    
    x_t_minus_1 = coef1 * (x_t - coef2 * noise_pred)
    
    # add du bruit si t > 0 (stochastique)
    if t[0] > 0:
        noise = tf.random.normal(tf.shape(x_t))
        sigma_t = tf.sqrt(beta_t)
        x_t_minus_1 = x_t_minus_1 + sigma_t * noise
    
    return x_t_minus_1


def generate_samples(model, num_samples, attributes, img_size=IMG_SIZE, img_channels=IMG_CHANNELS):
    """
    Génération complète : x_T (bruit pur) -> x_0 (image).
    
    Process :
    1. Partir de bruit gaussien x_T
    2. Itérer reverse_diffusion_step de T à 0
    3. Retourner x_0
    
    Args:
        model : U-Net conditionnel
        num_samples : nombre d'images à générer
        attributes : conditions (num_samples, 40)
    
    Returns:
        images générées (num_samples, 64, 64, 3)
    """
    # Partir de bruit pur
    x = tf.random.normal([num_samples, img_size, img_size, img_channels])
    
    # Débruitage itératif de T à 0
    for t in reversed(range(T)):
        t_batch = tf.fill([num_samples], t)
        x = reverse_diffusion_step(model, x, t_batch, attributes)
    
    return x


if __name__ == '__main__':
    print("Test U-Net Conditionnel\n")
    
    model = build_unet_conditional()
    model.summary()
    
    # Test forward
    x_0 = tf.random.normal([2, 64, 64, 3])
    t = tf.constant([10, 50])
    attributes = tf.random.uniform([2, 40], 0, 2, dtype=tf.int32)
    attributes = tf.cast(attributes, tf.float32)
    
    x_t, noise = forward_diffusion(x_0, t)
    print(f"\nForward diffusion OK : {x_t.shape}")
    
    # Test U-Net
    noise_pred = model([x_t, t, attributes])
    print(f"U-Net prediction OK : {noise_pred.shape}")
    
    print("\nTest terminé avec succès")