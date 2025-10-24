"""
Entraînement du GAN classique (V1).
Boucle complète avec :
- logs d'epoch flushés,
- logs de progression toutes les 100 itérations,
- sauvegardes d'images & checkpoints selon config.
"""

import os
import time
import tensorflow as tf

from models.generator import build_generator
from models.discriminator import build_discriminator
from utils.data_loader import load_celeba_dataset
from utils.visualization import plot_losses, save_generated_images
from utils.metrics import (
    log_training_metrics,
    log_epoch_summary,
    save_training_history,
)
from utils.config import *

# --- Pertes (D sort un sigmoid -> from_logits=False)
_bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

def discriminator_loss(real_output, fake_output):
    real_labels = tf.ones_like(real_output)
    fake_labels = tf.zeros_like(fake_output)
    real_loss = _bce(real_labels, real_output)
    fake_loss = _bce(fake_labels, fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return _bce(tf.ones_like(fake_output), fake_output)

# --- Un pas d'entraînement
@tf.function
def train_step(generator, discriminator, images, noise_dim, g_optimizer, d_optimizer):
    batch_size = tf.shape(images)[0]

    # 1) Update D
    z = tf.random.normal([batch_size, noise_dim])
    with tf.GradientTape() as d_tape:
        fake = generator(z, training=True)
        real_out = discriminator(images, training=True)
        fake_out = discriminator(fake, training=True)
        d_loss = discriminator_loss(real_out, fake_out)
    d_grads = d_tape.gradient(d_loss, discriminator.trainable_variables)
    d_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))

    # 2) Update G
    z2 = tf.random.normal([batch_size, noise_dim])
    with tf.GradientTape() as g_tape:
        gen = generator(z2, training=True)
        fake_out2 = discriminator(gen, training=True)
        g_loss = generator_loss(fake_out2)
    g_grads = g_tape.gradient(g_loss, generator.trainable_variables)
    g_optimizer.apply_gradients(zip(g_grads, generator.trainable_variables))

    return g_loss, d_loss

def train(dataset, epochs=EPOCHS, steps_log_interval=100):
    # Dossiers
    os.makedirs(GENERATED_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if CHECKPOINT_DIR:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Modèles & optis
    generator = build_generator()
    discriminator = build_discriminator()
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_G, beta_1=BETA_1, beta_2=BETA_2)
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_D, beta_1=BETA_1, beta_2=BETA_2)

    g_losses, d_losses = [], []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        epoch_g = tf.constant(0.0, dtype=tf.float32)
        epoch_d = tf.constant(0.0, dtype=tf.float32)
        steps = 0

        for images in dataset:
            g_loss, d_loss = train_step(generator, discriminator, images, NOISE_DIM, g_optimizer, d_optimizer)
            epoch_g += g_loss
            epoch_d += d_loss
            steps += 1

            # --- Log de progression intra-epoch
            if steps % steps_log_interval == 0:
                print(f"  step {steps:5d} | G {float(g_loss):.4f} | D {float(d_loss):.4f}", flush=True)

        # moyennes par epoch
        epoch_g = epoch_g / tf.cast(steps, tf.float32)
        epoch_d = epoch_d / tf.cast(steps, tf.float32)
        g_losses.append(float(epoch_g.numpy()))
        d_losses.append(float(epoch_d.numpy()))

        # log d'epoch (flush immédiat)
        log_training_metrics(epoch, g_losses[-1], d_losses[-1], time_per_epoch=time.time() - t0)

        # images & résumé périodique
        if epoch % SAVE_INTERVAL == 0 or epoch == 1:
            save_generated_images(epoch, generator, noise_dim=NOISE_DIM, save_dir=GENERATED_DIR)
            log_epoch_summary(epoch, g_losses, d_losses, save_interval=SAVE_INTERVAL)

        # checkpoints keras
        if CHECKPOINT_DIR and (epoch % CHECKPOINT_INTERVAL == 0):
            gen_path = os.path.join(CHECKPOINT_DIR, f"generator_epoch_{epoch}.keras")
            disc_path = os.path.join(CHECKPOINT_DIR, f"discriminator_epoch_{epoch}.keras")
            generator.save(gen_path)
            discriminator.save(disc_path)
            print(f"Checkpoint sauvegardé: {gen_path} | {disc_path}", flush=True)

    # fin d'entraînement : courbes + historique
    plot_losses(g_losses, d_losses, save_path=os.path.join(PLOTS_DIR, "loss_curves.png"))
    save_training_history(g_losses, d_losses, save_path=os.path.join(OUTPUT_DIR, "training_history.npz"))

if __name__ == '__main__':
    dataset = load_celeba_dataset()
    # steps_log_interval=100 -> un print toutes les 100 itérations
    train(dataset, epochs=EPOCHS, steps_log_interval=100)
