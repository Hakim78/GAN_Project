"""
Entraînement GAN avec Attention (V2).
Identique à V1 mais avec modèles attention.
"""

import tensorflow as tf
import time
import os
from datetime import datetime

from models.generator_attention import build_generator_attention
from models.discriminator_attention import build_discriminator_attention
from utils.data_loader import load_celeba_dataset
from utils.visualization import plot_losses, save_generated_images, compare_real_fake
from utils.metrics import log_training_metrics, save_training_history, check_training_stability
from utils.config import *


def discriminator_loss(real_output, fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss


def generator_loss(fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    return cross_entropy(tf.ones_like(fake_output), fake_output)


@tf.function
def train_step(generator, discriminator, images, noise_dim, g_optimizer, d_optimizer):
    batch_size = tf.shape(images)[0]
    noise = tf.random.normal([batch_size, noise_dim])
    
    with tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        disc_loss = discriminator_loss(real_output, fake_output)
    
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    d_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    noise = tf.random.normal([batch_size, noise_dim])
    
    with tf.GradientTape() as gen_tape:
        generated_images = generator(noise, training=True)
        fake_output = discriminator(generated_images, training=True)
        gen_loss = generator_loss(fake_output)
    
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    g_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    
    return gen_loss, disc_loss


def train(dataset, epochs=EPOCHS):
    print("\n" + "="*60)
    print("DÉMARRAGE ENTRAÎNEMENT GAN V2 (ATTENTION)")
    print("="*60)
    
    print("\nCréation des modèles avec attention...")
    generator = build_generator_attention()
    discriminator = build_discriminator_attention()
    
    print(f"Générateur : {generator.count_params():,} paramètres")
    print(f"Discriminateur : {discriminator.count_params():,} paramètres")
    
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_G, beta_1=BETA_1, beta_2=BETA_2)
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_D, beta_1=BETA_1, beta_2=BETA_2)
    
    print(f"\nOptimiseurs Adam :")
    print(f"  - Générateur : lr={LEARNING_RATE_G}, beta_1={BETA_1}, beta_2={BETA_2}")
    print(f"  - Discriminateur : lr={LEARNING_RATE_D}, beta_1={BETA_1}, beta_2={BETA_2}")
    
    # Dossiers de sauvegarde V2
    checkpoint_dir_v2 = 'outputs/checkpoints_v2/'
    generated_dir_v2 = 'outputs/generated_images_v2/'
    plots_dir_v2 = 'outputs/plots_v2/'
    
    os.makedirs(checkpoint_dir_v2, exist_ok=True)
    os.makedirs(generated_dir_v2, exist_ok=True)
    os.makedirs(plots_dir_v2, exist_ok=True)
    
    g_losses = []
    d_losses = []
    
    print(f"\nConfiguration :")
    print(f"  - Epochs : {epochs}")
    print(f"  - Batch size : {BATCH_SIZE}")
    print(f"  - Image size : {IMG_SIZE}x{IMG_SIZE}")
    print(f"  - Noise dim : {NOISE_DIM}")
    print(f"  - Save interval : {SAVE_INTERVAL}")
    
    print("\n" + "="*60)
    print("DÉBUT DE L'ENTRAÎNEMENT V2")
    print("="*60 + "\n")
    
    for sample_batch in dataset.take(1):
        sample_images = sample_batch
    
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        
        for image_batch in dataset:
            g_loss, d_loss = train_step(
                generator, discriminator, image_batch,
                NOISE_DIM, g_optimizer, d_optimizer
            )
        
        g_losses.append(float(g_loss.numpy()))
        d_losses.append(float(d_loss.numpy()))
        
        epoch_time = time.time() - start_time
        log_training_metrics(epoch, g_loss.numpy(), d_loss.numpy(), epoch_time)
        
        if epoch % SAVE_INTERVAL == 0:
            print(f"\n{'='*60}")
            print(f"SAUVEGARDE V2 - Epoch {epoch}")
            print(f"{'='*60}")
            
            save_generated_images(epoch, generator, save_dir=generated_dir_v2)
            print(f"Images V2 sauvegardées")
            
            plot_losses(g_losses, d_losses, save_path=f'{plots_dir_v2}/losses_epoch_{epoch}.png')
            
            stability = check_training_stability(g_losses, d_losses)
            if stability['status'] == 'warning':
                print("\nATTENTION - Problèmes détectés :")
                for warning in stability['warnings']:
                    print(f"  - {warning}")
            
            print(f"{'='*60}\n")
        
        if epoch % CHECKPOINT_INTERVAL == 0:
            generator.save(f'{checkpoint_dir_v2}/generator_epoch_{epoch}.h5')
            discriminator.save(f'{checkpoint_dir_v2}/discriminator_epoch_{epoch}.h5')
            print(f"Modèles V2 sauvegardés (epoch {epoch})")
    
    print("\n" + "="*60)
    print("ENTRAÎNEMENT V2 TERMINÉ")
    print("="*60)
    
    generator.save(f'{checkpoint_dir_v2}/generator_final.h5')
    discriminator.save(f'{checkpoint_dir_v2}/discriminator_final.h5')
    print("\nModèles finaux V2 sauvegardés")
    
    save_training_history(g_losses, d_losses, save_path=f'{OUTPUT_DIR}/training_history_v2.npz')
    compare_real_fake(sample_images, generator, save_path=f'{plots_dir_v2}/final_comparison.png')
    save_generated_images(epochs, generator, save_dir=generated_dir_v2)
    
    print("\nFichiers V2 créés :")
    print(f"  - Modèles : {checkpoint_dir_v2}")
    print(f"  - Images : {generated_dir_v2}")
    print(f"  - Graphiques : {plots_dir_v2}")


if __name__ == '__main__':
    print("Chargement du dataset CelebA...")
    dataset = load_celeba_dataset()
    
    train(dataset, epochs=600)