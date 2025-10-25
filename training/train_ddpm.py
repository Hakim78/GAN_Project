"""
Entraînement DDPM Conditionnel (V3).
Diffusion Model avec contrôle par attributs CelebA.
"""

import tensorflow as tf
import time
import os
import numpy as np

from models.diffusion import build_unet_conditional, forward_diffusion, generate_samples, T
from utils.data_loader import load_celeba_with_attributes
from utils.attributes import load_celeba_attributes, sample_random_attributes
from utils.visualization import save_generated_images
from utils.config import *


@tf.function
def train_step(model, images, attributes, optimizer):
    """
    Un pas d'entraînement DDPM.
    
    Process :
    1. Sample timestep t aléatoire
    2. Ajouter bruit à l'image selon t (forward)
    3. Prédire bruit avec U-Net
    4. Loss MSE entre bruit réel et prédit
    
    Args:
        model : U-Net conditionnel
        images : batch images réelles (batch, 64, 64, 3)
        attributes : batch attributs (batch, 40)
        optimizer : Adam optimizer
    
    Returns:
        loss : MSE loss
    """
    batch_size = tf.shape(images)[0]
    
    # Sample timesteps aléatoires pour chaque image
    t = tf.random.uniform([batch_size], 0, T, dtype=tf.int32)
    
    # Forward diffusion : ajouter bruit
    noise = tf.random.normal(tf.shape(images))
    x_t, _ = forward_diffusion(images, t, noise)
    
    with tf.GradientTape() as tape:
        # Prédire le bruit
        noise_pred = model([x_t, t, attributes], training=True)
        
        # Loss MSE (formule prof DDPM)
        loss = tf.reduce_mean(tf.square(noise - noise_pred))
    
    # Backprop
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss


def train(dataset, attributes_dict, epochs=300):
    """
    Boucle d'entraînement principale DDPM.
    
    Args:
        dataset : tf.data.Dataset avec images
        attributes_dict : dict {image_name: attributes}
        epochs : nombre d'epochs (300 recommandé)
    """
    print("\n" + "="*60)
    print("DÉMARRAGE ENTRAÎNEMENT DDPM V3 (CONDITIONNEL)")
    print("="*60)
    
    print("\nCréation du U-Net conditionnel...")
    model = build_unet_conditional()
    
    print(f"U-Net : {model.count_params():,} paramètres")
    
    # Optimizer avec learning rate réduit (DDPM plus stable avec LR faible)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    
    print(f"\nOptimiseur Adam :")
    print(f"  - Learning rate : 1e-4")
    print(f"  - Timesteps : {T}")
    
    # Dossiers V3
    checkpoint_dir_v3 = 'outputs/checkpoints_v3/'
    generated_dir_v3 = 'outputs/generated_images_v3/'
    plots_dir_v3 = 'outputs/plots_v3/'
    
    os.makedirs(checkpoint_dir_v3, exist_ok=True)
    os.makedirs(generated_dir_v3, exist_ok=True)
    os.makedirs(plots_dir_v3, exist_ok=True)
    
    losses = []
    
    print(f"\nConfiguration :")
    print(f"  - Epochs : {epochs}")
    print(f"  - Batch size : {BATCH_SIZE}")
    print(f"  - Image size : {IMG_SIZE}x{IMG_SIZE}")
    print(f"  - Attributs : 40")
    print(f"  - Diffusion steps : {T}")
    
    print("\n" + "="*60)
    print("DÉBUT DE L'ENTRAÎNEMENT V3")
    print("="*60 + "\n")
    
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        epoch_losses = []
        
        # Entraînement sur tous les batches
        for image_batch, attr_batch in dataset:
            loss = train_step(model, image_batch, attr_batch, optimizer)
            epoch_losses.append(float(loss.numpy()))
        
        # Moyenne des pertes
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch:05d} | Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")
        
        # Sauvegarde périodique
        if epoch % 50 == 0:
            print(f"\n{'='*60}")
            print(f"SAUVEGARDE V3 - Epoch {epoch}")
            print(f"{'='*60}")
            
            # Génération d'échantillons (16 images)
            # Utiliser attributs aléatoires pour diversité
            test_attrs = sample_random_attributes(16, seed=epoch)
            test_attrs = tf.constant(test_attrs, dtype=tf.float32)
            
            print("Génération de 16 échantillons (1000 steps)...")
            generated = generate_samples(model, 16, test_attrs)
            
            # Sauvegarder comme grille 4x4
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(4, 4, figsize=(8, 8))
            axes = axes.flatten()
            
            for i in range(16):
                img = (generated[i].numpy() + 1) / 2.0  # [-1,1] -> [0,1]
                img = np.clip(img, 0, 1)
                axes[i].imshow(img)
                axes[i].axis('off')
            
            plt.suptitle(f"DDPM V3 - Epoch {epoch}")
            plt.tight_layout()
            plt.savefig(f'{generated_dir_v3}/epoch_{epoch}.png', dpi=120)
            plt.close()
            
            print(f"Images V3 sauvegardées")
            
            # Courbe de perte
            plt.figure(figsize=(10, 5))
            plt.plot(losses, color='blue')
            plt.title("DDPM Training Loss")
            plt.xlabel("Epoch")
            plt.ylabel("MSE Loss")
            plt.grid(True)
            plt.savefig(f'{plots_dir_v3}/losses_epoch_{epoch}.png', dpi=120)
            plt.close()
            
            print(f"{'='*60}\n")
        
        # Sauvegarde modèle
        if epoch % 100 == 0:
            model.save(f'{checkpoint_dir_v3}/unet_epoch_{epoch}.h5')
            print(f"Modèle V3 sauvegardé (epoch {epoch})")
    
    print("\n" + "="*60)
    print("ENTRAÎNEMENT V3 TERMINÉ")
    print("="*60)
    
    # Sauvegarde finale
    model.save(f'{checkpoint_dir_v3}/unet_final.h5')
    print("\nModèle final V3 sauvegardé")
    
    # Sauvegarde historique pertes
    np.savez(f'{plots_dir_v3}/training_history_v3.npz', losses=np.array(losses))
    
    # Génération finale contrôlée
    print("\nGénération d'échantillons contrôlés...")
    
    # Créer 4 profils différents
    from utils.attributes import create_controlled_attributes, CELEBA_ATTRIBUTES
    
    profiles = [
        {'Male': 1, 'Smiling': 1, 'Eyeglasses': 1, 'Young': 1},  # Jeune homme souriant lunettes
        {'Male': 0, 'Smiling': 1, 'Wearing_Lipstick': 1, 'Young': 1},  # Jeune femme souriante maquillée
        {'Male': 1, 'Smiling': 0, 'No_Beard': 0, 'Young': 0},  # Homme âgé avec barbe
        {'Male': 0, 'Smiling': 0, 'Blond_Hair': 1, 'Young': 1}  # Jeune femme blonde
    ]
    
    controlled_attrs = []
    for profile in profiles:
        base = sample_random_attributes(4, seed=42)[0]
        for _ in range(4):
            controlled_attrs.append(create_controlled_attributes(base, profile))
    
    controlled_attrs = tf.constant(np.array(controlled_attrs), dtype=tf.float32)
    
    print("Génération contrôlée en cours...")
    controlled_imgs = generate_samples(model, 16, controlled_attrs)
    
    # Sauvegarder
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    titles = ['Male+Smiling+Glasses+Young'] * 4 + \
             ['Female+Smiling+Lipstick+Young'] * 4 + \
             ['Male+Beard+Old'] * 4 + \
             ['Female+Blonde+Young'] * 4
    
    for i in range(16):
        row, col = i // 4, i % 4
        img = (controlled_imgs[i].numpy() + 1) / 2.0
        img = np.clip(img, 0, 1)
        axes[row, col].imshow(img)
        axes[row, col].axis('off')
        if col == 0:
            axes[row, col].set_title(titles[i], fontsize=8)
    
    plt.suptitle("DDPM V3 - Génération Conditionnelle")
    plt.tight_layout()
    plt.savefig(f'{generated_dir_v3}/controlled_generation.png', dpi=150)
    plt.close()
    
    print("\nFichiers V3 créés :")
    print(f"  - Modèle : {checkpoint_dir_v3}")
    print(f"  - Images : {generated_dir_v3}")
    print(f"  - Graphiques : {plots_dir_v3}")


if __name__ == '__main__':
    print("Chargement du dataset CelebA avec attributs...")
    
    # Charger attributs
    attributes_dict = load_celeba_attributes('data/celeba/list_attr_celeba.txt')
    
    # Charger dataset avec attributs
    dataset = load_celeba_with_attributes(attributes_dict)
    
    # Entraînement
    train(dataset, attributes_dict, epochs=300)