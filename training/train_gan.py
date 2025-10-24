"""
Entraînement du GAN classique (V1).
Je fais s'affronter le générateur et le discriminateur.

Justifications des choix :
- BinaryCrossentropy : fonction de perte standard pour GANs (prof l'utilise)
- Adam optimizer : optimiseur standard pour deep learning (prof l'utilise)
- Alternance G/D : d'abord discriminateur, puis générateur (standard GANs)
- tf.function : accélère l'entraînement en compilant la fonction
"""

import tensorflow as tf
import time
import os
from datetime import datetime

# J'importe mes modules
from models.generator import build_generator
from models.discriminator import build_discriminator
from utils.data_loader import load_celeba_dataset
from utils.visualization import plot_losses, save_generated_images, compare_real_fake
from utils.metrics import log_training_metrics, save_training_history, check_training_stability
from utils.config import *

# TODO ENSEMBLE: Définir les fonctions de perte
def discriminator_loss(real_output, fake_output):
    """Perte du discriminateur (BinaryCrossentropy)
        - BinaryCrossentropy : mesure l'écart entre prédiction et vérité
        - from_logits=False : j'utilise sigmoid dans le modèle, donc pas de logits bruts
        - real_loss : le discriminateur doit prédire 1 (vrai) pour les vraies images
        - fake_loss : le discriminateur doit prédire 0 (faux) pour les fausses images
        - total_loss : somme des deux (le discriminateur doit réussir les deux tâches)
    """
    # je créé la focntion de perte
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    # pertes sur les vraies images
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)

    # pertes sur les fausse images
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)

    # perte totale 
    total_loss = real_loss + fake_loss

    return total_loss

def generator_loss(fake_output):
        """
    Calcule la perte du générateur.
    
    Justifications :
    - Le générateur veut tromper le discriminateur
    - Il veut que le discriminateur prédise 1 (vrai) pour ses fausses images
    - Plus le discriminateur est trompé, plus la perte est faible
    
    Args:
        fake_output : Prédictions du discriminateur sur fausses images
    
    Returns:
        float : Perte du générateur
    """
        # function de perte
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

        # Le générateur veut que le discriminateur prédise 1 (vrai) pour ses fausses images
        return cross_entropy(tf.ones_like(fake_output), fake_output)


#compile la fonction en graphe TensorFlow, beaucoup plus rapide
@tf.function
def train_step(generator, discriminator, images, noise_dim, g_optimizer, d_optimizer):
    """
    Un pas d'entraînement : entraîner D puis G sur un batch.
    
    Justifications :
    - GradientTape : enregistre les opérations pour calculer les gradients
    - Deux tapes séparés : un pour G, un pour D (calculs indépendants)
    - Ordre : D d'abord, puis G (standard dans les GANs)
    
    Args:
        generator : Modèle générateur
        discriminator : Modèle discriminateur
        images : Batch d'images réelles
        noise_dim : Dimension du bruit
        g_optimizer : Optimiseur du générateur
        d_optimizer : Optimiseur du discriminateur
    
    Returns:
        tuple : (g_loss, d_loss)
    """

    # je get la size du batch
    batch_size = tf.shape(images)[0]

    # je genere du bruit aléatoire
    noise = tf.random.normal([batch_size, noise_dim])

    # ÉTAPE 1 : Entraîner le discriminateur
    # J'enregistre les opérations dans un GradientTape
    with tf.GradientTape() as disc_tape:
        # Je génère des fausses images
        generated_images = generator(noise, training=True)
        
        # Le discriminateur évalue les vraies images
        real_output = discriminator(images, training=True)
        
        # Le discriminateur évalue les fausses images
        fake_output = discriminator(generated_images, training=True)
        
        # Je calcule la perte du discriminateur
        disc_loss = discriminator_loss(real_output, fake_output)
    
    # Je calcule les gradients du discriminateur
    # Justification : gradients = comment changer les poids pour réduire la perte
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    # J'applique les gradients au discriminateur
    # Justification : mise à jour des poids selon la règle d'optimisation (Adam)
    d_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    # ÉTAPE 2 : Entraîner le générateur
    # Je génère du nouveau bruit (indépendant du précédent)
    noise = tf.random.normal([batch_size, noise_dim])
    
    with tf.GradientTape() as gen_tape:
        # Je génère des fausses images
        generated_images = generator(noise, training=True)
        
        # Le discriminateur les évalue
        fake_output = discriminator(generated_images, training=True)
        
        # Je calcule la perte du générateur
        gen_loss = generator_loss(fake_output)
    
    # Je calcule les gradients du générateur
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    
    # J'applique les gradients au générateur
    g_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    
    return gen_loss, disc_loss

def train(dataset, epochs=EPOCHS):
    """
    Boucle d'entraînement principale.
    
    Justifications :
    - epochs : nombre de passages sur le dataset complet
    - Sauvegarde régulière : évite de perdre le progrès en cas de crash
    - Logging : permet de suivre l'évolution de l'entraînement
    
    Args:
        dataset : Dataset TensorFlow avec les images
        epochs : Nombre d'epochs d'entraînement
    """
    print("\n" + "="*60)
    print("DÉMARRAGE DE L'ENTRAÎNEMENT DU GAN")
    print("="*60)

    # create models
    print("\nCréations des modèless")
    generator = build_generator()
    discriminator = build_discriminator()

    print(f"Générateur : {generator.count_params():,} paramètres")
    print(f"Discriminateur : {discriminator.count_params():,} paramètres")

    # create les optimiseurs adma avec LR
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_G, beta_1=BETA_1, beta_2=BETA_2 )
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_D, beta_1=BETA_1, beta_2=BETA_2 )

    print(f"\nOptimiseurs Adam :")
    print(f"  - Générateur : lr={LEARNING_RATE_G}, beta_1={BETA_1}, beta_2={BETA_2}")
    print(f"  - Discriminateur : lr={LEARNING_RATE_D}, beta_1={BETA_1}, beta_2={BETA_2}")

    # Je crée les dossiers de sauvegarde
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(GENERATED_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Listes pour stocker les pertes
    g_losses = []
    d_losses = []

    print(f"\nConfiguration :")
    print(f"  - Epochs : {epochs}")
    print(f"  - Batch size : {BATCH_SIZE}")
    print(f"  - Image size : {IMG_SIZE}x{IMG_SIZE}")
    print(f"  - Noise dim : {NOISE_DIM}")
    print(f"  - Save interval : {SAVE_INTERVAL}")
    
    print("\n" + "="*60)
    print("DÉBUT DE L'ENTRAÎNEMENT")
    print("="*60 + "\n")

    # Je récupère un batch de vraies images pour comparaison
    for sample_batch in dataset.take(1):
         sample_images = sample_batch
    
    # Boucle d'entraînement principale
    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # CORRECTION : Indentation correcte de la boucle for
        # Je parcours tous les batches du dataset
        for image_batch in dataset:
            g_loss, d_loss = train_step(
                generator, discriminator, image_batch,
                NOISE_DIM, g_optimizer, d_optimizer
            )

        # Je stocke les pertes
        g_losses.append(float(g_loss.numpy()))
        d_losses.append(float(d_loss.numpy()))
    
        # Je calcule le temps par epoch
        epoch_time = time.time() - start_time
    
        # J'affiche les métriques
        log_training_metrics(epoch, g_loss.numpy(), d_loss.numpy(), epoch_time)

        # Je sauvegarde des images à intervalles réguliers
        if epoch % SAVE_INTERVAL == 0:
            print(f"\n{'='*60}")
            print(f"SAUVEGARDE - Epoch {epoch}")
            print(f"{'='*60}")
            
            # Je sauvegarde une grille d'images générées
            save_generated_images(epoch, generator, save_dir=GENERATED_DIR)
            print(f"Images générées sauvegardées")
            
            # Je sauvegarde les courbes de perte
            plot_losses(g_losses, d_losses, save_path=f'{PLOTS_DIR}/losses_epoch_{epoch}.png')
            
            # Je vérifie la stabilité
            stability = check_training_stability(g_losses, d_losses)
            if stability['status'] == 'warning':
                print("\n⚠️  ATTENTION - Problèmes détectés :")
                for warning in stability['warnings']:
                    print(f"  - {warning}")
            
            print(f"{'='*60}\n")
        
        # Je sauvegarde les modèles à intervalles réguliers
        if epoch % CHECKPOINT_INTERVAL == 0:
            generator.save(f'{CHECKPOINT_DIR}/generator_epoch_{epoch}.h5')
            discriminator.save(f'{CHECKPOINT_DIR}/discriminator_epoch_{epoch}.h5')
            print(f"Modèles sauvegardés (epoch {epoch})")
    
    print("\n" + "="*60)
    print("ENTRAÎNEMENT TERMINÉ")
    print("="*60)

    
    # Je sauvegarde les modèles finaux
    generator.save(f'{CHECKPOINT_DIR}/generator_final.h5')
    discriminator.save(f'{CHECKPOINT_DIR}/discriminator_final.h5')
    print("\nModèles finaux sauvegardés")
    
    # Je sauvegarde l'historique des pertes
    save_training_history(g_losses, d_losses, save_path=f'{OUTPUT_DIR}/training_history.npz')
    
    # Je crée une comparaison finale
    compare_real_fake(sample_images, generator, save_path=f'{PLOTS_DIR}/final_comparison.png')
    
    # Je sauvegarde la dernière grille
    save_generated_images(epochs, generator, save_dir=GENERATED_DIR)
    
    print("\nFichiers créés :")
    print(f"  - Modèles : {CHECKPOINT_DIR}/")
    print(f"  - Images : {GENERATED_DIR}/")
    print(f"  - Graphiques : {PLOTS_DIR}/")


if __name__ == '__main__':
    # Je charge le dataset
    print("Chargement du dataset CelebA...")
    dataset = load_celeba_dataset()
    
    # Je lance l'entraînement
    # Commence avec peu d'epochs pour tester (ex: 100)
    # Puis lance avec beaucoup d'epochs pour la nuit (ex: 10000)
    train(dataset, epochs=10)  # Change à 10000 pour la vraie version
    





if __name__ == '__main__':
    # TODO ENSEMBLE: Charger dataset
    # TODO ENSEMBLE: Créer générateur et discriminateur
    # TODO ENSEMBLE: Lancer train()
    pass