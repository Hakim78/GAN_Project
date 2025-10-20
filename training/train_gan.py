"""
Entraînement du GAN classique (V1).
Nous intégrons nos modules respectifs et lançons l'entraînement.
"""

import tensorflow as tf
from models.generator import build_generator
from models.discriminator import build_discriminator
from utils.data_loader import load_celeba_dataset
from utils.visualization import plot_losses, save_generated_images
from utils.metrics import log_training_metrics
from utils.config import *

# TODO ENSEMBLE: Définir les fonctions de perte
def discriminator_loss(real_output, fake_output):
    """Perte du discriminateur (BinaryCrossentropy)"""
    pass

def generator_loss(fake_output):
    """Perte du générateur"""
    pass

# TODO ENSEMBLE: Définir le train_step
@tf.function
def train_step(generator, discriminator, images, noise_dim, g_optimizer, d_optimizer):
    """Un pas d'entraînement"""
    pass

# TODO ENSEMBLE: Boucle d'entraînement principale
def train(dataset, epochs=EPOCHS):
    """Entraînement complet"""
    pass

if __name__ == '__main__':
    # TODO ENSEMBLE: Charger dataset
    # TODO ENSEMBLE: Créer générateur et discriminateur
    # TODO ENSEMBLE: Lancer train()
    pass