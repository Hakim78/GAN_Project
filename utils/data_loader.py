"""
Chargement et prétraitement du dataset CelebA.
Je m'assure que les images sont normalisées entre [-1, 1] pour fonctionner avec tanh.

Justification des choix techniques :
- Normalisation [-1, 1] : Compatible avec l'activation tanh du générateur Je m'assure que les images sont normalisées entre [-1, 1] pour fonctionner avec tanh.
- Shuffle buffer 10000 : Valeur standard pour datasets de grande taille (prof utilise 60000 pour MNIST)
- Prefetch AUTOTUNE : Optimisation TensorFlow pour charger les données en parallèle pendant l'entraînement
"""

import tensorflow as tf
import os
from utils.config import *


def preprocess_image(image_path):
    """
      Prétraite une image individuelle.
    
    Justifications :
    - decode_jpeg : Format des images CelebA (178x218 initialement)
    - resize bilinear : Méthode standard pour redimensionner sans perte de qualité majeure
    - Normalisation (pixel - 127.5) / 127.5 : Transforme [0, 255] en [-1, 1]
      Cette formule vient du code du prof (exemple MNIST et Pokemon)
    
    Args:
        image_path (str): Chemin vers l'image
    
    Returns:
        tf.Tensor: Image normalisée (IMG_SIZE, IMG_SIZE, 3)
    """
    
    # chargement de l'image depuis le fichier
    image = tf.io.read_file(image_path)
    
    # décodage de l'image JPEG en tensor value betwen 0-255
    image  = tf.image.decode_jpeg(image, channels=IMG_CHANNELS)

    # redimensionnement de l'image à la taille souhaitée (definie dans config.py) methode used bilinear method standard
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])

    # normalisation des pixels entre -1 et 1 l'activation tanh du générateur produit des valeurs dans [-1, 1] donc on normalise les images de la même manière
    image = (image - 127.5) / 127.5

    return image

def load_celeba_dataset(data_dir=DATA_DIR, batch_size=BATCH_SIZE, img_size=IMG_SIZE):
    """
    Chargement du dataset CelebA depuis le dossier local
    
    Justifications des paramètres :
    - batch_size=128 : Valeur standard pour GANs (prof utilise 256 pour MNIST, 64 pour Pokemon)
      128 est un bon compromis entre vitesse et utilisation mémoire GPU
    - shuffle_buffer=10000 Assez grand pour bien mélanger prof utilise 60000 pour MNIST qui a 60k images
      10000 est environ 3-5% je pense du dataset CelebA qui lui à 200k img
    - prefetch AUTOTUNE optimisation TensorFlow le dataset suivant se charge pendant l'entraînement
    """

    # je vérifie que le dossier de données existe
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Le dossier de données {data_dir} n'existe pas. Mec T sur d'avoir téléchargé et extrait CelebA ?")
    
    # Je récupère tous les chemins des images dans le dossier CelebA contient environ 202599 images au format .jpg
    image_paths = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('.jpg')]

    print(f"Je charge {len(image_paths)} images depuis {data_dir}")
    
    # la je crée un dataset TensorFlow à partir des chemins from_tensor_slices & crée un dataset où chaque élément est un chemin
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    # ici ce passe le mappe de la fonction de prétraitement sur chaque image
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Je mélange le dataset avec un buffer de 10000 images
    dataset = dataset.shuffle(buffer_size=10000)

    # Je crée des batches de taille batch_size
    # drop_remainder=True assure que tous les batches ont exactement la même taille
    dataset = dataset.batch(batch_size, drop_remainder=True)

    # Je précharge les données pour optimiser la vitesse d'entraînement
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset