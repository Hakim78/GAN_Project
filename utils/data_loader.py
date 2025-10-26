"""
Chargement et prétraitement du dataset CelebA.
Images normalisées en [-1, 1] (compatible avec une sortie générateur en tanh).

Robustesse :
- Recherche RÉCURSIVE des .jpg/.jpeg (gère les "double dossiers" type img_align_celeba/img_align_celeba).
- Extensions insensibles à la casse (.jpg/.JPG/.jpeg/.JPEG).
- Garde-fou : lève une erreur claire si 0 image détectée.
- Pipeline tf.data optimisé : shuffle → map(parallèle) → batch → prefetch.

Astuce : dans utils/config.py, gardez simplement
    DATA_DIR = "data/celeba/img_align_celeba"
Et créez un symlink si vos images sont ailleurs.
"""

import os
import glob
Images normalisées en [-1, 1] (compatible avec une sortie générateur en tanh).

Robustesse :
- Recherche RÉCURSIVE des .jpg/.jpeg (gère les "double dossiers" type img_align_celeba/img_align_celeba).
- Extensions insensibles à la casse (.jpg/.JPG/.jpeg/.JPEG).
- Garde-fou : lève une erreur claire si 0 image détectée.
- Pipeline tf.data optimisé : shuffle → map(parallèle) → batch → prefetch.

Astuce : dans utils/config.py, gardez simplement
    DATA_DIR = "data/celeba/img_align_celeba"
Et créez un symlink si vos images sont ailleurs.
"""

import os
import glob
import tensorflow as tf
import numpy as np
from utils.config import *

# ------------------------------------------------------------
# Prétraitement d'une image
# ------------------------------------------------------------
def preprocess_image(image_path: tf.Tensor) -> tf.Tensor:
    """
# ------------------------------------------------------------
# Prétraitement d'une image
# ------------------------------------------------------------
def preprocess_image(image_path: tf.Tensor) -> tf.Tensor:
    """
    Args:
        image_path (tf.Tensor): chemin (tf.string) vers une image .jpg/.jpeg

        image_path (tf.Tensor): chemin (tf.string) vers une image .jpg/.jpeg

    Returns:
        tf.Tensor: image float32 (IMG_SIZE, IMG_SIZE, 3) dans [-1, 1]
        tf.Tensor: image float32 (IMG_SIZE, IMG_SIZE, 3) dans [-1, 1]
    """
    # 1) lecture binaire
    image_bytes = tf.io.read_file(image_path)

    # 2) décodage JPEG → uint8 [H, W, C] (C=IMG_CHANNELS)
    image = tf.image.decode_jpeg(image_bytes, channels=IMG_CHANNELS)
    # 1) lecture binaire
    image_bytes = tf.io.read_file(image_path)

    # 2) décodage JPEG → uint8 [H, W, C] (C=IMG_CHANNELS)
    image = tf.image.decode_jpeg(image_bytes, channels=IMG_CHANNELS)

    # 3) cast en float32 [0,1] puis resize
    image = tf.image.convert_image_dtype(image, tf.float32)  # [0,1]
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR)

    # 4) normalisation [0,1] → [-1,1]
    image = (image * 2.0) - 1.0
    # 3) cast en float32 [0,1] puis resize
    image = tf.image.convert_image_dtype(image, tf.float32)  # [0,1]
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR)

    # 4) normalisation [0,1] → [-1,1]
    image = (image * 2.0) - 1.0
    return image


# ------------------------------------------------------------
# Chargement complet du dataset
# ------------------------------------------------------------
def load_celeba_dataset(
    data_dir: str = DATA_DIR,
    batch_size: int = BATCH_SIZE,
    img_size: int = IMG_SIZE,           # non utilisé directement (IMG_SIZE vient de config)
    shuffle_buffer: int = 10_000,
    limit: int | None = None,           # pour smoke tests (ex: 1000)
) -> tf.data.Dataset:
    """
    Args:
        data_dir: dossier contenant *directement* ou *indirectement* les .jpg
        batch_size: taille des batches
        img_size: taille cible (64/128) — param global déjà utilisé par preprocess
        shuffle_buffer: taille du buffer de mélange
        limit: limiter le nb d'images (pour tests rapides)

    Returns:
        tf.data.Dataset prêt pour l'entraînement (batches de float32 en [-1,1]).
    """

# ------------------------------------------------------------
# Chargement complet du dataset
# ------------------------------------------------------------
def load_celeba_dataset(
    data_dir: str = DATA_DIR,
    batch_size: int = BATCH_SIZE,
    img_size: int = IMG_SIZE,           # non utilisé directement (IMG_SIZE vient de config)
    shuffle_buffer: int = 10_000,
    limit: int | None = None,           # pour smoke tests (ex: 1000)
) -> tf.data.Dataset:
    """
    Args:
        data_dir: dossier contenant *directement* ou *indirectement* les .jpg
        batch_size: taille des batches
        img_size: taille cible (64/128) — param global déjà utilisé par preprocess
        shuffle_buffer: taille du buffer de mélange
        limit: limiter le nb d'images (pour tests rapides)

    Returns:
        tf.data.Dataset prêt pour l'entraînement (batches de float32 en [-1,1]).
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Le dossier de données n'existe pas : {data_dir}\n"
            f"Astuce : créez un lien symbolique vers le dossier réel contenant les .jpg."
        )

    # Recherche récursive et tolérante à la casse
    patterns = [
        os.path.join(data_dir, "**", "*.jpg"),
        os.path.join(data_dir, "**", "*.JPG"),
        os.path.join(data_dir, "**", "*.jpeg"),
        os.path.join(data_dir, "**", "*.JPEG"),
    ]
    image_paths: list[str] = []
    for p in patterns:
        image_paths.extend(glob.glob(p, recursive=True))
    image_paths = sorted(set(image_paths))  # unique + tri

    if len(image_paths) == 0:
        raise FileNotFoundError(
            "Aucune image .jpg/.jpeg trouvée.\n"
            f"- data_dir = {data_dir}\n"
            "- Cas fréquent : double dossier (img_align_celeba/img_align_celeba) "
            "ou archive non extraite."
        )

    if limit is not None and limit > 0:
        image_paths = image_paths[:limit]
        raise FileNotFoundError(
            f"Le dossier de données n'existe pas : {data_dir}\n"
            f"Astuce : créez un lien symbolique vers le dossier réel contenant les .jpg."
        )

    # Recherche récursive et tolérante à la casse
    patterns = [
        os.path.join(data_dir, "**", "*.jpg"),
        os.path.join(data_dir, "**", "*.JPG"),
        os.path.join(data_dir, "**", "*.jpeg"),
        os.path.join(data_dir, "**", "*.JPEG"),
    ]
    image_paths: list[str] = []
    for p in patterns:
        image_paths.extend(glob.glob(p, recursive=True))
    image_paths = sorted(set(image_paths))  # unique + tri

    if len(image_paths) == 0:
        raise FileNotFoundError(
            "Aucune image .jpg/.jpeg trouvée.\n"
            f"- data_dir = {data_dir}\n"
            "- Cas fréquent : double dossier (img_align_celeba/img_align_celeba) "
            "ou archive non extraite."
        )

    if limit is not None and limit > 0:
        image_paths = image_paths[:limit]

    print(f"Je charge {len(image_paths)} images depuis {data_dir}")

    # Pipeline tf.data
    ds = tf.data.Dataset.from_tensor_slices(tf.constant(image_paths, dtype=tf.string))
    ds = ds.shuffle(buffer_size=min(shuffle_buffer, len(image_paths)), reshuffle_each_iteration=True)
    ds = ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ------------------------------------------------------------
# Smoke test (exécution directe)
# ------------------------------------------------------------
if __name__ == "__main__":
    try:
        dataset = load_celeba_dataset(limit=512)  # limite pour test rapide
        for batch in dataset.take(1):
            print("Batch shape :", batch.shape, "| dtype:", batch.dtype)
            print("Min:", float(tf.reduce_min(batch).numpy()), "Max:", float(tf.reduce_max(batch).numpy()))
        print("OK data loader.")
    except Exception as e:
        print("Erreur loader:", e)


    # Pipeline tf.data
    ds = tf.data.Dataset.from_tensor_slices(tf.constant(image_paths, dtype=tf.string))
    ds = ds.shuffle(buffer_size=min(shuffle_buffer, len(image_paths)), reshuffle_each_iteration=True)
    ds = ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def load_celeba_with_attributes(
    attributes_dict: dict,
    data_dir: str = DATA_DIR,
    batch_size: int = BATCH_SIZE,
    shuffle_buffer: int = 10_000,
    limit: int | None = None,
) -> tf.data.Dataset:
    """
    Charge CelebA avec images ET attributs pour DDPM conditionnel.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dossier non trouvé : {data_dir}")
    
    patterns = [
        os.path.join(data_dir, "**", "*.jpg"),
        os.path.join(data_dir, "**", "*.JPG"),
    ]
    image_paths: list[str] = []
    for p in patterns:
        image_paths.extend(glob.glob(p, recursive=True))
    image_paths = sorted(set(image_paths))
    
    if len(image_paths) == 0:
        raise FileNotFoundError(f"Aucune image trouvée dans {data_dir}")
    
    if limit is not None and limit > 0:
        image_paths = image_paths[:limit]
    
    print(f"Chargement {len(image_paths)} images avec attributs")
    
    def load_image_and_attrs(img_path_tensor):
        img_path = img_path_tensor.numpy().decode('utf-8')
        img_bytes = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img_bytes, channels=IMG_CHANNELS)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = (img * 2.0) - 1.0
        
        img_name = os.path.basename(img_path)
        attrs = attributes_dict.get(img_name, np.zeros(40, dtype=np.float32))
        attrs = tf.constant(attrs, dtype=tf.float32)
        
        return img, attrs
    
    ds = tf.data.Dataset.from_tensor_slices(tf.constant(image_paths, dtype=tf.string))
    ds = ds.shuffle(buffer_size=min(shuffle_buffer, len(image_paths)))
    ds = ds.map(
        lambda path: tf.py_function(load_image_and_attrs, [path], [tf.float32, tf.float32]),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.map(lambda img, attrs: (
        tf.ensure_shape(img, [IMG_SIZE, IMG_SIZE, IMG_CHANNELS]),
        tf.ensure_shape(attrs, [40])
    ))
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    return ds

# ------------------------------------------------------------
# Smoke test (exécution directe)
# ------------------------------------------------------------
if __name__ == "__main__":
    try:
        dataset = load_celeba_dataset(limit=512)  # limite pour test rapide
        for batch in dataset.take(1):
            print("Batch shape :", batch.shape, "| dtype:", batch.dtype)
            print("Min:", float(tf.reduce_min(batch).numpy()), "Max:", float(tf.reduce_max(batch).numpy()))
        print("OK data loader.")
    except Exception as e:
        print("Erreur loader:", e)
