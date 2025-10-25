# frontend/fe_utils/loader.py
import os
import sys
from pathlib import Path
from typing import Optional
import numpy as np
import tensorflow as tf

# --- Localisation des dossiers ---
THIS_FILE = Path(__file__).resolve()
FRONTEND_DIR = THIS_FILE.parents[1]        # <repo>/frontend
REPO_ROOT = THIS_FILE.parents[2]           # <repo>  (où vivent models/ et utils/config.py)

# Assure que la racine du repo est dans le PYTHONPATH (pour importer models, utils.config, etc.)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --- Normalisation de chemins par rapport à frontend/ ---
def _resolve_path(p: str) -> Path:
    if not p:
        return Path("")
    p_norm = p.replace("\\", "/")
    path = Path(p_norm)
    if not path.is_absolute():
        path = (FRONTEND_DIR / path).resolve()
    return path

# Chemins par défaut (tes fichiers réels)
DEFAULT_V1_CKPT = "runs/v1/ckpt/generator_final.h5"
DEFAULT_V2_CKPT = "runs/v2/ckpt/generator_final.h5"

def _is_savedmodel(path: Path) -> bool:
    return path.is_dir() and (path / "saved_model.pb").exists()

def _load_generator_from_checkpoint(weights_path: str, attention: bool) -> tf.keras.Model:
    if not weights_path:
        raise ValueError("Aucun checkpoint fourni.")
    wp = _resolve_path(weights_path)
    if not wp.exists():
        raise FileNotFoundError(f"Checkpoint introuvable: {wp}")

    # Cas SavedModel (dossier)
    if _is_savedmodel(wp):
        return tf.keras.models.load_model(str(wp), compile=False)

    # Cas poids (.h5 / .keras) : reconstruire l'archi puis charger les poids
    if attention:
        from models.generator_attention import build_generator_attention as build_gen
    else:
        from models.generator import build_generator as build_gen

    G = build_gen()
    G.load_weights(str(wp))
    return G

def load_gan_generator(weights_path: Optional[str] = None, attention: bool = False) -> tf.keras.Model:
    if not weights_path or weights_path.strip() == "":
        weights_path = DEFAULT_V2_CKPT if attention else DEFAULT_V1_CKPT
    return _load_generator_from_checkpoint(weights_path, attention=attention)

def sample_gan(generator: tf.keras.Model, n: int, noise_dim: int, seed: Optional[int] = 42) -> np.ndarray:
    if generator is None:
        raise ValueError("Le générateur n'est pas chargé.")
    tf.random.set_seed(seed or 42)
    z = tf.random.normal((n, noise_dim))
    imgs = generator(z, training=False).numpy()
    return imgs

def load_diffusion_model(weights_path: str):
    from models.diffusion import build_diffusion_model
    if not weights_path:
        raise ValueError("Aucun checkpoint fourni pour la diffusion.")
    wp = _resolve_path(weights_path)
    if not wp.exists():
        raise FileNotFoundError(f"Checkpoint introuvable: {wp}")
    if _is_savedmodel(wp):
        return tf.keras.models.load_model(str(wp), compile=False)
    D = build_diffusion_model()
    D.load_weights(str(wp))
    return D

def sample_diffusion(model, n: int, steps: int, seed: Optional[int] = 42) -> np.ndarray:
    from models import diffusion as diffusion_mod
    if not hasattr(diffusion_mod, "sample_fn"):
        raise AttributeError(
            "models.diffusion doit exposer sample_fn(model, n, steps, seed) pour générer des images réelles."
        )
    return diffusion_mod.sample_fn(model, n=n, steps=steps, seed=seed)

def get_generator_z_dim(generator) -> int:
    """
    Détecte automatiquement la dimension du bruit (z_dim) depuis le modèle chargé.
    - Cas classique: Input shape = (None, zdim)
    - Si le modèle a plusieurs inputs, on prend le premier input vectoriel.
    """
    # Essaye via .input_shape
    try:
        shp = generator.input_shape
        # Keras peut renvoyer une liste si multi-input
        if isinstance(shp, list):
            for s in shp:
                if isinstance(s, tuple) and len(s) == 2 and s[1] is not None:
                    return int(s[1])
        else:
            if isinstance(shp, tuple) and len(shp) == 2 and shp[1] is not None:
                return int(shp[1])
    except Exception:
        pass

    # Fallback: inspecter les InputLayers
    try:
        for layer in generator.layers:
            if "InputLayer" in layer.__class__.__name__:
                conf = layer.get_config()
                shape = conf.get("batch_input_shape") or conf.get("input_shape")
                if shape and len(shape) >= 2 and shape[-1] is not None:
                    return int(shape[-1])
    except Exception:
        pass

    raise ValueError("Impossible de détecter automatiquement z_dim depuis le modèle générateur.")
