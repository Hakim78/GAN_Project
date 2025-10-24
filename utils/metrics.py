"""
Métriques & logging pour l'entraînement des GANs.
Les impressions sont flushées pour apparaître immédiatement même via un pipe (tee).
"""

import numpy as np
import tensorflow as tf
from datetime import datetime

def log_training_metrics(epoch, g_loss, d_loss, time_per_epoch=None):
    """
    Affiche les métriques pendant l'entraînement (flush immédiat).
    """
    msg = f"Epoch {epoch:05d} | G Loss: {g_loss:.4f} | D Loss: {d_loss:.4f}"
    if time_per_epoch is not None:
        msg += f" | Time: {time_per_epoch:.2f}s"
    print(msg, flush=True)

def log_epoch_summary(epoch, g_losses, d_losses, save_interval):
    """
    Affiche un résumé tous les N epochs.
    """
    recent_g = g_losses[-save_interval:] if len(g_losses) >= save_interval else g_losses
    recent_d = d_losses[-save_interval:] if len(d_losses) >= save_interval else d_losses
    avg_g = float(np.mean(recent_g)) if recent_g else float("nan")
    avg_d = float(np.mean(recent_d)) if recent_d else float("nan")

    print("\n" + "="*60, flush=True)
    print(f"Résumé Epoch {epoch:05d}", flush=True)
    print("="*60, flush=True)
    print(f"Moyenne G Loss (derniers {len(recent_g)} epochs): {avg_g:.4f}", flush=True)
    print(f"Moyenne D Loss (derniers {len(recent_d)} epochs): {avg_d:.4f}", flush=True)
    print("="*60 + "\n", flush=True)

def check_training_stability(g_losses, d_losses, window=100):
    """
    Vérifie la stabilité (détection rapide de mode collapse/divergence).
    """
    if len(g_losses) < window:
        return {"status": "not_enough_data"}

    recent_g = float(np.mean(g_losses[-window:]))
    recent_d = float(np.mean(d_losses[-window:]))

    warnings = []
    if recent_g > 5.0:
        warnings.append("Possible mode collapse : G loss très élevée")
    if recent_d < 0.1:
        warnings.append("Possible mode collapse : D loss très faible")
    if recent_g > 10.0 and recent_d > 10.0:
        warnings.append("Possible divergence : G et D explosent")

    status = "warning" if warnings else "ok"
    return {
        "status": status,
        "warnings": warnings,
        "recent_g_loss": recent_g,
        "recent_d_loss": recent_d,
    }

def save_training_history(g_losses, d_losses, save_path='outputs/training_history.npz'):
    """
    Sauvegarde l'historique des pertes pour analyse ultérieure.
    """
    np.savez(save_path,
             generator_losses=np.array(g_losses),
             discriminator_losses=np.array(d_losses))
    print(f"Historique d'entraînement sauvegardé dans {save_path}", flush=True)

def load_training_history(load_path='outputs/training_history.npz'):
    """
    Charge l'historique des pertes depuis un fichier .npz.
    """
    data = np.load(load_path)
    g_losses = data['generator_losses'].tolist()
    d_losses = data['discriminator_losses'].tolist()
    print(f"Historique chargé depuis {load_path}", flush=True)
    print(f"  - {len(g_losses)} epochs de pertes générateur", flush=True)
    print(f"  - {len(d_losses)} epochs de pertes discriminateur", flush=True)
    return g_losses, d_losses
