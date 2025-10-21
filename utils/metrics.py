"""
Métriques d'évaluation pour les GANs.
Pour l'instant, je crée juste la structure de base.
Le FID score sera ajouté en bonus.
"""

import numpy as np
import tensorflow as tf
from datetime import datetime

def log_training_metrics(epoch, g_loss, d_loss, time_per_epoch=None):
    """
    Affiche les métriques pendant l'entraînement.
    
    Args:
        epoch (int): Numéro d'epoch
        g_loss (float): Perte du générateur
        d_loss (float): Perte du discriminateur
        time_per_epoch (float, optional): Temps pris pour l'epoch en secondes
    """
    message = f"Epoch {epoch:05d} |`G Loss: {g_loss:.4f} | D Loss: {d_loss:.4f}"

    # time per epoch
    if time_per_epoch is not None:
        message+= f" | Time: {time_per_epoch:.2f}s"

    print(message)


def log_epoch_summary(epoch, g_losses, d_losses, save_interval):
    """
    Affiche un résumé tous les N epochs.
    
    Justifications :
    - Résumé périodique : évite de spammer la console à chaque epoch
    - Statistiques moyennes : donne une vue d'ensemble de la progression
    - save_interval : aligné avec la sauvegarde des images (cohérence)
    
    Args:
        epoch (int): Numéro d'epoch actuel
        g_losses (list): Liste des pertes du générateur
        d_losses (list): Liste des pertes du discriminateur
        save_interval (int): Intervalle de sauvegarde (ex: 500)
    """

    # calcul des moyennes
    recent_g_losses = g_losses[-save_interval:] if len(g_losses) >= save_interval else g_losses
    recent_d_losses = d_losses[-save_interval:] if len(d_losses) >= save_interval else d_losses

    avg_g_loss = np.mean(recent_g_losses)
    avg_d_loss = np.mean(recent_d_losses)

    print(f"\n{'='*60}")
    print(f"Résumé Epoch {epoch:05d}")
    print(f"{'='*60}")
    print(f"Moyenne G Loss (derniers {len(recent_g_losses)} epochs): {avg_g_loss:.4f}")
    print(f"Moyenne D Loss (derniers {len(recent_d_losses)} epochs): {avg_d_loss:.4f}")
    print(f"{'='*60}\n")

def check_training_stability(g_losses, d_losses, window=100):
    """
    Vérifie la stabilité de l'entraînement.
    
    Justifications :
    - window=100 : fenêtre glissante pour détecter les tendances
    - Détecte le mode collapse : G loss explose ou D loss proche de 0
    - Détecte la divergence : les deux pertes explosent
    
    Args:
        g_losses (list): Liste des pertes du générateur
        d_losses (list): Liste des pertes du discriminateur
        window (int): Taille de la fenêtre pour calculer les moyennes
    
    Returns:
        dict: Dictionnaire avec warnings et statistiques
    """

    if len(g_losses) < window:
        return {"status": "LA TEAM PLEASE OPEN YOUR EYES TA Pas assez de données pour évaluer la stabilité."}
    
    # calcul des moyennes récentes
    recent_g = np.mean(g_losses[-window:])
    recent_d = np.mean(d_losses[-window:])

    # detection des anomalies
    warnings = []
    
    # collapse mode
    if recent_g > 5.0:
        warnings.append("Possible mode collapse : G loss très élevée")
    
    if recent_d < 0.1:
        warnings.append("Possible mode collapse : D loss très faible")

    # divergenvce les 2 pertes explosent
    if recent_g > 10 and recent_d > 10.0:
        warnings.append("Possible divergence : les deux pertes explosent")
    
    # resultat
    status = "warning" if warnings else "its okay"

    return {
        "status": status,
        "warnings": warnings,
        "recent_g_loss": recent_g,
        "recent_d_loss": recent_d
    }

def calculate_inception_score(images, num_splits=10):
    """
    Calcule le Inception Score pour un ensemble d'images générées.
    - IS mesure la qualité et la diversité des images générées
    - num_splits=10 : valeur standard (prof mentionne cette métrique en bonus)
    - Nécessite un modèle Inception pré-entraîné
    
    Args:
        images (np.array): Tableau d'images générées
        num_splits (int): Nombre de splits pour le calcul du score
    
    Returns:
        float: Inception Score moyen
    """
    # TODO BONUS SEMAINE
    # Justification : métrique avancée, pas prioritaire pour V1
    # Implémentation nécessite :
    # 1. Charger InceptionV3 pré-entraîné
    # 2. Calculer les probabilités de classe pour chaque image
    # 3. Calculer KL divergence entre p(y|x) et p(y)
    raise NotImplementedError("IS score sera implémenté en bonus (semaine 3)")

def calculate_fid(real_images, fake_images):
    """
    Calcule le Fréchet Inception Distance (FID) - BONUS SEMAINE 3.
    
    Justifications :
    - FID mesure la distance entre distributions réelles et générées
    - Métrique standard pour évaluer les GANs (prof mentionne en bonus)
    - Plus le score est bas, meilleur est le générateur
    
    Args:
        real_images (np.array): Images réelles (N, H, W, 3)
        fake_images (np.array): Images générées (N, H, W, 3)
    
    Returns:
        float: FID score (plus bas = meilleur)
    """
    # TODO BONUS SEMAINE
    # Justification : métrique avancée, pas prioritaire pour V1
    # Implémentation nécessite :
    # 1. Charger InceptionV3 pré-entraîné
    # 2. Extraire les features des images réelles et générées
    # 3. Calculer la distance de Fréchet entre les deux distributions
    raise NotImplementedError("FID score sera implémenté en bonus (semaine 3)")


def save_training_history(g_losses, d_losses, save_path='outputs/training_history.npz'):
    """
    Sauvegarde l'historique des pertes pour analyse ultérieure.
    
    Justifications :
    - Format .npz : compact et rapide à charger avec NumPy
    - Permet de tracer les courbes après l'entraînement
    - Utile pour comparer différentes versions du modèle
    
    Args:
        g_losses (list): Liste des pertes du générateur
        d_losses (list): Liste des pertes du discriminateur
        save_path (str): Chemin de sauvegarde
    """
    # Je convertis les listes en arrays NumPy
    g_losses_array = np.array(g_losses)
    d_losses_array = np.array(d_losses)
    
    # Je sauvegarde au format .npz (compressé)
    # Justification : format standard NumPy, compact et rapide
    np.savez(save_path, 
             generator_losses=g_losses_array,
             discriminator_losses=d_losses_array)
    
    print(f"Historique d'entraînement sauvegardé dans {save_path}")


def load_training_history(load_path='outputs/training_history.npz'):
    """
    Charge l'historique des pertes depuis un fichier .npz.
    
    Args:
        load_path (str): Chemin du fichier
    
    Returns:
        tuple: (g_losses, d_losses)
    """
    # Je charge le fichier .npz
    data = np.load(load_path)
    
    g_losses = data['generator_losses'].tolist()
    d_losses = data['discriminator_losses'].tolist()
    
    print(f"Historique chargé depuis {load_path}")
    print(f"  - {len(g_losses)} epochs de pertes générateur")
    print(f"  - {len(d_losses)} epochs de pertes discriminateur")
    
    return g_losses, d_losses


if __name__ == '__main__':
    # Test des fonctions de logging
    print("=== Test des métriques ===\n")
    
    # Je teste log_training_metrics
    print("Test 1 : Logging basique")
    log_training_metrics(epoch=1, g_loss=1.234, d_loss=0.678)
    log_training_metrics(epoch=100, g_loss=0.543, d_loss=0.321, time_per_epoch=2.5)
    
    # Je teste log_epoch_summary
    print("\nTest 2 : Résumé d'epoch")
    fake_g_losses = [1.5 - 0.001 * i for i in range(1000)]
    fake_d_losses = [0.7 - 0.0003 * i for i in range(1000)]
    log_epoch_summary(epoch=1000, g_losses=fake_g_losses, d_losses=fake_d_losses, save_interval=500)
    
    # Je teste check_training_stability
    print("Test 3 : Vérification de stabilité")
    stable_check = check_training_stability(fake_g_losses, fake_d_losses)
    print(f"Statut : {stable_check['status']}")
    print(f"G Loss récente : {stable_check['recent_g_loss']:.4f}")
    print(f"D Loss récente : {stable_check['recent_d_loss']:.4f}")
    
    # Je teste avec un cas de mode collapse
    print("\nTest 4 : Détection de mode collapse")
    collapse_g_losses = [10.0] * 100
    collapse_d_losses = [0.05] * 100
    collapse_check = check_training_stability(collapse_g_losses, collapse_d_losses)
    print(f"Statut : {collapse_check['status']}")
    if collapse_check['warnings']:
        print("Warnings détectés :")
        for warning in collapse_check['warnings']:
            print(f"  - {warning}")
    
    # Je teste la sauvegarde/chargement
    print("\nTest 5 : Sauvegarde et chargement")
    save_training_history(fake_g_losses, fake_d_losses, save_path='outputs/test_history.npz')
    loaded_g, loaded_d = load_training_history(load_path='outputs/test_history.npz')
    print(f"Vérification : {len(loaded_g)} epochs chargés")
    
    print("\nTest terminé avec succès !")