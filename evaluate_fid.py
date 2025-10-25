"""
Script pour calculer les FID scores de mes modèles V1, V2 et V3.
Je lance ce script après l'entraînement pour évaluer la qualité quantitative.

EMPLACEMENT: À la racine du projet (même niveau que training/, models/, utils/)

Usage:
    python evaluate_fid.py --version v1
    python evaluate_fid.py --version v2
    python evaluate_fid.py --version v3
    python evaluate_fid.py --all  # Calcule pour les 3 versions
"""

import argparse
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

# J'importe mes modules
from utils.data_loader import load_celeba_dataset, load_celeba_with_attributes
from utils.metrics import evaluate_generator_fid, evaluate_ddpm_fid
from utils.attributes import load_celeba_attributes
from models.generator import build_generator
from models.generator_attention import build_generator_attention
from models.diffusion import build_unet_conditional


def load_model(version, checkpoint_path=None):
    """
    Je charge le modèle selon la version spécifiée.
    
    Args:
        version: 'v1', 'v2', ou 'v3'
        checkpoint_path: chemin custom (sinon j'utilise les chemins par défaut)
    """
    if checkpoint_path is None:
        checkpoint_path = {
            'v1': 'outputs/checkpoints/generator_final.h5',
            'v2': 'outputs/checkpoints_v2/generator_final.h5',
            'v3': 'outputs/checkpoints_v3/unet_final.h5'
        }[version]
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint introuvable: {checkpoint_path}\n"
            f"Je dois d'abord entraîner le modèle {version.upper()}"
        )
    
    print(f"\nChargement du modèle {version.upper()}...")
    print(f"Checkpoint: {checkpoint_path}")
    
    if version == 'v1':
        model = build_generator()
        model.load_weights(checkpoint_path)
        print(f"Paramètres: {model.count_params():,}")
        return model, 'gan'
    
    elif version == 'v2':
        model = build_generator_attention()
        model.load_weights(checkpoint_path)
        print(f"Paramètres: {model.count_params():,}")
        return model, 'gan'
    
    elif version == 'v3':
        model = build_unet_conditional()
        model.load_weights(checkpoint_path)
        print(f"Paramètres: {model.count_params():,}")
        return model, 'ddpm'
    
    else:
        raise ValueError(f"Version invalide: {version}. Je dois utiliser 'v1', 'v2' ou 'v3'")


def calculate_fid_for_version(version, num_samples=1000, checkpoint_path=None):
    """
    Je calcule le FID score pour une version spécifique.
    
    Args:
        version: 'v1', 'v2', ou 'v3'
        num_samples: nombre d'images pour le calcul (1000 recommandé)
        checkpoint_path: chemin custom du checkpoint
    
    Returns:
        float: FID score
    """
    print("\n" + "="*70)
    print(f"CALCUL FID SCORE - {version.upper()}")
    print("="*70)
    
    # Je charge le modèle
    model, model_type = load_model(version, checkpoint_path)
    
    # Je charge le dataset réel
    print("\nChargement du dataset CelebA...")
    if model_type == 'ddpm' and version == 'v3':
        # Pour DDPM, j'ai besoin des attributs
        attributes_dict = load_celeba_attributes('data/celeba/list_attr_celeba.txt')
        dataset = load_celeba_with_attributes(attributes_dict, limit=num_samples)
    else:
        dataset = load_celeba_dataset(limit=num_samples)
    
    # Je calcule le FID
    if model_type == 'gan':
        fid_score = evaluate_generator_fid(
            generator=model,
            real_dataset=dataset,
            num_samples=num_samples,
            noise_dim=100
        )
    else:  # ddpm
        attributes_dict = load_celeba_attributes('data/celeba/list_attr_celeba.txt')
        fid_score = evaluate_ddpm_fid(
            unet_model=model,
            real_dataset=dataset,
            attributes_dict=attributes_dict,
            num_samples=num_samples
        )
    
    return fid_score


def plot_fid_comparison(results):
    """
    Je crée un graphique de comparaison des FID scores.
    
    Args:
        results: dict {'v1': score, 'v2': score, 'v3': score}
    """
    versions = list(results.keys())
    scores = list(results.values())
    
    # Je crée le graphique
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Barres avec couleurs différentes
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    bars = ax.bar(versions, scores, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # J'ajoute les valeurs sur les barres
    for i, (bar, score) in enumerate(zip(bars, scores)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.2f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Customisation
    ax.set_ylabel('FID Score (plus bas = meilleur)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Version du modèle', fontsize=12, fontweight='bold')
    ax.set_title('Comparaison des FID Scores\n(Fréchet Inception Distance)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # J'ajoute une ligne horizontale de référence
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Seuil "Bon" (< 50)')
    ax.legend(loc='upper right')
    
    # Je sauvegarde
    output_path = 'outputs/plots/fid_comparison.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nGraphique sauvegardé: {output_path}")
    plt.show()


def save_fid_results(results, output_file='outputs/fid_scores.txt'):
    """
    Je sauvegarde les résultats FID dans un fichier texte.
    
    Args:
        results: dict {'v1': score, 'v2': score, 'v3': score}
        output_file: chemin du fichier de sortie
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("RÉSULTATS FID SCORES - ÉVALUATION DES MODÈLES\n")
        f.write("="*70 + "\n\n")
        
        for version, score in sorted(results.items()):
            f.write(f"{version.upper()}: {score:.2f}\n")
            
            # J'ajoute une interprétation
            if score < 10:
                quality = "Excellent"
            elif score < 30:
                quality = "Très bon"
            elif score < 50:
                quality = "Bon"
            elif score < 100:
                quality = "Acceptable"
            else:
                quality = "À améliorer"
            
            f.write(f"  → Qualité: {quality}\n\n")
        
        # Je trouve le meilleur modèle
        best_version = min(results, key=results.get)
        f.write(f"\n{'='*70}\n")
        f.write(f"MEILLEUR MODÈLE: {best_version.upper()} (FID = {results[best_version]:.2f})\n")
        f.write(f"{'='*70}\n")
    
    print(f"\nRésultats sauvegardés: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Je calcule les FID scores pour évaluer mes modèles génératifs"
    )
    parser.add_argument(
        '--version', 
        type=str, 
        choices=['v1', 'v2', 'v3'],
        help="Version spécifique à évaluer"
    )
    parser.add_argument(
        '--all', 
        action='store_true',
        help="Je calcule pour toutes les versions disponibles"
    )
    parser.add_argument(
        '--num_samples', 
        type=int, 
        default=1000,
        help="Nombre d'images pour le calcul FID (défaut: 1000)"
    )
    parser.add_argument(
        '--checkpoint', 
        type=str,
        help="Chemin custom du checkpoint (optionnel)"
    )
    
    args = parser.parse_args()
    
    # Je vérifie les arguments
    if not args.version and not args.all:
        parser.error("Je dois spécifier --version ou --all")
    
    results = {}
    
    if args.all:
        # Je calcule pour toutes les versions
        for version in ['v1', 'v2', 'v3']:
            try:
                fid = calculate_fid_for_version(version, args.num_samples)
                results[version] = fid
            except FileNotFoundError as e:
                print(f"\n{e}")
                print(f"Je passe à la version suivante...\n")
            except Exception as e:
                print(f"\nErreur lors du calcul FID pour {version}: {e}\n")
    else:
        # Je calcule pour une version spécifique
        try:
            fid = calculate_fid_for_version(args.version, args.num_samples, args.checkpoint)
            results[args.version] = fid
        except Exception as e:
            print(f"\nErreur: {e}")
            return
    
    if not results:
        print("\nAucun FID score calculé. Je vérifie mes checkpoints.")
        return
    
    # J'affiche un résumé
    print("\n" + "="*70)
    print("RÉSUMÉ DES FID SCORES")
    print("="*70)
    for version, score in sorted(results.items()):
        print(f"{version.upper()}: {score:.2f}")
    print("="*70)
    
    # Je crée les visualisations et sauvegardes
    if len(results) > 1:
        plot_fid_comparison(results)
    
    save_fid_results(results)
    
    print("\nÉvaluation FID terminée avec succès!")


if __name__ == '__main__':
    main()