# GAN CelebA Project — Génération d’Images avec Attention et Modèle de Diffusion

## Introduction

Dans le cadre du module *Deep Learning Avancé* du Master **Big Data & Intelligence Artificielle** à l’**IPSSI**, nous avons développé ce projet de génération d’images intitulé **GAN CelebA Project**.  
L’objectif principal était de concevoir, entraîner et comparer plusieurs architectures de réseaux génératifs, en intégrant des **mécanismes d’attention** et un **modèle de diffusion (DDPM)**.

Ce travail repose sur le dataset **CelebA**, un jeu d’images de visages annotés contenant plus de 200 000 exemples, largement utilisé dans les travaux de recherche sur les GANs.

---

## 1. Objectifs du projet

- Implémenter un **Generative Adversarial Network (GAN)** capable de produire des visages réalistes à partir de bruit aléatoire.  
- Intégrer un **mécanisme d’attention (CBAM)** dans le générateur pour améliorer la qualité des détails visuels.  
- Expérimenter avec un **modèle de diffusion (DDPM)** afin de comparer ses performances avec celles des GANs.  
- Évaluer la qualité des images générées à l’aide de **métriques avancées (FID, IS, PSNR, SSIM)**.  
- Développer une **interface Streamlit** pour visualiser et comparer les résultats de chaque architecture.

---

## 2. Structure du projet

```bash
GAN_CELEBA_PROJECT/
│
├── data/                           # Données CelebA (non versionnées)
│
├── frontend/                       # Interface Streamlit (visualisation et comparaison)
│   ├── app.py
│   ├── intro.html
│   ├── assets/
│   ├── pages/                      # Pages V1, V2, V3, Comparaison
│   ├── fe_utils/
│   ├── lotties/
│   └── ui/
│
├── models/                         # Architectures des modèles
│   ├── generator.py
│   ├── discriminator.py
│   ├── attention.py
│   ├── generator_attention.py
│   ├── discriminator_attention.py
│   ├── diffusion.py
│   └── __init__.py
│
├── notebooks/                      # Expérimentations et analyses
│   ├── 01_data_exploration.ipynb
│   ├── 02_v1_gan_baseline.ipynb
│   ├── 03_v2_gan_attention.ipynb
│   ├── 03_v2_diffusion.ipynb
│   └── 04_analysis.ipynb
│
├── outputs/                        # Résultats (plots, checkpoints, images)
│   ├── checkpoints_v1/
│   ├── checkpoints_v2/
│   ├── checkpoints_v3/
│   └── plots/
│
├── training/                       # Scripts d'entraînement
│   ├── train_gan.py
│   ├── train_gan_attention.py
│   └── train_ddpm.py
│
├── utils/                          # Fonctions utilitaires
│   ├── attributes.py
│   ├── config.py
│   ├── data_loader.py
│   ├── metrics.py
│   └── visualization.py
│
├── evaluate_fid.py                 # Script de calcul du score FID
├── main.py                         # Exécution principale
├── requirements.txt
└── README.md
```

---

## 3. Méthodologie

### Version 1 — GAN Baseline
- Architecture de base : Générateur + Discriminateur classiques.  
- Perte adversariale de type **Binary Cross-Entropy**.  
- Résultats cohérents mais encore flous.

### Version 2 — GAN avec Attention (CBAM)
- Intégration du **Convolutional Block Attention Module (CBAM)** dans le générateur.  
- Amélioration notable de la netteté et du contraste.  
- Convergence plus stable et images plus diversifiées.

### Version 3 — Modèle de Diffusion (DDPM)
- Implémentation d’un **U-Net conditionnel** basé sur un processus de diffusion inverse.  
- Génération plus lente (~25 s pour 16 images) mais de meilleure qualité visuelle.  
- Meilleur score FID et diversité des échantillons.

---

## 4. Interface Streamlit

L’interface Streamlit permet :
- La **visualisation côte à côte** des résultats des trois versions.  
- La **génération en temps réel** d’échantillons.  
- L’accès à des **métriques visuelles et statistiques**.  

L’interface intègre :
- Des **animations Lottie** pour l’esthétique,  
- Une navigation intuitive par onglets (V1 / V2 / V3 / Comparaison),  
- Un **design épuré** pour une utilisation pédagogique.

---

## 5. Travail collaboratif

| Membre | Rôle principal | Contributions |
|--------|----------------|----------------|
| **Hakim Djaalal** | Développeur & intégrateur IA | Implémentation des modèles (GAN, CBAM, DDPM), création de l’interface Streamlit, gestion du pipeline d’entraînement |
| **Assia Oumri** | Data Scientist & analyste | Implémentation des modèles (GAN, CBAM, DDPM), Préparation du dataset CelebA, expérimentation via notebooks, analyses comparatives et visualisation |

Cette collaboration a permis de croiser nos approches :  
**technique (optimisation, scripts)** et **analytique (évaluation, visualisation)**.

---

## 6. Résultats obtenus

| Modèle | Temps de génération (16 images) | Qualité visuelle | Diversité | Complexité |
|---------|--------------------------------|------------------|------------|-------------|
| **V1 (GAN)** | ≈ 1 s | Moyenne | Limitée | Faible |
| **V2 (GAN + CBAM)** | ≈ 1.2 s | Bonne | Correcte | Moyenne |
| **V3 (DDPM)** | ≈ 25 s | Excellente | Élevée | Haute |

Les échantillons générés et les courbes de perte sont disponibles dans `outputs/plots/`,  
et l’analyse complète dans `notebooks/04_analysis.ipynb`.

---

## 7. Installation et exécution

### a. Cloner le dépôt
```bash
git clone https://github.com/Hakim78/GAN_Project.git
cd GAN_CELEBA_PROJECT
```

### b. Créer l’environnement virtuel
```bash
python -m venv venv
source venv/bin/activate   # macOS / Linux
# ou
venv\Scripts\activate      # Windows
```

### c. Installer les dépendances
```bash
pip install -r requirements.txt
```

### d. Lancer l’interface Streamlit
```bash
streamlit run frontend/app.py
```

### e. (Optionnel) Réentraîner les modèles
```bash
python training/train_gan.py
python training/train_gan_attention.py
python training/train_ddpm.py
```

---

## 8. Technologies utilisées

- **Python 3.10+**  
- **TensorFlow / Keras**  
- **Streamlit**  
- **NumPy, Matplotlib, OpenCV**  
- **Lottie Animations**  
- **Google Colab GPU (Tesla T4)**  

---

## 9. Difficultés rencontrées

- Saturation de mémoire GPU sur Google Colab (notamment pour DDPM).  
- Limite de taille GitHub (fichiers > 100 Mo, checkpoints exclus).  
- Ajustement fin des taux d’apprentissage pour la stabilité du GAN.  
- Gestion des fusions multi-branches (Assia, Hakim, Main).  

---

## 10. Conclusion

Ce projet nous a permis de consolider nos compétences en **apprentissage génératif** et de comprendre les différences fondamentales entre **GANs** et **modèles de diffusion**.  
L’utilisation du module d’attention a démontré un gain clair en netteté et stabilité, tandis que le DDPM a permis d’obtenir les images les plus réalistes.  

Ce projet illustre un pipeline complet : **entraînement, analyse, comparaison et visualisation**.

---

## 11. Annexe : Bonus réalisés

Afin d’aller au-delà du cahier des charges initial, plusieurs bonus ont été implémentés :

| Bonus | Description | Impact |
|--------|--------------|---------|
| **Implémentation U-Net** | Architecture conditionnelle pour le DDPM | Amélioration de la cohérence spatiale |
| **Conditionnement d’attributs (CelebA)** | Génération guidée par des caractéristiques (sexe, sourire, lunettes, etc.) | Génération ciblée et contrôlable |
| **Métrique FID (Fréchet Inception Distance)** | Comparaison quantitative entre images réelles et synthétiques | Évaluation plus fine de la qualité générative |
| **Pipeline de comparaison automatisé** | Notebook comparatif (V2 vs V3) avec histogrammes et statistiques | Gain de temps et traçabilité |
| **Interface Streamlit interactive** | Interface multi-pages avec visualisation directe | Expérience utilisateur professionnelle |

---

## 12. Auteurs

Projet réalisé par :  
**Assia Oumri**  & **Hakim Djaalal** 
Master Big Data & Intelligence Artificielle — *IPSSI, Promotion MIA*

Encadré par :  
**Christopher Loisel** & **Sayf Bejaoui**
