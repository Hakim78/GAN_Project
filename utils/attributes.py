"""
Gestion des attributs CelebA pour conditionnement.
Dataset CelebA contient 40 attributs binaires par image.
"""

import os
import numpy as np
import pandas as pd


# Liste des 40 attributs CelebA (ordre du fichier list_attr_celeba.txt)
CELEBA_ATTRIBUTES = [
    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
    'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
    'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
    'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
    'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
    'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
    'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
    'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
    'Wearing_Necktie', 'Young'
]


def load_celeba_attributes(attr_file_path='data/celeba/list_attr_celeba.txt'):
    """
    Charge le fichier des attributs CelebA.
    
    Format fichier :
    202599           (nombre total d'images)
    image_name.jpg 5_o_Clock_Shadow Arched_Eyebrows ... Young
    000001.jpg -1 1 1 -1 ... 1
    000002.jpg 1 -1 -1 1 ... -1
    
    Valeurs : -1 (absent), 1 (présent)
    On convertit en 0/1 pour faciliter.
    
    Args:
        attr_file_path : chemin vers list_attr_celeba.txt
    
    Returns:
        dict : {image_name: numpy array (40,) de 0/1}
    """
    if not os.path.exists(attr_file_path):
        raise FileNotFoundError(
            f"Fichier attributs non trouvé : {attr_file_path}\n"
            f"Télécharge depuis Kaggle : jessicali9530/celeba-dataset\n"
            f"Fichier attendu : list_attr_celeba.txt"
        )
    
    print(f"Chargement attributs depuis {attr_file_path}...")
    
    # Lire le fichier (skip première ligne = nombre total)
    df = pd.read_csv(attr_file_path, delim_whitespace=True, skiprows=1)
    
    # Index = nom image, colonnes = attributs
    # Convertir -1/1 en 0/1
    attributes_dict = {}
    for idx, row in df.iterrows():
        image_name = idx  # Index = nom fichier
        # Convertir -1 -> 0, 1 -> 1
        attrs = ((row.values + 1) // 2).astype(np.float32)
        attributes_dict[image_name] = attrs
    
    print(f"✅ {len(attributes_dict)} images avec attributs chargés")
    return attributes_dict


def get_attributes_for_images(image_paths, attributes_dict):
    """
    Récupère les attributs pour une liste de chemins d'images.
    
    Args:
        image_paths : liste de chemins (ex: 'data/celeba/img_align_celeba/000001.jpg')
        attributes_dict : dict retourné par load_celeba_attributes()
    
    Returns:
        numpy array (N, 40) avec attributs
    """
    attributes = []
    for img_path in image_paths:
        # Extraire nom fichier
        img_name = os.path.basename(img_path)
        
        # Récupérer attributs (ou zeros si non trouvé)
        attrs = attributes_dict.get(img_name, np.zeros(40, dtype=np.float32))
        attributes.append(attrs)
    
    return np.array(attributes, dtype=np.float32)


def sample_random_attributes(num_samples=1, seed=None):
    """
    Génère des attributs aléatoires pour tests.
    Utile pour génération conditionnelle sans dataset.
    
    Args:
        num_samples : nombre de vecteurs d'attributs
        seed : seed aléatoire (reproductibilité)
    
    Returns:
        numpy array (num_samples, 40) de 0/1
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Probabilités réalistes pour certains attributs
    # Ex: Male ~50%, Smiling ~40%, Eyeglasses ~6%, etc.
    probs = np.array([
        0.11, 0.27, 0.51, 0.20, 0.02, 0.15, 0.24, 0.24, 0.24, 0.15,
        0.05, 0.21, 0.14, 0.05, 0.05, 0.06, 0.06, 0.04, 0.39, 0.45,
        0.58, 0.48, 0.04, 0.11, 0.83, 0.29, 0.04, 0.28, 0.08, 0.06,
        0.06, 0.48, 0.21, 0.28, 0.19, 0.05, 0.47, 0.12, 0.05, 0.77
    ])
    
    attributes = np.random.binomial(1, probs, size=(num_samples, 40)).astype(np.float32)
    return attributes


def create_controlled_attributes(base_attrs, modifications):
    """
    Crée des attributs contrôlés pour génération ciblée.
    
    Args:
        base_attrs : attributs de base (40,)
        modifications : dict {attr_name: 0 ou 1}
    
    Returns:
        numpy array (40,) modifié
    
    Exemple :
        attrs = sample_random_attributes(1)[0]
        attrs_male_smiling = create_controlled_attributes(attrs, {
            'Male': 1,
            'Smiling': 1,
            'Eyeglasses': 1
        })
    """
    attrs = base_attrs.copy()
    for attr_name, value in modifications.items():
        if attr_name in CELEBA_ATTRIBUTES:
            idx = CELEBA_ATTRIBUTES.index(attr_name)
            attrs[idx] = float(value)
    return attrs


def get_attribute_names():
    """Retourne la liste des noms d'attributs."""
    return CELEBA_ATTRIBUTES


if __name__ == '__main__':
    print("Test utils/attributes.py\n")
    
    # Test 1 : Génération aléatoire
    print("Test 1 : Génération aléatoire")
    attrs = sample_random_attributes(5, seed=42)
    print(f"Shape : {attrs.shape}")
    print(f"Exemple attributs image 1 : {attrs[0][:10]}")
    
    # Test 2 : Contrôle attributs
    print("\nTest 2 : Contrôle attributs")
    base = sample_random_attributes(1)[0]
    controlled = create_controlled_attributes(base, {
        'Male': 1,
        'Smiling': 1,
        'Eyeglasses': 1,
        'Young': 1
    })
    
    idx_male = CELEBA_ATTRIBUTES.index('Male')
    idx_smiling = CELEBA_ATTRIBUTES.index('Smiling')
    idx_glasses = CELEBA_ATTRIBUTES.index('Eyeglasses')
    idx_young = CELEBA_ATTRIBUTES.index('Young')
    
    print(f"Male : {controlled[idx_male]}")
    print(f"Smiling : {controlled[idx_smiling]}")
    print(f"Eyeglasses : {controlled[idx_glasses]}")
    print(f"Young : {controlled[idx_young]}")
    
    print("\n Tests terminés")