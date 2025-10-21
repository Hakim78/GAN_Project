"""
Test du data_loader pour vérifier que le chargement fonctionne correctement.
Je vérifie les shapes, les valeurs min/max, et j'affiche quelques images.
"""

from utils.data_loader import load_celeba_dataset
import matplotlib.pyplot as plt
import tensorflow as tf

# Je charge le dataset
print("Chargement du dataset CelebA...")
dataset = load_celeba_dataset()

# Je récupère un batch pour tester
print("\nRécupération d'un batch de test...")
for images in dataset.take(1):
    print(f"Shape du batch: {images.shape}")
    # Je vérifie que la shape est correcte : (batch_size, img_size, img_size, channels)
    # Attendu : (128, 64, 64, 3)
    
    print(f"Valeur minimale: {tf.reduce_min(images).numpy():.4f}")
    print(f"Valeur maximale: {tf.reduce_max(images).numpy():.4f}")
    # Je vérifie que les valeurs sont bien entre [-1, 1]
    # Si elles sont entre [0, 255], j'ai oublié la normalisation
    
    print(f"Moyenne: {tf.reduce_mean(images).numpy():.4f}")
    # La moyenne devrait être proche de 0 si la normalisation est correcte
    
    # J'affiche 8 images du batch pour vérifier visuellement
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    
    for i in range(8):
        # Je dénormalise pour l'affichage : [-1, 1] -> [0, 1]
        # Formule inverse : (image + 1) / 2
        image_display = (images[i] + 1) / 2
        axes[i].imshow(image_display)
        axes[i].axis('off')
        axes[i].set_title(f'Image {i+1}')
    
    plt.suptitle('Échantillon d\'images CelebA chargées')
    plt.tight_layout()
    plt.savefig('outputs/test_data_loader.png')
    print("\nImages sauvegardées dans outputs/test_data_loader.png")
    plt.show()

print("\nTest terminé avec succès!")