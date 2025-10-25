import numpy as np

def diversity_std(imgs: np.ndarray) -> float:
    """Proxy simple: écart-type moyen. Tu peux remplacer par ta vraie métrique interne."""
    return float(np.mean(np.std(imgs, axis=(1,2,3))))
