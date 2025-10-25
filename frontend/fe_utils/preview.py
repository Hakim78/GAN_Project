import numpy as np
from PIL import Image

def denorm_to_uint8(x):
    """Convertit [-1,1] -> [0,255]. Si tes outputs sont déjà [0,1], adapte ici."""
    x = (x + 1.0) * 127.5
    x = np.clip(x, 0, 255).astype(np.uint8)
    return x

def make_grid(imgs, ncols=4, pad=2):
    # imgs: list/array de (H,W,3) en [-1,1] (ou adapte si 0..1)
    imgs = [denorm_to_uint8(im) for im in imgs]
    h, w = imgs[0].shape[:2]
    n = len(imgs)
    nrows = (n + ncols - 1)//ncols
    grid = Image.new("RGB", (ncols*w + pad*(ncols-1), nrows*h + pad*(nrows-1)))
    for idx, im in enumerate(imgs):
        row, col = divmod(idx, ncols)
        grid.paste(Image.fromarray(im), (col*(w+pad), row*(h+pad)))
    return grid
