"""32x32 grayscale preprocessing and local NumPy logistic inference."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

BASE = Path(__file__).resolve().parent

def image_pre(upload):
    try:
        with Image.open(upload) as image:
            if image.format not in {'PNG', 'JPEG'}:
                raise ValueError('Only PNG and JPEG image contents are supported.')
            if image.width * image.height > 16000000:
                raise ValueError('Use an image with no more than 16 million pixels.')
            image = ImageOps.grayscale(image)
            image = image.resize((32, 32), resample=Image.Resampling.BILINEAR)
            pixels = np.asarray(image, dtype=np.float32)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError('The file is not a readable image.') from exc
    # Preserve training polarity.
    pixels = pixels / 255.0
    return pixels.reshape((1, 32, 32, 1))

def predict(data):
    """Run the shipped NumPy logistic baseline (class 1 = Healthy)."""
    path = BASE.parent / 'models' / 'baseline.npz'
    if not path.is_file():
        raise RuntimeError('The baseline artifact is missing; retrain with scripts/train_baseline.py.')
    with np.load(path, allow_pickle=False) as artifact:
        coef, intercept = artifact['coef'], artifact['intercept']
    features = np.asarray(data).reshape(1, -1)
    if features.shape != (1, 1024) or coef.shape != (1, 1024):
        raise RuntimeError('Invalid baseline input or artifact shape.')
    score = float((features @ coef.T + intercept).item())
    if not np.isfinite(score):
        raise RuntimeError('Invalid classifier output.')
    return int(score >= 0)
