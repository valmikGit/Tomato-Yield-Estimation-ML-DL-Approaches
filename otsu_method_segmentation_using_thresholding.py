#!/usr/bin/env python3
"""
Otsu's Thresholding Implementation (Standalone Script)

Usage:
    python otsu_threshold.py <image_path>
"""

import sys
import numpy as np
from PIL import Image


def otsu_threshold(gray: np.ndarray) -> int:
    """
    Compute Otsu's threshold for a grayscale image.

    Args:
        gray (np.ndarray): 2D numpy array (grayscale image)

    Returns:
        int: optimal threshold value (0-255)
    """
    # Compute histogram (256 bins for uint8 grayscale)
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    total = hist.sum()

    # Normalize histogram to probabilities
    prob = hist.astype(np.float64) / total

    # Cumulative sums (class probabilities)
    omega = np.cumsum(prob)

    # Cumulative means (class means numerator)
    mu = np.cumsum(prob * np.arange(256))

    # Global mean
    mu_total = mu[-1]

    # Between-class variance
    sigma_b2 = (mu_total * omega - mu) ** 2 / (omega * (1 - omega) + 1e-12)

    # Ignore invalid entries (at ends where omega=0 or 1)
    sigma_b2[omega == 0] = 0
    sigma_b2[omega == 1] = 0

    # Find the threshold that maximizes between-class variance
    threshold = np.argmax(sigma_b2)
    return threshold


def main():
    if len(sys.argv) != 2:
        print("Usage: python otsu_threshold.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    # Load image as grayscale
    img = Image.open(image_path).convert("L")
    gray = np.array(img)

    # Compute Otsu threshold
    t_star = otsu_threshold(gray)
    print(f"Otsu's optimal threshold: {t_star}")

    # Optional: apply thresholding and save binary image
    binary = (gray > t_star).astype(np.uint8) * 255
    Image.fromarray(binary).save("otsu_result.png")
    print("Binary image saved as otsu_result.png")


if __name__ == "__main__":
    main()