"""
Batch Grayscale Converter for Tomato Images

This script processes all images in a given input directory,
converts them to grayscale, and saves them in an output directory.

Author: Your Name
"""

import os
from PIL import Image

# ----------------------------
# CONFIG
# ----------------------------
INPUT_DIR = "./tomato_images"        # path to input directory with tomato images
OUTPUT_DIR = "./grayscale_outputs"   # path to save grayscale images

# ----------------------------
# Main batch processor
# ----------------------------
def process_images(input_dir: str, output_dir: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    supported_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(supported_ext):
            continue

        fpath = os.path.join(input_dir, fname)

        try:
            # Convert to grayscale
            img = Image.open(fpath).convert("L")

            # Save grayscale result
            out_name = os.path.splitext(fname)[0] + "_gray.png"
            out_path = os.path.join(output_dir, out_name)
            img.save(out_path)

            print(f"[OK] {fname} -> {out_name}")

        except Exception as e:
            print(f"[ERROR] Skipping {fname}: {e}")

    print(f"\n[COMPLETE] Grayscale images saved in {output_dir}")

# ----------------------------
# Entry point
# ----------------------------
if __name__ == "__main__":
    process_images(INPUT_DIR, OUTPUT_DIR)