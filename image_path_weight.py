import os
import csv
import random

# Input folder
input_dir = "images"
output_csv = "image_weights.csv"

# Extensions considered as images
valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"}

# Collect image paths
image_files = [
    os.path.join(input_dir, f)
    for f in os.listdir(input_dir)
    if os.path.splitext(f.lower())[1] in valid_exts
]

# Generate random weights (example: between 0.5 and 5.0)
data = [(img, round(random.uniform(0.5, 5.0), 2)) for img in image_files]

# Write to CSV
with open(output_csv, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "weight"])
    writer.writerows(data)

print(f"CSV file saved as {output_csv} with {len(data)} entries.")