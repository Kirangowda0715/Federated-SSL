"""
src/utils/generate_mock_data.py
------------------------------
Utility to generate dummy (random) image datasets for NIH, Shenzhen, and Montgomery.
Allows testing the 'real' pipeline (including splitter/loader logic) without 40GB of data.
"""

import os
import random
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def generate_random_image(size=(224, 224)):
    """Generate a random grayscale image."""
    arr = np.random.randint(0, 255, size, dtype=np.uint8)
    return Image.fromarray(arr)

def setup_nih(root_dir, num_images=100):
    print(f"Generating {num_images} mock NIH images...")
    img_dir = Path(root_dir) / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    
    for i in tqdm(range(num_images)):
        img = generate_random_image()
        # NIH images are usually named like 00000001_000.png
        img.save(img_dir / f"{i:08d}_000.png")

def setup_labeled(root_dir, name, num_tb=20, num_normal=20):
    print(f"Generating mock {name} labeled images...")
    root = Path(root_dir)
    tb_dir = root / "TB"
    normal_dir = root / "Normal"
    tb_dir.mkdir(parents=True, exist_ok=True)
    normal_dir.mkdir(parents=True, exist_ok=True)
    
    # TB images
    for i in tqdm(range(num_tb), desc=f"{name} TB"):
        img = generate_random_image()
        # Naming convention CXR_XXX_1.png (TB)
        img.save(tb_dir / f"CXR_{i:04d}_1.png")
        
    # Normal images
    for i in tqdm(range(num_normal), desc=f"{name} Normal"):
        img = generate_random_image()
        # Naming convention CXR_XXX_0.png (Normal)
        img.save(normal_dir / f"CXR_{i:04d}_0.png")

def main():
    base_data_dir = Path("data/raw")
    
    nih_path = base_data_dir / "NIH"
    shz_path = base_data_dir / "Shenzhen"
    mon_path = base_data_dir / "Montgomery"
    
    # 1. NIH
    setup_nih(nih_path, num_images=150)
    
    # 2. Shenzhen
    setup_labeled(shz_path, "Shenzhen", num_tb=30, num_normal=30)
    
    # 3. Montgomery
    setup_labeled(mon_path, "Montgomery", num_tb=20, num_normal=20)
    
    print("\n[Mock Data] Successfully generated dummy datasets in data/raw/")
    print("[Note] You can now run 'python src/federated/simulation.py --config configs/default.yaml'")

if __name__ == "__main__":
    main()
