"""
Step 2 of the proposed methodology: Frame Extraction and Preprocessing
-----------------------------------------------------------------------
Reads raw sign-language images (one image = one "frame"/isolated sign sample),
resizes them to a fixed size, normalizes pixel values, removes unreadable/corrupt
files, and produces a clean train/val/test split saved as compressed .npz arrays.

Dataset used: Sign-Language-Digits-Dataset (ardamavi, MIT license, free, GitHub)
  10 classes (digits 0-9), ~2060 images total.

Usage:
    python preprocess.py --data_dir ../data/Sign-Language-Digits-Dataset/Dataset \
                          --out_dir ../outputs --img_size 128
"""

import os
import argparse
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split


def load_and_preprocess(data_dir: str, img_size: int = 128):
    """Walk class-labeled subfolders, load images, resize + normalize."""
    data_dir = Path(data_dir)
    class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    images, labels = [], []
    skipped = 0

    for label_idx, class_name in enumerate(class_names):
        class_dir = data_dir / class_name
        for img_path in class_dir.iterdir():
            img = cv2.imread(str(img_path))
            if img is None:
                skipped += 1
                continue
            # Resize to a fixed size expected by downstream CNN encoder
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
            # Convert BGR (OpenCV default) -> RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Normalize to [0, 1]
            img = img.astype(np.float32) / 255.0
            images.append(img)
            labels.append(label_idx)

    print(f"Loaded {len(images)} images, skipped {skipped} unreadable files.")
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int64), class_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=128)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    X, y, class_names = load_and_preprocess(args.data_dir, args.img_size)

    # 70% train / 15% val / 15% test, stratified so each split has all classes
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    out_path = Path(args.out_dir) / "preprocessed_dataset.npz"
    np.savez_compressed(
        out_path,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        class_names=np.array(class_names),
    )

    print(f"\nSplit sizes -> train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")
    print(f"Saved preprocessed dataset to: {out_path}")


if __name__ == "__main__":
    main()
