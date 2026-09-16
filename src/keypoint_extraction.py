"""
Step 3 of the proposed methodology: Hand and Body Keypoint Detection
----------------------------------------------------------------------
Uses MediaPipe Hands (free, open-source, runs locally, no API key/cost)
to detect 21 hand landmarks (x, y, z) per detected hand in every
preprocessed image, and saves them as a structured array alongside labels.

This output (a sequence of keypoint vectors instead of raw pixels) is what
Step 4 (RepViT spatial feature extraction) and Step 5 (BiLSTM/GRU temporal
modeling) will consume next.

Usage:
    python keypoint_extraction.py --data_dir ../data/Sign-Language-Digits-Dataset/Dataset \
                                   --out_dir ../outputs
"""

import os
import argparse
import numpy as np
import cv2
import mediapipe as mp
from pathlib import Path

N_LANDMARKS = 21          # MediaPipe Hands: 21 landmarks per hand
N_COORDS = 3              # x, y, z per landmark
VECTOR_SIZE = N_LANDMARKS * N_COORDS  # 63 values per hand


def extract_keypoints_from_image(hands_model, img_bgr):
    """Return a 63-d keypoint vector for the first detected hand, or None."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = hands_model.process(img_rgb)
    if not results.multi_hand_landmarks:
        return None
    hand = results.multi_hand_landmarks[0]
    coords = []
    for lm in hand.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return np.array(coords, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--max_per_class", type=int, default=0,
                         help="0 = use all images; set e.g. 50 for a quick test run")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data_dir = Path(args.data_dir)
    class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])

    mp_hands = mp.solutions.hands
    keypoints, labels = [], []
    detected, missed = 0, 0

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
    ) as hands_model:
        for label_idx, class_name in enumerate(class_names):
            class_dir = data_dir / class_name
            files = sorted(os.listdir(class_dir))
            if args.max_per_class:
                files = files[: args.max_per_class]

            for fname in files:
                img_path = class_dir / fname
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                vec = extract_keypoints_from_image(hands_model, img)
                if vec is None:
                    missed += 1
                    continue
                keypoints.append(vec)
                labels.append(label_idx)
                detected += 1

            print(f"Class '{class_name}': processed {len(files)} images")

    keypoints = np.array(keypoints, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    out_path = Path(args.out_dir) / "hand_keypoints.npz"
    np.savez_compressed(
        out_path,
        keypoints=keypoints,
        labels=labels,
        class_names=np.array(class_names),
    )

    print(f"\nHand detected in {detected} images, no hand found in {missed} images.")
    print(f"Keypoint tensor shape: {keypoints.shape}  (N_samples x 63)")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
