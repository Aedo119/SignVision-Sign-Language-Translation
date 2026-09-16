"""
Improved version of keypoint_extraction.py.

The original script (min_detection_confidence=0.5, single pass) detected a
hand in 1,717 / 2,062 images (~83%). This version adds two cheap, free
improvements to recover some of the misses:

  1. Lower min_detection_confidence to 0.3 on a retry pass (trades a small
     risk of false positives for fewer missed detections).
  2. Horizontal-flip retry: some images have the hand mirrored relative to
     what MediaPipe's model was mostly trained on; if the first two passes
     fail, try again on a horizontally flipped copy (and flip the resulting
     x-coordinates back before saving).

Usage:
    python keypoint_extraction_v2.py --data_dir ../data/Sign-Language-Digits-Dataset/Dataset \
                                      --out_dir ../outputs
"""

import os
import argparse
import numpy as np
import cv2
import mediapipe as mp
from pathlib import Path

N_LANDMARKS = 21
N_COORDS = 3


def try_detect(hands_model, img_rgb):
    results = hands_model.process(img_rgb)
    if not results.multi_hand_landmarks:
        return None
    hand = results.multi_hand_landmarks[0]
    coords = []
    for lm in hand.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return np.array(coords, dtype=np.float32)


def extract_keypoints_robust(hands_high, hands_low, img_bgr):
    """Try normal confidence, then low confidence, then flipped, in order."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    vec = try_detect(hands_high, img_rgb)
    if vec is not None:
        return vec, "normal"

    vec = try_detect(hands_low, img_rgb)
    if vec is not None:
        return vec, "low_confidence"

    flipped_rgb = cv2.flip(img_rgb, 1)
    vec = try_detect(hands_low, flipped_rgb)
    if vec is not None:
        # un-flip x coordinates (every 3rd value starting at index 0) so
        # keypoints stay consistent with the original, unflipped image
        vec = vec.copy()
        vec[0::3] = 1.0 - vec[0::3]
        return vec, "flipped"

    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data_dir = Path(args.data_dir)
    class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])

    mp_hands = mp.solutions.hands
    keypoints, labels, sources = [], [], []
    missed = 0
    strategy_counts = {"normal": 0, "low_confidence": 0, "flipped": 0}

    with mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                         min_detection_confidence=0.5) as hands_high, \
         mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                         min_detection_confidence=0.25) as hands_low:

        for label_idx, class_name in enumerate(class_names):
            class_dir = data_dir / class_name
            files = sorted(os.listdir(class_dir))

            for fname in files:
                img_path = class_dir / fname
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                vec, strategy = extract_keypoints_robust(hands_high, hands_low, img)
                if vec is None:
                    missed += 1
                    continue
                keypoints.append(vec)
                labels.append(label_idx)
                sources.append(strategy)
                strategy_counts[strategy] += 1

            print(f"Class '{class_name}': processed {len(files)} images")

    keypoints = np.array(keypoints, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    total = len(keypoints) + missed

    out_path = Path(args.out_dir) / "hand_keypoints_v2.npz"
    np.savez_compressed(
        out_path,
        keypoints=keypoints,
        labels=labels,
        class_names=np.array(class_names),
        detection_strategy=np.array(sources),
    )

    print(f"\nTotal images: {total}")
    print(f"Detected: {len(keypoints)} ({100 * len(keypoints) / total:.1f}%)")
    print(f"Missed:   {missed} ({100 * missed / total:.1f}%)")
    print(f"Breakdown by strategy: {strategy_counts}")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()