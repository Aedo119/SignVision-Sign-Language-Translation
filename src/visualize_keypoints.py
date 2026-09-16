"""
Generates a visual proof-of-concept image: one sample per class with
MediaPipe hand landmarks drawn on top. Useful to paste into your report
or show your professor that keypoint detection is actually working.

Usage:
    python visualize_keypoints.py --data_dir ../data/Sign-Language-Digits-Dataset/Dataset \
                                   --out_dir ../outputs
"""

import argparse
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])

    thumbs = []
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                         min_detection_confidence=0.4) as hands_model:
        for class_name in class_names:
            class_dir = data_dir / class_name
            chosen_img = None
            for fname in sorted(class_dir.iterdir()):
                img = cv2.imread(str(fname))
                if img is None:
                    continue
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands_model.process(rgb)
                if results.multi_hand_landmarks:
                    for hand_lms in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            img, hand_lms, mp_hands.HAND_CONNECTIONS,
                            mp_styles.get_default_hand_landmarks_style(),
                            mp_styles.get_default_hand_connections_style(),
                        )
                    img = cv2.resize(img, (200, 200))
                    cv2.putText(img, class_name, (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2)
                    chosen_img = img
                    break
            if chosen_img is not None:
                thumbs.append(chosen_img)

    if not thumbs:
        print("No hands detected in any sample - try lowering confidence.")
        return

    # Arrange thumbnails in a grid (2 rows x 5 cols)
    rows = []
    per_row = 5
    for i in range(0, len(thumbs), per_row):
        row_imgs = thumbs[i:i + per_row]
        while len(row_imgs) < per_row:
            row_imgs.append(np.zeros_like(thumbs[0]))
        rows.append(np.hstack(row_imgs))
    grid = np.vstack(rows)

    out_path = Path(args.out_dir) / "keypoint_detection_demo.jpg"
    cv2.imwrite(str(out_path), grid)
    print(f"Saved visual proof to: {out_path}")


if __name__ == "__main__":
    main()
