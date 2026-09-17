"""
Failure analysis for keypoint detection.

Categorizes every image into one of four buckets based on which pass
detected a hand (or didn't):
  - normal            : detected on the first pass (confidence >= 0.5)
  - low_confidence     : only detected after lowering confidence to 0.25
  - flipped            : only detected after a horizontal-flip retry
  - missed             : not detected by any pass

Then:
  1. Computes basic image statistics (brightness, contrast) per category to
     check, quantitatively, whether "harder" categories actually look
     different from the easy ones - rather than just eyeballing samples.
  2. Builds a labeled visual grid with sample images (and landmarks, where
     detected) from each category, side by side, for direct inspection.

Usage:
    python failure_analysis.py --data_dir ../data/Sign-Language-Digits-Dataset/Dataset \
                                --out_dir ../outputs
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


def classify_all_images(data_dir, class_names):
    """Run the 3-pass detection cascade on every image, recording category,
    the detected landmarks (if any), and basic brightness/contrast stats."""
    records = []

    with mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                         min_detection_confidence=0.5) as hands_high, \
         mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                         min_detection_confidence=0.25) as hands_low:

        for class_name in class_names:
            class_dir = data_dir / class_name
            for fname in sorted(class_dir.iterdir()):
                img = cv2.imread(str(fname))
                if img is None:
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                brightness = float(np.mean(gray))
                contrast = float(np.std(gray))
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                category, landmarks_source = None, None
                res = hands_high.process(rgb)
                if res.multi_hand_landmarks:
                    category, landmarks_source = "normal", rgb
                else:
                    res = hands_low.process(rgb)
                    if res.multi_hand_landmarks:
                        category, landmarks_source = "low_confidence", rgb
                    else:
                        flipped = cv2.flip(rgb, 1)
                        res_f = hands_low.process(flipped)
                        if res_f.multi_hand_landmarks:
                            category, landmarks_source = "flipped", flipped
                        else:
                            category, landmarks_source = "missed", None

                records.append({
                    "path": fname,
                    "class": class_name,
                    "category": category,
                    "brightness": brightness,
                    "contrast": contrast,
                    "hand_landmarks": res.multi_hand_landmarks if category in
                        ("normal", "low_confidence") else
                        (res_f.multi_hand_landmarks if category == "flipped" else None),
                    "landmarks_image_rgb": landmarks_source,
                })
    return records


def plot_stat_comparison(records, out_path):
    cats = ["normal", "low_confidence", "flipped", "missed"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, stat, title in zip(axes, ["brightness", "contrast"],
                                 ["Brightness (mean pixel value)", "Contrast (pixel std dev)"]):
        data = [[r[stat] for r in records if r["category"] == c] for c in cats]
        ax.boxplot(data, labels=cats)
        ax.set_title(title)
        ax.set_xticklabels(cats, rotation=20)
    fig.suptitle("Image statistics by detection outcome")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print("\nMean +/- std by category:")
    for c in cats:
        b = [r["brightness"] for r in records if r["category"] == c]
        k = [r["contrast"] for r in records if r["category"] == c]
        if not b:
            continue
        print(f"  {c:15s} n={len(b):4d}  brightness={np.mean(b):6.1f}+/-{np.std(b):5.1f}"
              f"   contrast={np.mean(k):6.1f}+/-{np.std(k):5.1f}")


def build_sample_grid(records, out_path, n_per_cat=4):
    cats = ["normal", "low_confidence", "flipped", "missed"]
    rng = np.random.default_rng(42)
    rows = []

    for cat in cats:
        cat_records = [r for r in records if r["category"] == cat]
        if not cat_records:
            continue
        chosen = rng.choice(len(cat_records), size=min(n_per_cat, len(cat_records)), replace=False)
        thumbs = []
        for idx in chosen:
            r = cat_records[idx]
            img = cv2.imread(str(r["path"]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if r["category"] == "flipped":
                img = cv2.flip(img, 1)  # show the version that actually got detected
            if r["hand_landmarks"]:
                for hand_lms in r["hand_landmarks"]:
                    mp_drawing.draw_landmarks(
                        img, hand_lms, mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )
            img = cv2.resize(img, (160, 160))
            thumbs.append(img)
        while len(thumbs) < n_per_cat:
            thumbs.append(np.full((160, 160, 3), 230, dtype=np.uint8))
        row = np.hstack(thumbs)
        label_bar = np.full((30, row.shape[1], 3), 40, dtype=np.uint8)
        cv2.putText(label_bar, f"{cat}  (n={len(cat_records)})", (5, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        rows.append(np.vstack([label_bar, row]))

    grid = np.vstack(rows)
    grid_bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), grid_bgr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])

    print("Classifying all images by detection outcome (3-pass cascade)...")
    records = classify_all_images(data_dir, class_names)

    counts = {}
    for r in records:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
    print("Category counts:", counts)

    plot_stat_comparison(records, out_dir / "failure_analysis_stats.png")
    build_sample_grid(records, out_dir / "failure_analysis_grid.jpg")

    print(f"\nSaved stats comparison to {out_dir / 'failure_analysis_stats.png'}")
    print(f"Saved sample grid to {out_dir / 'failure_analysis_grid.jpg'}")


if __name__ == "__main__":
    main()
