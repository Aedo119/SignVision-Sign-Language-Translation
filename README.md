# Sign Language Recognition & Translation

A system that translates sign language video into text, following a data pipeline →
deep learning → language processing architecture. This repository currently
implements the **data pipeline stage**: dataset acquisition, preprocessing,
hand keypoint detection, and a baseline evaluation of that representation.

**Team 29** — Christine Mary Paul, Kavya Nair Puthiyedath, Hema Sudabathula, Joanna Sara Jipson

---

## Project architecture

```
Sign language video
        ↓
Frame preprocessing            ┐
        ↓                      │  Data pipeline  (implemented)
Keypoint detection             ┘
        ↓
Spatial feature extraction     ┐
        ↓                      │
Temporal modeling              │  Deep learning  (planned — see ROADMAP.md)
        ↓                      │
Sign classification            ┘
        ↓
Language processing             — Grammar/context refinement (planned)
        ↓
Text output
```

## Current implementation status

| Stage | Description | Script |
|---|---|---|
| Frame preprocessing | Load, resize, normalize images; stratified train/val/test split | `src/preprocess.py` |
| Keypoint detection | Extract 21-point hand landmarks per frame via MediaPipe, with confidence/flip retries to maximize detection rate | `src/keypoint_extraction.py` |
| Verification | Visual overlay of detected landmarks on sample frames | `src/visualize_keypoints.py` |
| Baseline evaluation | Sanity-check classifiers trained on the extracted keypoints | `src/baseline_classifier.py` |

Everything else in the architecture (spatial feature extraction, temporal
modeling, final classification, language processing, and the web interface)
is not yet implemented. See [`ROADMAP.md`](./ROADMAP.md) for the detailed
plan, required tools, and task breakdown for the remaining work.

---

## Dataset

**[Sign-Language-Digits-Dataset](https://github.com/ardamavi/Sign-Language-Digits-Dataset)**
(Ankara Ayrancı Anadolu High School / ardamavi, MIT License) — 2,062 images
across 10 classes (digits 0–9), ~205–208 images per class, hosted freely on
GitHub with no registration required.

### Why this dataset, for now

The architecture is designed around full sign-language *video* (continuous
motion feeding a temporal BiLSTM/GRU model). We're deliberately starting with
a static-image, isolated-sign dataset instead, for three reasons:

1. **De-risking the pipeline before adding video complexity.** Preprocessing
   and keypoint detection logic is identical whether the input is a single
   frame or a frame sampled from video — validating it on images first lets
   us catch bugs early and cheaply, before dealing with the much larger
   storage and processing cost of video.
3. **A working baseline.** A static-image dataset alone is sufficient for
   isolated sign/digit classification, so it can support an early end-to-end
   demo (including real evaluation numbers, see below) while the
   video-based dataset (for continuous signing) is prepared in parallel.

The switch to a video dataset (WLASL or INCLUDE) happens at the temporal
modeling stage — details and rationale in `ROADMAP.md`.

---

## Setup

Requires Python 3.9–3.12.

```bash
pip install "mediapipe==0.10.14" opencv-python-headless numpy pandas scikit-learn matplotlib
```

> **Version note:** MediaPipe is pinned to `0.10.14`. Versions from ~0.10.2x
> onward removed the bundled `mp.solutions` API in favor of one that fetches
> model files from Google's servers at runtime. Pinning to `0.10.14` keeps
> everything working fully offline with no extra downloads.

use a virtual environment instead:
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install "mediapipe==0.10.14" opencv-python-headless numpy pandas scikit-learn matplotlib
```

## Getting the dataset

```bash
cd data
git clone https://github.com/ardamavi/Sign-Language-Digits-Dataset.git
cd ..
```

## Running the pipeline

```bash
cd src

python preprocess.py --data_dir ../data/Sign-Language-Digits-Dataset/Dataset --out_dir ../outputs --img_size 128

python keypoint_extraction.py --data_dir ../data/Sign-Language-Digits-Dataset/Dataset --out_dir ../outputs

python visualize_keypoints.py --data_dir ../data/Sign-Language-Digits-Dataset/Dataset --out_dir ../outputs

python baseline_classifier.py --keypoints ../outputs/hand_keypoints.npz --out_dir ../outputs
```

(run each command on one line as above, or use a backtick
`` ` `` instead of `\` for line continuation — PowerShell doesn't use `\`.)

## Outputs

| File | Contents |
|---|---|
| `outputs/preprocessed_dataset.npz` | `X_train/val/test`, `y_train/val/test`, `class_names` — resized, normalized image tensors, stratified 70/15/15 split |
| `outputs/hand_keypoints.npz` | `keypoints` (N × 63 array: 21 landmarks × x,y,z), `labels`, `class_names` |
| `outputs/keypoint_detection_demo.jpg` | Visual grid of landmark detection, one sample per class |
| `outputs/baseline_metrics.json` | Full per-class precision/recall/F1 for both baseline models |
| `outputs/confusion_matrix_logreg.png`, `confusion_matrix_rf.png` | Confusion matrices for the baseline classifiers |

---

## Results from the reference run

### Preprocessing & keypoint detection

- **Preprocessing:** all 2,062 images loaded successfully (0 corrupt files),
  resized to 128×128, normalized to [0, 1]. Split: 1,443 train / 309 val / 310
  test, stratified by class.
- **Keypoint detection:** a single-pass, default-confidence detector found a
  hand in 1,717 of 2,062 images (~83%). Adding two free retry strategies —
  a lower-confidence retry pass, and a horizontally-flipped retry (with
  x-coordinates un-flipped afterward) for images that still failed — raised
  this to **1,883 / 2,062 images (91.3%)**, recovering 166 extra usable
  samples with no new dependencies and no manual labeling.

| | Detected |
|---|---|
| Normal-confidence pass | 1,717 |
| + low-confidence retry | +115 |
| + flipped-image retry | +51 |
| **Total** | **1,883 / 2,062 (91.3%)** |

### Baseline evaluation — is the keypoint representation actually useful?

We trained two lightweight baseline classifiers directly on the 63-d keypoint vectors
already extracted, to get real, reportable numbers now and to sanity-check
that the keypoint representation is actually working well before investing
time in RepViT/BiLSTM.

**This is a baseline, not the final model** — static per-image
classification with no temporal component. It's a lower bound.

80/20 stratified split, plus 5-fold cross-validation, on the improved
(1,883-sample) keypoint set:

| Model | Test accuracy | Macro F1 | 5-fold CV accuracy |
|---|---|---|---|
| Logistic Regression | 98.7% | 0.986 | 98.5% ± 0.6% |
| Random Forest | 98.1% | 0.981 | 97.8% ± 0.6% |

Confusion matrices (`confusion_matrix_logreg.png`, `confusion_matrix_rf.png`)
are near-perfectly diagonal, confirming the 21-point hand landmark
representation cleanly separates these 10 digit classes — a good sign for
the harder, video-based continuous-sign task ahead.

> 98%+ accuracy on isolated static digit signs is a much easier problem 
> than continuous video-based sign recognition. This result is evidence the
>  *keypoint representation* is sound, not a preview of the final system's accuracy.

---

## Repository structure

```
.
├── README.md               this file
├── ROADMAP.md               remaining work, tools, and task division
├── data/
│   └── Sign-Language-Digits-Dataset/   (clone separately, see above)
├── src/
│   ├── preprocess.py
│   ├── keypoint_extraction.py
│   ├── visualize_keypoints.py
│   └── baseline_classifier.py
└── outputs/                 generated by running the scripts above
```