# Sign Language Recognition & Translation

A system that translates sign language video into text, following a data pipeline →
deep learning → language processing architecture. This repository currently
implements the **data pipeline stage**: dataset acquisition, preprocessing, and
hand keypoint detection.

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

This repository implements the **first three stages** of the architecture above:

| Stage | Description | Script |
|---|---|---|
| Frame preprocessing | Load, resize, normalize images; stratified train/val/test split | `src/preprocess.py` |
| Keypoint detection | Extract 21-point hand landmarks per frame via MediaPipe | `src/keypoint_extraction.py` |
| Verification | Visual overlay of detected landmarks on sample frames | `src/visualize_keypoints.py` |

Everything else in the architecture (spatial feature extraction, temporal
modeling, classification, language processing, and the web interface) is not
yet implemented. See [`ROADMAP.md`](./ROADMAP.md) for the detailed plan,
required tools, and task breakdown for the remaining work.

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

1. **Immediate, frictionless access.** It's a direct `git clone` with no
   account, quota, or manual video download step, so the whole team can
   reproduce results instantly.
2. **De-risking the pipeline before adding video complexity.** Preprocessing
   and keypoint detection logic is identical whether the input is a single
   frame or a frame sampled from video — validating it on images first lets
   us catch bugs early and cheaply, before dealing with the much larger
   storage and processing cost of video.
3. **A working baseline.** A static-image dataset alone is sufficient for
   isolated sign/digit classification, so it can support an early end-to-end
   demo while the video-based dataset (for continuous signing) is prepared in
   parallel.

The switch to a video dataset (WLASL or INCLUDE) happens at the temporal
modeling stage — details and rationale in `ROADMAP.md`.

---

## Setup

Requires Python 3.9–3.12.

```bash
pip install "mediapipe==0.10.14" opencv-python-headless numpy pandas scikit-learn
```

> **Version note:** MediaPipe is pinned to `0.10.14`. Versions from ~0.10.2x
> onward removed the bundled `mp.solutions` API in favor of one that fetches
> model files from Google's servers at runtime. Pinning to `0.10.14` keeps
> everything working fully offline with no extra downloads.

On some Linux setups pip may require `--break-system-packages`, or use a
virtual environment instead:
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install "mediapipe==0.10.14" opencv-python-headless numpy pandas scikit-learn
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

python preprocess.py \
    --data_dir ../data/Sign-Language-Digits-Dataset/Dataset \
    --out_dir ../outputs \
    --img_size 128

python keypoint_extraction.py \
    --data_dir ../data/Sign-Language-Digits-Dataset/Dataset \
    --out_dir ../outputs

python visualize_keypoints.py \
    --data_dir ../data/Sign-Language-Digits-Dataset/Dataset \
    --out_dir ../outputs
```

## Outputs

| File | Contents |
|---|---|
| `outputs/preprocessed_dataset.npz` | `X_train/val/test`, `y_train/val/test`, `class_names` — resized, normalized image tensors, stratified 70/15/15 split |
| `outputs/hand_keypoints.npz` | `keypoints` (N × 63 array: 21 landmarks × x,y,z), `labels`, `class_names` |
| `outputs/keypoint_detection_demo.jpg` | Visual grid of landmark detection, one sample per class |

```
.
├── README.md              this file
├── ROADMAP.md              remaining work, tools, and task division
├── data/
│   └── Sign-Language-Digits-Dataset/   (clone separately, see above)
├── src/
│   ├── preprocess.py
│   ├── keypoint_extraction.py
│   └── visualize_keypoints.py
└── outputs/                generated by running the scripts above
```
