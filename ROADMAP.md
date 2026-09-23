# Roadmap — Remaining Work

This covers everything after the data pipeline stage documented in `README.md`:
spatial feature extraction, temporal modeling, classification, language
processing, and deployment. Roughly 75% of the project by architecture stage.

## 1. Switch to the WLASL video dataset

The current dataset (Sign-Language-Digits) is static images of isolated signs.
Temporal modeling (Section 3 below) needs actual motion across frames, so the
next step is moving to a video dataset. The project has standardized on
**WLASL (Word-Level ASL)** for that phase.

**Selected dataset:** WLASL — a large English ASL word dataset with clip-based
examples. The videos are distributed as links to external sources (for example,
YouTube) rather than a single bundled download, so fetching and verification
need to be built into the data pipeline.

`src/preprocess.py` needs a video-aware variant: sample N frames per clip at a
fixed rate (e.g. via OpenCV `VideoCapture`) instead of treating each file as a
single frame. The resize/normalize logic stays the same.

---

## 2. Spatial feature extraction — RepViT

**Goal:** replace/augment the raw 63-d keypoint vectors with learned visual
features per frame (hand shape, finger configuration, orientation, posture).

**Tools needed (all free):**
- RepViT reference implementation — available on GitHub from the original
  paper's authors; can be used as a fixed pretrained feature extractor or
  fine-tuned.
- PyTorch or TensorFlow (whichever the team is more comfortable with;
  PyTorch has more readily available RepViT ports).
- GPU access — Google Colab's free tier (T4 GPU) is enough at this dataset
  scale.

**Implementation notes:**
- Start by using RepViT as a frozen feature extractor (no fine-tuning) to get
  a working end-to-end pipeline faster; fine-tune only if time/accuracy needs
  demand it.
- Output per frame should be a fixed-length feature vector — this is what
  gets stacked into a sequence for Section 3.
- Consider concatenating RepViT features with the MediaPipe keypoint vector
  already produced by `src/keypoint_extraction.py` — landmarks give precise
  geometric structure, RepViT gives visual context; combining both often
  outperforms either alone.

---

## 3. Temporal modeling — BiLSTM / GRU

**Goal:** learn how per-frame features change across a clip to capture motion,
not just static hand shape.

**Tools needed:**
- PyTorch (`nn.LSTM` / `nn.GRU`, bidirectional) or TensorFlow/Keras
  equivalents — both free, no extra setup beyond what Section 2 already needs.

**Implementation notes:**
- Input: sequence of per-frame feature vectors from Section 2, padded/masked
  to a fixed max sequence length per batch.
- Start with a single BiLSTM layer (128–256 hidden units) before adding
  depth — easier to debug, and often sufficient at small dataset scale.
- Log training/validation loss per epoch from the start; this is required
  evidence for the results section of the final report.

---

## 4. Sign classification

**Goal:** map the BiLSTM/GRU's output to a predicted sign/word.

**Implementation notes:**
- A softmax layer over the BiLSTM's final (or pooled) hidden state, trained
  with cross-entropy loss.
- Evaluate with accuracy, a confusion matrix, and per-class F1 — not just
  overall accuracy, since class imbalance is likely with a larger vocabulary.
- Keep the train/val/test split strategy consistent with `preprocess.py`
  (stratified, fixed random seed) so results are reproducible and comparable
  across experiments.

---

## 5. Language processing / post-processing

**Goal:** for continuous signing, turn a raw predicted word sequence into
coherent text (grammar, word order, context).

**Implementation notes:**
- Start simple: an n-gram language model or basic rule-based reordering is
  enough to demonstrate the concept and is fast to implement.
- If time allows, a small pretrained transformer (e.g. a lightweight
  sequence-to-sequence model available via Hugging Face, run locally/free)
  can be explored as a stretch goal.
- This stage is only meaningful once continuous (multi-word) signing is
  supported — for isolated single-sign classification it can be skipped
  entirely for the demo.

---

## 6. Text (and optional speech) output

**Tools needed (all free):**
- A minimal UI: Streamlit or Flask are both quick to set up.
- Optional text-to-speech: `pyttsx3` (runs fully offline, no API key, no cost).

---

## 7. Web/mobile integration and deployment

**Tools needed (all free tiers):**
- **Hugging Face Spaces** or **Streamlit Community Cloud** — host the demo
  for free, good fit for a Streamlit/Gradio front end.
- **Render** free tier — alternative if a Flask app is preferred over
  Streamlit/Gradio.

**Implementation notes:**
- Webcam input in-browser requires the deployment platform to support
  `getUserMedia`-style access (Streamlit and Gradio both have built-in
  components for this — no custom JS needed).
- Keep inference lightweight (or offer "upload a short clip" as a fallback)
  since free-tier hosting typically has limited CPU/GPU and may not sustain
  real-time webcam inference well.

---

## 8. Evaluation and ablation studies

**Done:** a keypoints-only baseline (Logistic Regression / Random Forest on
the 63-d MediaPipe vectors, no RepViT, no temporal modeling) has already
been trained and evaluated — ~98.7% test accuracy, 98.5% 5-fold CV accuracy.
See `README.md` "Baseline evaluation" for the full numbers and the explicit
caveat that this is a floor, not a preview of final system accuracy.

**Also done:** a failure analysis of the keypoint-detection step itself
(`FAILURE_ANALYSIS.md`) found that missed detections are concentrated in
specific hand poses (digits 1, 2, 6 — thin/sparse finger configurations),
not explained by image brightness, contrast, or framing. This means the
remaining ~8.7% of undetected images (179/2,062) are a targeted problem, not
a generic "improve image quality" one.

**Still remaining, once Sections 2–4 exist:**
- Keypoints-only (done, above) vs. RepViT features vs. combined.
- With vs. without the BiLSTM temporal stage (i.e. per-frame classification
  vs. sequence classification).
- Effect of dataset size / augmentation on accuracy.
- Worth revisiting given Finding 2 in `FAILURE_ANALYSIS.md`: a targeted
  fourth detection pass (e.g. cropping tighter around the detected
  skin-color region before re-running MediaPipe) aimed specifically at the
  low-detection-rate poses (classes 1, 2, 6), rather than further blind
  threshold/orientation tuning.

---

## Suggested task division (4 members)

This assumes roughly equal effort per person and lets each person own one
architecture stage end-to-end, with everyone contributing to integration and
the final report.

| Member | Primary ownership | Also contributes to |
|---|---|---|
| **A** | Video dataset setup (Section 1) + Spatial feature extraction / RepViT (Section 2) | Data pipeline extension for video (updating `preprocess.py`) |
| **B** | Temporal modeling / BiLSTM-GRU (Section 3) + Classification head (Section 4) | Evaluation & ablation studies (Section 8) |
| **C** | Language processing / post-processing (Section 5) + Text/speech output (Section 6) | Report write-up, results section |
| **D** | Web/mobile integration and deployment (Section 7) | Integration testing across all stages, demo prep |

Adjust based on actual team strengths (e.g. whoever is most comfortable with
PyTorch should likely pair on Sections 2–4 regardless of the split above).
Regardless of the split, agree early on:
1. The chosen video dataset (Section 1) — this blocks everyone else.
2. A shared feature-vector format/interface between Sections 2 → 3 → 4, so
   people can build their piece independently and integrate at the end.