# Decision Log — Sign Language Recognition & Translation (Team 29)

A running record of the significant technical and process decisions made on
this project, why they were made, and what alternatives were considered.

Status key: **Accepted** = currently in effect · **Superseded** = replaced by a later entry · **Open** = revisit later

---

### DEC-001 — Use a static-image dataset for Milestone 1, not video
**Status:** Accepted (scoped to Milestone 1 only — revisit before temporal modeling)

**Context:** The proposed architecture is built around sign-language *video*
feeding a BiLSTM/GRU temporal model. A true video dataset (WLASL, INCLUDE)
matches the architecture but requires resolving/downloading individual clips
from external sources — slow to bootstrap and adds failure points before the
core pipeline logic is even validated.

**Decision:** Use **Sign-Language-Digits-Dataset** (10 classes, 2,062 static
images, MIT license, single `git clone`, no account needed) to build and
validate Steps 1–3 of the pipeline (preprocessing, keypoint detection) and
get an early baseline evaluation signal.

**Alternatives considered:**
- WLASL / INCLUDE directly — rejected for Milestone 1 due to slower,
  higher-friction data acquisition; not rejected long-term (see DEC-009).
- Kaggle-hosted datasets (e.g. ASL Alphabet CSV) — rejected because they
  require a Kaggle account/API key, adding setup friction for teammates.

**Consequence:** This dataset cannot validate temporal modeling or
continuous/sentence-level signing — it structurally contains no motion. A
video dataset is required before Steps 5+ (see `ROADMAP.md` Section 1).

---

### DEC-002 — Fixed resize + normalization in preprocessing
**Status:** Accepted

**Context:** Needed to confirm whether the dataset's images were already
uniform before deciding whether a resize step was necessary overhead or a
requirement.

**Decision:** Resize every image to 128×128 and normalize to [0,1] in
`preprocess.py`, regardless of source resolution.

**Evidence:** A scan of the dataset (see `notebooks/dataset_analysis.ipynb`,
Section 2) found 2,059 images at a uniform 100×100, but **3 images at
3024×3024** (full phone-camera resolution, not resized on export). Without a
forced resize step, these three would break batched model input.

**Alternatives considered:** Skipping resize and filtering out
non-conforming images instead — rejected, since it throws away usable data
for no benefit; resizing is cheap and keeps the full dataset.

---

### DEC-003 — Stratified 70/15/15 split, no class weighting
**Status:** Accepted

**Context:** Needed to decide whether class imbalance handling (oversampling,
class weights) was necessary.

**Decision:** Use a stratified split with no additional balancing.

**Evidence:** Class counts range 204–208 images (std. dev. ~1.08), under 2%
spread — close enough to balanced that weighting would add complexity with
no measurable benefit.

**Revisit if:** a future video dataset (DEC-009) turns out to be less
balanced — recheck class counts before assuming this decision still holds.

---

### DEC-004 — MediaPipe for keypoint detection, pinned to version 0.10.14
**Status:** Accepted

**Context:** MediaPipe versions ~0.10.2x and later removed the legacy
`mp.solutions.hands` API in favor of a Tasks API that downloads model files
from Google's servers at runtime. That download requires network access to a
domain not available in every dev/CI environment, and adds a setup step.

**Decision:** Pin `mediapipe==0.10.14`, the last version with the legacy API
and bundled model weights — fully offline after `pip install`, no extra
downloads, no API key.

**Alternatives considered:**
- Latest MediaPipe + manual `.task` model download — rejected due to the
  extra network dependency and setup step for teammates.
- OpenPose or another pose-estimation library — not evaluated in depth;
  MediaPipe was chosen primarily for its ease of setup and real-time
  suitability (already named in the original proposed methodology).

---

### DEC-005 — Multi-pass retry strategy for keypoint detection
**Status:** Accepted

**Context:** A single pass at default confidence (0.5) only detected a hand
in 1,717 / 2,062 images (83.3%) — meaning ~17% of the dataset would be
silently dropped before any downstream stage saw it.

**Decision:** Add two free retry passes when the first fails: (1) a
lower-confidence retry (0.25), (2) a horizontally-flipped retry (with
x-coordinates un-flipped afterward). No new dependencies, no manual
labeling.

**Result:** Detection rate rose to 1,883 / 2,062 (91.3%) — 166 additional
usable samples (115 via low-confidence retry, 51 via flip retry).

**Trade-off accepted knowingly:** the recovered samples are, by
construction, the harder cases. Baseline classifier accuracy (DEC-006)
dropped slightly (99.7% → 98.7%) when evaluated on the larger, harder set —
accepted as a more honest number rather than reporting accuracy on the easy
83% only.

**Why this approach, and prior work it's grounded in:** MediaPipe Hands is
a two-stage pipeline (palm detector → landmark model) described in
Zhang, F., Bazarevsky, V., Vakunov, A., Tkachenka, A., Sung, G., Chang,
C-L., & Grundmann, M. (2020), *MediaPipe Hands: On-device Real-time Hand
Tracking*, arXiv:2006.10214 — 0.5 is the library's own default confidence
threshold, used unmodified as the first pass. The retry-on-transformed-input
approach (flip retry) is an instance of **test-time augmentation (TTA)**, a
well-established technique: Krizhevsky, Sutskever & Hinton (2012, NeurIPS,
the AlexNet paper) averaged predictions over crops and horizontal flips at
test time; Simonyan & Zisserman (2014, VGGNet) used flip-averaging
explicitly. See also Shanmugam, D. et al. (2021), *When and Why Test-Time
Augmentation Works*, arXiv:2011.11156, for a more critical analysis of when
TTA helps vs. doesn't — worth reading before assuming this generalizes to a
future video dataset without re-checking.

**Root cause, investigated further (see `FAILURE_ANALYSIS.md`):** the
original hypotheses above (borderline confidence / orientation bias)
correctly predicted that retries would recover *some* samples, but a deeper
analysis found the better explanation is **hand pose itself** — detection
rate varies from 80% to 100% by class, concentrated in thin/sparse-finger
poses (digits 1, 2, 6), and is *not* explained by brightness, contrast, or
framing (all statistically similar across detected vs. missed images). The
retries worked as a mechanism, but not fully for the reason originally
assumed — recorded here rather than quietly correcting the earlier
reasoning.

---

### DEC-006 — Add a baseline classifier before RepViT/BiLSTM exist
**Status:** Accepted

**Context:** The architecture's evaluation metrics (accuracy, F1, confusion
matrix) were originally scoped to the *final* classifier, which sits after
spatial (RepViT) and temporal (BiLSTM/GRU) stages that don't exist yet —
meaning there was no way to produce real evaluation numbers at Milestone 1.

**Decision:** Train lightweight baseline classifiers (Logistic Regression,
Random Forest) directly on the extracted keypoint vectors, purely as a
sanity check that the representation is discriminative — explicitly
documented as a floor, not the project's final result.

**Result:** ~98.7% test accuracy, 98.5% 5-fold CV accuracy (Logistic
Regression) — confirms the keypoint representation is sound before
investing time in RepViT/BiLSTM.

**Explicit caveat recorded alongside this decision:** isolated static-digit
classification is a materially easier task than continuous video-based sign
recognition; this number must not be quoted as expected final system
accuracy.

---

## Open decisions (not yet made — tracked here so they aren't forgotten)

| # | Decision needed | Notes |
|---|---|---|
| OPEN-1 | Which video dataset: WLASL vs. INCLUDE | Blocks all of `ROADMAP.md` Sections 2–4; see DEC-001's revisit note |
| OPEN-2 | Shared feature-vector interface between spatial (RepViT) and temporal (BiLSTM/GRU) stages | Needed so team members can build Sections 2–4 independently and integrate at the end (`ROADMAP.md`) |
| OPEN-3 | Deployment target for the final web app | Hugging Face Spaces vs. Streamlit Community Cloud vs. Render — all free-tier candidates, not yet decided |
| OPEN-4 | Whether to fine-tune RepViT or use it frozen | Affects compute budget and Colab free-tier feasibility |

---

*When a new significant decision is made, add a new `DEC-0XX` entry above
rather than editing past entries — if a decision is later reversed, mark the
old entry "Superseded" and link to the new one, don't delete it. The point of
this log is to preserve *why*, including decisions that were later changed.*
