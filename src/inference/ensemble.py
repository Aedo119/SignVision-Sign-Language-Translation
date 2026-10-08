# src/inference/ensemble.py
"""WLASL100 Transformer ensemble (ported from src/notebooks/video_to_gloss.ipynb).

The checkpoint ``outputs/ensemble.pt`` holds ``{"states": [state_dict, ...], "T": 32}``:
the weights of several ``SignTransformer`` models trained with different seeds.
Normalization, feature construction and the model definition below must stay
identical to the notebook, otherwise the weights will not produce valid predictions.
"""

import json
import logging
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("signvision.ensemble")

NPTS = 7 + 21 + 21
POSE_D = 7 * 2 * 2 + 7                 # xy + vel + mask                              = 35
HAND_D = 21 * 2 * 2 + 21 * 3 * 2 + 1   # global xy + vel, local xyz + vel, presence   = 211

# WLASL100 class order (indices 0-99 of wlasl_class_list.txt). Overridden by
# outputs/glosses.json (saved by the notebook) when that file is present.
WLASL100_GLOSSES = [
    "book", "drink", "computer", "before", "chair", "go", "clothes", "who", "candy", "cousin",
    "deaf", "fine", "help", "no", "thin", "walk", "year", "yes", "all", "black",
    "cool", "finish", "hot", "like", "many", "mother", "now", "orange", "table", "thanksgiving",
    "what", "woman", "bed", "blue", "bowling", "can", "dog", "family", "fish", "graduate",
    "hat", "hearing", "kiss", "language", "later", "man", "shirt", "study", "tall", "white",
    "wrong", "accident", "apple", "bird", "change", "color", "corn", "cow", "dance", "dark",
    "doctor", "eat", "enjoy", "forget", "give", "last", "meet", "pink", "pizza", "play",
    "school", "secretary", "short", "time", "want", "work", "africa", "basketball", "birthday", "brown",
    "but", "cheat", "city", "cook", "decide", "full", "how", "jacket", "letter", "medicine",
    "need", "paint", "paper", "pull", "purple", "right", "same", "son", "tell", "thursday",
]


# ---------- normalization + features (same as the notebook) ----------

def normalize(kp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """kp: (L, 49, 3) raw, NaN = missing  ->  X (L, 49, 3), M (L, 49).
    X: shoulder-centred, shoulder-width-scaled, missing = 0.  M: 1 where the point was detected."""
    L = len(kp)
    kp = (pd.DataFrame(kp.astype(np.float32).reshape(L, -1))
            .interpolate(limit=3, limit_area="inside").values.reshape(L, NPTS, 3))
    sh = kp[:, 1:3, :2]                                                   # shoulders
    with warnings.catch_warnings():                                       # no shoulders -> "Mean of empty slice"
        warnings.simplefilter("ignore", RuntimeWarning)
        ctr = pd.DataFrame(np.nanmean(sh, axis=1)).interpolate(limit_direction="both").values
        ctr = np.nan_to_num(ctr, nan=0.5)
        wid = np.nanmean(np.linalg.norm(sh[:, 0] - sh[:, 1], axis=-1))
    wid = 0.25 if (not np.isfinite(wid) or wid < 1e-3) else wid
    out = kp.copy()
    out[..., :2] = (kp[..., :2] - ctr[:, None, :]) / wid
    out[..., 2] = kp[..., 2] / wid
    M = np.isfinite(out[..., 0]).astype(np.float32)
    X = np.clip(np.nan_to_num(out, nan=0.0), -5, 5).astype(np.float32)
    return X, M


def _vel(a, valid):
    v = np.zeros_like(a)
    v[1:] = (a[1:] - a[:-1]) * 5 * (valid[1:] & valid[:-1])[:, None]
    return v


def _local(h, hm):
    size = np.linalg.norm(h[:, 9, :2] - h[:, 0, :2], axis=-1)[:, None, None]   # wrist -> middle-finger base
    return np.clip((h - h[:, :1]) / np.maximum(size, 0.05), -6, 6) * hm[:, :, None]


def make_features(x: np.ndarray, m: np.ndarray) -> np.ndarray:
    n = len(x)
    pose = x[:, :7, :2].reshape(n, -1)
    parts = [pose, _vel(pose, m[:, :7].mean(1) > 0.5), m[:, :7]]
    for s, e in ((7, 28), (28, 49)):
        h, hm = x[:, s:e], m[:, s:e]
        ok = hm[:, 0] > 0
        g = h[..., :2].reshape(n, -1)
        loc = _local(h, hm).reshape(n, -1)
        parts += [g, _vel(g, ok), loc, _vel(loc, ok), hm[:, :1]]
    return np.concatenate(parts, -1).astype(np.float32)


# ---------- model (same as the notebook) ----------

class SignTransformer(nn.Module):
    def __init__(self, n_classes=100, d=256, layers=4, heads=8, drop=0.2, hid=128, T=32):
        super().__init__()
        def branch(n_in):
            return nn.Sequential(nn.LayerNorm(n_in), nn.Linear(n_in, hid), nn.GELU(), nn.Dropout(drop))
        self.pose, self.lh, self.rh = branch(POSE_D), branch(HAND_D), branch(HAND_D)
        self.fuse = nn.Sequential(nn.Linear(3 * hid, d), nn.LayerNorm(d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.randn(1, T + 1, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, heads, d * 2, drop, batch_first=True, norm_first=True, activation="gelu")
        self.temporal = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(drop), nn.Linear(d, n_classes))

    def forward(self, x):                         # x: (B, T, POSE_D + 2*HAND_D)
        z = torch.cat([self.pose(x[..., :POSE_D]),
                       self.lh(x[..., POSE_D:POSE_D + HAND_D]),
                       self.rh(x[..., POSE_D + HAND_D:])], -1)
        z = self.fuse(z)
        z = torch.cat([self.cls.expand(len(z), -1, -1), z], 1) + self.pos
        return self.head(self.temporal(z)[:, 0])


class SignEnsemble:
    """Loads every model in ensemble.pt and averages their softmax outputs."""

    def __init__(self, ckpt_path: Path, glosses_path: Optional[Path] = None):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if not (isinstance(ckpt, dict) and "states" in ckpt):
            raise ValueError(f"{ckpt_path} is not an ensemble checkpoint (expected keys 'states', 'T')")
        self.T = int(ckpt.get("T", 32))
        self.models: List[SignTransformer] = []
        for state in ckpt["states"]:
            n_classes = state["head.2.weight"].shape[0]
            model = SignTransformer(n_classes=n_classes, T=self.T)
            model.load_state_dict(state)
            self.models.append(model.eval())
        self.glosses = self._load_glosses(glosses_path, n_classes)
        logger.info("Loaded ensemble of %d models (%d classes, T=%d)",
                    len(self.models), n_classes, self.T)

    @staticmethod
    def _load_glosses(path: Optional[Path], n_classes: int) -> List[str]:
        if path is not None and path.exists():
            glosses = json.loads(path.read_text(encoding="utf-8"))
            if len(glosses) == n_classes:
                return glosses
            logger.warning("%s has %d glosses, model has %d classes; using built-in list",
                           path, len(glosses), n_classes)
        if n_classes != len(WLASL100_GLOSSES):
            raise ValueError(f"No gloss list for {n_classes} classes; provide outputs/glosses.json")
        return WLASL100_GLOSSES

    @torch.no_grad()
    def predict(self, kp: np.ndarray) -> Tuple[str, float]:
        """kp: (L, 49, 3) raw landmarks (NaN = missing). Returns (gloss, confidence)."""
        X, M = normalize(kp)
        idx = np.linspace(0, len(X) - 1, self.T).round().astype(int)
        xb = torch.from_numpy(make_features(X[idx], M[idx])).unsqueeze(0)
        probs = torch.stack([F.softmax(m(xb), -1) for m in self.models]).mean(0)[0]
        conf, pred = probs.max(-1)
        return self.glosses[int(pred)], float(conf)
