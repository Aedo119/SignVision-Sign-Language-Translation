"""
Baseline evaluation: is the keypoint representation actually good enough to
classify signs?

This is NOT the final classifier from the proposed architecture (that comes
after RepViT spatial features + BiLSTM/GRU temporal modeling, once we move to
video). It's a lightweight sanity-check classifier trained directly on the
63-d MediaPipe hand-keypoint vectors already extracted, to get real,
reportable evaluation numbers at this stage of the project rather than
deferring evaluation entirely to a later milestone.

Two models are compared:
  - Logistic Regression (simple linear baseline)
  - Random Forest (nonlinear baseline, usually stronger on landmark data)

Usage:
    python baseline_classifier.py --keypoints ../outputs/hand_keypoints.npz \
                                   --out_dir ../outputs
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)


def evaluate_model(name, model, X_train, y_train, X_test, y_test, class_names):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    cv_scores = cross_val_score(model, np.vstack([X_train, X_test]),
                                 np.concatenate([y_train, y_test]), cv=5)

    report = classification_report(y_test, y_pred, target_names=class_names,
                                    output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n=== {name} ===")
    print(f"Test accuracy:      {acc:.4f}")
    print(f"Macro F1:           {macro_f1:.4f}")
    print(f"Weighted F1:        {weighted_f1:.4f}")
    print(f"5-fold CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    return {
        "model": name,
        "test_accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "cv_accuracy_mean": cv_scores.mean(),
        "cv_accuracy_std": cv_scores.std(),
        "per_class_report": report,
    }, cm


def plot_confusion_matrix(cm, class_names, out_path, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keypoints", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.keypoints, allow_pickle=True)
    X, y = data["keypoints"], data["labels"]
    class_names = [str(c) for c in data["class_names"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = []

    lr = LogisticRegression(max_iter=2000)
    res_lr, cm_lr = evaluate_model("Logistic Regression", lr, X_train, y_train,
                                    X_test, y_test, class_names)
    results.append(res_lr)
    plot_confusion_matrix(cm_lr, class_names, out_dir / "confusion_matrix_logreg.png",
                           "Logistic Regression - Confusion Matrix")

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    res_rf, cm_rf = evaluate_model("Random Forest", rf, X_train, y_train,
                                    X_test, y_test, class_names)
    results.append(res_rf)
    plot_confusion_matrix(cm_rf, class_names, out_dir / "confusion_matrix_rf.png",
                           "Random Forest - Confusion Matrix")

    with open(out_dir / "baseline_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved metrics to {out_dir / 'baseline_metrics.json'}")
    print(f"Saved confusion matrix plots to {out_dir}")


if __name__ == "__main__":
    main()