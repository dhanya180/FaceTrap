#!/usr/bin/env python3
"""Build train-only ArcFace centroids and evaluate held-out manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import time
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from insightface import model_zoo
from insightface.utils import face_align
from insightface.utils.storage import ensure_available
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


IDENTITIES = ("you", "teammate", "professor")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, default=Path("dataset/splits"))
    parser.add_argument("--detector", type=Path, default=Path("app/src/main/assets/det_10g.onnx"))
    parser.add_argument("--recognizer", type=Path, default=Path("app/src/main/assets/w600k_r50.onnx"))
    parser.add_argument("--output", type=Path, default=Path("report/results"))
    parser.add_argument("--references", type=Path, default=Path("reference_embeddings"))
    parser.add_argument("--android-assets", type=Path, default=Path("app/src/main/assets"))
    parser.add_argument(
        "--export-references",
        action="store_true",
        help="Explicitly replace reference .npy files with train-only centroids.",
    )
    parser.add_argument("--cache", type=Path, default=Path(".cache/evaluation_embeddings"))
    parser.add_argument("--recognition-threshold", type=float, default=0.30)
    parser.add_argument("--trigger-threshold", type=float, default=0.85)
    parser.add_argument("--calibration-slope", type=float, default=10.582016)
    parser.add_argument("--calibration-intercept", type=float, default=-4.672831)
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def aligned_face(detector, path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path))
    if image is None:
        return None
    boxes, landmarks = detector.detect(image, max_num=1)
    if boxes.shape[0] == 0:
        return None
    return face_align.norm_crop(image, landmark=landmarks[0], image_size=112)


def cache_path(cache: Path, image_path: str) -> Path:
    key = hashlib.sha256(image_path.encode("utf-8")).hexdigest()
    return cache / f"{key}.npy"


def extract(
    rows: list[dict[str, str]], detector, recognizer, split: str, cache: Path
) -> list[dict[str, object]]:
    detected: list[tuple[dict[str, str], np.ndarray]] = []
    extracted: list[dict[str, object]] = []
    for index, row in enumerate(rows, 1):
        saved = cache_path(cache, row["path"])
        if saved.exists():
            extracted.append({**row, "embedding": np.load(saved)})
            continue
        crop = aligned_face(detector, Path(row["path"]))
        if crop is not None:
            detected.append((row, crop))
        if index % 50 == 0 or index == len(rows):
            print(
                f"{split}: {index}/{len(rows)} scanned, "
                f"{len(extracted)} cached + {len(detected)} newly detected",
                flush=True,
            )

    batch_size = 32
    for start in range(0, len(detected), batch_size):
        batch = detected[start:start + batch_size]
        vectors = recognizer.get_feat([crop for _, crop in batch]).astype(np.float32)
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        for (row, _), vector in zip(batch, vectors):
            np.save(cache_path(cache, row["path"]), vector)
            extracted.append({**row, "embedding": vector})
        print(f"{split}: embedded {len(extracted)}/{len(rows)}", flush=True)
    print(f"{split}: embedded {len(extracted)} faces", flush=True)
    return extracted


def make_references(train: list[dict[str, object]]) -> dict[str, np.ndarray]:
    references: dict[str, np.ndarray] = {}
    for identity in IDENTITIES:
        vectors = np.stack([row["embedding"] for row in train if row["identity"] == identity])
        centroid = vectors.mean(axis=0, dtype=np.float32)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
        references[identity] = centroid
    return references


def evaluate(
    rows: list[dict[str, object]],
    references: dict[str, np.ndarray],
    recognition_threshold: float,
    trigger_threshold: float,
    calibration_slope: float,
    calibration_intercept: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    truth_identity: list[str] = []
    predicted_identity: list[str] = []
    truth_target: list[int] = []
    predicted_target: list[int] = []
    professor_scores: list[float] = []
    records: list[dict[str, object]] = []

    for row in rows:
        scores = {name: float(np.dot(row["embedding"], ref)) for name, ref in references.items()}
        best_identity = max(scores, key=scores.get)
        accepted = best_identity if scores[best_identity] >= recognition_threshold else "unknown"
        target = int(row["identity"] == "professor")
        target_confidence = float(
            1.0 / (1.0 + np.exp(-(calibration_slope * scores["professor"] + calibration_intercept)))
        )
        triggered = int(best_identity == "professor" and target_confidence > trigger_threshold)
        truth_identity.append(str(row["identity"]))
        predicted_identity.append(accepted)
        truth_target.append(target)
        predicted_target.append(triggered)
        professor_scores.append(scores["professor"])
        records.append({
            "path": row["path"],
            "true_identity": row["identity"],
            "predicted_identity": accepted,
            "you_score": scores["you"],
            "teammate_score": scores["teammate"],
            "professor_score": scores["professor"],
            "target_confidence": target_confidence,
            "target_triggered": triggered,
        })

    target_counts = confusion_matrix(truth_target, predicted_target, labels=[0, 1])
    tn, fp, fn, tp = target_counts.ravel()
    metrics: dict[str, object] = {
        "evaluated": len(rows),
        "identity_accuracy": accuracy_score(truth_identity, predicted_identity),
        "identity_counts": dict(Counter(truth_identity)),
        "target_trigger": {
            "threshold": trigger_threshold,
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
            "precision": precision_score(truth_target, predicted_target, zero_division=0),
            "recall": recall_score(truth_target, predicted_target, zero_division=0),
            "f1": f1_score(truth_target, predicted_target, zero_division=0),
            "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
            "roc_auc": roc_auc_score(truth_target, professor_scores),
        },
    }
    return metrics, records


def write_records(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)


def plot_results(output: Path, test_records: list[dict[str, object]]) -> None:
    truth = [str(row["true_identity"]) for row in test_records]
    predicted = [str(row["predicted_identity"]) for row in test_records]
    labels = [*IDENTITIES, "unknown"]
    matrix = confusion_matrix(truth, predicted, labels=labels)
    display = ConfusionMatrixDisplay(matrix, display_labels=labels)
    display.plot(cmap="Blues", colorbar=False)
    plt.title("Held-out identity confusion matrix")
    plt.tight_layout()
    plt.savefig(output / "confusion_matrix.png", dpi=180)
    plt.close()

    target = np.array([row["true_identity"] == "professor" for row in test_records], dtype=int)
    scores = np.array([row["professor_score"] for row in test_records], dtype=float)
    fpr, tpr, _ = roc_curve(target, scores)
    auc = roc_auc_score(target, scores)
    plt.figure(figsize=(5.2, 4.2))
    plt.plot(fpr, tpr, label=f"ArcFace score (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False-positive rate")
    plt.ylabel("True-positive rate")
    plt.title("Professor trigger ROC on held-out test set")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output / "roc_curve.png", dpi=180)
    plt.close()


def ablation_study(
    train: list[dict[str, object]],
    test: list[dict[str, object]],
    recognition_threshold: float,
    trigger_threshold: float,
    calibration_slope: float,
    calibration_intercept: float,
) -> list[dict[str, float]]:
    results: list[dict[str, float]] = []
    for samples_per_identity in (1, 5, 25, 100, 400):
        subset: list[dict[str, object]] = []
        for identity in IDENTITIES:
            subset.extend(
                [row for row in train if row["identity"] == identity][:samples_per_identity]
            )
        references = make_references(subset)
        metrics, _ = evaluate(
            test, references, recognition_threshold, trigger_threshold,
            calibration_slope, calibration_intercept,
        )
        results.append({
            "samples_per_identity": samples_per_identity,
            "identity_accuracy": float(metrics["identity_accuracy"]),
            "target_roc_auc": float(metrics["target_trigger"]["roc_auc"]),
        })
    return results


def plot_ablation(output: Path, results: list[dict[str, float]]) -> None:
    x = [row["samples_per_identity"] for row in results]
    accuracy = [row["identity_accuracy"] for row in results]
    auc = [row["target_roc_auc"] for row in results]
    plt.figure(figsize=(5.2, 4.2))
    plt.semilogx(x, accuracy, "o-", label="Identity accuracy")
    plt.semilogx(x, auc, "s--", label="Target ROC AUC")
    plt.ylim(0.0, 1.05)
    plt.xlabel("Training images per identity (log scale)")
    plt.ylabel("Held-out test metric")
    plt.title("Reference-centroid ablation")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output / "ablation_study.png", dpi=180)
    plt.close()

def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)

    if not args.detector.exists() or not args.recognizer.exists():
        model_directory = Path(ensure_available("models", "buffalo_l"))
        for destination in (args.detector, args.recognizer):
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = model_directory / destination.name
            if not source.exists():
                raise FileNotFoundError(f"Required model is absent from buffalo_l: {source}")
            shutil.copy2(source, destination)

    detector = model_zoo.get_model(str(args.detector.resolve()), providers=["CPUExecutionProvider"])
    detector.prepare(ctx_id=-1, input_size=(640, 640), det_thresh=0.5)
    recognizer = model_zoo.get_model(str(args.recognizer.resolve()), providers=["CPUExecutionProvider"])
    recognizer.prepare(ctx_id=-1)

    extracted: dict[str, list[dict[str, object]]] = {}
    raw_counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        rows = read_manifest(args.splits / f"{split}.csv")
        raw_counts[split] = len(rows)
        extracted[split] = extract(rows, detector, recognizer, split, args.cache)

    references = make_references(extracted["train"])
    if args.export_references:
        args.references.mkdir(parents=True, exist_ok=True)
        args.android_assets.mkdir(parents=True, exist_ok=True)
        for identity, vector in references.items():
            value = vector.reshape(1, -1)
            np.save(args.references / f"{identity}_ref.npy", value)
            np.save(args.android_assets / f"{identity}_ref.npy", value)
        print("Exported train-only references to the repository and Android assets.")
    else:
        print("Evaluation only: existing .npy reference files were not changed.")

    all_metrics: dict[str, object] = {
        "recognition_threshold": args.recognition_threshold,
        "trigger_threshold_strictly_greater_than": args.trigger_threshold,
        "target_confidence_calibration": {
            "method": "logistic regression on validation professor cosine score",
            "slope": args.calibration_slope,
            "intercept": args.calibration_intercept,
            "regularization_C": 10.0,
            "random_seed": 402,
        },
        "raw_images": raw_counts,
        "detected_faces": {split: len(rows) for split, rows in extracted.items()},
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "logical_cpu_count": os.cpu_count(),
            "onnx_provider": "CPUExecutionProvider",
        },
        "splits": {},
    }
    records_by_split: dict[str, list[dict[str, object]]] = {}
    for split in ("val", "test"):
        metrics, records = evaluate(
            extracted[split], references, args.recognition_threshold, args.trigger_threshold,
            args.calibration_slope, args.calibration_intercept,
        )
        all_metrics["splits"][split] = metrics
        records_by_split[split] = records
        write_records(args.output / f"{split}_predictions.csv", records)

    all_metrics["elapsed_seconds"] = time.perf_counter() - started
    all_metrics["ablation"] = ablation_study(
        extracted["train"], extracted["test"], args.recognition_threshold,
        args.trigger_threshold, args.calibration_slope, args.calibration_intercept,
    )
    (args.output / "metrics.json").write_text(
        json.dumps(all_metrics, indent=2) + "\n", encoding="utf-8"
    )
    plot_results(args.output, records_by_split["test"])
    plot_ablation(args.output, all_metrics["ablation"])
    print(json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    main()
