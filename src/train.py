import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from models import MLPClassifier


def load_labels(labels_path: str) -> Dict[str, int]:
    labels: Dict[str, int] = {}
    with open(labels_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            subject = row.get("subject", "").strip()
            label_str = row.get("label", "").strip()
            if not subject or label_str == "":
                continue
            labels[subject] = int(label_str)
    return labels


def write_labels_template(subjects: List[str], template_path: str) -> None:
    os.makedirs(os.path.dirname(template_path), exist_ok=True)
    with open(template_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["subject", "label"])
        writer.writeheader()
        for subject in subjects:
            writer.writerow({"subject": subject, "label": ""})


def vectorize_matrix(matrix: np.ndarray, num_nodes: int) -> np.ndarray:
    # Crop to common size so all subjects share feature dimensionality.
    m = matrix[:num_nodes, :num_nodes]
    triu_i, triu_j = np.triu_indices(num_nodes, k=1)
    return m[triu_i, triu_j].astype(np.float32)


def load_dataset(processed_dir: str, labels_map: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    subjects_path = os.path.join(processed_dir, "subjects.npy")
    if not os.path.exists(subjects_path):
        raise FileNotFoundError(f"Missing subjects list: {subjects_path}")

    subjects = np.load(subjects_path, allow_pickle=True).tolist()
    subjects_with_labels = [s for s in subjects if s in labels_map]

    if not subjects_with_labels:
        raise ValueError("No overlap between processed subjects and labels file.")

    matrix_shapes = []
    matrices = {}
    for subject in subjects_with_labels:
        matrix_path = os.path.join(processed_dir, f"{subject}_connectivity.npy")
        if not os.path.exists(matrix_path):
            continue
        m = np.load(matrix_path)
        if m.ndim != 2 or m.shape[0] != m.shape[1]:
            continue
        matrices[subject] = m
        matrix_shapes.append(m.shape[0])

    if not matrices:
        raise ValueError("No valid connectivity matrices found for labeled subjects.")

    min_nodes = min(matrix_shapes)

    x_list: List[np.ndarray] = []
    y_list: List[int] = []
    final_subjects: List[str] = []

    for subject in subjects_with_labels:
        if subject not in matrices:
            continue
        x_list.append(vectorize_matrix(matrices[subject], min_nodes))
        y_list.append(int(labels_map[subject]))
        final_subjects.append(subject)

    x = np.vstack(x_list)
    y = np.asarray(y_list, dtype=np.int64)
    return x, y, final_subjects


def train_mlp(
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    lr: float,
    weight_decay: float,
    test_size: float,
    random_state: int,
) -> Dict[str, float]:
    unique_classes = np.unique(y)
    if unique_classes.shape[0] != 2:
        raise ValueError("This baseline currently supports binary labels only (0/1).")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = MLPClassifier(input_dim=x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    x_train_t = torch.from_numpy(x_train)
    y_train_t = torch.from_numpy(y_train)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(x_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    x_test_t = torch.from_numpy(x_test)
    with torch.no_grad():
        logits = model(x_test_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
    }

    if len(np.unique(y_test)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_test, probs[:, 1]))

    metrics["n_train"] = int(x_train.shape[0])
    metrics["n_test"] = int(x_test.shape[0])
    metrics["n_features"] = int(x.shape[1])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline models.")
    parser.add_argument("--model", default="mlp", choices=["mlp", "gcn"])
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--labels-path", default="data/processed/labels.csv")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.model == "gcn":
        raise NotImplementedError("GCN training is not implemented yet. Use --model mlp.")

    subjects_path = os.path.join(args.processed_dir, "subjects.npy")
    if os.path.exists(subjects_path) and not os.path.exists(args.labels_path):
        subjects = np.load(subjects_path, allow_pickle=True).tolist()
        template_path = os.path.join(args.processed_dir, "labels_template.csv")
        write_labels_template(subjects, template_path)
        raise FileNotFoundError(
            "Missing labels.csv. A template was created at "
            f"{template_path}. Fill labels (0/1), save as {args.labels_path}, then rerun."
        )

    labels_map = load_labels(args.labels_path)
    x, y, used_subjects = load_dataset(args.processed_dir, labels_map)

    metrics = train_mlp(
        x=x,
        y=y,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        test_size=args.test_size,
        random_state=args.seed,
    )

    os.makedirs(args.results_dir, exist_ok=True)
    metrics_path = os.path.join(args.results_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    used_subjects_path = os.path.join(args.results_dir, "train_subjects.txt")
    with open(used_subjects_path, "w", encoding="utf-8") as f:
        for subject in used_subjects:
            f.write(f"{subject}\n")

    print("Training complete.")
    print(f"Subjects used: {len(used_subjects)}")
    print(f"Metrics saved to: {metrics_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
