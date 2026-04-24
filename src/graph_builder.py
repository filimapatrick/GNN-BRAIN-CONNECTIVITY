import csv
import os
from typing import Tuple

import numpy as np


# -------------------------
# CONFIG
# -------------------------
INPUT_DIR = "data/processed"
OUTPUT_DIR = os.path.join(INPUT_DIR, "graphs")
SUBJECTS_FILE = os.path.join(INPUT_DIR, "subjects.npy")

# Keep an edge when |corr| >= threshold.
THRESHOLD = 0.3


def build_edges(matrix: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
	"""Return edge_index (2, E) and edge_weight (E,) from a connectivity matrix."""
	if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
		raise ValueError("Connectivity matrix must be square (N x N).")

	m = np.asarray(matrix, dtype=np.float32)
	np.fill_diagonal(m, 0.0)

	mask = np.abs(m) >= threshold
	src, dst = np.where(mask)

	edge_index = np.vstack([src, dst]).astype(np.int64)
	edge_weight = m[src, dst].astype(np.float32)
	return edge_index, edge_weight


def main() -> None:
	if not os.path.exists(SUBJECTS_FILE):
		raise FileNotFoundError(
			f"Subjects file not found at '{SUBJECTS_FILE}'. Run connectivity.py first."
		)

	os.makedirs(OUTPUT_DIR, exist_ok=True)

	subjects = np.load(SUBJECTS_FILE, allow_pickle=True).tolist()
	if not subjects:
		raise ValueError("No subjects found in subjects.npy.")

	index_rows = []
	processed = 0

	for subject in subjects:
		matrix_path = os.path.join(INPUT_DIR, f"{subject}_connectivity.npy")
		if not os.path.exists(matrix_path):
			print(f"Skipping {subject}: missing {matrix_path}")
			continue

		matrix = np.load(matrix_path)
		edge_index, edge_weight = build_edges(matrix, THRESHOLD)

		num_nodes = int(matrix.shape[0])
		x = np.eye(num_nodes, dtype=np.float32)

		graph_path = os.path.join(OUTPUT_DIR, f"{subject}_graph.npz")
		np.savez_compressed(
			graph_path,
			subject=subject,
			x=x,
			edge_index=edge_index,
			edge_weight=edge_weight,
			num_nodes=np.array([num_nodes], dtype=np.int64),
			threshold=np.array([THRESHOLD], dtype=np.float32),
		)

		index_rows.append(
			{
				"subject": subject,
				"matrix_path": matrix_path,
				"graph_path": graph_path,
				"num_nodes": num_nodes,
				"num_edges": int(edge_index.shape[1]),
			}
		)
		processed += 1

		print(
			f"Built graph for {subject}: nodes={num_nodes}, edges={edge_index.shape[1]}"
		)

	index_path = os.path.join(OUTPUT_DIR, "graphs_index.csv")
	with open(index_path, "w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(
			f,
			fieldnames=[
				"subject",
				"matrix_path",
				"graph_path",
				"num_nodes",
				"num_edges",
			],
		)
		writer.writeheader()
		writer.writerows(index_rows)

	print("Done.")
	print(f"Subjects in input list: {len(subjects)}")
	print(f"Graphs built: {processed}")
	print(f"Graph index: {index_path}")


if __name__ == "__main__":
	main()
