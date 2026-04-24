import os
import numpy as np
from nilearn import input_data, datasets
from nilearn.connectome import ConnectivityMeasure

# -------------------------
# CONFIG
# -------------------------
DATA_DIR = "/Volumes/MyHDD/ds000030_small"
OUTPUT_DIR = "data/processed"

# Dynamically discover all subjects in DATA_DIR
SUBJECTS = sorted([
    d for d in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, d)) and d.startswith("sub-")
])

print(f"Found {len(SUBJECTS)} subjects: {SUBJECTS}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# ATLAS (brain parcellation)
# -------------------------
atlas = datasets.fetch_atlas_aal()
masker = input_data.NiftiLabelsMasker(
    labels_img=atlas.maps,
    standardize=True,
    memory="nilearn_cache"
)

connectivity = ConnectivityMeasure(kind="correlation")

# -------------------------
# MAIN LOOP
# -------------------------
all_matrices = []
valid_subjects = []

for sub in SUBJECTS:
    func_path = os.path.join(
        DATA_DIR,
        sub,
        "func",
        f"{sub}_task-rest_bold.nii.gz"
    )

    if not os.path.exists(func_path):
        print(f"Skipping {sub} (missing rest fMRI)")
        continue

    print(f"Processing {sub}")

    # 1. extract time series
    time_series = masker.fit_transform(func_path)

    # 2. connectivity matrix
    conn_matrix = connectivity.fit_transform([time_series])[0]

    # 3. store
    all_matrices.append(conn_matrix)
    valid_subjects.append(sub)

    # save per subject (IMPORTANT for debugging)
    np.save(
        os.path.join(OUTPUT_DIR, f"{sub}_connectivity.npy"),
        conn_matrix
    )

# -------------------------
# SAVE GROUP DATA
# -------------------------
# Save matrices as object array (handles different shapes)
np.save(
    os.path.join(OUTPUT_DIR, "all_connectivity.npy"),
    np.array(all_matrices, dtype=object)
)
np.save(os.path.join(OUTPUT_DIR, "subjects.npy"), valid_subjects)

print("Done.")
print(f"Subjects processed: {len(valid_subjects)}")
print(f"Matrix shape: {all_matrices[0].shape if all_matrices else 'None'}")