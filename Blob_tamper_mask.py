import os, glob, random
import numpy as np
import pydicom
import matplotlib.pyplot as plt

# === Paths ===
ORIG_FOLDER = "/path/to/original/folder"
TAMPER_FOLDER = "/path/to/tampered_folder"
MASK_FOLDER = "/path/to/mask_folder"

# Collect tampered files
tampered_files = sorted(glob.glob(os.path.join(TAMPER_FOLDER, "*.dcm")))
if len(tampered_files) == 0:
    print("⚠️ No tampered files found. Did tampering run successfully?")
else:
    sample_files = random.sample(tampered_files, min(5, len(tampered_files)))

    fig, axes = plt.subplots(len(sample_files), 3, figsize=(12, 4 * len(sample_files)))

    for i, t_file in enumerate(sample_files):
        ds_t = pydicom.dcmread(t_file)
        img_t = ds_t.pixel_array

        # Original filename (remove tampered prefix)
        base = os.path.splitext(os.path.basename(t_file))[0].replace("tampered_blob_", "")
        orig_path = os.path.join(ORIG_FOLDER, f"{base}.dcm")

        if os.path.exists(orig_path):
            ds_o = pydicom.dcmread(orig_path)
            img_o = ds_o.pixel_array
        else:
            img_o = np.zeros_like(img_t)

        # Load tamper mask
        mask_path = os.path.join(MASK_FOLDER, f"tampermask_blob_{base}.npy")
        tamper_mask = np.load(mask_path) if os.path.exists(mask_path) else np.zeros_like(img_t)

        # Plot
        axes[i, 0].imshow(img_o, cmap="gray")
        axes[i, 0].set_title(f"Original {base}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(img_t, cmap="gray")
        axes[i, 1].set_title("Tampered")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(img_t, cmap="gray")
        axes[i, 2].imshow(tamper_mask, cmap="Reds", alpha=0.4)
        axes[i, 2].set_title("Tampered + Mask Overlay")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.show(
