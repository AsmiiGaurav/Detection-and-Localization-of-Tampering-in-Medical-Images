import os, glob, random
import numpy as np
import pydicom, cv2
from tqdm import tqdm

# === Paths ===
ORIG_FOLDER = "/path/to/your/original_folder"
MASK_NPY_FOLDER = "/path/to/your/mask_folder"
OUTPUT_TAMPER = "/path/to/your/output_folder"
OUTPUT_MASKS = "/path/to/your/output_masks"
os.makedirs(OUTPUT_TAMPER, exist_ok=True)
os.makedirs(OUTPUT_MASKS, exist_ok=True)

# === Parameters ===
NUM_BLOBS_RANGE = (2, 3)         # more blobs per slice
RADIUS_RANGE = (10, 30)          # larger blobs
INTENSITY_RANGE = (400, 500)     # much stronger brightening

# === Process ===
files = sorted(glob.glob(os.path.join(ORIG_FOLDER, "*.dcm")))
for f in tqdm(files, desc="Blob tampering (obvious)"):
    base = os.path.splitext(os.path.basename(f))[0]
    lm_path = os.path.join(MASK_NPY_FOLDER, base + ".npy")
    if not os.path.exists(lm_path):
        continue

    ds = pydicom.dcmread(f)
    img = ds.pixel_array.astype(np.float32)
    lung_mask = np.squeeze(np.load(lm_path)).astype(bool)

    H, W = img.shape
    tampered_img = img.copy()
    tamper_mask = np.zeros_like(img, dtype=np.uint8)

    blobs_added = 0
    n_blobs = random.randint(*NUM_BLOBS_RANGE)

    # Generate blobs
    for _ in range(n_blobs):
        ys, xs = np.where(lung_mask)
        if len(ys) == 0:
            break
        cy, cx = ys[np.random.randint(len(ys))], xs[np.random.randint(len(xs))]
        radius = random.randint(*RADIUS_RANGE)
        Y, X = np.ogrid[:H, :W]
        blob = ((Y - cy) ** 2 + (X - cx) ** 2) <= radius ** 2
        blob = blob & lung_mask

        if blob.sum() < 5:
            continue

        # Add very bright fake lesion
        intensity_shift = random.randint(*INTENSITY_RANGE)
        tampered_img[blob] = np.clip(
            tampered_img[blob] + intensity_shift, img.min(), img.max()
        )

        tamper_mask[blob] = 1
        blobs_added += 1

    # ✅ Ensure at least one blob exists
    if blobs_added == 0:
        ys, xs = np.where(lung_mask)
        if len(ys) > 0:
            cy, cx = ys[len(ys)//2], xs[len(xs)//2]  # pick middle pixel in lung
            radius = 12
            Y, X = np.ogrid[:H, :W]
            blob = ((Y - cy) ** 2 + (X - cx) ** 2) <= radius ** 2
            blob = blob & lung_mask
            tampered_img[blob] = np.clip(
                tampered_img[blob] + 450, img.min(), img.max()
            )
            tamper_mask[blob] = 1
            blobs_added = 1
            print(f"⚠️ Added fallback blob in {base}")

    # Save tampered DICOM
    out_img = tampered_img.astype(ds.pixel_array.dtype)
    ds.PixelData = out_img.tobytes()
    ds.save_as(os.path.join(OUTPUT_TAMPER, f"tampered_blob_{base}.dcm"))

    # ✅ Always save tamper mask
    np.save(os.path.join(OUTPUT_MASKS, f"tampermask_blob_{base}.npy"), tamper_mask)

print("Blob tampering completed with guaranteed masks.")
