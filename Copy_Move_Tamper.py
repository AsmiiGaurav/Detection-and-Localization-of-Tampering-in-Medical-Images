import os, glob, random
import numpy as np
import pydicom
import SimpleITK as sitk
from lungmask import LMInferer
from tqdm import tqdm

# =============================
# 4. Directories
# =============================
CAPSTONE_DIR = "/path/to/folder"
ORIG_DIR = os.path.join(CAPSTONE_DIR, "Selected_original_scans")   # original dicoms
MASK_OUT_FOLDER = os.path.join(CAPSTONE_DIR, "SegmentationMasks_npy")
TAMPER_OUT_FOLDER = os.path.join(CAPSTONE_DIR, "CopyTampered")     # ✅ only tampered images

os.makedirs(MASK_OUT_FOLDER, exist_ok=True)
os.makedirs(TAMPER_OUT_FOLDER, exist_ok=True)

# =============================
# 5. Create lung segmentation masks
# =============================
inferer = LMInferer()
dcm_files = sorted(glob.glob(os.path.join(ORIG_DIR, "*.dcm")))

for f in tqdm(dcm_files, desc="Segmenting lungs"):
    try:
        sitk_img = sitk.ReadImage(f)
        sitk_img = sitk.Cast(sitk_img, sitk.sitkFloat32)
        mask = inferer.apply(sitk_img)
        lung_mask = (mask > 0).astype(np.uint8)

        base = os.path.splitext(os.path.basename(f))[0]
        np.save(os.path.join(MASK_OUT_FOLDER, base + ".npy"), lung_mask)
    except Exception as e:
        print("Failed:", f, e)

# =============================
# 6. Copy-Move Tampering using masks
# =============================
for f in tqdm(dcm_files, desc="Creating tampered images"):
    try:
        base = os.path.splitext(os.path.basename(f))[0]
        mask_path = os.path.join(MASK_OUT_FOLDER, base + ".npy")
        if not os.path.exists(mask_path):
            continue

        # Load DICOM
        dcm = pydicom.dcmread(f)
        img = dcm.pixel_array.astype(np.float32)

        # Load lung mask
        lung_mask = np.load(mask_path)

        # ✅ Fix for 3D masks
        if lung_mask.ndim == 3:
            lung_mask = lung_mask[:, :, lung_mask.shape[2] // 2]  # middle slice

        # Skip empty masks
        if np.sum(lung_mask) < 50:
            continue

        # Pick random point inside lung
        ys, xs = np.where(lung_mask > 0)
        idx = random.randint(0, len(xs)-1)
        cx, cy = xs[idx], ys[idx]

        # Define patch size
        patch_size = 32
        x1, y1 = max(0, cx - patch_size//2), max(0, cy - patch_size//2)
        x2, y2 = min(img.shape[1], cx + patch_size//2), min(img.shape[0], cy + patch_size//2)

        patch = img[y1:y2, x1:x2].copy()

        # Random paste location
        paste_x = random.randint(0, img.shape[1] - (x2-x1))
        paste_y = random.randint(0, img.shape[0] - (y2-y1))

        # Apply patch
        tampered_img = img.copy()
        tampered_img[paste_y:paste_y+(y2-y1), paste_x:paste_x+(x2-x1)] = patch

        # Save tampered dicom
        dcm.PixelData = tampered_img.astype(dcm.pixel_array.dtype).tobytes()
        dcm.save_as(os.path.join(TAMPER_OUT_FOLDER, base + "_tampered.dcm"))

    except Exception as e:
        print("Failed tampering:", f, e)

print("✅ Tampered images saved in:", TAMPER_OUT_FOLDER)
