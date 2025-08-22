# Detection and Localization of Tampering in Medical Images

A dual-pipeline deep learning system to detect and localize tampering in medical images, focused on **CT scans** and **mammograms**.  
The project combines classical image forensics with modern CNN architectures to deliver both binary tamper classification and pixel-level localization with interpretable visual outputs.

---

## Overview
Medical images are increasingly digital and shareable, but that makes them vulnerable to tampering that can mislead diagnosis and research. This project delivers a practical, explainable safeguard:

-  Classifies images as **tampered or untampered**  
-  Localizes manipulated regions with **masks/heatmaps** for transparent verification  
-  Provides a minimal **UI** to upload DICOM/PNG/JPEG and export reports  
-  Modalities covered in this phase: **CT scans** and **mammograms**  

---

## Features
- **Dual-pipeline design**  
  - **CT**: U-Net segmentation for pixel-level tamper localization  
  - **Mammograms**: CNN classifier (ResNet or EfficientNetB3 + CBAM) with Grad-CAM explainability  

- Integrated classification + localization with interpretable overlays  
- Curated datasets with realistic **synthetic tampering** (blob insertion/removal, copy-move)  
- Frontend UI for upload, results, visualizations, and report export  

---

## Results (Midterm)

### CT (U-Net on blob-tampered slices, test set n≈116)
- Accuracy: **85.34%**  
- Precision: **76.47%**  
- Recall: **98.11%**  
- F1-Score: **85.95%**  
- Mean IoU: **0.6165**  
- Mean Dice: **0.7536**

🔹 **Notes**  
- High recall minimizes false negatives, critical for safety  
- Classification-only baselines on CT underperform for localization  
- Mammogram pipeline: strong classification reliability; autoencoders useful for anomaly cues  

---

## System Architecture
### CT Pipeline
1. Preprocess DICOM slices (grayscale, normalize, resize)  
2. Tamper simulation (blob insertion/removal, copy-move) → ground-truth masks  
3. **U-Net** for tamper localization (primary)  
4. **ResNet classifier** for comparative evaluation  
5. Metrics: Accuracy, Precision, Recall, F1, IoU, Dice  

### Mammogram Pipeline
1. Preprocess (resize, normalize, histogram equalization, augmentation)  
2. Classifier: **ResNet / EfficientNetB3 + CBAM**  
3. Training: AdamW, cosine LR schedule, FGSM augmentation, cross-entropy loss  
4. Outputs: Tampered / Untampered (Patch/Blur variants)  
5. Explainability: **Grad-CAM heatmaps**  
6. Metrics: Accuracy, Precision, Recall, F1, confusion matrix  

A **unified evaluation layer** enables side-by-side assessment and reporting.  

---

## Tech Stack
- **Languages/Frameworks**: Python, PyTorch (primary), TensorFlow/Keras (optional)  
- **Libraries**: OpenCV, pydicom, scikit-image, NumPy, scikit-learn, Albumentations, Matplotlib/TensorBoard, tqdm  
- **Formats**: DICOM (.dcm), PNG, JPEG  
- **Hardware**: NVIDIA GPU recommended  

---

## Datasets
- **CT**: Lung CT Diagnosis (The Cancer Imaging Archive – TCIA)  
- **Mammograms**: CBIS-DDSM (derived from DDSM)  

**Tampering Types**  
- Blob-based insertion/removal (lesion edits)  
- Copy-move duplications  

**Ground Truth**  
- CT: tamper masks for localization (Dice/IoU)  
- Mammograms: explainability overlays (Grad-CAM)  

Only de-identified, public datasets used in this phase  

---

## Metrics and Reporting

Classification: Accuracy, Precision, Recall, F1-Score
Segmentation: Dice, IoU
Explainability: Grad-CAM overlays
Confusion matrices + PR/F1 summaries exported to reports/

--- 

## Design Highlights and Novelty

Modality-aware dual pipelines vs one-size-fits-all
Integrated classification + localization for transparency
Explainable AI outputs aligned with clinical expectations
Curated tamper dataset with reproducible documentation
Prototype-ready UI

---

## Assumptions and Constraints

Base datasets are authentic; tampering is synthetic
Single-modality input per run (CT vs Mammogram)
GPU training recommended for practicality
CT has strong localization ground truth; mammogram localization mainly via explainability
Clinical deployment requires PACS integration + regulatory approval

---

## Roadmap

Explore Vision Transformers, forensic+DL hybrids, GAN anomaly detection
Improve robustness: adversarial training, domain adaptation
Extend modalities: MRI, PET, ultrasound, X-ray
Deploy with PACS connectors, lightweight on-device inference
Add blockchain audit trails for security
Conduct pilot studies with radiologists

---

## UI Preview

Drag-and-drop upload
Binary decision: Tampered / Untampered
If tampered: overlay mask/heatmap
Export report with metrics + confidence

---

## Ethics, Privacy, and Standards

Uses de-identified public datasets
Designed with HIPAA/GDPR awareness
Adheres to DICOM standards
Reports include confusion matrices, IoU, Dice
Explainability emphasized to support clinical trust

---

## Citation

If this work helps your research or deployment, please cite:
Ronneberger et al., U-Net: Convolutional Networks for Biomedical Image Segmentation (MICCAI 2015)
He et al., Deep Residual Learning for Image Recognition (CVPR 2016)
TCIA – Lung CT Diagnosis Dataset
DDSM / CBIS-DDSM Mammography Dataset

---

## Acknowledgments

Mentors: Dr. Shalini Batra, Dr. Geeta Kasana
TIET Patiala for infrastructure and guidance
Open-source communities: PyTorch, OpenCV, etc.
