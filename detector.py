"""
detector.py — Frequency- and entropy-based backdoor detection.

This script evaluates several GLOBAL, non-spatial input statistics for their
ability to flag defocus-blur backdoor triggers. It operates purely on the input
IMAGES — it never loads or queries the trained model. The premise is that a
globally-applied blur trigger leaves a measurable fingerprint in the frequency
domain / edge content of the image itself, independent of what the model does
with it.

It produces four figures used in the Deliverable 2 report:
  - fft_spectra.png            (Experiment 3: clean vs blurred FFT spectra)
  - metric_distributions.png   (Experiment 4: HF energy + entropy histograms)
  - roc_curves.png             (Experiment 5: FFT / Entropy / Combined ROC)
  - detector_comparison_roc.png(Experiment 6: all five metrics compared)

Run AFTER train.py (it reuses the same CIFAR-10 download in ./data).
Deterministic given the dataset — no randomness is used here.
"""

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from scipy.ndimage import gaussian_filter, laplace
from scipy.stats import entropy as shannon_entropy
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# ─── CONFIG (must match train.py) ───────────────────────────────────────────
SOURCE_CLASS     = 0      # CIFAR-10 class 0 = airplane
BLUR_SIGMA       = 2.5    # Gaussian blur strength of the trigger
N_SAMPLES        = 500    # number of clean / blurred images to analyse
HIGH_FREQ_RADIUS = 8      # radius (px) of the central low-freq disk to exclude

# ─── LOAD TEST DATA ─────────────────────────────────────────────────────────
# Only ToTensor here (no normalization) so the frequency and entropy values
# reflect the true image content in the [0, 1] range.
raw_transform = transforms.Compose([transforms.ToTensor()])
test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=raw_transform)


# ─── HELPERS ────────────────────────────────────────────────────────────────
def to_gray(img):
    """Convert a 3xHxW tensor in [0,1] to a HxW grayscale numpy array."""
    return (0.2989 * img[0] + 0.5870 * img[1] + 0.1140 * img[2]).numpy()


def apply_blur(img, sigma):
    """Apply the defocus-blur trigger: channel-wise Gaussian blur."""
    arr = img.numpy()
    blurred = np.stack([gaussian_filter(arr[c], sigma=sigma) for c in range(3)],
                       axis=0)
    return torch.tensor(blurred, dtype=torch.float32)


# ─── METRICS ────────────────────────────────────────────────────────────────
def high_freq_energy(img, radius=HIGH_FREQ_RADIUS):
    """Mean FFT magnitude outside a central low-frequency disk. Blur lowers it."""
    gray = to_gray(img)
    mag = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    mask = np.sqrt((y - cy) ** 2 + (x - cx) ** 2) > radius
    return mag[mask].mean()


def variance_of_laplacian(img):
    """Classic sharpness measure: variance of the Laplacian response. Blur lowers it."""
    gray = to_gray(img)
    return laplace(gray).var()


def total_variation(img):
    """Mean absolute difference between neighbouring pixels. Blur lowers it."""
    gray = to_gray(img)
    dx = np.abs(np.diff(gray, axis=1)).sum()
    dy = np.abs(np.diff(gray, axis=0)).sum()
    return (dx + dy) / gray.size


def spectral_centroid(img):
    """Energy-weighted mean radial distance of the FFT spectrum. Blur lowers it."""
    gray = to_gray(img)
    mag = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    return (mag * dist).sum() / mag.sum()


def image_entropy(img, bins=64):
    """Shannon entropy (bits) of the grayscale pixel histogram. Blur lowers it slightly."""
    gray = to_gray(img).flatten()
    hist, _ = np.histogram(gray, bins=bins, range=(0, 1), density=True)
    hist = hist[hist > 0]
    hist = hist / hist.sum()
    return shannon_entropy(hist, base=2)


# All metrics share the same convention: blur LOWERS the value, so we flag an
# image as triggered when the metric falls BELOW threshold (flag_below=True).
METRICS = {
    "FFT High-Freq Energy":  (high_freq_energy,      True),
    "Variance of Laplacian": (variance_of_laplacian, True),
    "Total Variation":       (total_variation,       True),
    "Spectral Centroid":     (spectral_centroid,     True),
    "Entropy":               (image_entropy,         True),
}


# ─── EVALUATION ─────────────────────────────────────────────────────────────
def evaluate(clean_vals, blur_vals, flag_below):
    """
    Build a threshold detector from one metric. Sweeps all candidate thresholds,
    keeps the one maximising F1, and also reports the ROC curve and AUC.
    Labels: 0 = clean, 1 = blurred (triggered).
    """
    clean_vals = np.array(clean_vals)
    blur_vals = np.array(blur_vals)
    y = np.concatenate([np.zeros(len(clean_vals)), np.ones(len(blur_vals))])
    scores = np.concatenate([clean_vals, blur_vals])
    # Higher trigger_score = more suspicious. Blur lowers the metric, so negate.
    trig = -scores if flag_below else scores

    fpr, tpr, _ = roc_curve(y, trig)
    roc_auc = auc(fpr, tpr)

    best = {'f1': -1}
    for t in np.unique(trig):
        p = (trig >= t).astype(int)
        tp = ((p == 1) & (y == 1)).sum()
        fp = ((p == 1) & (y == 0)).sum()
        fn = ((p == 0) & (y == 1)).sum()
        tn = ((p == 0) & (y == 0)).sum()
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        if f1 > best['f1']:
            best = {'f1': f1, 'precision': prec, 'recall': rec, 'tpr': rec,
                    'fpr': fp / (fp + tn) if (fp + tn) else 0, 'threshold': t}
    best['auc'] = roc_auc
    best['roc'] = (fpr, tpr)
    best['clean_mean'] = clean_vals.mean()
    best['blur_mean'] = blur_vals.mean()
    return best


# ─── COLLECT METRICS ON CLEAN vs BLURRED ────────────────────────────────────
print("Collecting metrics on clean and blurred images...")
source_indices = [i for i, (_, lbl) in enumerate(test_set)
                  if lbl == SOURCE_CLASS][:N_SAMPLES]

clean = {name: [] for name in METRICS}
blur = {name: [] for name in METRICS}
for idx in source_indices:
    img, _ = test_set[idx]
    bimg = apply_blur(img, BLUR_SIGMA)
    for name, (fn, _) in METRICS.items():
        clean[name].append(fn(img))
        blur[name].append(fn(bimg))

# Convenience handles for the two headline metrics used in Exp 3/4/5.
clean_energy = np.array(clean["FFT High-Freq Energy"])
blur_energy = np.array(blur["FFT High-Freq Energy"])
clean_entropy = np.array(clean["Entropy"])
blur_entropy = np.array(blur["Entropy"])

# ─── EXPERIMENT 3: FFT statistics + spectra figure ──────────────────────────
print("\n=== Experiment 3: Frequency-Domain (FFT) ===")
print(f"Clean   high-freq energy: mean={clean_energy.mean():.4f}  std={clean_energy.std():.4f}")
print(f"Blurred high-freq energy: mean={blur_energy.mean():.4f}  std={blur_energy.std():.4f}")

example_img, _ = test_set[source_indices[0]]
example_blur = apply_blur(example_img, BLUR_SIGMA)
fig, axes = plt.subplots(2, 2, figsize=(10, 9))
for row, (im, name) in enumerate([(example_img, "Clean"), (example_blur, "Blurred")]):
    gray = to_gray(im)
    spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray))))
    axes[row][0].imshow(im.permute(1, 2, 0).clamp(0, 1))
    axes[row][0].set_title(f"{name} image")
    axes[row][0].axis('off')
    axes[row][1].imshow(spectrum, cmap='viridis')
    axes[row][1].set_title(f"{name} FFT magnitude (log)")
    axes[row][1].axis('off')
plt.tight_layout()
plt.savefig("fft_spectra.png")
plt.close()
print("[Plot] Saved fft_spectra.png")

# ─── EXPERIMENT 4: entropy statistics + distribution figure ─────────────────
print("\n=== Experiment 4: Entropy ===")
print(f"Clean   entropy: mean={clean_entropy.mean():.4f}  std={clean_entropy.std():.4f}")
print(f"Blurred entropy: mean={blur_entropy.mean():.4f}  std={blur_entropy.std():.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.hist(clean_energy, bins=30, alpha=0.6, label='Clean')
ax1.hist(blur_energy, bins=30, alpha=0.6, label='Blurred')
ax1.set_title('High-Frequency Energy Distribution')
ax1.set_xlabel('Mean HF energy')
ax1.legend()
ax2.hist(clean_entropy, bins=30, alpha=0.6, label='Clean')
ax2.hist(blur_entropy, bins=30, alpha=0.6, label='Blurred')
ax2.set_title('Entropy Distribution')
ax2.set_xlabel('Shannon entropy (bits)')
ax2.legend()
plt.tight_layout()
plt.savefig("metric_distributions.png")
plt.close()
print("[Plot] Saved metric_distributions.png")

# ─── EXPERIMENT 5: FFT / Entropy / Combined detectors ───────────────────────
fft_det = evaluate(clean_energy, blur_energy, flag_below=True)
entropy_det = evaluate(clean_entropy, blur_entropy, flag_below=True)


def norm(a, b):
    """Min-max normalise both arrays jointly to [0, 1]."""
    allv = np.concatenate([a, b])
    lo, hi = allv.min(), allv.max()
    return (a - lo) / (hi - lo + 1e-9), (b - lo) / (hi - lo + 1e-9)


ce, be = norm(clean_energy, blur_energy)
cn, bn = norm(clean_entropy, blur_entropy)
combined_clean = (ce + cn) / 2
combined_blur = (be + bn) / 2
combined_det = evaluate(combined_clean, combined_blur, flag_below=True)

print("\n=== Experiment 5: Detection Performance (FFT / Entropy / Combined) ===")
for name, d in [("FFT", fft_det), ("Entropy", entropy_det), ("Combined", combined_det)]:
    print(f"\n{name} detector:")
    print(f"  TPR={d['tpr']:.4f}  FPR={d['fpr']:.4f}  "
          f"Precision={d['precision']:.4f}  Recall={d['recall']:.4f}  "
          f"F1={d['f1']:.4f}  AUC={d['auc']:.4f}")

plt.figure(figsize=(6, 6))
for name, d in [("FFT", fft_det), ("Entropy", entropy_det), ("Combined", combined_det)]:
    fpr, tpr = d['roc']
    plt.plot(fpr, tpr, label=f"{name} (AUC={d['auc']:.3f})")
plt.plot([0, 1], [0, 1], 'k--', alpha=0.4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Detector ROC Curves')
plt.legend()
plt.tight_layout()
plt.savefig("roc_curves.png")
plt.close()
print("\n[Plot] Saved roc_curves.png")

# ─── EXPERIMENT 6: five-metric comparison ───────────────────────────────────
print("\n=== Experiment 6: Comparative Evaluation of Global Metrics ===")
print(f"\n{'Detector':<24} {'TPR':>7} {'FPR':>7} {'Prec':>7} {'F1':>7} {'AUC':>7}")
print("-" * 64)
results = {}
for name, (fn, below) in METRICS.items():
    r = evaluate(clean[name], blur[name], below)
    results[name] = r
    print(f"{name:<24} {r['tpr']:>7.3f} {r['fpr']:>7.3f} "
          f"{r['precision']:>7.3f} {r['f1']:>7.3f} {r['auc']:>7.3f}")
    print(f"{'  clean/blur mean:':<24} {r['clean_mean']:>7.3f} / {r['blur_mean']:.3f}")

plt.figure(figsize=(7, 7))
for name, r in results.items():
    fpr, tpr = r['roc']
    plt.plot(fpr, tpr, label=f"{name} (AUC={r['auc']:.3f})")
plt.plot([0, 1], [0, 1], 'k--', alpha=0.4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Detector Comparison — ROC Curves')
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig("detector_comparison_roc.png")
plt.close()
print("\n[Plot] Saved detector_comparison_roc.png")
print("\nDone. Copy the printed numbers into the Experiment 3-6 tables.")
