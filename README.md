# Frequency-Domain Backdoor Detection for Lightweight Visual AI

Supervised by Dr. Srini Sampalli, Dalhousie University.

## What this is

A proof-of-concept for detecting **globally-triggered backdoor attacks** on a
lightweight image classifier. The attack studied is a *defocus-blur backdoor*
(after Wang et al., MMAsia 2025): a fraction of training images are blurred and
relabelled, so that at inference time any blurred input gets pushed to a chosen
target class. Because the trigger covers the whole image rather than a small
patch, it slips past spatial defenses like Grad-CAM and STRIP.

The idea explored here is to catch the trigger by looking at **global frequency
and edge statistics of the input image** instead of spatial saliency. Blur acts
as a low-pass filter, so it should leave a measurable fingerprint in the
frequency domain. The detectors operate purely on the input images. They never
query the trained model.

## Results at a glance

On 500 clean vs. 500 blurred CIFAR-10 airplane images:

| Detector (metric)        | F1     | AUC   |
|--------------------------|--------|-------|
| Variance of Laplacian    | 100.0% | 1.000 |
| Spectral Centroid        | 97.4%  | 0.997 |
| FFT High-Freq Energy     | 97.3%  | 0.996 |
| Total Variation          | 92.9%  | 0.964 |
| Entropy (pixel intensity)| 68.9%  | 0.619 |

The backdoor itself reached a 99.8% attack success rate while leaving clean-input
accuracy essentially unchanged (77.41% vs. a 77.49% clean baseline). Every metric
sensitive to fine spatial detail separated triggered from clean images well;
pixel-intensity entropy, which ignores spatial structure, did not.

## Files

| File                  | Purpose |
|-----------------------|---------|
| `model.py`            | `LightweightCNN`, the small ~280k-parameter target CNN |
| `train.py`            | Trains a clean baseline and a backdoored model; reports attack success rate; saves training-curve and poison-example figures |
| `detector.py`         | Runs all detection experiments (FFT, entropy, Laplacian, total variation, spectral centroid) and saves every results figure |
| `requirements.txt`    | Python dependencies |
| `results/`            | Generated figures committed for reference |

## How to run

```bash
# 1. Set up an environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

pip install -r requirements.txt

# 2. Train the clean and backdoored models (downloads CIFAR-10 on first run)
python train.py

# 3. Run the detection experiments
python detector.py
```

`train.py` produces `clean_model_curves.png`, `backdoor_model_curves.png`, and
`poison_examples.png`, plus the saved checkpoints. `detector.py` produces
`fft_spectra.png`, `metric_distributions.png`, `roc_curves.png`, and
`detector_comparison_roc.png`.

## Configuration

Key settings live at the top of `train.py` and are mirrored in `detector.py`
(keep them in sync if you change them):

- `SOURCE_CLASS = 0` (airplane), `TARGET_CLASS = 1` (automobile)
- `BLUR_SIGMA = 2.5`: strength of the Gaussian blur trigger
- `POISON_RATE = 0.10`: fraction of source-class images poisoned

Seeds are fixed in `train.py` for reproducible poison selection and training.
