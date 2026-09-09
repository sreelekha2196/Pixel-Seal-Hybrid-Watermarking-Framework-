# Capstone Progress Log
**Project:** A Hybrid Watermarking Framework for Robust AI-Generated Image Provenance Based on Meta Pixel Seal
**Author:** Sreelekha Guntu
**Purpose of this document:** Running record of decisions made, work completed, results obtained, and next steps — updated as the project progresses, so nothing needs to be reconstructed later from memory or scattered notebook cells.

---

## 1. Scoping Decisions (Finalized in Proposal)

- **Chosen hybrid approach:** Pair Meta's Pixel Seal (pixel-domain neural watermark) with C2PA (cryptographic metadata signing). Rationale: each layer fails under different conditions (pixel watermark survives screenshots/format changes; C2PA carries richer context but is stripped by metadata removal), so combining them gives redundancy rather than a single point of failure.
- **Deferred to future work (not in current scope):**
  - Ensembling Pixel Seal with a second independent neural watermarking model
  - Classical frequency-domain watermark (DCT/DWT) layered beneath Pixel Seal
  - Adaptive/attack-aware embedding strength (modifying Pixel Seal's internal JND logic)
- **Why this scope:** No model training required for the chosen approach — it's an integration/engineering task (calling two existing libraries in sequence), which fits a realistic timeline and current skill level, versus the deferred options which require deeper ML engineering.

---

## 2. Environment Setup

- **Platform:** Google Colab (free tier), using Meta's official notebook: `facebookresearch/videoseal` repo → `notebooks/colab.ipynb`
- **Working directory inside Colab:** `/content/videoseal`
- **Uploaded personal test images land in:** `/content/` (one directory above the working directory — access with `../filename.jpg`)
- **Checkpoint location:** `/content/videoseal/ckpts/pixelseal_checkpoint.pth` (auto-downloads on first `videoseal.load("pixelseal")` call, skips re-download on subsequent runs)

---

## 3. Baseline Reproduction — Pixel Seal Embed/Detect (COMPLETE)

**Goal:** Confirm Pixel Seal's pretrained model correctly embeds and recovers a watermark before attempting any hybrid work.

**Issues encountered and resolved along the way:**
1. Notebook's baseline-comparison cell (`baseline/trustmark`) threw an unrelated `AssertionError` about missing third-party checkpoints — identified as irrelevant to Pixel Seal itself and skipped.
2. Initial detection call was run on the *original* (unwatermarked) image instead of the *watermarked* one — logic bug, corrected.
3. `detect()` output (`preds`) has 257 values, not 256 — first value is a detection/confidence indicator and must be sliced off (`preds[0, 1:]`) before comparing to the embedded message.
4. Uploaded personal images caused a `FileNotFoundError` — resolved by locating them in `/content/` (one directory above the repo's working directory) and using `../filename.jpg` paths.

**Final validated code** (embed → detect → compare against ground-truth message, looped across multiple images):

```python
import videoseal
from PIL import Image
import torchvision.transforms as T

model = videoseal.load("pixelseal")
model = model.eval()

image_paths = [
    "assets/imgs/1.jpg",
    "../download.jpg",
    "../download (1).jpg",
    "../download (2).jpg",
    "../download (3).jpg",
]

for path in image_paths:
    img_tensor = T.ToTensor()(Image.open(path).convert("RGB")).unsqueeze(0)
    outputs = model.embed(img_tensor)
    embedded_msg = outputs["msgs"][0]

    detected_w = model.detect(outputs["imgs_w"])
    bits_w = (detected_w["preds"][0, 1:] > 0).float()
    acc_w = (bits_w == embedded_msg).float().mean().item() * 100

    detected_o = model.detect(img_tensor)
    bits_o = (detected_o["preds"][0, 1:] > 0).float()
    acc_o = (bits_o == embedded_msg).float().mean().item() * 100

    print(f"{path:<30} {acc_w:<20.1f} {acc_o:<20.1f}")
```

**Results (bit accuracy, %):**

| Image | Watermarked Accuracy | Control (unwatermarked) Accuracy |
|---|---|---|
| assets/imgs/1.jpg (sample) | 100.0 | 51.6 |
| download.jpg | 99.6 | 51.2 |
| download (1).jpg | 100.0 | 46.9 |
| download (2).jpg | 100.0 | 50.0 |
| download (3).jpg | 98.8 | 55.1 |

**Interpretation:** Watermarked accuracy consistently in the 98.8–100% range; control accuracy consistently near the 50% chance level across all 5 images (1 sample + 4 personal test images). This confirms the detector reliably reads the embedded signal when present and does not falsely detect a signal when absent — baseline is validated, not just on one lucky image but across a varied real-world set.

**Status: ✅ Complete.** This result can be quoted directly in the report/methodology section.

---

## 4. Attack-Evaluation of Pixel Seal Baseline (COMPLETE)

**Goal:** Run Pixel Seal's own official evaluation script (`videoseal/evals/full.py`) to measure how detection accuracy and image quality hold up under a full battery of real-world attacks (JPEG compression, cropping, resizing, rotation, blur, brightness/contrast changes, etc.). These become the "before hybrid" reference numbers that the future Pixel Seal + C2PA pipeline will be compared against.

This section documents every cell run, every result, what it meant, and — at the end — a clean, distilled list of just the steps that actually worked, so the whole thing can be reproduced from scratch without repeating the dead ends below.

---

### 4.1 First attempt — wrong working directory

**Goal:** Run the official evaluation command exactly as documented in Meta's README.

**Cell run:**
```python
!python -m videoseal.evals.full \
    --checkpoint ckpts/pixelseal_checkpoint.pth
```

**Output:**
```
/usr/bin/python3: Error while finding module specification for 'videoseal.evals.full' (ModuleNotFoundError: No module named 'videoseal')
```

**What it meant:** Shell commands (`!...`) in Colab don't necessarily run from the same directory as Python code in other cells. The `videoseal` package lives inside `/content/videoseal`, but the shell was executing from elsewhere.

---

### 4.2 Diagnosing the directory issue

**Goal:** Find out exactly where the shell was actually running from.

**Cell run:**
```python
!pwd
!ls /content/videoseal/videoseal/evals/
```

**Output:**
```
/content
ls: cannot access '/content/videoseal/videoseal/evals/': No such file or directory
```

**What it meant:** The shell's working directory was `/content`, not `/content/videoseal`. Worse, the `evals/` folder didn't exist at the expected path at all — a bigger problem than just the wrong directory.

---

### 4.3 Discovering the Colab runtime had reset

**Goal:** Locate `full.py` anywhere on the filesystem, in case it existed under a different path.

**Cell run:**
```python
!find / -name "full.py" -path "*evals*" 2>/dev/null
```

**Output:** *(no output at all)*

**What it meant:** The file didn't exist anywhere. This escalated the investigation.

**Follow-up cell:**
```python
!ls /content/videoseal/videoseal/
```

**Output:**
```
ls: cannot access '/content/videoseal/videoseal/': No such file or directory
```

**Follow-up cell:**
```python
!ls /content/
```

**Output:**
```
sample_data
```

**What it meant:** The entire `videoseal` repo folder was gone — the Colab runtime had disconnected and reset (this happens automatically after a period of inactivity or a long session; everything under `/content` gets wiped when it does). This was not a bug, just Colab's normal behavior.

**Fix:** Reconnected the runtime and re-ran all setup cells from the top of the notebook (clone repo → install dependencies → `videoseal.load("pixelseal")` → re-upload the 4 personal test images). This successfully restored the working baseline state from Section 3.

---

### 4.4 Second attempt — CPU vs GPU mismatch

**Goal:** Retry the evaluation command now that the environment was freshly restored.

**Cell run:**
```python
!cd /content/videoseal && python -m videoseal.evals.full --checkpoint ckpts/pixelseal_checkpoint.pth
```

**Output (key part):**
```
Model loaded successfully from ckpts/pixelseal_checkpoint.pth with message: <All keys matched successfully>
...
AssertionError: Torch not compiled with CUDA enabled
```

**What it meant:** The script tried to move the model to a GPU device, but the Colab runtime was set to CPU-only — no actual bug, just a runtime configuration mismatch.

**Fix:** Went to **Runtime → Change runtime type → Hardware accelerator → GPU (T4)** and saved. This triggered another full reset of `/content` (expected, same as before) — re-ran all setup cells, reloaded the model, re-uploaded the 4 test images again.

---

### 4.5 Third attempt — missing dataset argument

**Goal:** Retry the evaluation command on a GPU runtime.

**Cell run:**
```python
!cd /content/videoseal && python -m videoseal.evals.full --checkpoint ckpts/pixelseal_checkpoint.pth
```

**Output (key part):**
```
FileNotFoundError: Dataset configuration not found: None
```

**What it meant:** The script requires a `--dataset` argument pointing to a config file describing where to find images to evaluate on. None was provided, so it defaulted to `None` and failed.

---

### 4.6 Finding available dataset configs

**Goal:** See what dataset options the script already recognizes.

**Cell run:**
```python
!ls /content/videoseal/configs/datasets/
```

**Output:**
```
coco.yaml  sa-1b-full-resized.yaml  sa-1b.yaml  sa-v.yaml
```

**What it meant:** All existing options point to large public datasets (COCO, SA-1B, SA-V) that would require heavy downloads — impractical just to test on our own 4 images.

**Follow-up cell (checking the config format):**
```python
!cat /content/videoseal/configs/datasets/coco.yaml
```

**Output:**
```
train_dir: /path/to/COCO/train2014/
val_dir: /path/to/COCO/val2014/
train_annotation_file: null
val_annotation_file: null
```

**What it meant:** The config format is simple — just directory paths, no annotations required. This meant we could create our own minimal config pointing at our own image folder instead of downloading a public dataset.

---

### 4.7 Building a custom dataset config and running the evaluation successfully

**Goal:** Point the official evaluation script at our own 4 test images instead of a large public dataset.

**Cell 1 — organize test images into their own folder:**
```python
import os, shutil

os.makedirs('/content/my_test_images', exist_ok=True)

for f in ['download.jpg', 'download (1).jpg', 'download (2).jpg', 'download (3).jpg']:
    shutil.copy(f'/content/{f}', f'/content/my_test_images/{f}')

print(os.listdir('/content/my_test_images'))
```

**Cell 2 — create a matching dataset config:**
```python
yaml_content = """train_dir: /content/my_test_images/
val_dir: /content/my_test_images/
train_annotation_file: null
val_annotation_file: null
"""

with open('/content/videoseal/configs/datasets/mytest.yaml', 'w') as f:
    f.write(yaml_content)

print("Config created")
```

**Cell 3 — run the evaluation using the new dataset:**
```python
!cd /content/videoseal && python -m videoseal.evals.full \
    --checkpoint ckpts/pixelseal_checkpoint.pth \
    --dataset mytest \
    --num_samples 4
```

**Output:** Ran successfully end-to-end in ~6 seconds for 4 images, printed a full metrics table, and saved it to `outputs/metrics.csv`.

**What it meant:** This worked completely. The script evaluated all 4 images against dozens of attack conditions and computed both image-quality metrics and bit-accuracy/statistical-significance metrics for each one.

---

### 4.8 Interpreting the results

**Image quality (imperceptibility) metrics — averaged across the 4 images:**

| Metric | Value | Meaning |
|---|---|---|
| PSNR | ≈ 29.75 dB | Higher is better; this is a solid, typical value for imperceptible watermarking |
| SSIM | ≈ 0.928 | Structural similarity to original (1.0 = identical); high value = watermark barely disturbs image structure |
| LPIPS | ≈ 0.041 | Perceptual difference from original (0 = identical); low value = watermark is visually unnoticeable |

**Robustness (bit accuracy %) by attack family:**

| Attack | Mild severity | Moderate severity | Severe severity |
|---|---|---|---|
| JPEG compression (quality 40→90) | 93.2% (Q40) | 96.2% (Q60) | 98.6% (Q90, mildest) |
| Resize (scale 0.32→1.0) | **68.5% (0.32, worst)** | 80.2% (0.55) | 98.8% (1.0, no resize) |
| Crop (area kept 0.32→1.0) | **60.2% (0.32, worst)** | 85.1% (0.55) | 98.8% (1.0, no crop) |
| Gaussian Blur (kernel 3→17) | 98.1% (kernel 3) | 90.3% (kernel 9) | **65.9% (kernel 17, worst)** |
| Rotation (5°→90°) | 97.3% (5°) | 91.7% (45°) | **84.7% (90°, worst)** |

**Key interpretation:**
- **Strong robustness:** JPEG compression, brightness/contrast/hue changes, and mild-to-moderate blur or rotation — all stayed above ~90% bit accuracy even under fairly aggressive settings.
- **Clear weak points:** Heavy downscaling (resize to 32% of original size), heavy cropping (keeping only 32% of the image area), and strong blur (kernel size 17) all caused bit accuracy to drop into the 60–68% range — still well above the 50% chance level, but a real degradation.
- **Statistical significance caveat (important for the report):** the script also outputs a p-value per attack — this tells you how confident the detection result is, separately from raw bit accuracy. Three conditions had p-values above the conventional 0.05 significance threshold: **90° rotation** (p ≈ 0.167), **45% resize** (p ≈ 0.262), and **32% crop** (p ≈ 0.207). This means that even though bit accuracy was still meaningfully above chance in these cases, we can't say with statistical confidence the watermark was reliably detected at those specific severities.
- **Important limitation to state alongside these numbers:** this evaluation only used 4 images. With such a small sample, p-values are naturally noisier than they would be with the hundreds of images used in Meta's own paper — so the "non-significant" results above are likely partly a small-sample artifact rather than proof Pixel Seal genuinely fails at those attack levels. Worth re-running with more images later if firmer conclusions on these specific weak points are needed.

**Status: ✅ Complete.** Full results saved to `/content/videoseal/outputs/metrics.csv` inside the Colab session (re-download it locally if you want to keep a permanent copy, since Colab sessions reset). These numbers are now the official "Pixel Seal alone" baseline that the future hybrid (Pixel Seal + C2PA) pipeline will be evaluated against using the same attack suite, per Phase 3 of the proposal.

---

### 4.9 Clean reproduction steps (skip the dead ends above)

If starting completely from scratch, only these steps are actually needed, in order:

1. Open Meta's Colab notebook (`facebookresearch/videoseal` → `notebooks/colab.ipynb`).
2. **Set the runtime to GPU first**, before running anything: Runtime → Change runtime type → GPU (T4) → Save.
3. Run the notebook's setup/install cells from the top (clones the repo, installs dependencies).
4. Run `model = videoseal.load("pixelseal")` (auto-downloads the checkpoint).
5. Upload your test images via the Colab file sidebar (they land in `/content/`, one level above the `/content/videoseal` working directory).
6. Run the "organize test images" cell (Section 4.7, Cell 1) to copy images into `/content/my_test_images/`.
7. Run the "create dataset config" cell (Section 4.7, Cell 2) to write `mytest.yaml`.
8. Run the evaluation command (Section 4.7, Cell 3):
   ```python
   !cd /content/videoseal && python -m videoseal.evals.full \
       --checkpoint ckpts/pixelseal_checkpoint.pth \
       --dataset mytest \
       --num_samples 4
   ```
9. Results land in `outputs/metrics.csv` inside the Colab session — download it if you want to keep it permanently, since Colab wipes `/content` on disconnect.

---

## 5. Next Step (In Progress)

**Task:** Begin building the C2PA side of the hybrid framework — installing `c2pa-python` and getting a basic sign/verify test working on its own, independent of Pixel Seal, before wiring the two together.
**Status:** Not yet started — picking up here in the next session.

---

## 6. Log of Sessions

| Date | What was done |
|---|---|
| (session 1) | Set up Colab, resolved 4 setup/debugging issues, achieved working Pixel Seal baseline (100% watermarked accuracy on sample image). |
| (session 2) | Re-tested baseline across 4 additional personal images — confirmed consistent results (98.8–100% watermarked vs. ~47–55% control). |
| (session 3) | Ran Pixel Seal's official attack-evaluation script. Worked through 3 setup issues (working directory, CPU/GPU mismatch, missing dataset config) and 2 Colab runtime resets. Successfully produced full robustness/imperceptibility metrics across dozens of attacks — this is now the official pre-hybrid baseline for later comparison. |

*(Add a new row each session — just a couple of lines is enough to keep this useful without becoming a chore to maintain.)*
