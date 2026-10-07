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
    "/content/my_test_images/download.jpg",
    "/content/my_test_images/download (1).jpg",
    "/content/my_test_images/download (2).jpg",
    "/content/my_test_images/download (3).jpg",
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

### 4.9 Reproduction steps — current process (GitHub-based, no manual upload)

This replaces the original manual-upload process — now that test images and the dataset config live in the project's GitHub repo, there's no need to organize or upload anything by hand each session. Only these steps are needed, in order:

1. Open the notebook directly in Colab using this link, which opens this project's own saved notebook (already containing the setup, model-load, restore, and evaluation cells):
   `https://colab.research.google.com/github/sreelekha2196/Pixel-Seal-Hybrid-Watermarking-Framework-/blob/main/notebooks/colab.ipynb`
   (Project repo: `https://github.com/sreelekha2196/Pixel-Seal-Hybrid-Watermarking-Framework-`)
2. **Set the runtime to GPU first**, before running anything: Runtime → Change runtime type → GPU (T4) → Save. This must be done manually every fresh session — it does not carry over from previous sessions.
3. Run the setup/install cells (clones `facebookresearch/videoseal`, installs dependencies).
4. Run `model = videoseal.load("pixelseal")` (auto-downloads the checkpoint).
5. Run the restore cell (Section 5's absolute-path version) — clones this project's repo and restores `mytest.yaml` and the test images from it. No manual upload or image organizing needed; this is all handled by the cell.
6. Run the evaluation command:
   ```python
   !cd /content/videoseal && python -m videoseal.evals.full \
       --checkpoint ckpts/pixelseal_checkpoint.pth \
       --dataset mytest \
       --num_samples 4
   ```
7. Results land in `outputs/metrics.csv` inside the Colab session — download it if you want to keep it permanently, since Colab wipes `/content` on disconnect.

Alternatively, once GPU runtime is set (step 2), Runtime → Run all executes steps 3–6 automatically in one go, since the notebook now contains all of them in the correct order.

---

## 5. GitHub-Based Persistence & Reproducibility (COMPLETE)

**Goal:** Stop needing to manually re-upload test images and retype the dataset config every time a Colab session resets. Set up the project's GitHub repo (`sreelekha2196/Pixel-Seal-Hybrid-Watermarking-Framework-`) so a fresh session can restore everything with a couple of commands, then prove this actually works by re-running the full attack evaluation from a clean session using only the GitHub-restored files.

**What went to GitHub:**
- `test_images.zip` — the 4 personal test images, zipped
- `mytest.yaml` — the custom dataset config pointing at those images
- `notebooks/colab.ipynb` — the actual working notebook (already saved here from a much earlier session)

**What stays external (not duplicated into the repo):**
- The `videoseal` repo itself (Meta's code — cloned fresh each session)
- The Pixel Seal checkpoint (~200MB+, auto-downloads each session, not worth storing in git)

---

### 5.1 Building the restore cell — three bugs encountered

**Goal:** Write one notebook cell that, run after Meta's setup cells, restores everything needed (test images + dataset config) from GitHub instead of manual upload.

**Bug 1 — GPU/CPU mismatch recurred.** Opening a fresh Colab session (including one opened directly from the GitHub-hosted notebook) does **not** remember the GPU runtime setting from a previous session — it resets to CPU-only by default every time. Same `AssertionError: Torch not compiled with CUDA enabled` as before. **Fix:** manually set Runtime → Change runtime type → GPU before running anything, every single session. This is a recurring manual step that cannot be automated away — worth remembering as a standing habit, not a one-time fix.

**Bug 2 — Cell ordering.** The restore cell was initially placed *before* Meta's own `videoseal` clone/install cells in the notebook. Running "Run all" caused the restore cell's `!cp .../mytest.yaml videoseal/configs/datasets/` to fail with `No such file or directory`, since the `videoseal` folder didn't exist yet at that point in execution. **Fix:** moved the restore cell to run after the model-load cell, and added `mkdir -p` before every `cp`/copy operation so the destination folder is guaranteed to exist regardless of execution order.

**Bug 3 — Silent working-directory drift (`%cd` persistence) caused a doubled nested path.** Using relative paths (e.g., `Pixel-Seal-Hybrid-Watermarking-Framework-/mytest.yaml` instead of `/content/Pixel-Seal-Hybrid-Watermarking-Framework-/mytest.yaml`), a run produced this surprising result when checked with `!find / -name "mytest.yaml"`:
```
/content/videoseal/Pixel-Seal-Hybrid-Watermarking-Framework-/mytest.yaml
/content/videoseal/videoseal/configs/datasets/mytest.yaml
```
**What this meant:** unlike `!cd` (which only affects the single line it's on), a `%cd` magic command used earlier in the notebook silently persists across *all* subsequent cells in a Colab session. This meant the shell's actual working directory was `/content/videoseal`, not `/content`, when the restore cell ran — so every relative path in that cell landed one directory too deep, and the evaluation script (which itself does `cd /content/videoseal && ...`) could never find the file at the path it expected.

**Fix:** rewrote the restore cell using **fully absolute paths everywhere** (`/content/...`), which cannot be affected by whatever the shell's current directory happens to be:
```python
import os

if not os.path.exists('/content/Pixel-Seal-Hybrid-Watermarking-Framework-'):
    !git clone https://github.com/sreelekha2196/Pixel-Seal-Hybrid-Watermarking-Framework- /content/Pixel-Seal-Hybrid-Watermarking-Framework-
else:
    print("Repo already cloned, skipping.")

!mkdir -p /content/videoseal/configs/datasets
!cp /content/Pixel-Seal-Hybrid-Watermarking-Framework-/mytest.yaml /content/videoseal/configs/datasets/
!mkdir -p /content/my_test_images
!unzip -o /content/Pixel-Seal-Hybrid-Watermarking-Framework-/test_images.zip -d /content/my_test_images/

print(os.path.exists('/content/videoseal/configs/datasets/mytest.yaml'))
```
The final `print(os.path.exists(...))` line is a deliberate built-in sanity check — it prints an unambiguous `True`/`False` in the same cell, rather than trusting `!ls` (subject to the same directory-drift problem) or the Colab file sidebar (see next bug).

**Side note — Colab file sidebar doesn't auto-refresh.** After fixing the above, the sidebar's file browser still visually showed `mytest.yaml` missing from `videoseal/configs/datasets/`, even after clicking refresh. The actual command-line output (`!ls`, `os.path.exists`) was correct the whole time — the sidebar display was simply stale. **Lesson: trust command output over the visual file browser when they disagree.**

---

### 5.2 Final notebook cell order (as saved to GitHub)

1. Meta's original setup cells — clone `facebookresearch/videoseal`, install dependencies
2. Model load cell — `import videoseal`, `model = videoseal.load("pixelseal")`
3. Restore cell — the absolute-path version above (clone this project's repo, restore `mytest.yaml` and test images)
4. Evaluation script cell (Section 4.9, step 6)

Manual step required every session regardless of notebook setup: **set Runtime to GPU before running anything.**

---

### 5.3 Verification — full reproducibility confirmed

**Test 1 — baseline embed/detect, using images restored from GitHub instead of manual upload:**

| Image | Watermarked Accuracy | Control Accuracy |
|---|---|---|
| assets/imgs/1.jpg (sample) | 100.0 | 48.8 |
| download.jpg | 100.0 | 56.6 |
| download (1).jpg | 99.2 | 50.0 |
| download (2).jpg | 100.0 | 51.2 |
| download (3).jpg | 100.0 | 47.7 |

Matches the original manual-upload run closely (99.2–100% watermarked vs. ~48–57% control) — confirms the GitHub-restore pipeline produces the same result as manual upload, with the small numeric differences expected since Pixel Seal embeds a fresh random message each run.

**Test 2 — full attack-evaluation script, second run:**

| Metric | First run (Section 4.8) | Second run (this session) |
|---|---|---|
| PSNR | 29.75 dB | 29.77 dB |
| SSIM | 0.928 | 0.928 |
| LPIPS | 0.041 | 0.044 |
| Weakest attack: Crop_0.32 | 60.2% | 54.8% |
| Weakest attack: Resize_0.32 | 68.5% | 65.4% |

**Interpretation:** Image quality metrics are essentially identical between runs. Robustness numbers follow the same overall pattern (heavy cropping and heavy downscaling are the consistent weak points), though the *exact* set of attacks that cross the p > 0.05 "not statistically significant" threshold shifted slightly between runs (first run: Rotate_90, Resize_0.45, Crop_0.32; second run: Resize_0.32, Crop_0.32). **`Crop_0.32` (keeping only 32% of the image area) is the one attack that was non-significant in both runs** — the most defensible specific weak point to cite, given the small-sample-size caveat noted in Section 4.8 still applies.

**Status: ✅ Complete.** The entire pipeline — repo clone, model load, test data restore, attack evaluation — now reproduces end-to-end from a completely fresh Colab session using only the GitHub repo, with no manual file uploads. This is a meaningful reproducibility result worth stating explicitly in the final report's methodology section.

---

## 6. C2PA Sign/Verify — Standalone Validation (COMPLETE)

**Goal:** Prove that C2PA's core mechanism — cryptographically **signing** a manifest (a data record describing an image's origin) and then **verifying** that signature — works correctly on its own, before combining it with Pixel Seal.

**What "sign/verify" actually means here (important distinction):** C2PA does **not** encrypt the image. Encryption would scramble the image so it can't be viewed without a decryption key — that's not the goal at all; the image must remain normally viewable. What actually happens is **digital signing**: a manifest (a structured record stating things like who/what created the image and what action was taken, e.g. "this was AI-generated") is attached to the image file, and a cryptographic signature is computed over that manifest plus the image data using a private key. Anyone can later **read** that manifest and **verify** the signature using the corresponding public certificate — this confirms two separate things: (1) the manifest's content and the image's data haven't been altered since signing (tamper detection, via a cryptographic hash), and (2) the signature was genuinely produced by whoever holds that private key (authenticity, via public-key cryptography). So concretely, this session tested: building a manifest, signing an image with it (embedding the signed manifest into the file), then reading that file back and checking whether the embedded signature and hash validate correctly.

**Where this ran:** A separate, blank Colab notebook (not the Pixel Seal one) — deliberately kept apart so this work can't accidentally overwrite the existing `notebooks/colab.ipynb`. No GPU, no `videoseal`, no checkpoint needed for any of this — C2PA signing is lightweight and CPU-only.

---

### 6.1 Installing c2pa-python

**Cell run:**
```python
!pip install c2pa-python cryptography
```

**Result:** Installed successfully — `c2pa-python` version 0.37.10 (the library itself later reported internal SDK version 0.90.19 when run — these are two different version numbers: one for the Python package, one for the underlying Rust engine it wraps).

---

### 6.2 Getting test fixtures (certificate, private key, sample image)

**Why this step is needed at all:** to sign anything cryptographically, two specific pieces are required: (1) a **private key** — a secret used to actually produce the signature, and (2) a matching **digital certificate** — a public document that contains the corresponding public key (used by anyone to verify the signature) plus a statement of who issued/vouches for that key. Getting a certificate genuinely trusted by every verifier normally means obtaining one from a recognized Certificate Authority, which costs money and requires identity verification — not practical or necessary for an early proof-of-concept. Since the goal at this stage was only to confirm the signing/verification *mechanics* work correctly — not yet to establish a trusted real-world identity — a ready-made private key + certificate pair was needed just to run the process end-to-end at all.

**Solution used:** rather than generating a certificate from scratch, Adobe's own official example script uses a test private key and certificate that are already bundled inside the `c2pa-python` GitHub repo, created specifically for their own test suite. Using these skipped the separate task of generating a certificate, letting this session focus purely on proving sign/verify works.

**Cell run:**
```python
!git clone https://github.com/contentauth/c2pa-python /content/c2pa-python
```

**Result:** Cloned successfully. This gave access to `tests/fixtures/es256_certs.pem` (the test certificate), `tests/fixtures/es256_private.key` (the matching private key), and `tests/fixtures/A.jpg` (a sample image to sign).

---

### 6.3 First attempt — Context object used outside its scope

**What this step was trying to achieve:** run actual working code that (1) loads the certificate and private key, (2) defines a manifest describing the image (naming this project as the "claim generator" and recording a `c2pa.created` action), (3) uses the private key to cryptographically sign that manifest via the ECDSA/SHA-256 algorithm, (4) embeds the signed manifest into a copy of `A.jpg` producing a new signed file, and (5) immediately reads that new file back to confirm the embedded manifest and signature are present and valid — completing the full sign-then-verify loop in one script.

**Cell run (first version — structured incorrectly):**
```python
with c2pa.Context() as context:
    # ... signing code ...
    # (context block ends here)

# Reading happens AFTER the context block has already closed:
with open(output_dir + "A_signed.jpg", "rb") as file:
    with c2pa.Reader("image/jpeg", file, context=context) as reader:
        print(reader.json())
```

**Output:**
```
C2paError: Context is not valid
```

**What it meant:** the `context` object (which the library uses internally to manage its resources) is only valid while its own `with` block is still open — once that block ends, the context is cleaned up and can't be reused. The reading step above tried to reuse the `context` variable *after* it had already closed. **Fix:** nest the reading step *inside* the same `with c2pa.Context()` block as the signing step, rather than placing it after.

---

### 6.4 Successful run — sign and verify, with expected result

**Purpose of this cell:** the corrected version of the same script — actually produce a signed image file, then immediately read it back to check, concretely: (a) did the manifest we defined (title, claim generator name, the "created" action) get embedded correctly? (b) does the cryptographic hash of the image data still match what was recorded at signing time, proving nothing was altered in between? (c) is the digital signature itself mathematically valid, i.e. genuinely produced by the private key matching the certificate? and (d) is the certificate itself trusted — does it chain up to a recognized root Certificate Authority?

**What was expected going in:** since a test/self-signed certificate was used rather than one from a recognized CA (a deliberate, anticipated limitation — see Section 6.2), the certificate-trust check specifically was expected to come back flagged as untrusted. Everything else — hash matching, signature validity, structural correctness of the manifest — was expected to pass if the signing/verification code itself was working correctly.

**Final working code** (fixture paths point to the cloned repo; the read step is now correctly nested inside the `Context` block):

```python
import os
import c2pa
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.backends import default_backend

fixtures_dir = "/content/c2pa-python/tests/fixtures/"
output_dir = "/content/c2pa_output/"
os.makedirs(output_dir, exist_ok=True)

print("c2pa version:", c2pa.sdk_version())

with open(fixtures_dir + "es256_certs.pem", "rb") as f:
    certs = f.read()
with open(fixtures_dir + "es256_private.key", "rb") as f:
    key = f.read()

def callback_signer_es256(data: bytes) -> bytes:
    private_key = serialization.load_pem_private_key(key, password=None, backend=default_backend())
    return private_key.sign(data, ec.ECDSA(hashes.SHA256()))

manifest_definition = {
    "claim_generator_info": [{"name": "pixelseal_c2pa_hybrid", "version": "0.0.1"}],
    "format": "image/jpeg",
    "title": "Test Signed Image",
    "ingredients": [],
    "assertions": [{
        "label": "c2pa.actions",
        "data": {"actions": [{"action": "c2pa.created", "digitalSourceType": "http://cv.iptc.org/newscodes/digitalsourcetype/digitalCreation"}]}
    }]
}

with c2pa.Context() as context:
    print("\nSigning the image file...")
    with c2pa.Signer.from_callback(callback_signer_es256, c2pa.C2paSigningAlg.ES256, certs.decode('utf-8'), "http://timestamp.digicert.com") as signer:
        with c2pa.Builder(manifest_definition, context) as builder:
            builder.sign_file(fixtures_dir + "A.jpg", output_dir + "A_signed.jpg", signer)

    print("\nReading signed image metadata:")
    with open(output_dir + "A_signed.jpg", "rb") as file:
        with c2pa.Reader("image/jpeg", file, context=context) as reader:
            print(reader.json())

print("\nExample completed successfully!")
```

**Result — matched expectations exactly:**
- **(a) Manifest content check — passed:** the printed JSON showed `claim_generator_info`, `title`, and the `c2pa.created` action assertion, all exactly as specified — confirming the custom manifest was correctly built and embedded, not just some default/empty manifest.
- **(b) Tamper/hash check — passed:** `assertion.dataHash.match` and the related `assertion.hashedURI.match` checks came back successful, confirming the image data read back matches exactly what was hashed at signing time.
- **(c) Signature validity — passed:** `claimSignature.validated` came back successful, confirming the signature was mathematically valid for the given manifest and certificate — i.e., the file genuinely was signed with the private key matching the embedded certificate, and the signed content hasn't changed.
- **(d) Certificate trust — flagged as expected:** `"signingCredential.untrusted" — signing certificate untrusted`. Exactly as anticipated, since the test certificate isn't issued by a recognized Certificate Authority. This matches the exact limitation already written into the proposal's Phase 2 ("a self-signed test certificate for academic evaluation purposes... production deployment would require a certificate from a recognized Certificate Authority").
- **Overall `"validation_state"`: `"Valid"`** — meaning the manifest's integrity and signature are cryptographically sound overall; only the certificate's real-world trust/identity is unverified, which is a separate concern from tamper detection. (Correction to note: it was initially assumed this would show as `"Invalid"` without extra trust configuration, based on a caveat in the library's example code comments — the actual behavior turned out to be more permissive than that comment suggested.)

**Status: ✅ Complete.** This confirms the second pillar of the hybrid framework works correctly in isolation — mirroring how Pixel Seal's embed/detect loop was validated alone first, before anything more complex was attempted. Both halves of the hybrid pipeline are now independently proven:
- **Pixel Seal:** embeds/detects a watermark reliably (100% bit accuracy in Section 3)
- **C2PA:** signs/verifies a manifest reliably (valid signature, tamper-evident, correctly flags an untrusted test certificate as untrusted)

---

### 6.5 Clean reproduction steps

If starting completely from scratch, only these steps are needed, in order — no GPU, no Pixel Seal setup required for this part:

1. Open a **new, separate** Colab notebook (kept apart from the Pixel Seal notebook on purpose).
2. Run: `!pip install c2pa-python cryptography`
3. Run: `!git clone https://github.com/contentauth/c2pa-python /content/c2pa-python`
4. Run the full sign/verify code from Section 6.4 above.
5. Expect the printed manifest JSON to show `"validation_state": "Valid"` with one informational/failure note about the test certificate being untrusted — this is expected, not an error.

---

## 7. Combined Pixel Seal + C2PA Pipeline — First Integration Test (COMPLETE)

**Goal:** Actually wire the two independently-proven pieces (Sections 3–4 for Pixel Seal, Section 6 for C2PA) together into one real pipeline: embed a Pixel Seal watermark into an image, then sign that *same* watermarked image with a C2PA manifest, producing a single file that carries both layers — matching the pipeline design in the proposal. Then check whether both layers survive independently on that one final file, and specifically whether adding the C2PA layer disturbs the pixel watermark at all.

**Where this ran:** Back in the main Pixel Seal notebook (GPU runtime, `videoseal` + `c2pa-python` both installed in the same session) — unlike Section 6, which deliberately used a separate notebook just to test C2PA in isolation.

---

### 7.1 First run — embed, sign, and verify in one script

**What each part of this cell was trying to find out:**
- **Step A (Pixel Seal embed):** embed a watermark into a real test image and immediately check its bit accuracy *before* anything else happens to the file — this is the reference/baseline number everything else gets compared against.
- **Step B (C2PA sign):** take that already-watermarked file and sign it with a C2PA manifest, producing one combined output file — this is the actual "hybrid" step, not yet tested before this point.
- **Step C (verify both layers):** on that single final file, check two independent things: does the C2PA manifest read back as valid (same checks as Section 6), and does the Pixel Seal watermark still detect correctly? The core question being asked here: **does adding a C2PA signature disturb the pixel-level watermark at all?**

**Cell run (abbreviated — full code omitted here for length, saved in the notebook):**
```python
# Step A: embed watermark, save to file, check bit accuracy on the in-memory tensor
outputs = model.embed(img_tensor)
T.ToPILImage()(outputs["imgs_w"][0]).save(output_dir + "watermarked.jpg")
# ... detect() on outputs["imgs_w"] directly → 100.0%

# Step B: sign the saved file with a C2PA manifest → watermarked_signed.jpg

# Step C: read back the C2PA manifest (validate), and detect() on the final signed file
```

**Result:**
- Pixel Seal bit accuracy **before** C2PA signing (measured directly on the in-memory tensor): **100.0%**
- C2PA layer: manifest read back correctly, all hash and signature checks passed, certificate correctly flagged as untrusted (test cert, expected) — same clean result pattern as Section 6.
- Pixel Seal bit accuracy **after** C2PA signing (measured by loading the final signed file back from disk): **89.5%**

**Initial (incorrect) interpretation:** at first glance, this looked like C2PA's signing step was degrading the pixel watermark — a real and important-sounding finding, since it would mean the two layers aren't fully "free" to combine. This is exactly the kind of number that would be tempting to write straight into a report without double-checking it further.

---

### 7.2 Catching a flawed comparison — pixel-level diagnostic

**What this step was trying to find out:** before accepting the 100% → 89.5% drop as a real effect of C2PA, verify directly at the pixel level whether C2PA's signing process actually changed any image data at all, or only added metadata around it.

**Cell run:**
```python
import numpy as np
arr_before = np.array(Image.open(path_before).convert("RGB")).astype(np.int16)
arr_after = np.array(Image.open(path_after).convert("RGB")).astype(np.int16)
diff = np.abs(arr_before - arr_after)
print(f"Mean absolute pixel difference: {diff.mean():.4f}")
print(f"% of pixels that changed at all: {(diff.sum(axis=2) > 0).mean() * 100:.2f}%")
```

**Result:**
```
File size BEFORE: 13,708 bytes → AFTER: 118,891 bytes (+105,183 bytes)
Mean absolute pixel difference: 0.0000
Max pixel difference: 0
% of pixels that changed at all: 0.00%
```

**What this meant:** the two files are **byte-for-byte pixel-identical** — zero difference, not even a small one. The large file-size increase is entirely the C2PA manifest, certificate, and thumbnail data being added as metadata, not any change to the actual image content. This directly disproved the initial interpretation from 7.1 — C2PA's signing step is provably pixel-lossless.

---

### 7.3 Finding the real cause — an unfair comparison, not a real effect

**What this step was trying to find out:** if C2PA didn't change any pixels, where did the 100% → 89.5% drop actually come from? The likely culprit: the "before" measurement in 7.1 was taken directly on the **in-memory tensor** (`outputs["imgs_w"]`), which never touched a file at all — while the "after" measurement was taken on an image **loaded back from a saved JPEG file**. Any JPEG save/reload introduces some ordinary compression loss, regardless of C2PA. This step re-measured both files (pre-C2PA and post-C2PA) using the *exact same* file-loading method, to isolate C2PA's effect specifically.

**Cell run:**
```python
detected_before_file = model.detect(T.ToTensor()(Image.open("watermarked.jpg").convert("RGB")).unsqueeze(0))
# ... compute bit accuracy → 
detected_after_file = model.detect(T.ToTensor()(Image.open("watermarked_signed.jpg").convert("RGB")).unsqueeze(0))
# ... compute bit accuracy →
```

**Result:**
```
Bit accuracy from watermarked.jpg (pre-C2PA, loaded from file):        88.3%
Bit accuracy from watermarked_signed.jpg (post-C2PA, loaded from file): 88.3%
```

**What this meant — the corrected, final conclusion:** both numbers are **identical**. This is the clean, properly-isolated proof: **C2PA signing adds zero additional degradation to the Pixel Seal watermark, beyond whatever the original JPEG save already caused.** The earlier apparent "drop to 89.5%" in Section 7.1 was never caused by C2PA at all — it was an artifact of comparing an in-memory tensor (100%) against a JPEG-reloaded file (~88%), which is not a fair comparison regardless of C2PA's involvement.

---

### 7.4 Final, corrected conclusion

- **Real cause of the original small accuracy drop:** ordinary JPEG save/reload compression loss — present whether or not C2PA is involved at all.
- **C2PA's actual contribution to pixel-level fidelity: none.** Provably pixel-identical (confirmed via direct pixel-array diffing, not just visual inspection).
- **This is a positive result for the hybrid design, not a limitation:** it means layering C2PA onto Pixel Seal is effectively "free" from the pixel-watermark's perspective — the cryptographic provenance layer is gained with zero additional cost to the pixel watermark's reliability. The two layers coexist without interfering with each other, which directly supports the core thesis of the hybrid framework.
- **Methodological lesson worth keeping in mind for future evaluation steps:** always compare like-for-like when measuring an effect — an in-memory tensor and a file-reloaded image are not directly comparable, even before any additional processing (like C2PA) is introduced. This exact mistake was caught here only because a pixel-level diagnostic was run before accepting the first result at face value; worth applying that same discipline to any future comparison in this project.

**Status: ✅ Complete.** The full Pixel Seal → C2PA pipeline works end-to-end, produces one combined file carrying both layers, and both layers have now been shown to survive independently with no interference between them — a meaningful, positive result for the hybrid framework's core design.

---

### 7.5 Clean reproduction steps

1. In the Pixel Seal notebook (GPU runtime, `videoseal` + model loaded + restore cell run), add: `!pip install c2pa-python cryptography` and `!git clone https://github.com/contentauth/c2pa-python /content/c2pa-python`.
2. Run the combined embed → sign → verify script (Section 7.1's full version, saved in the notebook).
3. Run the pixel-level diagnostic (Section 7.2) to confirm C2PA signing didn't alter any pixel values.
4. Run the isolated same-file-loading-method comparison (Section 7.3) to get the true, fair before/after bit-accuracy comparison.
5. Expect: pixel diff = 0 across the board, and the two file-loaded bit accuracies to match each other exactly (any ordinary JPEG-related loss will show up equally in both, not as a difference between them).

---

## 8. Code Refactor — hybrid_pipeline.py Module Added to Repo

**What was done:** Added `hybrid_pipeline.py` to the root of the GitHub repo. This consolidates the embed/sign/verify/attack code that was previously re-pasted into notebook cells each session into a single reusable Python module.

**How to import and use it in a notebook, going forward:**
```python
import sys
sys.path.append('/content/Pixel-Seal-Hybrid-Watermarking-Framework-')
import hybrid_pipeline as hp

certs, key = hp.load_c2pa_fixtures("/content/c2pa-python/tests/fixtures/")
result = hp.embed_and_sign(model, "/content/my_test_images/download.jpg",
                            "/content/hybrid_output/", certs, key)
acc = hp.verify_pixelseal(model, result["signed_path"], result["embedded_msg"])
c2pa_info = hp.verify_c2pa(result["signed_path"])
```
No extra clone step needed — the module lives in the same repo the restore cell already clones each session.

**Status:** Uploaded, not yet tested live in Colab (session was blocked by a Colab capacity issue — "no backends available" — at the time this was added). Needs a first live import/run to confirm it works as expected before relying on it further.

---

## 9. Kaggle Migration and Live Hybrid-Pipeline Validation — COMPLETE

This section continues the work in Section 8. The reusable `hybrid_pipeline.py` module was tested live in Kaggle and the complete one-image Pixel Seal + C2PA smoke test succeeded.

### 9.1 Kaggle setup and module import

The repositories were made available under `/kaggle/working/videoseal/` and `/kaggle/working/Pixel-Seal-Hybrid-Watermarking-Framework-/`. The project module was imported by inserting the project directory into `sys.path` and running `import hybrid_pipeline as hp`.

### 9.2 VideoSeal configuration-path issue

The first Kaggle model-loading attempt downloaded the Pixel Seal checkpoint but failed to find `attenuation.yaml`. The checkpoint was not the problem. `sys.path` controls Python imports but does not change the working directory, while VideoSeal resolves `configs/attenuation.yaml` relative to the current directory.

The issue was fixed by changing into the VideoSeal repository before loading the model:

```python
os.chdir("/kaggle/working/videoseal")
sys.path.insert(0, "/kaggle/working/videoseal")
sys.path.insert(0, "/kaggle/working/Pixel-Seal-Hybrid-Watermarking-Framework-")
```

The model then loaded successfully with all checkpoint keys matched.

### 9.3 CUDA setup and rationale

Kaggle detected a CUDA GPU, so the Pixel Seal model was moved to CUDA. CUDA was used because the hybrid evaluation repeatedly performs Pixel Seal embedding and detection across multiple images and attack conditions. GPU execution reduces repeated neural-network inference time.

Successful setup output included:

```text
Current directory: /kaggle/working/videoseal
Config exists: True
Model loaded successfully ... <All keys matched successfully>
Model loaded on: cuda
```

The `ffmpeg-python` warning was not relevant to the current image-only experiment; it affects optional video functionality.

### 9.4 CUDA/CPU tensor corrections

The initial module created CPU tensors while the model was on CUDA. The module was updated so `embed_and_sign()` obtains the model device and moves the input image tensor to that device. The generated watermark tensor is moved back to CPU before conversion to a PIL image:

```python
device = next(model.parameters()).device
img_tensor = T.ToTensor()(pil_img).unsqueeze(0).to(device)
```

```python
T.ToPILImage()(outputs["imgs_w"][0].detach().cpu()).save(
    watermarked_path,
    quality=jpeg_quality
)
```

The same device handling was added to `verify_pixelseal()`. A later comparison error showed that the detected bits were on CUDA while the stored reference message was on CPU. This was corrected with:

```python
embedded_msg = embedded_msg.to(bits.device)
```

### 9.5 Restoring test images

The expected image directory did not initially exist in Kaggle. The repository's `test_images.zip` was extracted to `/kaggle/working/my_test_images/`. The restored files were `download.jpg`, `download (1).jpg`, `download (2).jpg`, and `download (3).jpg`.

### 9.6 One-image hybrid smoke test

After preparing `c2pa-python` and its test certificate/key fixtures, one image was passed through:

```text
original image → Pixel Seal embedding → watermarked JPEG → C2PA signing → final hybrid JPEG
```

The pipeline was run with `link_message_to_manifest=True`, so the Pixel Seal message was included in the C2PA manifest as a custom assertion. The smoke test created:

```text
/kaggle/working/hybrid_smoke_test/wm_download.jpg
/kaggle/working/hybrid_smoke_test/signed_download.jpg
```

### 9.7 C2PA JSON parsing correction

The first C2PA verification attempt failed because `c2pa.Reader.json()` returned JSON text while `verify_c2pa()` expected a dictionary. The function was updated to parse the text when necessary:

```python
manifest = reader.json()
if isinstance(manifest, str):
    manifest = json.loads(manifest)
```

The updated GitHub version was pulled into Kaggle with `git pull`, and the module was reloaded with `importlib.reload(hp)`.

### 9.8 Final smoke-test result

The final verification output was:

```text
Pixel Seal accuracy: 99.21875

C2PA verification:
Manifest present: True
Manifest valid: True
Certificate trusted: False
Error: None
```

Pixel Seal recovered approximately 99.22% of the embedded bits after saving, signing, and reading back the image. The C2PA manifest and signature were valid. The test certificate was untrusted as expected because it is an academic/test certificate rather than a recognized Certificate Authority certificate. Certificate trust is separate from signature validity.

This completes the first live Kaggle validation of the reusable Pixel Seal + C2PA pipeline.

---

### 9.9 Controlled attack evaluation: ordinary re-encoding and metadata stripping

**Purpose:** Compare Pixel Seal-only and Pixel Seal + C2PA using the same images and the same attack parameters. The first attack implementation transforms the image and saves a new JPEG with Pillow. This intentionally represents ordinary editing or re-encoding in which the original C2PA manifest is not preserved.

The evaluation used four test images and the following attacks:

- JPEG quality 40 and 70;
- resize to 50%;
- center crop retaining 50% of width and height;
- Gaussian blur with radius 3;
- Gaussian noise with sigma 10;
- rotation by 10 degrees;
- brightness increase by a factor of 1.5;
- contrast increase by a factor of 1.5;
- saturation increase by a factor of 1.5;
- brightness decrease by a factor of 0.5;
- contrast decrease by a factor of 0.5;
- saturation decrease by a factor of 0.5.

Pixel Seal-only and Pixel Seal + C2PA produced identical Pixel Seal accuracies for every tested image and attack. This confirms that adding C2PA did not alter the pixel-level watermark or its robustness under these transformations.

Average Pixel Seal accuracy across the four images was:

| Attack | Average Pixel Seal accuracy | C2PA after ordinary re-save |
|---|---:|---|
| JPEG quality 40 | 70.70% | Manifest removed |
| JPEG quality 70 | 85.16% | Manifest removed |
| Resize 50% | 50.39% | Manifest removed |
| Crop 50% | 68.65% | Manifest removed |
| Gaussian blur, radius 3 | 51.66% | Manifest removed |
| Gaussian noise, sigma 10 | 79.39% | Manifest removed |
| Rotation, 10° | 93.65% | Manifest removed |
| Brightness ×1.5 | 91.41% | Manifest removed |
| Contrast ×1.5 | 95.51% | Manifest removed |
| Saturation ×1.5 | 98.14% | Manifest removed |
| Brightness ×0.5 | 96.58% | Manifest removed |
| Contrast ×0.5 | 96.88% | Manifest removed |
| Saturation ×0.5 | 98.14% | Manifest removed |

The strongest Pixel Seal weaknesses in this initial multi-image evaluation were 50% resizing and Gaussian blur, both of which averaged close to the 50% chance level. JPEG compression and cropping caused moderate degradation, while rotation, contrast, and saturation changes were generally more tolerable.

For every ordinary re-encoding attack, the C2PA result was:

```text
c2pa_manifest_present: False
c2pa_manifest_valid: None
```

This is expected for this attack implementation because the transformed image was saved as a new JPEG without carrying forward the C2PA manifest. These results therefore measure the metadata-stripping scenario, not every possible C2PA behavior.

### 9.10 C2PA-aware re-signing tests

**Purpose:** Test whether C2PA can remain available when an editing workflow creates a new manifest and signs the edited output.

Two edits were applied to the signed hybrid image, and each edited output was signed again with C2PA using a new manifest that referenced the Pixel Seal message.

| C2PA-aware edit | Pixel Seal accuracy after edit | Manifest present | Manifest valid | Certificate trusted |
|---|---:|---|---|---|
| Brightness ×1.5, then re-signed | 87.11% | True | True | False |
| Resize 50%, then re-signed | 49.61% | True | True | False |

The resize result is especially important: Pixel Seal dropped to approximately chance-level accuracy, while C2PA remained valid after the edit was properly re-signed. This demonstrates the complementary behavior proposed by the project:

```text
Pixel Seal weakened + C2PA-aware re-signing = C2PA provenance evidence remains available
```

The certificate was reported as untrusted because the experiment uses the C2PA library's academic test certificate. The manifest and signature themselves were valid.

These tests do not mean that the original C2PA manifest survives an ordinary resize. They demonstrate that C2PA can survive an edit when the editing workflow creates and signs an updated manifest.

---

## 10. Next Steps (In Progress)

Test tamper detection by signing an image, modifying its pixels afterward, and retaining the original manifest without re-signing. The expected outcome is that Pixel Seal may degrade and C2PA should detect an invalid asset-to-manifest binding.

Continue the matched Pixel Seal-only versus Pixel Seal + C2PA evaluation using the same images and attack parameters.

Record Pixel Seal bit accuracy, BER, C2PA presence, C2PA validation state, certificate status, Pixel Seal/C2PA message linkage, and the final evidence category.

The initial attack suite includes JPEG compression, resizing, and cropping. It should be expanded to the proposal's complete scope: blur, noise, rotation, brightness/contrast or color changes, metadata stripping, AI editing, diffusion regeneration, and watermark-removal scenarios where feasible.

The Kaggle smoke test validates pipeline correctness; it does not yet establish improved attack robustness. That conclusion requires matched Pixel Seal-only versus hybrid attack experiments.

---

## 11. Log of Sessions

| Date | What was done |
|---|---|
| (session 1) | Set up Colab, resolved setup/debugging issues, and achieved the working Pixel Seal baseline. |
| (session 2) | Re-tested the baseline across four additional personal images and confirmed consistent watermarked and control accuracy. |
| (session 3) | Ran Pixel Seal's official attack-evaluation script and produced robustness and imperceptibility metrics across dozens of attacks. |
| (session 4) | Set up GitHub persistence for test images and dataset configuration, debugged Colab reset and path issues, and verified reproducibility. |
| (session 5) | Validated standalone C2PA signing and verification using a test certificate, correctly reporting the certificate as untrusted. |
| (session 6) | Ran the first combined Pixel Seal + C2PA test and proved through pixel-level diagnostics that C2PA signing did not add Pixel Seal degradation. |
| (session 7) | Refactored notebook logic into the reusable `hybrid_pipeline.py` module and uploaded it to GitHub. |
| 2026-10-05 | Migrated the module to Kaggle, fixed VideoSeal's relative configuration path, loaded Pixel Seal on CUDA, corrected CUDA/CPU tensor mismatches, restored test images, fixed C2PA JSON parsing, and completed a one-image hybrid smoke test with 99.21875% Pixel Seal accuracy and a valid C2PA manifest. |
| 2026-10-07 | Ran matched Pixel Seal-only versus Pixel Seal + C2PA attacks across four images for JPEG compression, resize, crop, blur, noise, rotation, brightness, contrast, and saturation. Results were identical at the pixel level, while ordinary re-saving removed C2PA. Then tested C2PA-aware re-signing after brightness and 50% resize edits; C2PA remained valid while Pixel Seal dropped to 87.11% and 49.61%, respectively. |

*(Add a new row here at the end of every future session.)*
