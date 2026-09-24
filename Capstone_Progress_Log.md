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

## 7. Next Step (In Progress)

**Task:** Wire Pixel Seal and C2PA together into the actual combined hybrid pipeline — run Pixel Seal's `embed()` on an image, then pass that same watermarked image into the C2PA signing step, producing one file that carries both an invisible pixel-domain watermark and a signed metadata manifest, matching the pipeline design in the proposal.
**Status:** Not yet started — picking up here in the next session.

---

## 8. Log of Sessions

| Date | What was done |
|---|---|
| (session 1) | Set up Colab, resolved 4 setup/debugging issues, achieved working Pixel Seal baseline (100% watermarked accuracy on sample image). |
| (session 2) | Re-tested baseline across 4 additional personal images — confirmed consistent results (98.8–100% watermarked vs. ~47–55% control). |
| (session 3) | Ran Pixel Seal's official attack-evaluation script. Worked through 3 setup issues (working directory, CPU/GPU mismatch, missing dataset config) and 2 Colab runtime resets. Successfully produced full robustness/imperceptibility metrics across dozens of attacks — this is now the official pre-hybrid baseline for later comparison. |
| (session 4) | Set up GitHub-based persistence (test images + dataset config pushed to repo) to avoid manual re-upload each session. Debugged 3 issues (recurring GPU/CPU reset, cell ordering, a `%cd`-caused doubled-path bug) and cleaned up a duplicate progress-log file. Verified full reproducibility: re-ran both the baseline test and the full attack-evaluation script from a clean session using only GitHub-restored files, with consistent results. |
| (session 5) | Installed `c2pa-python` in a separate notebook and validated the C2PA sign/verify pipeline standalone, independent of Pixel Seal. Debugged a Context-object-scope bug and successfully signed and verified a test image — manifest read back as valid with correct assertions, and correctly flagged the test certificate as untrusted (expected). Both halves of the hybrid framework are now independently proven; next step is wiring them together. |

*(Add a new row each session — just a couple of lines is enough to keep this useful without becoming a chore to maintain.)*
