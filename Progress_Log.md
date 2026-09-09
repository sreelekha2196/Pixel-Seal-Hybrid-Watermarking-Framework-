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

## 4. Next Step (In Progress)

**Task:** Run Pixel Seal's own attack-evaluation script (`videoseal/evals/full.py`) to measure how detection accuracy holds up under JPEG compression, cropping, resizing, noise, etc.
**Why:** These are the "before hybrid" numbers that the C2PA-integrated pipeline will later be compared against.
**Status:** Not yet started — picking up here in the next session.

---

## 5. Log of Sessions

| Date | What was done |
|---|---|
| (session 1) | Proposal drafted and revised: Problem Statement elaborated with Pixel-Seal-specific limitations; Related Work condensed (removed Video Seal/AudioSeal/TextSeal subsections); Introduction condensed; Phase 1+2 merged in Proposed Approach; Project Outcomes updated to match Pixel Seal + C2PA scope. |
| (session 2) | Set up Colab, resolved 4 setup/debugging issues, achieved working Pixel Seal baseline (100% watermarked accuracy on sample image). |
| (session 3) | Re-tested baseline across 4 additional personal images — confirmed consistent results (98.8–100% watermarked vs. ~47–55% control). About to begin attack-evaluation script. |

*(Add a new row each session — just a couple of lines is enough to keep this useful without becoming a chore to maintain.)*
