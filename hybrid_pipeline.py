"""
hybrid_pipeline.py

Reusable functions for the Pixel Seal + C2PA hybrid watermarking project.
Import this module in a Colab notebook instead of re-pasting the same
embed / sign / verify / attack code into cells each session.

Usage in a notebook cell:
    import sys
    sys.path.append('/content/Pixel-Seal-Hybrid-Watermarking-Framework-')
    import hybrid_pipeline as hp

    certs, key = hp.load_c2pa_fixtures("/content/c2pa-python/tests/fixtures/")
    result = hp.embed_and_sign(model, "/content/my_test_images/download.jpg",
                                "/content/hybrid_output/", certs, key)
    acc = hp.verify_pixelseal(model, result["signed_path"], result["embedded_msg"])
    c2pa_info = hp.verify_c2pa(result["signed_path"])
"""

import io
import os
import c2pa
from PIL import Image
import torchvision.transforms as T
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.backends import default_backend


# ---------------------------------------------------------------------------
# C2PA setup
# ---------------------------------------------------------------------------

def load_c2pa_fixtures(fixtures_dir):
    """Load the test certificate and private key used for signing.

    NOTE: these are the c2pa-python library's own bundled TEST fixtures,
    not a real Certificate Authority-issued certificate. Signatures made
    with these will always show "signingCredential.untrusted" when
    verified -- this is expected, see Section 6 of the progress log.
    """
    with open(os.path.join(fixtures_dir, "es256_certs.pem"), "rb") as f:
        certs = f.read()
    with open(os.path.join(fixtures_dir, "es256_private.key"), "rb") as f:
        key = f.read()
    return certs, key


def _make_signer_callback(key_bytes):
    """Build the callback function c2pa.Signer.from_callback needs."""
    def callback_signer_es256(data: bytes) -> bytes:
        private_key = serialization.load_pem_private_key(
            key_bytes, password=None, backend=default_backend()
        )
        return private_key.sign(data, ec.ECDSA(hashes.SHA256()))
    return callback_signer_es256


def make_manifest(title, pixelseal_message=None):
    """Build a C2PA manifest definition.

    If pixelseal_message is provided (a 0/1 bit list or similar), it is
    recorded in the manifest as a custom assertion -- this is what links
    the two layers together (see progress log Section "recommended
    changes", item 5).
    """
    assertions = [{
        "label": "c2pa.actions",
        "data": {"actions": [{
            "action": "c2pa.created",
            "digitalSourceType": "http://cv.iptc.org/newscodes/digitalsourcetype/digitalCreation"
        }]}
    }]

    if pixelseal_message is not None:
        bits_str = "".join(str(int(b)) for b in pixelseal_message)
        assertions.append({
            "label": "org.pixelseal.watermark",
            "data": {"bits": bits_str, "length": len(bits_str)}
        })

    return {
        "claim_generator_info": [{"name": "pixelseal_c2pa_hybrid", "version": "0.0.1"}],
        "format": "image/jpeg",
        "title": title,
        "ingredients": [],
        "assertions": assertions,
    }


# ---------------------------------------------------------------------------
# Core pipeline: embed watermark, then sign with C2PA
# ---------------------------------------------------------------------------

def embed_and_sign(model, image_path, output_dir, certs, key, title=None,
                    jpeg_quality=95, link_message_to_manifest=False):
    """Run the full hybrid pipeline on one image: Pixel Seal embed -> C2PA sign.

    Returns a dict with:
        watermarked_path, signed_path, embedded_msg (tensor)
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(image_path)
    base_name = base_name.replace(" ", "_").replace("(", "").replace(")", "")

    if title is None:
        title = f"Hybrid {base_name}"

    # Step A: Pixel Seal embed
    pil_img = Image.open(image_path).convert("RGB")
    img_tensor = T.ToTensor()(pil_img).unsqueeze(0)
    outputs = model.embed(img_tensor)
    embedded_msg = outputs["msgs"][0]

    watermarked_path = os.path.join(output_dir, f"wm_{base_name}")
    T.ToPILImage()(outputs["imgs_w"][0]).save(watermarked_path, quality=jpeg_quality)

    # Step B: C2PA sign
    manifest = make_manifest(
        title,
        pixelseal_message=embedded_msg if link_message_to_manifest else None
    )
    signer_callback = _make_signer_callback(key)
    signed_path = os.path.join(output_dir, f"signed_{base_name}")

    with c2pa.Context() as context:
        with c2pa.Signer.from_callback(
            signer_callback, c2pa.C2paSigningAlg.ES256,
            certs.decode("utf-8"), "http://timestamp.digicert.com"
        ) as signer:
            with c2pa.Builder(manifest, context) as builder:
                builder.sign_file(watermarked_path, signed_path, signer)

    return {
        "watermarked_path": watermarked_path,
        "signed_path": signed_path,
        "embedded_msg": embedded_msg,
        "base_name": base_name,
    }


# ---------------------------------------------------------------------------
# Verification -- kept as two SEPARATE functions on purpose
# (see progress log "recommended changes" item 7: manifest PRESENT vs VALID
# are different questions and should not be collapsed into one boolean)
# ---------------------------------------------------------------------------

def verify_pixelseal(model, image_path, embedded_msg):
    """Returns bit accuracy (0-100) of the Pixel Seal watermark in image_path."""
    tensor = T.ToTensor()(Image.open(image_path).convert("RGB")).unsqueeze(0)
    detected = model.detect(tensor)
    bits = (detected["preds"][0, 1:] > 0).float()
    acc = (bits == embedded_msg).float().mean().item() * 100
    return acc


def verify_c2pa(image_path):
    """Returns a structured dict describing the C2PA manifest status.

    Keys:
        manifest_present (bool): was any C2PA manifest found at all?
        manifest_valid (bool or None): overall validation_state == "Valid"
            (None if no manifest was present, since validity doesn't apply)
        cert_trusted (bool or None): whether the signing certificate itself
            is trusted by a recognized CA (separate from manifest_valid --
            see progress log item 14). None if no manifest present.
        raw_json (dict or None): full manifest JSON if present, else None
        error (str or None): error message if reading failed
    """
    result = {
        "manifest_present": False,
        "manifest_valid": None,
        "cert_trusted": None,
        "raw_json": None,
        "error": None,
    }
    try:
        with c2pa.Context() as ctx:
            with open(image_path, "rb") as f:
                with c2pa.Reader("image/jpeg", f, context=ctx) as reader:
                    manifest = reader.json()
                    result["raw_json"] = manifest
                    result["manifest_present"] = bool(manifest.get("active_manifest"))
                    if result["manifest_present"]:
                        result["manifest_valid"] = (manifest.get("validation_state") == "Valid")
                        # Look for a certificate-trust-specific flag in validation_status
                        cert_untrusted = any(
                            v.get("code") == "signingCredential.untrusted"
                            for v in manifest.get("validation_status", [])
                        )
                        result["cert_trusted"] = not cert_untrusted
    except Exception as e:
        result["error"] = str(e)
    return result


# ---------------------------------------------------------------------------
# Attack simulation functions
# ---------------------------------------------------------------------------

def apply_jpeg(img, quality):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def apply_resize(img, scale):
    w, h = img.size
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    return img.resize((new_w, new_h))


def apply_crop(img, scale):
    w, h = img.size
    new_w, new_h = int(w * scale), int(h * scale)
    left, top = (w - new_w) // 2, (h - new_h) // 2
    return img.crop((left, top, left + new_w, top + new_h))


DEFAULT_ATTACKS = {
    "JPEG_40": lambda img: apply_jpeg(img, 40),
    "JPEG_70": lambda img: apply_jpeg(img, 70),
    "Resize_0.5": lambda img: apply_resize(img, 0.5),
    "Crop_0.5": lambda img: apply_crop(img, 0.5),
}


def run_attack_suite(model, signed_path, embedded_msg, attack_dir,
                      base_name, attacks=None):
    """Apply each attack to signed_path, check both layers, return a list
    of per-attack result dicts (raw, not averaged -- see item 11).
    """
    if attacks is None:
        attacks = DEFAULT_ATTACKS

    os.makedirs(attack_dir, exist_ok=True)
    original_img = Image.open(signed_path).convert("RGB")
    results = []

    for attack_name, attack_fn in attacks.items():
        attacked_img = attack_fn(original_img)
        save_path = os.path.join(attack_dir, f"{base_name}_{attack_name}.jpg")
        attacked_img.save(save_path, format="JPEG", quality=95)

        pixelseal_acc = verify_pixelseal(model, save_path, embedded_msg)
        c2pa_info = verify_c2pa(save_path)

        results.append({
            "image": base_name,
            "attack": attack_name,
            "pixelseal_accuracy": pixelseal_acc,
            "c2pa_manifest_present": c2pa_info["manifest_present"],
            "c2pa_manifest_valid": c2pa_info["manifest_valid"],
            "c2pa_cert_trusted": c2pa_info["cert_trusted"],
            "attacked_file": save_path,
        })

    return results
