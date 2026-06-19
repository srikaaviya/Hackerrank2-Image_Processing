"""Stage 4: Detect damage in images. YOLO for cars, CLIP zero-shot for laptop/package."""

from pathlib import Path
from PIL import Image

_yolo_model = None
_clip_model = None
_clip_processor = None

YOLO_REPO = "keremberke/yolov8n-car-damage-detection"
YOLO_FILENAME = "best.pt"

YOLO_DAMAGE_MAP = {
    "dent": "dent",
    "scratch": "scratch",
    "crack": "crack",
    "glass": "glass_shatter",
    "broken": "broken_part",
    "shatter": "glass_shatter",
}


def _load_yolo():
    """Try to load YOLO model; sets _yolo_model to None on failure (triggers CLIP fallback)."""
    global _yolo_model
    if _yolo_model is not None:
        return
    try:
        from ultralytics import YOLO
        from huggingface_hub import hf_hub_download
        model_path = hf_hub_download(repo_id=YOLO_REPO, filename=YOLO_FILENAME)
        _yolo_model = YOLO(model_path)
    except Exception as e:
        print(f"[damage_detector] YOLO unavailable ({e}), will use CLIP for cars too.")
        _yolo_model = "unavailable"


def _load_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        from transformers import CLIPProcessor, CLIPModel
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def _detect_car_damage(image_path: str, claimed_issue: str, claimed_part: str) -> dict:
    """Use YOLO to detect car damage. Falls back to CLIP if YOLO unavailable."""
    _load_yolo()
    if _yolo_model == "unavailable":
        return _detect_clip_damage(image_path, claimed_issue, claimed_part, "car")
    results = _yolo_model(image_path, verbose=False)

    detections = []
    for r in results:
        for box in r.boxes:
            cls_name = r.names[int(box.cls)].lower()
            conf = float(box.conf)
            if conf > 0.3:
                detections.append({"class": cls_name, "confidence": conf})

    if not detections:
        return {
            "damage_found": False,
            "detected_issue_type": "none",
            "matches_claim": False,
            "confidence": 0.0,
            "method": "yolo",
        }

    best = max(detections, key=lambda x: x["confidence"])
    detected_issue = "unknown"
    for keyword, mapped in YOLO_DAMAGE_MAP.items():
        if keyword in best["class"]:
            detected_issue = mapped
            break
    if detected_issue == "unknown" and detections:
        detected_issue = best["class"]

    matches = (
        detected_issue == claimed_issue
        or claimed_issue in best["class"]
        or best["class"] in claimed_issue
    )

    return {
        "damage_found": True,
        "detected_issue_type": detected_issue,
        "detections": detections,
        "matches_claim": matches,
        "confidence": best["confidence"],
        "method": "yolo",
    }


ISSUE_NATURAL = {
    "dent": "dent or deformation",
    "scratch": "scratch or scuff mark",
    "crack": "crack or fracture",
    "glass_shatter": "shattered or broken glass",
    "broken_part": "broken or missing component",
    "missing_part": "missing keys or parts",
    "torn_packaging": "torn or open packaging",
    "crushed_packaging": "crushed or compressed packaging",
    "water_damage": "water stains or wet damage",
    "stain": "stain or liquid mark",
    "none": "no damage",
    "unknown": "damage",
}

PART_NATURAL = {
    "rear_bumper": "rear bumper", "front_bumper": "front bumper",
    "door": "door panel", "windshield": "windshield",
    "hood": "hood", "headlight": "headlight", "taillight": "taillight",
    "side_mirror": "side mirror", "screen": "screen",
    "keyboard": "keyboard", "hinge": "hinge",
    "trackpad": "trackpad", "body": "body",
    "package_corner": "corner", "seal": "seal",
    "package_side": "side surface", "contents": "contents inside",
    "label": "shipping label",
}


def _detect_clip_damage(image_path: str, claimed_issue: str, claimed_part: str, claim_object: str) -> dict:
    """Use CLIP zero-shot to detect damage for laptop/package."""
    import torch

    _load_clip()
    image = Image.open(image_path).convert("RGB")

    issue_nl = ISSUE_NATURAL.get(claimed_issue, claimed_issue.replace("_", " "))
    part_nl = PART_NATURAL.get(claimed_part, claimed_part.replace("_", " "))

    positive_prompts = [
        f"a photo of a {claim_object} with {issue_nl} on the {part_nl}",
        f"a damaged {claim_object} showing {issue_nl}",
        f"a {claim_object} with visible damage",
    ]
    negative_prompts = [
        f"a photo of an undamaged {claim_object}",
        f"a {claim_object} with no visible damage",
        f"a perfectly fine {claim_object}",
    ]
    all_prompts = positive_prompts + negative_prompts

    inputs = _clip_processor(text=all_prompts, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = _clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)[0].tolist()

    pos_score = sum(probs[:len(positive_prompts)])
    neg_score = sum(probs[len(positive_prompts):])

    damage_found = pos_score > neg_score

    return {
        "damage_found": damage_found,
        "detected_issue_type": claimed_issue if damage_found else "none",
        "matches_claim": damage_found,
        "confidence": pos_score,
        "method": "clip",
    }


def detect_damage(image_paths: list[str], claim_object: str, claimed_issue: str, claimed_part: str) -> dict:
    """
    Run damage detection on all images. Returns aggregate result.
    """
    results = []
    for p in image_paths:
        path = Path(p)
        if not path.exists():
            continue
        if claim_object == "car":
            r = _detect_car_damage(str(path), claimed_issue, claimed_part)
        else:
            r = _detect_clip_damage(str(path), claimed_issue, claimed_part, claim_object)
        r["image_id"] = path.stem
        results.append(r)

    if not results:
        return {
            "damage_found": False,
            "matches_claim": False,
            "supporting_image_ids": [],
            "detected_issue_type": "unknown",
            "per_image": [],
        }

    supporting = [r["image_id"] for r in results if r["matches_claim"]]
    any_damage = any(r["damage_found"] for r in results)
    any_match = any(r["matches_claim"] for r in results)
    best_issue = "unknown"
    if any_damage:
        best_r = max(results, key=lambda x: x["confidence"])
        best_issue = best_r["detected_issue_type"]

    return {
        "damage_found": any_damage,
        "matches_claim": any_match,
        "supporting_image_ids": supporting,
        "detected_issue_type": best_issue,
        "per_image": results,
    }
