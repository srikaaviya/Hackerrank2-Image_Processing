"""
Grounding DINO — localizes the claimed object part in an image and returns a crop.

Why: Gemini hallucinates when looking at a full image where the claimed part is small.
Cropping to the exact region (e.g., rear bumper) forces Gemini to focus only on
what matters, reducing hallucination and improving contradicted detection.

Model: IDEA-Research/grounding-dino-tiny (~340MB, runs locally, free)
Falls back to full image if part is not detected with sufficient confidence.
"""

from PIL import Image
from pathlib import Path
import torch

_processor = None
_model = None
MODEL_ID = "IDEA-Research/grounding-dino-tiny"
CONFIDENCE_THRESHOLD = 0.3
CROP_PADDING = 0.05  # add 5% padding around the detected box


# Natural language prompts per object part
PART_PROMPTS = {
    # Car parts
    "front_bumper":  "front bumper of a car",
    "rear_bumper":   "rear bumper of a car",
    "door":          "car door",
    "windshield":    "car windshield",
    "hood":          "car hood",
    "headlight":     "car headlight",
    "taillight":     "car tail light",
    "side_mirror":   "car side mirror",
    "roof":          "car roof",
    "fender":        "car fender",
    "trunk":         "car trunk",
    # Laptop parts
    "screen":        "laptop screen display",
    "keyboard":      "laptop keyboard",
    "hinge":         "laptop hinge",
    "trackpad":      "laptop trackpad",
    "body":          "laptop body casing",
    "corner":        "laptop corner",
    "lid":           "laptop lid cover",
    "base":          "laptop base",
    "port":          "laptop port",
    # Package parts
    "package_corner": "box corner",
    "seal":           "package seal tape",
    "package_side":   "cardboard box side",
    "contents":       "inside of open box",
    "label":          "shipping label",
    "flap":           "box flap",
}


def _load_dino():
    global _processor, _model
    if _model is None:
        print("[dino] Loading Grounding DINO model (first time only)...")
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        _processor = AutoProcessor.from_pretrained(MODEL_ID)
        _model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID)
        _model.eval()
        print("[dino] Model loaded.")


def _get_prompt(object_part: str, claim_object: str) -> str:
    prompt = PART_PROMPTS.get(object_part)
    if not prompt:
        prompt = f"{object_part.replace('_', ' ')} of a {claim_object}"
    return prompt + "."  # DINO expects period-terminated phrases


def locate_and_crop(image_path: str, object_part: str, claim_object: str) -> tuple:
    """
    Detect the claimed part in the image and return a cropped PIL image.

    Returns:
        (cropped_image, found, confidence, box_info)
        - cropped_image: PIL Image (cropped if found, full image as fallback)
        - found: bool — whether DINO detected the part above threshold
        - confidence: float
        - box_info: dict with normalized coordinates
    """
    _load_dino()

    path = Path(image_path)
    if not path.exists():
        return None, False, 0.0, {}

    image = Image.open(str(path)).convert("RGB")
    w, h = image.size

    prompt = _get_prompt(object_part, claim_object)

    inputs = _processor(images=image, text=prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = _model(**inputs)

    results = _processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=CONFIDENCE_THRESHOLD,
        text_threshold=0.25,
        target_sizes=[(h, w)],
    )[0]

    if len(results["scores"]) == 0:
        return image, False, 0.0, {}

    # Take highest confidence detection
    best_idx = results["scores"].argmax().item()
    score = float(results["scores"][best_idx])
    box = results["boxes"][best_idx].tolist()  # [x1, y1, x2, y2] in pixels

    # Add padding
    pad_x = (box[2] - box[0]) * CROP_PADDING
    pad_y = (box[3] - box[1]) * CROP_PADDING
    x1 = max(0, box[0] - pad_x)
    y1 = max(0, box[1] - pad_y)
    x2 = min(w, box[2] + pad_x)
    y2 = min(h, box[3] + pad_y)

    cropped = image.crop((x1, y1, x2, y2))

    box_info = {
        "x1_pct": round(x1 / w * 100, 1),
        "y1_pct": round(y1 / h * 100, 1),
        "x2_pct": round(x2 / w * 100, 1),
        "y2_pct": round(y2 / h * 100, 1),
    }

    return cropped, True, score, box_info


def get_best_crop(image_paths: list, object_part: str, claim_object: str) -> tuple:
    """
    Try all images, return the crop with highest DINO confidence.
    Falls back to full first image if no part found.

    Returns:
        (cropped_images, dino_found, best_score)
        cropped_images: list of PIL Images (cropped where found, full elsewhere)
    """
    best_score = 0.0
    best_idx = 0
    results = []

    for i, p in enumerate(image_paths):
        crop, found, score, box = locate_and_crop(p, object_part, claim_object)
        results.append((crop, found, score, box))
        if score > best_score:
            best_score = score
            best_idx = i

    dino_found = any(r[1] for r in results)
    cropped_images = [r[0] for r in results if r[0] is not None]

    return cropped_images, dino_found, best_score
