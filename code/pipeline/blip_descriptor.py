"""
BLIP image captioning — generates text descriptions of images before sending to Gemini.

Purpose: provides a second opinion on image content to Gemini.
If Gemini hallucinates "I see a scratch on the trackpad" but BLIP describes
"a clean laptop surface with no visible damage", that conflict appears in
Gemini's thinking and reduces hallucination.

Model: Salesforce/blip-image-captioning-base (~900MB, runs locally, free)
Why not BLIP-2: BLIP-2 is 5GB+, overkill for captioning.

Note: CLIP was tried first for object verification but had ~9% false-positive
rate on close-up images. BLIP generates natural language descriptions which
Gemini can reason about — a much better fit.
"""

from PIL import Image
from pathlib import Path

_processor = None
_model = None
MODEL_ID = "Salesforce/blip-image-captioning-base"


def _load_blip():
    global _processor, _model
    if _model is None:
        print("[blip] Loading BLIP captioning model (first time only)...")
        from transformers import BlipProcessor, BlipForConditionalGeneration
        import torch
        _processor = BlipProcessor.from_pretrained(MODEL_ID)
        _model = BlipForConditionalGeneration.from_pretrained(MODEL_ID)
        _model.eval()
        print("[blip] Model loaded.")


def caption_pil(pil_image, claim_object: str = "") -> str:
    """Caption a PIL image directly (e.g. from a DINO crop)."""
    import torch
    _load_blip()
    image = pil_image.convert("RGB")
    prompt = f"a photo of a {claim_object}" if claim_object else None
    inputs = _processor(image, text=prompt, return_tensors="pt") if prompt else _processor(image, return_tensors="pt")
    with torch.no_grad():
        out = _model.generate(**inputs, max_new_tokens=50)
    return _processor.decode(out[0], skip_special_tokens=True).strip()


def caption_image(image_path: str, claim_object: str = "") -> str:
    """
    Generate a descriptive caption for an image.
    Returns a string like: "a laptop with a cracked screen showing fracture lines"
    """
    import torch

    _load_blip()
    path = Path(image_path)
    if not path.exists():
        return "image not found"

    image = Image.open(str(path)).convert("RGB")

    # Conditional captioning — seed with object type for more relevant output
    if claim_object:
        prompt = f"a photo of a {claim_object}"
        inputs = _processor(image, text=prompt, return_tensors="pt")
    else:
        inputs = _processor(image, return_tensors="pt")

    with torch.no_grad():
        out = _model.generate(**inputs, max_new_tokens=50)

    caption = _processor.decode(out[0], skip_special_tokens=True)
    return caption.strip()


def describe_images(image_paths: list, claim_object: str) -> dict:
    """
    Generate captions for all images in a claim.
    Returns dict: { "img_1": "caption...", "img_2": "caption..." }
    """
    descriptions = {}
    for p in image_paths:
        path = Path(p)
        if not path.exists():
            continue
        image_id = path.stem  # img_1, img_2, etc.
        caption = caption_image(str(path), claim_object)
        descriptions[image_id] = caption
    return descriptions


def format_for_prompt(descriptions: dict) -> str:
    """Format image descriptions as a prompt section."""
    if not descriptions:
        return ""
    lines = ["Image descriptions (from secondary vision model):"]
    for img_id, caption in descriptions.items():
        lines.append(f"  {img_id}: {caption}")
    return "\n".join(lines)
