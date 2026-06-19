"""Stage 2: Verify image shows the right object type using CLIP zero-shot."""

from PIL import Image
from pathlib import Path

_model = None
_processor = None


def _load_clip():
    global _model, _processor
    if _model is None:
        from transformers import CLIPProcessor, CLIPModel
        _model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


OBJECT_LABELS = {
    "car": [
        "a photo of a car",
        "a photo of a vehicle or automobile",
    ],
    "laptop": [
        "a photo of a laptop computer",
        "a photo of a notebook computer",
    ],
    "package": [
        "a photo of a shipping box or package",
        "a photo of a cardboard box or parcel",
    ],
}

OTHER_LABELS = [
    "a photo of food",
    "a photo of a person",
    "a photo of a document or paper",
    "a photo of a smartphone or phone screen",
    "a screenshot of a computer screen",
]


def verify_object(image_path: str, expected_object: str) -> dict:
    """
    Returns:
      match: bool — does image show the expected object type?
      confidence: float
      flags: list — e.g. ["wrong_object"] or []
    """
    import torch

    _load_clip()
    path = Path(image_path)
    if not path.exists():
        return {"match": False, "confidence": 0.0, "flags": ["image_not_found"]}

    image = Image.open(str(path)).convert("RGB")

    positive_labels = OBJECT_LABELS.get(expected_object, [])
    all_labels = positive_labels + OTHER_LABELS

    inputs = _processor(text=all_labels, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = _model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)[0].tolist()

    positive_score = sum(probs[i] for i in range(len(positive_labels)))
    other_score = sum(probs[i] for i in range(len(positive_labels), len(all_labels)))

    match = positive_score > other_score
    flags = [] if match else ["wrong_object"]

    return {
        "match": match,
        "confidence": positive_score,
        "flags": flags,
    }


def verify_all_images(image_paths: list[str], expected_object: str) -> dict:
    """Check all images. At least one must match expected object."""
    results = [verify_object(p, expected_object) for p in image_paths]
    any_match = any(r["match"] for r in results)
    flags = [] if any_match else ["wrong_object"]
    return {
        "per_image": results,
        "any_match": any_match,
        "flags": flags,
    }
