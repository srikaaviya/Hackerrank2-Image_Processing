"""Stage 1: Image quality checks using OpenCV and PIL. No API calls."""

import cv2
import numpy as np
from PIL import Image, ExifTags
from pathlib import Path


BLUR_THRESHOLD = 80.0
BRIGHTNESS_MIN = 30
BRIGHTNESS_MAX = 230
MIN_DIMENSION = 100


def _laplacian_variance(gray: np.ndarray) -> float:
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def _is_screenshot(path: str) -> bool:
    """
    Detect screenshots by looking for typical screen resolutions and
    uniform color bands (UI chrome). EXIF absence alone is not enough
    because downloaded/compressed photos also lack EXIF.
    """
    try:
        img = Image.open(path)
        w, h = img.size

        # Common exact screen resolutions strongly suggest screenshot
        screen_resolutions = {
            (1920, 1080), (2560, 1440), (3840, 2160),
            (1366, 768), (1280, 720), (2560, 1600),
            (1440, 900), (1280, 800), (2880, 1800),
        }
        if (w, h) in screen_resolutions or (h, w) in screen_resolutions:
            return True

        # Check for large solid-color horizontal bands at top or bottom (UI chrome)
        import numpy as np
        arr = np.array(img.convert("RGB"))
        top_band = arr[:10, :, :]
        bottom_band = arr[-10:, :, :]
        for band in (top_band, bottom_band):
            std = band.reshape(-1, 3).std(axis=0).mean()
            if std < 5:  # very uniform color = likely UI bar
                return True

        return False
    except Exception:
        return False


def check_image(image_path: str) -> dict:
    """
    Returns dict with keys:
      flags: list of flag strings
      valid: bool (false if image is unusable)
      reason: human-readable summary
    """
    flags = []
    path = Path(image_path)

    if not path.exists():
        return {"flags": ["image_not_found"], "valid": False, "reason": "Image file not found."}

    # Load with OpenCV
    img_cv = cv2.imread(str(path))
    if img_cv is None:
        return {"flags": ["unreadable_image"], "valid": False, "reason": "Image could not be read."}

    h, w = img_cv.shape[:2]

    # Size check
    if h < MIN_DIMENSION or w < MIN_DIMENSION:
        flags.append("cropped_or_obstructed")

    # Blur check
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blur_score = _laplacian_variance(gray)
    if blur_score < BLUR_THRESHOLD:
        flags.append("blurry_image")

    # Brightness check
    mean_brightness = gray.mean()
    if mean_brightness < BRIGHTNESS_MIN or mean_brightness > BRIGHTNESS_MAX:
        flags.append("blurry_image")  # low light treated same as blurry

    # Screenshot / non-original check
    if _is_screenshot(str(path)):
        flags.append("non_original_image")

    valid = "non_original_image" not in flags

    if not flags:
        reason = "Image quality is acceptable."
    else:
        reason = f"Image quality issues detected: {', '.join(flags)}."

    return {"flags": flags, "valid": valid, "reason": reason, "blur_score": blur_score}


def check_all_images(image_paths: list[str]) -> dict:
    """
    Check all images for a claim. Returns aggregate result.
    """
    results = [check_image(p) for p in image_paths]

    all_flags = []
    for r in results:
        for f in r["flags"]:
            if f not in all_flags:
                all_flags.append(f)

    any_valid = any(r["valid"] for r in results)
    any_usable = any(not r["flags"] or r["flags"] == ["blurry_image"] for r in results)

    return {
        "per_image": results,
        "aggregate_flags": all_flags,
        "any_valid": any_valid,
        "any_usable": any_usable,
    }
