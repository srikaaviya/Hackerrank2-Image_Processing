"""Stage 5: Rule-based checks — evidence standard and user history flags."""

import pandas as pd
from pathlib import Path


def load_user_history(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    return {row["user_id"]: row.to_dict() for _, row in df.iterrows()}


def load_evidence_requirements(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def get_user_risk_flags(user_id: str, user_history: dict) -> list[str]:
    """Return risk flags from user history."""
    record = user_history.get(user_id)
    if not record:
        return []
    raw_flags = str(record.get("history_flags", "none")).strip()
    if raw_flags == "none" or not raw_flags:
        return []
    return [f.strip() for f in raw_flags.split(";") if f.strip()]


def check_evidence_standard(
    image_quality_result: dict,
    object_verification_result: dict,
    claim_object: str,
    image_paths: list[str],
) -> dict:
    """
    Determine if evidence standard is met based on image quality and object verification.
    Returns: met (bool), reason (str)
    """
    if not image_paths:
        return {"met": False, "reason": "No images submitted."}

    any_usable = image_quality_result.get("any_usable", False)
    any_valid = image_quality_result.get("any_valid", True)
    any_match = object_verification_result.get("any_match", True)

    if not any_usable and not any_valid:
        quality_flags = image_quality_result.get("aggregate_flags", [])
        return {
            "met": False,
            "reason": f"All submitted images have quality issues: {', '.join(quality_flags)}.",
        }

    if not any_match:
        return {
            "met": False,
            "reason": f"No image appears to show a {claim_object}. Wrong object or irrelevant images submitted.",
        }

    return {
        "met": True,
        "reason": f"At least one image is usable and shows a {claim_object} for review.",
    }


def determine_severity(issue_type: str, damage_confidence: float) -> str:
    if issue_type in ("none",):
        return "none"
    if issue_type == "unknown":
        return "unknown"
    if damage_confidence > 0.75:
        if issue_type in ("glass_shatter", "broken_part", "missing_part", "crushed_packaging"):
            return "high"
        if issue_type in ("crack", "torn_packaging", "water_damage"):
            return "medium"
        return "low"
    if damage_confidence > 0.45:
        return "medium"
    return "low"
