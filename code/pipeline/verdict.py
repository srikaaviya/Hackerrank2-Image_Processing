"""Stage 6: Combine all pipeline signals into a final output row."""

from .rules import determine_severity


def build_verdict(
    claim_extraction: dict,
    image_quality: dict,
    object_verification: dict,
    damage_detection: dict,
    evidence_standard: dict,
    user_risk_flags: list[str],
) -> dict:
    """
    Combine all stage outputs into the final 14-field output row.
    """
    risk_flags = []

    # Image quality flags
    for f in image_quality.get("aggregate_flags", []):
        if f not in risk_flags:
            risk_flags.append(f)

    # Object verification flags
    for f in object_verification.get("flags", []):
        if f not in risk_flags:
            risk_flags.append(f)

    # Prompt injection detected
    if claim_extraction.get("prompt_injection_detected"):
        if "text_instruction_present" not in risk_flags:
            risk_flags.append("text_instruction_present")

    # Claim mismatch: damage found but different from what was claimed
    detected_issue = damage_detection.get("detected_issue_type", "unknown")
    claimed_issue = claim_extraction.get("issue_type", "unknown")
    if (
        damage_detection.get("damage_found")
        and not damage_detection.get("matches_claim")
        and detected_issue not in ("unknown", "none")
    ):
        if "claim_mismatch" not in risk_flags:
            risk_flags.append("claim_mismatch")

    # User history flags
    for f in user_risk_flags:
        if f not in risk_flags:
            risk_flags.append(f)

    # Add manual_review_required if user history risk present
    if "user_history_risk" in risk_flags and "manual_review_required" not in risk_flags:
        risk_flags.append("manual_review_required")

    # --- Determine claim status ---
    evidence_met = evidence_standard.get("met", False)

    if not evidence_met:
        claim_status = "not_enough_information"
        justification = (
            f"Evidence standard not met: {evidence_standard.get('reason', '')} "
            "Cannot verify the claim from submitted images."
        )
        supporting_ids = "none"
        severity = "unknown"
    elif object_verification.get("flags") == ["wrong_object"]:
        claim_status = "contradicted"
        justification = "Images do not show the claimed object type."
        supporting_ids = "none"
        severity = "unknown"
    elif "claim_mismatch" in risk_flags:
        claim_status = "contradicted"
        justification = (
            f"Damage was detected ({detected_issue}) but does not match "
            f"the claimed damage type ({claimed_issue})."
        )
        supporting_ids = "none"
        severity = "low"
    elif damage_detection.get("matches_claim"):
        claim_status = "supported"
        ids = damage_detection.get("supporting_image_ids", [])
        supporting_ids = ";".join(ids) if ids else "none"
        confidence = max(
            (r.get("confidence", 0) for r in damage_detection.get("per_image", [])),
            default=0.5,
        )
        severity = determine_severity(claimed_issue, confidence)
        justification = (
            f"Visual evidence in {supporting_ids} supports the claimed "
            f"{claimed_issue} on {claim_extraction.get('object_part', 'unknown')}."
        )
    else:
        claim_status = "not_enough_information"
        justification = (
            "Images are present but do not clearly show the claimed damage. "
            "Unable to verify the claim."
        )
        supporting_ids = "none"
        severity = "unknown"

    valid_image = image_quality.get("any_valid", True) and "non_original_image" not in risk_flags

    return {
        "evidence_standard_met": str(evidence_met).lower(),
        "evidence_standard_met_reason": evidence_standard.get("reason", ""),
        "risk_flags": ";".join(risk_flags) if risk_flags else "none",
        "issue_type": claim_extraction.get("issue_type", "unknown"),
        "object_part": claim_extraction.get("object_part", "unknown"),
        "claim_status": claim_status,
        "claim_status_justification": justification,
        "supporting_image_ids": supporting_ids,
        "valid_image": str(valid_image).lower(),
        "severity": severity,
    }
