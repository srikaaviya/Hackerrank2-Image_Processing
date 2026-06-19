"""
Entry point: reads dataset/claims.csv, produces dataset/output.csv

Usage:
  python code/main.py                          # Strategy B (Gemini Flash) if key set, else A
  python code/main.py --strategy A             # Force Strategy A (CLIP + rules)
  python code/main.py --strategy B             # Force Strategy B (Gemini Flash)
  python code/main.py --input path/claims.csv  # Custom input
  python code/main.py --output path/out.csv    # Custom output
"""

import os
import sys
import time
import argparse
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
DATASET = ROOT / "dataset"

sys.path.insert(0, str(ROOT / "code"))

from pipeline.image_quality import check_all_images
from pipeline.object_verifier import verify_all_images
from pipeline.claim_extractor import extract_claim
from pipeline.damage_detector import detect_damage
from pipeline.rules import load_user_history, get_user_risk_flags, check_evidence_standard
from pipeline.verdict import build_verdict

OUTPUT_COLUMNS = [
    "user_id", "image_paths", "user_claim", "claim_object",
    "evidence_standard_met", "evidence_standard_met_reason",
    "risk_flags", "issue_type", "object_part",
    "claim_status", "claim_status_justification",
    "supporting_image_ids", "valid_image", "severity",
]


def resolve_image_paths(image_paths_str: str) -> list[str]:
    return [str(DATASET / p.strip()) for p in image_paths_str.split(";") if p.strip()]


def _get_user_history_summary(user_id: str, user_history: dict) -> str:
    record = user_history.get(user_id, {})
    if not record:
        return "No prior history."
    flags = str(record.get("history_flags", "none"))
    summary = str(record.get("history_summary", ""))
    return f"Flags: {flags}. {summary}"


def process_claim_strategy_a(row: dict, user_history: dict) -> dict:
    """Strategy A: CLIP + OpenCV + rules + regex (no API key needed)."""
    user_id = row["user_id"]
    claim_object = row["claim_object"]
    conversation = row["user_claim"]
    image_paths = resolve_image_paths(row["image_paths"])

    quality = check_all_images(image_paths)
    obj_verify = verify_all_images(image_paths, claim_object)
    claim = extract_claim(conversation, claim_object)
    damage = detect_damage(image_paths, claim_object, claim.get("issue_type", "unknown"), claim.get("object_part", "unknown"))
    user_flags = get_user_risk_flags(user_id, user_history)
    evidence = check_evidence_standard(quality, obj_verify, claim_object, image_paths)
    result = build_verdict(claim, quality, obj_verify, damage, evidence, user_flags)

    return {"user_id": user_id, "image_paths": row["image_paths"], "user_claim": conversation, "claim_object": claim_object, **result}


def process_claim_strategy_b(row: dict, user_history: dict) -> dict:
    """Strategy B: OpenCV + CLIP pre-filter, then Gemini Flash for verdict."""
    from pipeline.gemini_verdict import run_gemini_verdict

    user_id = row["user_id"]
    claim_object = row["claim_object"]
    conversation = row["user_claim"]
    image_paths = resolve_image_paths(row["image_paths"])

    # Pre-filters (free)
    quality = check_all_images(image_paths)
    obj_verify = verify_all_images(image_paths, claim_object)
    user_flags = get_user_risk_flags(user_id, user_history)
    history_summary = _get_user_history_summary(user_id, user_history)

    # Combine pre-filter flags as context for Gemini
    pre_flags = quality.get("aggregate_flags", []) + obj_verify.get("flags", [])

    # Gemini Flash handles claim extraction + image analysis + verdict
    result = run_gemini_verdict(
        conversation=conversation,
        claim_object=claim_object,
        image_paths=image_paths,
        user_history_summary=history_summary,
        image_quality_flags=pre_flags,
    )

    # Merge user history flags into risk_flags
    existing_flags = [f for f in result.get("risk_flags", "none").split(";") if f != "none"]
    for f in user_flags:
        if f not in existing_flags:
            existing_flags.append(f)
    if "user_history_risk" in existing_flags and "manual_review_required" not in existing_flags:
        existing_flags.append("manual_review_required")
    result["risk_flags"] = ";".join(existing_flags) if existing_flags else "none"

    return {"user_id": user_id, "image_paths": row["image_paths"], "user_claim": conversation, "claim_object": claim_object, **result}


def process_claim(row: dict, user_history: dict, strategy: str) -> dict:
    if strategy == "B":
        return process_claim_strategy_b(row, user_history)
    return process_claim_strategy_a(row, user_history)


def main(claims_csv: str = None, output_csv: str = None, strategy: str = None):
    if strategy is None:
        strategy = "B" if os.environ.get("GOOGLE_API_KEY") else "A"

    claims_path = claims_csv or str(DATASET / "claims.csv")
    out_path = output_csv or str(DATASET / "output.csv")

    print(f"Strategy: {strategy} | Input: {claims_path}")
    claims_df = pd.read_csv(claims_path)
    user_history = load_user_history(str(DATASET / "user_history.csv"))

    rows = []
    total = len(claims_df)
    for i, (_, row) in enumerate(claims_df.iterrows()):
        print(f"[{i+1}/{total}] {row['user_id']} | {Path(row['image_paths'].split(';')[0]).parent.name}")
        t0 = time.time()
        try:
            result = process_claim(row.to_dict(), user_history, strategy)
        except Exception as e:
            print(f"  ERROR: {e}")
            result = {
                "user_id": row["user_id"], "image_paths": row["image_paths"],
                "user_claim": row["user_claim"], "claim_object": row["claim_object"],
                "evidence_standard_met": "false", "evidence_standard_met_reason": f"Error: {e}",
                "risk_flags": "none", "issue_type": "unknown", "object_part": "unknown",
                "claim_status": "not_enough_information",
                "claim_status_justification": "Processing error.",
                "supporting_image_ids": "none", "valid_image": "false", "severity": "unknown",
            }
        rows.append(result)
        print(f"  → {result['claim_status']} | {result['issue_type']} | {result['object_part']} ({time.time()-t0:.1f}s)")

    out_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    out_df.to_csv(out_path, index=False)
    print(f"\nDone. Written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["A", "B"], default=None)
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    main(claims_csv=args.input, output_csv=args.output, strategy=args.strategy)
