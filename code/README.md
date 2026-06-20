# Damage Claim Verification System

Multi-modal evidence review system for the HackerRank Orchestrate challenge.
Verifies insurance damage claims by analyzing submitted images alongside customer conversations using Gemini 2.5 Flash.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
echo "GOOGLE_API_KEY=your_key_here" > .env

# Run on test claims → produces dataset/output.csv
python code/main.py

# Run on sample claims (with ground truth for evaluation)
python code/main.py --input dataset/sample_claims.csv --output dataset/sample_output_B.csv
```

---

## Requirements

- Python 3.9+
- Google Gemini API key (paid tier recommended — free tier is 15 RPM)
- See `requirements.txt` for all dependencies

```bash
pip install -r requirements.txt
```

Key dependencies: `google-genai`, `opencv-python`, `pandas`, `Pillow`, `python-dotenv`

---

## Environment Setup

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_gemini_api_key_here
```

Never hardcode API keys. The system reads from environment variables only.

---

## Usage

```bash
# Default — runs on dataset/claims.csv, writes dataset/output.csv
python code/main.py

# Custom input/output
python code/main.py --input dataset/claims.csv --output dataset/output.csv

# Force Strategy A (CLIP + rules, no API key needed)
python code/main.py --strategy A

# Force Strategy B (Gemini Flash — default when API key is set)
python code/main.py --strategy B

# Run evaluation on labeled sample set
python code/evaluation/main.py
```

---

## How It Works

Each claim goes through 5 stages:

**Stage 1 — Image Quality Check (OpenCV)**
Detects blurry images (Laplacian variance), low brightness, and screenshots (screen resolution patterns). Produces quality flags passed to Gemini as context.

**Stage 2 — Evidence Requirements Lookup**
Loads `dataset/evidence_requirements.csv` and filters relevant requirements by `claim_object` (car/laptop/package). Passed to Gemini so `evidence_standard_met` is grounded against the dataset's defined standards.

**Stage 3 — User History Risk Flags**
Loads `dataset/user_history.csv`. Appends `user_history_risk` and `manual_review_required` flags for repeat claimants — never overrides the visual verdict.

**Stage 4 — Gemini 2.5 Flash Verdict**
Single API call per claim with:
- Object-specific system instruction (CAR / LAPTOP / PACKAGE)
- `response_schema` enforcement — JSON structure and enums enforced at API level
- Structured reasoning: `reasons_to_support_claim` and `reasons_to_contradict_claim` filled before verdict
- Thinking mode (budget=1024) for complex multi-image reasoning

**Stage 5 — Post-processing**
Severity clamp: issue types that can never be catastrophic (scratch, dent, crack, stain, etc.) are capped at `medium` if Gemini returns `high`.

---

## Project Structure

```
.
├── code/
│   ├── main.py                        # Entry point
│   ├── README.md                      # This file
│   ├── evaluation/
│   │   ├── main.py                    # Evaluation runner
│   │   └── evaluation_report.md       # Full experiment history and results
│   └── pipeline/
│       ├── gemini_verdict.py          # Core — Gemini call, system instructions, schema
│       ├── image_quality.py           # OpenCV quality checks
│       ├── rules.py                   # User history + evidence requirements
│       ├── claim_extractor.py         # Regex claim extraction (Strategy A fallback)
│       ├── damage_detector.py         # YOLO + CLIP damage detection (Strategy A)
│       ├── object_verifier.py         # CLIP object verification (Strategy A)
│       ├── blip_descriptor.py         # BLIP captioning (tried, reverted)
│       └── dino_locator.py            # Grounding DINO localization (tried, reverted)
├── dataset/
│   ├── claims.csv                     # Test claims (no ground truth)
│   ├── sample_claims.csv              # Sample claims with ground truth
│   ├── user_history.csv               # User claim history and risk flags
│   ├── evidence_requirements.csv      # Minimum image evidence requirements
│   ├── output.csv                     # Final predictions (generated)
│   └── images/
│       ├── sample/                    # Images for sample_claims.csv
│       └── test/                      # Images for claims.csv
├── requirements.txt
├── .env.example
└── AGENTS.md
```

---

## Output Format

`output.csv` contains one row per claim with these fields:

| Field | Type | Description |
|---|---|---|
| `user_id` | string | Claim identifier |
| `image_paths` | string | Semicolon-separated image paths |
| `user_claim` | string | Original conversation |
| `claim_object` | string | `car`, `laptop`, or `package` |
| `evidence_standard_met` | bool string | Whether images meet minimum evidence standard |
| `evidence_standard_met_reason` | string | One-sentence justification |
| `risk_flags` | string | Semicolon-separated flags (e.g. `blurry_image;user_history_risk`) |
| `issue_type` | string | Damage type (e.g. `dent`, `crack`, `torn_packaging`) |
| `object_part` | string | Specific part claimed (e.g. `windshield`, `screen`) |
| `claim_status` | string | `supported`, `contradicted`, or `not_enough_information` |
| `claim_status_justification` | string | Explanation grounded in image IDs |
| `supporting_image_ids` | string | Semicolon-separated image IDs that support the verdict |
| `valid_image` | bool string | Whether submitted images are valid |
| `severity` | string | `none`, `low`, `medium`, `high`, or `unknown` |

---

## Performance

Evaluated on 20 labeled sample claims:

| Field | Accuracy |
|---|---|
| claim_status | 80% |
| issue_type | 65% |
| object_part | 85% |
| severity | 65% |

See `code/evaluation/evaluation_report.md` for full experiment history.

---

## Cost

- ~$0.003 per claim (Gemini 2.5 Flash + thinking mode)
- ~$0.13 for 44 test claims
- ~$3.00 for 1,000 claims
- Average latency: 6 seconds per claim
