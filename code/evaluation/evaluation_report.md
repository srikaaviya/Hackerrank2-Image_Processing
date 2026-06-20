# Evaluation Report — Multi-Modal Damage Claim Verification

**Challenge:** HackerRank Orchestrate — June 2026  
**Participant:** srikaaviya  
**Evaluated on:** `dataset/sample_claims.csv` (20 labeled claims)  
**Final output:** `dataset/output.csv` (44 test claims)

---

## Final System Performance

| Field | Accuracy | Correct / Total |
|---|---|---|
| claim_status | **80%** | 16/20 |
| issue_type | **65%** | 13/20 |
| object_part | **85%** | 17/20 |
| severity | **65%** | 13/20 |
| evidence_standard_met | **80%** | 16/20 |
| **Average** | **75%** | — |

---

## System Architecture

```
claims.csv + images
       │
       ▼
[Stage 1] OpenCV Image Quality Check
  - Blur detection (Laplacian variance < 80)
  - Brightness check
  - Screenshot detection (screen resolution + color band analysis)
  - Outputs: aggregate_flags (blurry_image, non_original_image, etc.)
       │
       ▼
[Stage 2] Evidence Requirements Lookup
  - Loads dataset/evidence_requirements.csv
  - Filters requirements by claim_object (car/laptop/package)
  - Passes relevant REQ_* text to Gemini for evidence_standard_met grounding
       │
       ▼
[Stage 3] User History Risk Flags
  - Loads dataset/user_history.csv
  - Returns history_flags per user_id
  - Appended to risk_flags in output (never changes verdict)
       │
       ▼
[Stage 4] Gemini 2.5 Flash — Core Verdict
  - Object-specific system instruction (CAR / LAPTOP / PACKAGE)
  - Object-specific response_schema with enum-constrained issue_type
  - Structured reasoning: reasons_to_support + reasons_to_contradict first
  - Thinking mode (budget=1024 tokens)
  - Single API call per claim
       │
       ▼
[Stage 5] Post-processing
  - Severity clamp: scratch/dent/crack/stain/water_damage/packaging → never "high"
  - User history flags merged into risk_flags
  - Boolean fields normalized to string ("true"/"false")
       │
       ▼
output.csv
```

---

## Experiment History

All experiments run on the 20-sample labeled set. Baseline = first working Gemini Flash run.

| # | Experiment | claim% | issue% | object% | sev% | Avg | Decision |
|---|---|---|---|---|---|---|---|
| 1 | Strategy A: CLIP + OpenCV + rules | 50% | 45% | 65% | — | — | Replaced by Gemini |
| 2 | Strategy B baseline (Gemini Flash) | 65% | 35% | 75% | 30% | 51% | Starting point |
| 3 | + Prompt fixes (relax evidence std, contradicted detection) | 80% | 40% | 85% | 35% | 60% | +15% claim_status |
| 4 | + Thinking mode (budget=1024) + system instruction separation | 80% | 40% | 85% | 35% | 60% | Same accuracy, more reliable |
| 5 | + BLIP image captioning | 75% | 40% | 85% | 30% | 58% | ❌ Removed — hurt accuracy |
| 6 | + Grounding DINO part localization + crops | 75% | 40% | 90% | 30% | 59% | ❌ Removed — marginal gain, slow |
| 7 | + CLIP evidence hints | 75% | 40% | 85% | 30% | 58% | ❌ Removed — false positives |
| 8 | + Few-shot examples (5 labeled) | 65% | 35% | 65% | 35% | 50% | ❌ Removed — worst result |
| 9 | + Deep issue_type definitions | 80% | 45% | 85% | 30% | 60% | Minimal gain |
| 10 | + Fix glass_shatter definition (remove spider-web example) | 75% | 65% | 85% | 55% | 70% | ✅ Major issue_type gain |
| 11 | + response_schema + structured reasoning fields | 75% | 55% | 85% | 55% | 68% | ✅ Zero parse errors |
| 12 | + Severity clamp (NEVER_HIGH set) | 75% | 55% | 85% | 60% | 69% | ✅ Small severity gain |
| 13 | + Evidence requirements in prompt | 80% | 55% | 85% | 50% | 68% | ✅ claim_status +5% |
| 14 | + Object-specific prompts (CAR/LAPTOP/PACKAGE) | 85% | 60% | 85% | 60% | 73% | ✅ Best claim_status |
| 15 | + crack added to NEVER_HIGH clamp | 80% | 65% | 85% | 65% | 74% | ✅ Best overall avg |
| 16 | + Light/Shadow dent vs scratch rule | 80% | 65% | 85% | 65% | **74%** | ✅ Kept — final config |
| 17 | + Negative prompting (list 3 reasons) | 70% | 40% | 70% | 30% | 53% | ❌ Removed — broke JSON |
| 18 | + Skeptical Adjuster (aggressive) | 75% | 60% | 90% | 65% | 73% | ❌ Removed — over-fired |
| 19 | + State-machine contradicted routing | 80% | 55% | 85% | 60% | 70% | ❌ Removed — GT conflict |
| 20 | + Functional severity framework | 80% | 65% | 85% | 65% | **74%** | ✅ Kept — final config |

---

## Key Findings

### What Worked

**1. Object-specific system instructions**  
Splitting into CAR/LAPTOP/PACKAGE system instructions with object-specific issue_type enums (enforced at API level via `response_schema`) prevented cross-object hallucination. A package claim can no longer return `dent` or `glass_shatter`; a car claim cannot return `torn_packaging`.

**2. Fixing the glass_shatter definition**  
The original definition listed "spider-web pattern" as an example of `glass_shatter`. GT classifies spider-web windshield cracks as `crack` (glass still intact). Removing this example and clarifying that `glass_shatter` = physically separate pieces drove issue_type from 40% → 65%.

**3. response_schema enforcement**  
Using `response_schema` + `response_mime_type=application/json` in GenerateContentConfig eliminated all JSON parse errors across 44+ test runs. `claim_status`, `issue_type`, and `severity` are enum-constrained — Gemini cannot return invalid values.

**4. Structured reasoning before verdict**  
Adding `reasons_to_support_claim` and `reasons_to_contradict_claim` as the first fields in the schema forces Gemini to weigh both sides before deciding `claim_status`. More reliable than free-text chain-of-thought which broke JSON output.

**5. Evidence requirements grounding**  
Including the relevant `REQ_*` entries from `evidence_requirements.csv` in the prompt gave Gemini a concrete standard for `evidence_standard_met` — e.g. "REQ_CAR_BODY_PANEL: The claimed car panel should be visible from an angle where surface marks can be assessed."

**6. Severity clamp on impossible combinations**  
Post-processing clamp: if `issue_type` ∈ {scratch, dent, crack, stain, water_damage, torn_packaging, crushed_packaging, missing_part} and `severity = high` → override to `medium`. These types cannot be catastrophic by definition. Improved severity accuracy without introducing hardcoded rules for specific cases.

### What Did Not Work

**BLIP image captioning** — Generic captions like "a car with damage" biased Gemini toward `supported` without specifying part location. Accuracy dropped 80% → 75%.

**Grounding DINO part localization** — Cropping to claimed part and sending to Gemini showed marginal benefit. DINO confidence was too low on unusual angles, and the extra processing added 8-12s per claim.

**CLIP object verification** — ~91% accuracy but ~9% false-positive rate on close-up images. Flagging a real laptop as `wrong_object` was worse than no check at all.

**Few-shot examples** — 5 labeled examples in the prompt caused over-application of `not_enough_information` (65% accuracy, worst result). Too many tokens also slowed responses to 40s/call.

**Negative prompting (free-text)** — Asking Gemini to "list 3 reasons for and against" before answering caused JSON parse errors when the reasoning appeared before the JSON block.

**Aggressive Skeptical Adjuster** — "Assume the object is pristine" caused genuine supported claims to be marked contradicted.

---

## Edge Cases Handled

| Edge Case | Handling |
|---|---|
| Multi-language conversations (Hindi, Spanish, mixed) | System instruction: "Understand all" |
| Prompt injection in claim text/images | Rule 2: flag as `text_instruction_present` |
| Blurry / low-quality images | OpenCV Laplacian variance check |
| Screenshots vs real photos | Screen resolution + uniform color band detection |
| Multiple images = different angles of same object | Rule 5: never treat as different vehicles |
| User says "crack" but image shows worse damage | Rule 7: still "supported" — damage is real |
| Repeat claimants with fraud history | `user_history.csv` → `user_history_risk` flag |
| API rate limiting | 4s delay between calls, 3 retries with backoff |
| API timeouts (socket) | Retry with 10s × attempt backoff |
| API 503 service unavailable | Retry with 5s × attempt backoff |

---

## Cost Analysis

| Scale | API Calls | Estimated Cost | Avg Latency |
|---|---|---|---|
| 20 sample claims | 20 | ~$0.06 | 6s/claim |
| 44 test claims | 44 | ~$0.13 | 6s/claim |
| 1,000 claims | 1,000 | ~$3.00 | 6s/claim |
| 10,000 claims | 10,000 | ~$30.00 | 6s/claim |

*Based on Gemini 2.5 Flash pricing with thinking mode (budget=1024). Single API call per claim.*

---

## Persistent Failure Analysis

Four claims remain incorrect after all optimizations:

| user_id | GT | Prediction | Root Cause |
|---|---|---|---|
| user_005 | contradicted/scratch/low | supported/dent/medium | Image shows rear bumper area; Gemini sees surface deformation and says supported. GT says claimed scratch is not visible — ambiguous angle. |
| user_020 | contradicted/none/none | supported/scratch/low | Trackpad image appears clean to human but Gemini finds a faint mark. |
| user_032 | not_enough_info/unknown/unknown | supported/missing_part/medium | Package contents image is ambiguous; Gemini confidently identifies missing item. |
| user_034 | contradicted/none/none | supported/torn_packaging/medium | Package seal image looks intact but Gemini sees edge artifacts as tearing. |

These failures share a pattern: **Gemini over-confirms damage on borderline images.** A second verification call (ensemble approach) was considered but rejected due to 2× cost. These cases likely require human review in production.

---

## Files

| File | Purpose |
|---|---|
| `code/main.py` | Entry point — orchestrates full pipeline |
| `code/pipeline/gemini_verdict.py` | Core: 3 system instructions, dynamic schema, Gemini call |
| `code/pipeline/image_quality.py` | OpenCV blur/brightness/screenshot detection |
| `code/pipeline/rules.py` | User history flags, evidence requirements loader |
| `code/pipeline/claim_extractor.py` | Regex-based claim extraction (fallback) |
| `code/pipeline/object_verifier.py` | CLIP object verification (Strategy A only) |
| `code/pipeline/damage_detector.py` | YOLO + CLIP damage detection (Strategy A only) |
| `code/pipeline/blip_descriptor.py` | BLIP captioning (tried, reverted — documented) |
| `code/pipeline/dino_locator.py` | Grounding DINO localization (tried, reverted — documented) |
| `dataset/output.csv` | Final predictions on 44 test claims |
| `dataset/sample_output_B.csv` | Development eval on 20 labeled claims |

---

*Generated: 2026-06-19*
