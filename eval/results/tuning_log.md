# Detection & Compliance Tuning Log

Tracks every named constant in `src/compliance.py`, `src/detector.py`, and `src/video.py` that
affects a detection, compliance, or video-processing decision.

**Starting values** are taken from the initial git commit `df563b0` (2026-11-17)
and confirmed by `PROJECT_CONTEXT.md` (line 41).  
**Final values** are read directly from the current working-tree code.  
No code was modified to produce this file.

---

## 1. `src/compliance.py` (Zones & Core Policy)

| Constant | File | Starting value | Final value | Why it changed |
|---|---|---|---|---|
| `HEAD_ZONE_TOP` | compliance.py | `0.0` | `0.0` | Unchanged — top of the person box is the natural start of the head zone. |
| `HEAD_ZONE_BOTTOM` | compliance.py | `0.35` | `0.35` | Unchanged — top 35 % of the person box reliably covers the helmet region. |
| `TORSO_ZONE_TOP` | compliance.py | `0.20` | `0.15` | Widened upward by 5 pp. Per user request (prompt: "widen the torso zone to the middle 15%–75%") to capture vests on workers with backpacks/tool bags that push the vest detection box higher. Confirmed in git diff and code comment. |
| `TORSO_ZONE_BOTTOM` | compliance.py | `0.65` | `0.75` | Widened downward by 10 pp. Same request as above — allows partial vest boxes occluded by bags to still fall inside the torso zone. Confirmed in git diff and code comment. |
| `OVERLAP_THRESHOLD` | compliance.py | `0.20` | `0.30` | Raised from 0.20 to 0.30. Per user request: "change the overlap rule for gear to use intersection area / gear box area (IoG) … with a named constant threshold of 0.3". The higher threshold reduces false associations while the IoG metric (instead of IoU) makes partial gear detections viable. Confirmed in git diff. |
| `SITE_SAFE_THRESHOLD` | compliance.py | `85.0` | `85.0` | Unchanged — 85 % compliance rate is the Safe boundary as originally specified. |
| `SITE_WARNING_THRESHOLD` | compliance.py | `60.0` | `60.0` | Unchanged — 60 % compliance rate is the Warning/Critical boundary as originally specified. |

### Overlap metric note

The initial commit already implemented **IoG** (Intersection over Gear area)
as the overlap formula (`compute_overlap_ratio` divides by gear box area, not
union area). The comment in the initial code read:

> *"Metric used: Intersection over Gear Area (IoG = intersection_area / gear_area). This is preferred over IoU because gear boxes (hardhat/vest) are much smaller than person body zones."*

A subsequent prompt formalised this choice and raised the threshold to `0.30`.

---

## 2. `src/compliance.py` (Reliability & Strict Hardhat Policy)

| Constant | File | Starting value | Final value | Why it changed |
|---|---|---|---|---|
| `STRICT_HARDHAT_POLICY` | compliance.py | `False` | `True` | In construction safety, a reliable worker observed with no hardhat detection is presumed missing head protection rather than unknown. Can be toggled in Streamlit sidebar. |
| `MIN_PERSON_CONF` | compliance.py | N/A | `0.40` | Worker reliability gate: Person detections with confidence < 0.40 are routed to "Needs review: low confidence" to prevent background clutter false violations. |
| `MIN_PERSON_HEIGHT_FRAC` | compliance.py | N/A | `0.12` | Worker reliability gate: Workers whose bounding box height is < 12% of total image height are routed to "Needs review: small" because tiny background figures cannot reliably show PPE. |
| `MIN_UPRIGHT_ASPECT` | compliance.py | N/A | `1.2` | Worker reliability gate: Height-to-width aspect ratio < 1.2 indicates overhead camera angles or severe crouching where torso/head geometry is unreliable. Routed to "Needs review: view not upright" and vest status overridden to unknown. |

---

## 3. `src/detector.py` (Two-Stage Confidence Filtering)

| Constant | File | Starting value | Final value | Why it changed |
|---|---|---|---|---|
| `BASE_YOLO_CONFIDENCE` | detector.py | N/A (single 0.25) | `0.10` | Two-stage filtering design: YOLO runs at a permissive base threshold (0.10) to retain candidates before per-class filters apply specific thresholds. |
| `CONF_PERSON` | detector.py | N/A | `0.30` | Per-class threshold for Person: set higher to prevent clutter and spurious false person boxes. |
| `CONF_HARDHAT` | detector.py | N/A | `0.30` | Per-class threshold for Hardhat: positive-class gear requires high certainty. |
| `CONF_NO_HARDHAT` | detector.py | N/A | `0.20` | Lower threshold favors violation recall (better to flag a potential violation than miss it). |
| `CONF_SAFETY_VEST` | detector.py | N/A | `0.20` | Lower threshold improves recall for partially occluded vests (straps, harnesses, bags). |
| `CONF_NO_SAFETY_VEST` | detector.py | N/A | `0.20` | Lower threshold favors violation recall for missing vests. |
| `CONF_DEFAULT` | detector.py | N/A | `0.20` | Fallback threshold for unlisted contextual classes (machinery, vehicles, cones). |

---

## 4. `src/video.py` (CCTV Video Processing & Smoothing)

| Constant | File | Starting value | Final value | Why it changed |
|---|---|---|---|---|
| `MAX_VIDEO_SECONDS` | video.py | N/A | `30` | Caps analysis to the first 30 seconds to keep CPU inference under acceptable latency. |
| `FALLBACK_FPS` | video.py | N/A | `25.0` | Default FPS when video metadata reports 0, negative, or invalid framerate. |
| `PROCESS_EVERY_N` | video.py | `2` | `1` | Stride of analyzed frames. Set to 1 for maximum smoothness and per-frame detection accuracy. |
| `MAX_FRAME_WIDTH` | video.py | `640` | `416` | Standard YOLO inference width, ~30% faster on CPU with minimal accuracy difference. |
| `OUTPUT_MAX_FPS` | video.py | N/A | `30` | Caps exported H.264 MP4 playback framerate. |
| `TOP_VIOLATION_FRAMES_COUNT` | video.py | N/A | `3` | Number of peak violation moments displayed in executive summary gallery. |
| `TEMPORAL_SMOOTHING_WINDOW` | video.py | N/A | `7` | 7-frame rolling median window to eliminate 1–2 frame transient flicker from rapid head turns and occlusions. |

---

## 5. Summary of Net Effects

| Category | Constants Affected | Net Effect |
|---|---|---|
| Torso zone widening | `TORSO_ZONE_TOP`, `TORSO_ZONE_BOTTOM` | Captures vests occluded by tool belts, bags, or backpacks. |
| Overlap metric | `OVERLAP_THRESHOLD` (IoG = 0.30) | Measures fraction of gear inside anatomical zone, preventing small gear penalties. |
| Reliability guards | `MIN_PERSON_CONF`, `MIN_PERSON_HEIGHT_FRAC`, `MIN_UPRIGHT_ASPECT` | Small, distant, or overhead persons are routed to "Needs review" rather than penalized falsely. |
| Strict hardhat | `STRICT_HARDHAT_POLICY` | Reliable persons without hardhat boxes are flagged as violations. |
| Temporal smoothing | `TEMPORAL_SMOOTHING_WINDOW = 7` | Stabilizes video metrics against inter-frame posture fluctuations without inter-frame tracking. |

---

*Last updated: 2026-11-24 | Source: Current code inspection of `src/compliance.py`, `src/detector.py`, and `src/video.py`.*
