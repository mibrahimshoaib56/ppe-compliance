# Detection & Compliance Tuning Log

Tracks every named constant in `src/compliance.py` and `src/detector.py` that
affects a detection or compliance decision.

**Starting values** are taken from the initial git commit `df563b0` (2026-11-17)
and confirmed by `PROJECT_CONTEXT.md` (line 41).  
**Final values** are read directly from the current working-tree code.  
No code was modified to produce this file.

---

## `src/compliance.py`

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

## `src/detector.py`

> [!NOTE]
> The initial commit (`df563b0`) had **no per-class confidence constants**.
> The original `run_detection()` accepted a single `conf_threshold: float = 0.25`
> parameter passed directly to `model.predict()`. All constants below are **new**,
> introduced in an uncommitted working-tree edit per the user prompt:
> *"run YOLO at a low base confidence (0.10), then filter detections per class
> using named constants"*.

| Constant | File | Starting value | Final value | Why it changed |
|---|---|---|---|---|
| `BASE_YOLO_CONFIDENCE` | detector.py | N/A (not present in initial commit) | `0.10` | New two-stage filtering design. YOLO is run at a permissive base threshold (0.10) so more raw candidates are returned; per-class filters then apply the real thresholds. The original single `conf_threshold=0.25` was replaced by this constant. |
| `CONF_PERSON` | detector.py | N/A (not present in initial commit) | `0.30` | Per-class threshold for the Person class. Set higher than the base to avoid false person detections from background objects. |
| `CONF_HARDHAT` | detector.py | N/A (not present in initial commit) | `0.30` | Per-class threshold for Hardhat. Same rationale as `CONF_PERSON` — positive-class gear needs high certainty. |
| `CONF_NO_HARDHAT` | detector.py | N/A (not present in initial commit) | `0.20` | Per-class threshold for NO-Hardhat. Set lower than positive class to favour false-negative detection (better to flag a potential violation than to miss it). |
| `CONF_SAFETY_VEST` | detector.py | N/A (not present in initial commit) | `0.20` | Per-class threshold for Safety Vest. Lower threshold improves recall for partially occluded vests (bags, harnesses). |
| `CONF_NO_SAFETY_VEST` | detector.py | N/A (not present in initial commit) | `0.20` | Per-class threshold for NO-Safety Vest. Same rationale as `CONF_NO_HARDHAT`. |
| `CONF_DEFAULT` | detector.py | N/A (not present in initial commit) | `0.20` | Fallback threshold for any class name not explicitly listed in `CLASS_CONF_THRESHOLDS`. Guards against future model classes that lack a named constant. |

---

## Summary

| Change category | Constants affected | Net effect |
|---|---|---|
| Torso zone widening | `TORSO_ZONE_TOP`, `TORSO_ZONE_BOTTOM` | Vest detections for bag-wearing workers are more likely to be associated with the correct worker. |
| Overlap threshold tightening | `OVERLAP_THRESHOLD` | Requires 30 % of the gear box to overlap a body zone (up from 20 %), reducing spurious assignments of distant gear boxes. |
| Two-stage confidence filtering | All `CONF_*` and `BASE_YOLO_CONFIDENCE` | Low base threshold keeps more candidates; per-class gates apply the real minimum, giving independent control over precision/recall per class. |
| Unchanged | `HEAD_ZONE_TOP`, `HEAD_ZONE_BOTTOM`, `SITE_SAFE_THRESHOLD`, `SITE_WARNING_THRESHOLD` | No evidence of change in git history or code comments. |

---

*Last updated: 2026-11-19 | Source: `git log -p src/compliance.py src/detector.py` + `PROJECT_CONTEXT.md` line 41*
