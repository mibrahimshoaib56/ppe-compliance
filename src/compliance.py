"""Pure Python rule-based compliance logic for PPE detection.

This module evaluates Personal Protective Equipment (PPE) compliance for
construction workers based on YOLOv8 detections. It has no dependencies
on Streamlit or YOLO, making it fast, robust, and completely unit-testable.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

# ============================================================================
# NAMED CONSTANTS & THRESHOLDS
# ============================================================================

# Anatomical body zone ratios relative to worker box height (top=0.0, bottom=1.0)
HEAD_ZONE_TOP: float = 0.0
HEAD_ZONE_BOTTOM: float = 0.35  # Top ~35% of the person box

# Widened torso zone covering the middle 15% to 75% of person height.
# This accommodates workers wearing tool bags, backpacks, or equipment harnesses
# over their vests, as well as partial vest detections.
TORSO_ZONE_TOP: float = 0.15
TORSO_ZONE_BOTTOM: float = 0.75

# Minimum overlap ratio required to associate a gear detection with a body zone.
# Metric: Intersection over Gear Area (IoG = intersection_area / gear_box_area),
# which measures how much of the gear box lies inside the person's anatomical zone
# instead of standard IoU. A named threshold of 0.30 ensures partial gear detections
# (e.g. vests partially occluded by bags) are reliably captured.
OVERLAP_THRESHOLD: float = 0.30

# Site safety status rating thresholds based on compliance rate percentage
SITE_SAFE_THRESHOLD: float = 85.0  # >= 85% compliant -> Safe
SITE_WARNING_THRESHOLD: float = 60.0  # >= 60% and < 85% -> Warning, < 60% -> Critical

# Standard class names used by the compliance logic
CLASS_PERSON: str = "Person"
CLASS_HARDHAT: str = "Hardhat"
CLASS_NO_HARDHAT: str = "NO-Hardhat"
CLASS_VEST: str = "Safety Vest"
CLASS_NO_VEST: str = "NO-Safety Vest"

# Mapping from class ID integers to normalized class names
CLASS_ID_TO_NAME: Dict[int, str] = {
    0: CLASS_HARDHAT,
    2: CLASS_NO_HARDHAT,
    4: CLASS_NO_VEST,
    5: CLASS_PERSON,
    7: CLASS_VEST,
}

# Gear classification types
GEAR_HEAD_POSITIVE: str = CLASS_HARDHAT
GEAR_HEAD_NEGATIVE: str = CLASS_NO_HARDHAT
GEAR_TORSO_POSITIVE: str = CLASS_VEST
GEAR_TORSO_NEGATIVE: str = CLASS_NO_VEST


# ============================================================================
# GEOMETRY HELPERS
# ============================================================================

def compute_box_area(box: Union[List[float], Tuple[float, float, float, float]]) -> float:
    """Compute area of a bounding box [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = box
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    return width * height


def compute_intersection_area(
    box_a: Union[List[float], Tuple[float, float, float, float]],
    box_b: Union[List[float], Tuple[float, float, float, float]],
) -> float:
    """Compute the intersection area between two bounding boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_width = max(0.0, x2 - x1)
    inter_height = max(0.0, y2 - y1)
    return inter_width * inter_height


def compute_overlap_ratio(
    gear_box: Union[List[float], Tuple[float, float, float, float]],
    zone_box: Union[List[float], Tuple[float, float, float, float]],
) -> float:
    """Compute the fraction of the gear box that lies inside the zone box.

    Formula: Intersection(gear, zone) / Area(gear)
    Returns a value between 0.0 and 1.0.
    """
    gear_area = compute_box_area(gear_box)
    if gear_area <= 0.0:
        return 0.0

    inter_area = compute_intersection_area(gear_box, zone_box)
    return inter_area / gear_area


def extract_person_zones(
    person_box: Union[List[float], Tuple[float, float, float, float]],
) -> Tuple[List[float], List[float]]:
    """Extract head and torso zone boxes from a person bounding box.

    Returns:
        Tuple of (head_zone_box, torso_zone_box), where each box is [x1, y1, x2, y2].
    """
    px1, py1, px2, py2 = person_box
    p_height = py2 - py1

    head_zone = [
        px1,
        py1 + (HEAD_ZONE_TOP * p_height),
        px2,
        py1 + (HEAD_ZONE_BOTTOM * p_height),
    ]

    torso_zone = [
        px1,
        py1 + (TORSO_ZONE_TOP * p_height),
        px2,
        py1 + (TORSO_ZONE_BOTTOM * p_height),
    ]

    return head_zone, torso_zone


# ============================================================================
# DETECTION NORMALIZATION
# ============================================================================

def normalize_detection(det: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalize a raw detection dictionary into a standard format.

    Accepts detection dicts with keys like 'class_name', 'class', 'label',
    'class_id', 'confidence', 'conf', 'score', and 'box' or 'bbox'.
    Returns None if class is not one of our 5 relevant compliance classes.
    """
    raw_class = det.get("class_name") or det.get("class") or det.get("label")
    raw_id = det.get("class_id")

    normalized_class = None
    if isinstance(raw_id, int) and raw_id in CLASS_ID_TO_NAME:
        normalized_class = CLASS_ID_TO_NAME[raw_id]
    elif isinstance(raw_class, str):
        cleaned = raw_class.strip().lower()
        mapping = {
            "person": CLASS_PERSON,
            "hardhat": CLASS_HARDHAT,
            "no-hardhat": CLASS_NO_HARDHAT,
            "safety vest": CLASS_VEST,
            "safety-vest": CLASS_VEST,
            "vest": CLASS_VEST,
            "no-safety vest": CLASS_NO_VEST,
            "no-safety-vest": CLASS_NO_VEST,
            "no-vest": CLASS_NO_VEST,
        }
        normalized_class = mapping.get(cleaned)

    if normalized_class is None:
        return None

    conf = det.get("confidence", det.get("conf", det.get("score", 1.0)))
    box = det.get("box", det.get("bbox"))
    if not box or len(box) != 4:
        return None

    return {
        "class_name": normalized_class,
        "confidence": float(conf),
        "box": [float(b) for b in box],
    }


# ============================================================================
# COMPLIANCE EVALUATION LOGIC
# ============================================================================

def evaluate_site_compliance(
    detections: List[Dict[str, Any]],
    required_gear: str = "both",
    overlap_threshold: float = OVERLAP_THRESHOLD,
) -> Dict[str, Any]:
    """Evaluate compliance of all detected workers in an image.

    Args:
        detections: List of detection dictionaries (class name, confidence, box).
        required_gear: 'both' (Hardhat + Vest) or 'hardhat_only'.
        overlap_threshold: Minimum overlap ratio for zone association.

    Returns:
        Dictionary containing:
            - 'workers': List of evaluated worker dictionaries.
            - 'summary': Aggregate site statistics dictionary.
            - 'unassigned_gear': Gear detections not linked to any worker.
    """
    # 1. Normalize and filter detections
    normalized = []
    for d in detections:
        nd = normalize_detection(d)
        if nd is not None:
            normalized.append(nd)

    # 2. Separate Person boxes from Gear boxes
    person_detections: List[Dict[str, Any]] = []
    gear_detections: List[Dict[str, Any]] = []

    for d in normalized:
        if d["class_name"] == CLASS_PERSON:
            person_detections.append(d)
        else:
            gear_detections.append(d)

    # Initialize workers list
    workers: List[Dict[str, Any]] = []
    for idx, p in enumerate(person_detections):
        head_zone, torso_zone = extract_person_zones(p["box"])
        workers.append({
            "worker_id": idx + 1,
            "box": p["box"],
            "confidence": p["confidence"],
            "head_zone": head_zone,
            "torso_zone": torso_zone,
            "assigned_gear": [],
            "hardhat_status": "unknown",  # "present", "missing", "unknown"
            "vest_status": "unknown",     # "present", "missing", "unknown"
            "status": "Unknown",          # "Compliant", "Missing ...", "Unknown"
        })

    # 3. Disambiguation: Assign each gear box to at most one person
    # Gear boxes are matched to the person zone where overlap is greatest
    unassigned_gear: List[Dict[str, Any]] = []

    for gear in gear_detections:
        g_name = gear["class_name"]
        g_box = gear["box"]
        is_head_gear = g_name in (GEAR_HEAD_POSITIVE, GEAR_HEAD_NEGATIVE)

        best_worker_idx: Optional[int] = None
        best_overlap: float = 0.0

        for w_idx, worker in enumerate(workers):
            zone_box = worker["head_zone"] if is_head_gear else worker["torso_zone"]
            overlap = compute_overlap_ratio(g_box, zone_box)

            if overlap >= overlap_threshold and overlap > best_overlap:
                best_overlap = overlap
                best_worker_idx = w_idx

        if best_worker_idx is not None:
            assigned_item = dict(gear)
            assigned_item["overlap"] = best_overlap
            workers[best_worker_idx]["assigned_gear"].append(assigned_item)
        else:
            unassigned_gear.append(gear)

    # 4. Resolve status for each worker
    for worker in workers:
        head_pos = [g for g in worker["assigned_gear"] if g["class_name"] == GEAR_HEAD_POSITIVE]
        head_neg = [g for g in worker["assigned_gear"] if g["class_name"] == GEAR_HEAD_NEGATIVE]
        torso_pos = [g for g in worker["assigned_gear"] if g["class_name"] == GEAR_TORSO_POSITIVE]
        torso_neg = [g for g in worker["assigned_gear"] if g["class_name"] == GEAR_TORSO_NEGATIVE]

        # Evaluate Hardhat
        if head_neg and not head_pos:
            worker["hardhat_status"] = "missing"
        elif head_pos and not head_neg:
            worker["hardhat_status"] = "present"
        elif head_pos and head_neg:
            # Conflict: choose the detection with higher confidence
            best_pos_conf = max(g["confidence"] for g in head_pos)
            best_neg_conf = max(g["confidence"] for g in head_neg)
            worker["hardhat_status"] = "present" if best_pos_conf >= best_neg_conf else "missing"
        else:
            worker["hardhat_status"] = "unknown"

        # Evaluate Vest
        if torso_neg and not torso_pos:
            worker["vest_status"] = "missing"
        elif torso_pos and not torso_neg:
            worker["vest_status"] = "present"
        elif torso_pos and torso_neg:
            # Conflict: choose the detection with higher confidence
            best_pos_conf = max(g["confidence"] for g in torso_pos)
            best_neg_conf = max(g["confidence"] for g in torso_neg)
            worker["vest_status"] = "present" if best_pos_conf >= best_neg_conf else "missing"
        else:
            worker["vest_status"] = "unknown"

        # Overall worker compliance status
        check_vest = (required_gear == "both")

        h_stat = worker["hardhat_status"]
        v_stat = worker["vest_status"]

        if check_vest:
            if h_stat == "missing" and v_stat == "missing":
                worker["status"] = "Missing both"
            elif h_stat == "missing":
                worker["status"] = "Missing hardhat"
            elif v_stat == "missing":
                worker["status"] = "Missing vest"
            elif h_stat == "present" and v_stat == "present":
                worker["status"] = "Compliant"
            elif v_stat == "unknown":
                # Workers without confirmed vest evidence require manual on-site review
                worker["status"] = "Vest not visible: manual check"
            else:
                worker["status"] = "Unknown"
        else:
            # Hardhat only
            if h_stat == "missing":
                worker["status"] = "Missing hardhat"
            elif h_stat == "present":
                worker["status"] = "Compliant"
            else:
                worker["status"] = "Unknown"

    # 5. Calculate Site Summary Metrics
    total_workers = len(workers)
    compliant_count = sum(1 for w in workers if w["status"] == "Compliant")
    violation_count = sum(
        1 for w in workers if w["status"] in ("Missing hardhat", "Missing vest", "Missing both")
    )
    needs_review_count = sum(
        1 for w in workers if w["status"] in ("Vest not visible: manual check", "Unknown")
    )
    unknown_count = needs_review_count

    if total_workers > 0:
        compliance_rate = round((compliant_count / total_workers) * 100.0, 1)
    else:
        compliance_rate = 100.0

    if total_workers == 0:
        site_status = "Safe"
    elif compliance_rate >= SITE_SAFE_THRESHOLD:
        site_status = "Safe"
    elif compliance_rate >= SITE_WARNING_THRESHOLD:
        site_status = "Warning"
    else:
        site_status = "Critical"

    summary = {
        "total_workers": total_workers,
        "compliant_count": compliant_count,
        "violation_count": violation_count,
        "needs_review_count": needs_review_count,
        "unknown_count": unknown_count,
        "compliance_rate": compliance_rate,
        "site_status": site_status,
    }

    return {
        "workers": workers,
        "summary": summary,
        "unassigned_gear": unassigned_gear,
    }
