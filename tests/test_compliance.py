"""Unit tests for pure Python PPE compliance evaluation logic.

These tests use synthetic bounding boxes to exhaustively verify all compliance
rules, body zone overlap calculations, and edge cases defined in PROJECT_CONTEXT.md.
"""

import pytest
from src.compliance import (
    CLASS_HARDHAT,
    CLASS_NO_HARDHAT,
    CLASS_NO_VEST,
    CLASS_PERSON,
    CLASS_VEST,
    STRICT_HARDHAT_POLICY,
    MIN_PERSON_CONF,
    MIN_PERSON_HEIGHT_FRAC,
    MIN_UPRIGHT_ASPECT,
    assess_person_reliability,
    evaluate_site_compliance,
    extract_person_zones,
    compute_overlap_ratio,
)


@pytest.fixture
def base_person_box():
    """Standard synthetic person box: width=100, height=200."""
    return [100.0, 100.0, 200.0, 300.0]


def test_zone_extraction(base_person_box):
    """Verify head (0% to 35%) and widened torso (15% to 75%) zone calculation."""
    head_zone, torso_zone = extract_person_zones(base_person_box)

    # Height = 200. Head zone: py1 + (0.0 to 0.35) * 200 = 100 to 170
    assert head_zone == [100.0, 100.0, 200.0, 170.0]

    # Widened torso zone: py1 + (0.15 to 0.75) * 200 = 130 to 250
    assert torso_zone == [100.0, 130.0, 200.0, 250.0]


def test_fully_compliant_worker(base_person_box):
    """Test worker wearing both a hardhat and a safety vest."""
    detections = [
        {"class_name": CLASS_PERSON, "confidence": 0.95, "box": base_person_box},
        {"class_name": CLASS_HARDHAT, "confidence": 0.90, "box": [120.0, 105.0, 180.0, 150.0]},
        {"class_name": CLASS_VEST, "confidence": 0.92, "box": [110.0, 150.0, 190.0, 220.0]},
    ]

    result = evaluate_site_compliance(detections)
    workers = result["workers"]
    summary = result["summary"]

    assert len(workers) == 1
    w = workers[0]
    assert w["hardhat_status"] == "present"
    assert w["vest_status"] == "present"
    assert w["status"] == "Compliant"

    assert summary["total_workers"] == 1
    assert summary["compliant_count"] == 1
    assert summary["violation_count"] == 0
    assert summary["compliance_rate"] == 100.0
    assert summary["site_status"] == "Safe"


def test_missing_hardhat(base_person_box):
    """Test worker with NO-Hardhat detected in head zone and Safety Vest in torso zone."""
    detections = [
        {"class_name": CLASS_PERSON, "confidence": 0.95, "box": base_person_box},
        {"class_name": CLASS_NO_HARDHAT, "confidence": 0.88, "box": [120.0, 105.0, 180.0, 150.0]},
        {"class_name": CLASS_VEST, "confidence": 0.90, "box": [110.0, 150.0, 190.0, 220.0]},
    ]

    result = evaluate_site_compliance(detections)
    workers = result["workers"]
    summary = result["summary"]

    assert len(workers) == 1
    w = workers[0]
    assert w["hardhat_status"] == "missing"
    assert w["vest_status"] == "present"
    assert w["status"] == "Missing hardhat"

    assert summary["total_workers"] == 1
    assert summary["compliant_count"] == 0
    assert summary["violation_count"] == 1
    assert summary["compliance_rate"] == 0.0
    assert summary["site_status"] == "Critical"


def test_missing_vest(base_person_box):
    """Test worker with Hardhat in head zone and NO-Safety Vest in torso zone."""
    detections = [
        {"class_name": CLASS_PERSON, "confidence": 0.95, "box": base_person_box},
        {"class_name": CLASS_HARDHAT, "confidence": 0.91, "box": [120.0, 105.0, 180.0, 150.0]},
        {"class_name": CLASS_NO_VEST, "confidence": 0.87, "box": [110.0, 150.0, 190.0, 220.0]},
    ]

    result = evaluate_site_compliance(detections)
    workers = result["workers"]
    summary = result["summary"]

    assert len(workers) == 1
    w = workers[0]
    assert w["hardhat_status"] == "present"
    assert w["vest_status"] == "missing"
    assert w["status"] == "Missing vest"

    assert summary["violation_count"] == 1
    assert summary["site_status"] == "Critical"


def test_missing_both(base_person_box):
    """Test worker with both NO-Hardhat and NO-Safety Vest detected."""
    detections = [
        {"class_name": CLASS_PERSON, "confidence": 0.95, "box": base_person_box},
        {"class_name": CLASS_NO_HARDHAT, "confidence": 0.89, "box": [120.0, 105.0, 180.0, 150.0]},
        {"class_name": CLASS_NO_VEST, "confidence": 0.86, "box": [110.0, 150.0, 190.0, 220.0]},
    ]

    result = evaluate_site_compliance(detections)
    workers = result["workers"]
    summary = result["summary"]

    assert len(workers) == 1
    w = workers[0]
    assert w["hardhat_status"] == "missing"
    assert w["vest_status"] == "missing"
    assert w["status"] == "Missing both"

    assert summary["violation_count"] == 1
    assert summary["compliance_rate"] == 0.0


def test_unknown_evidence(base_person_box):
    """With strict_hardhat=False, no-gear worker stays UNKNOWN (not a violation).

    This tests the non-strict path: reliable person + no gear boxes.
    With strict OFF, hardhat stays unknown; vest stays unknown -> Vest not visible.
    """
    detections = [
        {"class_name": CLASS_PERSON, "confidence": 0.95, "box": base_person_box},
    ]

    result = evaluate_site_compliance(detections, strict_hardhat=False)
    workers = result["workers"]
    summary = result["summary"]

    assert len(workers) == 1
    w = workers[0]
    assert w["hardhat_status"] == "unknown"
    assert w["vest_status"] == "unknown"
    # reliable person, strict OFF: status is "Vest not visible: manual check"
    assert w["status"] == "Vest not visible: manual check"
    assert w["reliability_reason"] == ""  # person IS reliable; strict is just off

    assert summary["total_workers"] == 1
    assert summary["compliant_count"] == 0
    assert summary["violation_count"] == 0
    assert summary["needs_review_count"] == 1
    assert summary["unknown_count"] == 1  # unknown_count mirrors needs_review_count


def test_one_gear_box_near_two_people():
    """Verify disambiguation: one gear box shared near two people is given only to the one with greatest overlap."""
    person_a = [100.0, 100.0, 200.0, 300.0]  # head zone: [100, 100, 200, 170]
    person_b = [180.0, 100.0, 280.0, 300.0]  # head zone: [180, 100, 280, 170]

    # Hardhat box located at [160, 105, 210, 155]:
    # Overlap with Person A: x in [160, 200] (40px) * y in [105, 155] (50px) = 2000 / 2500 = 80%
    # Overlap with Person B: x in [180, 210] (30px) * y in [105, 155] (50px) = 1500 / 2500 = 60%
    # Person A has greater overlap (0.80 > 0.60), so Person A gets the hardhat, Person B does not.
    shared_hardhat = [160.0, 105.0, 210.0, 155.0]

    detections = [
        {"class_name": CLASS_PERSON, "confidence": 0.95, "box": person_a},
        {"class_name": CLASS_PERSON, "confidence": 0.95, "box": person_b},
        {"class_name": CLASS_HARDHAT, "confidence": 0.90, "box": shared_hardhat},
    ]

    # Pass strict_hardhat=False so worker_b (no hardhat box) stays "Unknown"
    # rather than being promoted to "Missing hardhat" by the strict policy.
    result = evaluate_site_compliance(
        detections, required_gear="hardhat_only", strict_hardhat=False
    )
    workers = result["workers"]

    assert len(workers) == 2
    worker_a = workers[0]
    worker_b = workers[1]

    assert len(worker_a["assigned_gear"]) == 1
    assert worker_a["hardhat_status"] == "present"
    assert worker_a["status"] == "Compliant"

    assert len(worker_b["assigned_gear"]) == 0
    assert worker_b["hardhat_status"] == "unknown"
    assert worker_b["status"] == "Unknown"


def test_no_persons_detected():
    """Test site evaluation when no persons are detected in the image."""
    detections = [
        {"class_name": CLASS_HARDHAT, "confidence": 0.90, "box": [50.0, 50.0, 100.0, 100.0]},
        {"class_name": CLASS_VEST, "confidence": 0.88, "box": [60.0, 120.0, 140.0, 200.0]},
    ]

    result = evaluate_site_compliance(detections)
    workers = result["workers"]
    summary = result["summary"]
    unassigned = result["unassigned_gear"]

    assert len(workers) == 0
    assert len(unassigned) == 2
    assert summary["total_workers"] == 0
    assert summary["compliant_count"] == 0
    assert summary["violation_count"] == 0
    assert summary["compliance_rate"] == 100.0
    assert summary["site_status"] == "Safe"


def test_small_partial_vest_fully_inside_torso_zone(base_person_box):
    """Verify that a small partial vest box (e.g. worker wearing a bag over vest)
    is successfully assigned using intersection / gear box area (IoG).

    Person box: [100, 100, 200, 300], Torso zone: [100, 130, 200, 250].
    Small partial vest box: [120, 140, 160, 180] (area = 1600).
    IoG = 1600 / 1600 = 1.0 >= 0.3 named threshold.
    """
    small_partial_vest = [120.0, 140.0, 160.0, 180.0]
    hardhat = [120.0, 105.0, 180.0, 150.0]

    detections = [
        {"class_name": CLASS_PERSON, "confidence": 0.95, "box": base_person_box},
        {"class_name": CLASS_HARDHAT, "confidence": 0.90, "box": hardhat},
        {"class_name": CLASS_VEST, "confidence": 0.85, "box": small_partial_vest},
    ]

    result = evaluate_site_compliance(detections)
    workers = result["workers"]
    summary = result["summary"]

    assert len(workers) == 1
    w = workers[0]
    assert w["hardhat_status"] == "present"
    assert w["vest_status"] == "present"
    assert w["status"] == "Compliant"
    assert len(w["assigned_gear"]) == 2

    # Check that IoG overlap ratio recorded is 1.0 (100% of gear is in torso zone)
    vest_entry = [g for g in w["assigned_gear"] if g["class_name"] == CLASS_VEST][0]
    assert vest_entry["overlap"] == 1.0

    assert summary["compliant_count"] == 1
    assert summary["violation_count"] == 0
    assert summary["site_status"] == "Safe"


def test_small_partial_vest_straddling_torso_boundary(base_person_box):
    """Verify partial vest box overlapping torso zone boundary by >= 30% is assigned.

    Person box: [100, 100, 200, 300], Torso zone: [100, 130, 200, 250].
    Partial vest box: [120, 115, 160, 160] (width=40, height=45, area=1800).
    Intersection with torso zone (y from 130 to 160 = 30):
    Intersection area = 40 * 30 = 1200.
    IoG = 1200 / 1800 = 0.667 >= 0.3 named threshold.
    """
    partial_vest = [120.0, 115.0, 160.0, 160.0]
    hardhat = [120.0, 105.0, 180.0, 150.0]

    detections = [
        {"class_name": CLASS_PERSON, "confidence": 0.95, "box": base_person_box},
        {"class_name": CLASS_HARDHAT, "confidence": 0.92, "box": hardhat},
        {"class_name": CLASS_VEST, "confidence": 0.88, "box": partial_vest},
    ]

    result = evaluate_site_compliance(detections)
    w = result["workers"][0]

    assert w["vest_status"] == "present"
    assert w["status"] == "Compliant"
    vest_entry = [g for g in w["assigned_gear"] if g["class_name"] == CLASS_VEST][0]
    assert pytest.approx(vest_entry["overlap"], 0.01) == 0.667


def test_partial_vest_below_overlap_threshold(base_person_box):
    """Verify partial gear box with < 30% area inside torso zone is rejected.

    Person box: [100, 100, 200, 300], Torso zone: [100, 130, 200, 250].
    Stray/partial box: [120, 240, 160, 290] (width=40, height=50, area=2000).
    Intersection with torso zone (y from 240 to 250 = 10):
    Intersection area = 40 * 10 = 400.
    IoG = 400 / 2000 = 0.20 < 0.30 named threshold.
    """
    stray_vest = [120.0, 240.0, 160.0, 290.0]
    hardhat = [120.0, 105.0, 180.0, 150.0]

    detections = [
        {"class_name": CLASS_PERSON, "confidence": 0.95, "box": base_person_box},
        {"class_name": CLASS_HARDHAT, "confidence": 0.92, "box": hardhat},
        {"class_name": CLASS_VEST, "confidence": 0.80, "box": stray_vest},
    ]

    result = evaluate_site_compliance(detections)
    w = result["workers"][0]

    # Vest is not assigned because IoG 0.20 < 0.30
    assert w["vest_status"] == "unknown"
    assert len(w["assigned_gear"]) == 1
    assert len(result["unassigned_gear"]) == 1



# =============================================================================
# NEW TESTS: Strict Hardhat Policy & Person Reliability
# =============================================================================

def test_strict_on_reliable_person_no_hardhat(base_person_box):
    """Strict ON + reliable person + no hardhat box -> hardhat_status = 'missing'.

    base_person_box: [100, 100, 200, 300] -> width=100, height=200, aspect=2.0 >= 1.2.
    image_height=1000 -> height_frac=0.20 >= 0.12. confidence=0.90 >= 0.40.
    Person is reliable -> strict ON promotes unknown to missing -> violation.
    """
    detections = [
        {"class_name": CLASS_PERSON, "confidence": 0.90, "box": base_person_box},
        # No hardhat or NO-hardhat box in scene
    ]
    result = evaluate_site_compliance(
        detections,
        image_height=1000.0,
        strict_hardhat=True,
    )
    w = result["workers"][0]
    summary = result["summary"]

    assert w["hardhat_status"] == "missing"       # promoted by strict policy
    assert w["reliability_reason"] == ""          # person is reliable
    assert "Missing" in w["status"]               # counted as violation
    assert summary["violation_count"] == 1
    assert summary["compliant_count"] == 0


def test_strict_off_reliable_person_no_hardhat(base_person_box):
    """Strict OFF: even a reliable person with no hardhat box stays UNKNOWN (not a violation)."""
    detections = [
        {"class_name": CLASS_PERSON, "confidence": 0.90, "box": base_person_box},
    ]
    result = evaluate_site_compliance(
        detections,
        image_height=1000.0,
        strict_hardhat=False,
    )
    w = result["workers"][0]
    summary = result["summary"]

    assert w["hardhat_status"] == "unknown"       # stays unknown; strict is off
    assert w["reliability_reason"] == ""          # person IS reliable
    assert "Missing" not in w["status"]           # must NOT be a violation
    assert summary["violation_count"] == 0
    assert summary["needs_review_count"] == 1


def test_small_person_stays_unknown():
    """Small person box (height < MIN_PERSON_HEIGHT_FRAC * image_height) -> unreliable.

    Box: [0, 0, 50, 100] -> height=100, width=50, aspect=2.0 (upright OK).
    image_height=2000 -> height_frac = 100/2000 = 0.05 < 0.12 -> "small".
    Even with strict ON, small person stays UNKNOWN (needs review), not a violation.
    """
    small_box = [0.0, 0.0, 50.0, 100.0]  # height=100, width=50, aspect=2.0
    detections = [
        {"class_name": CLASS_PERSON, "confidence": 0.90, "box": small_box},
    ]
    result = evaluate_site_compliance(
        detections,
        image_height=2000.0,
        strict_hardhat=True,
    )
    w = result["workers"][0]
    summary = result["summary"]

    assert w["hardhat_status"] == "unknown"
    assert w["reliability_reason"] == "small"
    assert w["status"] == "Needs review: small"
    assert summary["violation_count"] == 0
    assert summary["needs_review_count"] == 1


def test_low_confidence_person_stays_unknown():
    """Low-confidence person detection (< MIN_PERSON_CONF) -> unreliable -> UNKNOWN.

    Box: [0, 0, 100, 300] -> aspect=3.0, height_frac=0.30, but conf=0.35 < 0.40.
    Strict ON still cannot promote to violation because reliability fails.
    """
    low_conf_box = [0.0, 0.0, 100.0, 300.0]  # aspect=3.0, height_frac=0.30
    detections = [
        {"class_name": CLASS_PERSON, "confidence": 0.35, "box": low_conf_box},
    ]
    result = evaluate_site_compliance(
        detections,
        image_height=1000.0,
        strict_hardhat=True,
    )
    w = result["workers"][0]
    summary = result["summary"]

    assert w["hardhat_status"] == "unknown"
    assert w["reliability_reason"] == "low confidence"
    assert w["status"] == "Needs review: low confidence"
    assert summary["violation_count"] == 0
    assert summary["needs_review_count"] == 1


def test_squashed_person_box_stays_unknown_vest_overridden():
    """Squashed / overhead-view person box (aspect < MIN_UPRIGHT_ASPECT=1.2).

    Box: [0, 0, 300, 150] -> height=150, width=300, aspect=0.5 < 1.2 -> 'view not upright'.
    Expected:
      - hardhat stays UNKNOWN (not reliable -> strict cannot promote)
      - vest is OVERRIDDEN to UNKNOWN even though a vest box is successfully
        assigned to the torso zone (geometry is unreliable for non-upright persons)
      - status = 'Needs review: view not upright'
      - NOT counted as a violation
    """
    squashed_box = [0.0, 0.0, 300.0, 150.0]   # height=150, width=300, aspect=0.5
    # Vest box that falls inside the torso zone of the squashed person:
    # Torso zone: y = 0.15*150=22.5 to 0.75*150=112.5; full x span
    # Vest: [50, 25, 250, 110] -> intersection with torso [50, 25, 250, 110] = 200*85=17000
    # Vest area = 200*85 = 17000; IoG = 1.0 >= 0.30 -> would normally be assigned
    vest_box = [50.0, 25.0, 250.0, 110.0]
    detections = [
        {"class_name": CLASS_PERSON, "confidence": 0.90, "box": squashed_box},
        {"class_name": CLASS_VEST,   "confidence": 0.80, "box": vest_box},
    ]
    result = evaluate_site_compliance(
        detections,
        image_height=1000.0,
        strict_hardhat=True,
    )
    w = result["workers"][0]
    summary = result["summary"]

    # Hardhat stays unknown (not reliable due to non-upright)
    assert w["hardhat_status"] == "unknown"

    # Vest overridden to unknown despite vest box being assigned to torso zone
    assert w["vest_status"] == "unknown"
    # Vest box IS in assigned_gear (assignment happened before override)
    assert any(g["class_name"] == CLASS_VEST for g in w["assigned_gear"])

    assert w["reliability_reason"] == "view not upright"
    assert w["status"] == "Needs review: view not upright"
    assert summary["violation_count"] == 0        # NOT a violation
    assert summary["needs_review_count"] == 1


# =============================================================================
# NEW TESTS: Conflicting Detections (Positive + Negative for same item)
# =============================================================================

def test_conflicting_hardhat_detections(base_person_box):
    """When both Hardhat and NO-Hardhat match the head zone, mark hardhat as UNKNOWN
    with reason 'conflicting detections', and overall status as
    'Needs review: conflicting detections'. Do not choose by confidence.
    """
    from src.utils import format_worker_table

    # Positive confidence is HIGHER than negative, but confidence must NOT be used to choose.
    detections = [
        {"class_name": CLASS_PERSON, "confidence": 0.95, "box": base_person_box},
        {"class_name": CLASS_HARDHAT, "confidence": 0.95, "box": [120.0, 105.0, 180.0, 150.0]},
        {"class_name": CLASS_NO_HARDHAT, "confidence": 0.70, "box": [125.0, 110.0, 175.0, 145.0]},
        {"class_name": CLASS_VEST, "confidence": 0.90, "box": [110.0, 150.0, 190.0, 220.0]},
    ]
    result = evaluate_site_compliance(detections)
    w = result["workers"][0]
    summary = result["summary"]

    assert w["hardhat_status"] == "unknown"
    assert w["hardhat_reason"] == "conflicting detections"
    assert w["vest_status"] == "present"
    assert w["status"] == "Needs review: conflicting detections"
    assert summary["compliant_count"] == 0
    assert summary["violation_count"] == 0
    assert summary["needs_review_count"] == 1

    # Verify formatting in worker table
    table_records = format_worker_table(result["workers"])
    assert table_records[0]["Hardhat"] == "Needs review: conflicting detections"
    assert table_records[0]["Safety Vest"] == "Present"
    assert table_records[0]["Overall Status"] == "Needs review: conflicting detections"


def test_conflicting_vest_detections(base_person_box):
    """When both Safety Vest and NO-Safety Vest match the torso zone, mark vest as UNKNOWN
    with reason 'conflicting detections', and overall status as
    'Needs review: conflicting detections'. Do not choose by confidence.
    """
    from src.utils import format_worker_table

    # Safety Vest confidence is HIGHER than NO-Safety Vest, but must not choose by confidence.
    detections = [
        {"class_name": CLASS_PERSON, "confidence": 0.95, "box": base_person_box},
        {"class_name": CLASS_HARDHAT, "confidence": 0.90, "box": [120.0, 105.0, 180.0, 150.0]},
        {"class_name": CLASS_VEST, "confidence": 0.92, "box": [110.0, 150.0, 190.0, 220.0]},
        {"class_name": CLASS_NO_VEST, "confidence": 0.65, "box": [115.0, 155.0, 185.0, 215.0]},
    ]
    result = evaluate_site_compliance(detections)
    w = result["workers"][0]
    summary = result["summary"]

    assert w["hardhat_status"] == "present"
    assert w["vest_status"] == "unknown"
    assert w["vest_reason"] == "conflicting detections"
    assert w["status"] == "Needs review: conflicting detections"
    assert summary["compliant_count"] == 0
    assert summary["violation_count"] == 0
    assert summary["needs_review_count"] == 1

    table_records = format_worker_table(result["workers"])
    assert table_records[0]["Hardhat"] == "Present"
    assert table_records[0]["Safety Vest"] == "Needs review: conflicting detections"
    assert table_records[0]["Overall Status"] == "Needs review: conflicting detections"


def test_conflicting_both_hardhat_and_vest(base_person_box):
    """When both Hardhat/NO-Hardhat and Vest/NO-Vest match, both items are UNKNOWN
    with reason 'conflicting detections'.
    """
    from src.utils import format_worker_table

    detections = [
        {"class_name": CLASS_PERSON, "confidence": 0.95, "box": base_person_box},
        {"class_name": CLASS_HARDHAT, "confidence": 0.88, "box": [120.0, 105.0, 180.0, 150.0]},
        {"class_name": CLASS_NO_HARDHAT, "confidence": 0.85, "box": [125.0, 110.0, 175.0, 145.0]},
        {"class_name": CLASS_VEST, "confidence": 0.80, "box": [110.0, 150.0, 190.0, 220.0]},
        {"class_name": CLASS_NO_VEST, "confidence": 0.82, "box": [115.0, 155.0, 185.0, 215.0]},
    ]
    result = evaluate_site_compliance(detections)
    w = result["workers"][0]

    assert w["hardhat_status"] == "unknown"
    assert w["hardhat_reason"] == "conflicting detections"
    assert w["vest_status"] == "unknown"
    assert w["vest_reason"] == "conflicting detections"
    assert w["status"] == "Needs review: conflicting detections"

    table_records = format_worker_table(result["workers"])
    assert table_records[0]["Hardhat"] == "Needs review: conflicting detections"
    assert table_records[0]["Safety Vest"] == "Needs review: conflicting detections"
    assert table_records[0]["Overall Status"] == "Needs review: conflicting detections"


def test_conflict_not_chosen_by_confidence_negative_higher(base_person_box):
    """Even if negative detection confidence is significantly higher than positive,
    it must NOT choose by confidence (do not mark as missing). It must stay UNKNOWN
    with reason 'conflicting detections'.
    """
    detections = [
        {"class_name": CLASS_PERSON, "confidence": 0.95, "box": base_person_box},
        # NO-Hardhat confidence 0.99 vs Hardhat confidence 0.40
        {"class_name": CLASS_HARDHAT, "confidence": 0.40, "box": [120.0, 105.0, 180.0, 150.0]},
        {"class_name": CLASS_NO_HARDHAT, "confidence": 0.99, "box": [125.0, 110.0, 175.0, 145.0]},
        # NO-Vest confidence 0.98 vs Vest confidence 0.35
        {"class_name": CLASS_VEST, "confidence": 0.35, "box": [110.0, 150.0, 190.0, 220.0]},
        {"class_name": CLASS_NO_VEST, "confidence": 0.98, "box": [115.0, 155.0, 185.0, 215.0]},
    ]
    result = evaluate_site_compliance(detections)
    w = result["workers"][0]

    # Neither should be marked as "missing" despite much higher negative confidence
    assert w["hardhat_status"] == "unknown"
    assert w["hardhat_reason"] == "conflicting detections"
    assert w["vest_status"] == "unknown"
    assert w["vest_reason"] == "conflicting detections"
    assert w["status"] == "Needs review: conflicting detections"
    assert result["summary"]["violation_count"] == 0
    assert result["summary"]["needs_review_count"] == 1


# =============================================================================
# NEW TESTS: Unreliable Guards & Zero Workers Image Annotation
# =============================================================================

def test_unreliable_person_never_deleted_stays_needs_review():
    """Unreliable person with gear boxes detected is NEVER deleted from the list,
    and their status must be 'Needs review: <reason>' (not Compliant or Violation).
    """
    # Low confidence person (0.30 < MIN_PERSON_CONF 0.40) who has Hardhat + Vest
    low_conf_box = [100.0, 100.0, 200.0, 300.0]
    detections = [
        {"class_name": CLASS_PERSON, "confidence": 0.30, "box": low_conf_box},
        {"class_name": CLASS_HARDHAT, "confidence": 0.90, "box": [120.0, 105.0, 180.0, 150.0]},
        {"class_name": CLASS_VEST, "confidence": 0.88, "box": [110.0, 150.0, 190.0, 220.0]},
    ]
    result = evaluate_site_compliance(detections, image_height=1000.0)
    workers = result["workers"]
    summary = result["summary"]

    # Must NOT be deleted
    assert len(workers) == 1
    w = workers[0]
    assert w["reliability_reason"] == "low confidence"
    assert w["status"] == "Needs review: low confidence"
    # Must NOT be counted as compliant and NOT as violation
    assert summary["compliant_count"] == 0
    assert summary["violation_count"] == 0
    assert summary["needs_review_count"] == 1


def test_zero_workers_with_loose_gear_and_annotation():
    """When zero workers are detected, evaluate_site_compliance handles it safely
    without division errors, and annotate_image retains and draws loose gear boxes.
    """
    import numpy as np
    from src.utils import annotate_image

    # Gear boxes exist, but zero persons detected (e.g. overhead or empty site)
    detections = [
        {"class_name": CLASS_HARDHAT, "confidence": 0.85, "box": [50.0, 50.0, 100.0, 100.0]},
        {"class_name": CLASS_VEST, "confidence": 0.90, "box": [150.0, 150.0, 250.0, 250.0]},
    ]
    result = evaluate_site_compliance(detections)
    workers = result["workers"]
    summary = result["summary"]
    unassigned = result["unassigned_gear"]

    assert len(workers) == 0
    assert len(unassigned) == 2
    assert summary["total_workers"] == 0
    assert summary["compliant_count"] == 0
    assert summary["violation_count"] == 0
    assert summary["needs_review_count"] == 0
    assert summary["compliance_rate"] == 100.0
    assert summary["site_status"] == "Safe"

    # Test annotation with zero workers: unassigned gear must still be drawn on canvas
    blank_image = np.zeros((400, 400, 3), dtype=np.uint8)
    annotated = annotate_image(
        image=blank_image,
        workers=workers,
        unassigned_gear=unassigned,
        show_gear_boxes=True,
    )
    assert annotated.shape == (400, 400, 3)
    # The canvas should have drawn boxes and text for the loose gear
    assert np.any(annotated > 0)


# ============================================================================
# SAFETY VEST ONLY MODE TESTS
# ============================================================================

def test_vest_only_mode_compliant(base_person_box):
    """Test that in vest_only mode, a worker with Safety Vest is Compliant even without Hardhat."""
    detections = [
        {"class_name": CLASS_PERSON, "confidence": 0.95, "box": base_person_box},
        {"class_name": CLASS_NO_HARDHAT, "confidence": 0.88, "box": [120.0, 105.0, 180.0, 150.0]},
        {"class_name": CLASS_VEST, "confidence": 0.90, "box": [110.0, 150.0, 190.0, 220.0]},
    ]
    # In 'both' mode, this worker would be a violation ('Missing hardhat')
    res_both = evaluate_site_compliance(detections, required_gear="both")
    assert res_both["workers"][0]["status"] == "Missing hardhat"
    assert res_both["summary"]["compliant_count"] == 0

    # In 'vest_only' mode, this worker must be Compliant
    res_vest = evaluate_site_compliance(detections, required_gear="vest_only")
    assert res_vest["workers"][0]["status"] == "Compliant"
    assert res_vest["summary"]["compliant_count"] == 1
    assert res_vest["summary"]["violation_count"] == 0
    assert res_vest["summary"]["compliance_rate"] == 100.0


def test_vest_only_mode_violation(base_person_box):
    """Test that in vest_only mode, a worker without Safety Vest is a violation."""
    detections = [
        {"class_name": CLASS_PERSON, "confidence": 0.95, "box": base_person_box},
        {"class_name": CLASS_HARDHAT, "confidence": 0.90, "box": [120.0, 105.0, 180.0, 150.0]},
        {"class_name": CLASS_NO_VEST, "confidence": 0.90, "box": [110.0, 150.0, 190.0, 220.0]},
    ]
    res = evaluate_site_compliance(detections, required_gear="vest_only")
    assert res["workers"][0]["status"] == "Missing vest"
    assert res["summary"]["compliant_count"] == 0
    assert res["summary"]["violation_count"] == 1
    assert res["summary"]["compliance_rate"] == 0.0
