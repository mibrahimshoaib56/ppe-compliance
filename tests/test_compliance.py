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
    evaluate_site_compliance,
    extract_person_zones,
    compute_overlap_ratio,
)


@pytest.fixture
def base_person_box():
    """Standard synthetic person box: width=100, height=200."""
    return [100.0, 100.0, 200.0, 300.0]


def test_zone_extraction(base_person_box):
    """Verify head (0% to 35%) and torso (20% to 65%) zone calculation."""
    head_zone, torso_zone = extract_person_zones(base_person_box)

    # Height = 200. Head zone: py1 + (0.0 to 0.35) * 200 = 100 to 170
    assert head_zone == [100.0, 100.0, 200.0, 170.0]

    # Torso zone: py1 + (0.20 to 0.65) * 200 = 140 to 230
    assert torso_zone == [100.0, 140.0, 200.0, 230.0]


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
    """Test worker with no gear detections in either head or torso zone."""
    detections = [
        {"class_name": CLASS_PERSON, "confidence": 0.95, "box": base_person_box},
    ]

    result = evaluate_site_compliance(detections)
    workers = result["workers"]
    summary = result["summary"]

    assert len(workers) == 1
    w = workers[0]
    assert w["hardhat_status"] == "unknown"
    assert w["vest_status"] == "unknown"
    assert w["status"] == "Unknown"

    assert summary["total_workers"] == 1
    assert summary["compliant_count"] == 0
    assert summary["violation_count"] == 0
    assert summary["unknown_count"] == 1


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

    result = evaluate_site_compliance(detections, required_gear="hardhat_only")
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
