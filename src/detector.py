"""Object detection module using Ultralytics YOLOv8.

Handles model loading (with CPU support and fallback handling) and formats
inference outputs into structured detection dictionaries.
"""

import os
from typing import Any, Dict, List, Optional, Union
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Default weights path. Falls back to pretrained yolov8n.pt if best.pt is not present.
DEFAULT_MODEL_PATH: str = os.path.join("models", "best.pt")
FALLBACK_MODEL_PATH: str = "yolov8n.pt"

# The 10 official dataset class IDs specified in PROJECT_CONTEXT.md
EXPECTED_CLASSES: Dict[int, str] = {
    0: "Hardhat",
    1: "Mask",
    2: "NO-Hardhat",
    3: "NO-Mask",
    4: "NO-Safety Vest",
    5: "Person",
    6: "Safety Cone",
    7: "Safety Vest",
    8: "machinery",
    9: "vehicle",
}

# ============================================================================
# NAMED PER-CLASS CONFIDENCE THRESHOLDS & INFERENCE CONFIGURATION
# ============================================================================

# Low base confidence threshold for the initial YOLO prediction pass.
# Running YOLO at 0.10 ensures small equipment (hardhats, partial vests) are not
# prematurely eliminated by internal non-maximum suppression (NMS).
BASE_YOLO_CONFIDENCE: float = 0.10

# Per-class confidence thresholds applied after the base YOLO pass:
# Higher threshold for Person (0.30) to prevent background clutter false positives.
# Lower thresholds (0.20) for negative and vest classes to maximize safety sensitivity.
CONF_PERSON: float = 0.30
CONF_HARDHAT: float = 0.30
CONF_NO_HARDHAT: float = 0.20
CONF_SAFETY_VEST: float = 0.20
CONF_NO_SAFETY_VEST: float = 0.20

# Default threshold for contextual / unlisted classes (e.g. machinery, vehicle, cone, mask)
CONF_DEFAULT: float = 0.20

CLASS_CONF_THRESHOLDS: Dict[str, float] = {
    "Person": CONF_PERSON,
    "Hardhat": CONF_HARDHAT,
    "NO-Hardhat": CONF_NO_HARDHAT,
    "Safety Vest": CONF_SAFETY_VEST,
    "NO-Safety Vest": CONF_NO_SAFETY_VEST,
}


def get_class_threshold(class_name: str, scale: float = 1.0) -> float:
    """Get the effective confidence threshold for a given class name.

    Args:
        class_name: Normalized or raw class name string (e.g. 'Person', 'Hardhat').
        scale: Multiplier from UI sensitivity slider (1.0 = base named thresholds).

    Returns:
        float: Effective threshold clamped between 0.05 and 0.95.
    """
    cleaned = class_name.strip().lower()
    mapping = {
        "person": CONF_PERSON,
        "hardhat": CONF_HARDHAT,
        "no-hardhat": CONF_NO_HARDHAT,
        "safety vest": CONF_SAFETY_VEST,
        "safety-vest": CONF_SAFETY_VEST,
        "vest": CONF_SAFETY_VEST,
        "no-safety vest": CONF_NO_SAFETY_VEST,
        "no-safety-vest": CONF_NO_SAFETY_VEST,
        "no-vest": CONF_NO_SAFETY_VEST,
    }
    base_thresh = mapping.get(cleaned, CONF_DEFAULT)
    scaled_thresh = base_thresh * scale
    return max(0.05, min(0.95, round(scaled_thresh, 3)))


def verify_model_classes(model: YOLO) -> bool:
    """Verify that model.names strictly matches the 10 classes in PROJECT_CONTEXT.md.

    Returns:
        bool: True if all 10 class IDs and names match exactly, False otherwise.
    """
    model_names = model.names
    if not isinstance(model_names, dict):
        return False

    for class_id, class_name in EXPECTED_CLASSES.items():
        if model_names.get(class_id) != class_name:
            return False

    return len(model_names) == len(EXPECTED_CLASSES)


def resolve_model_path(model_path: Optional[str] = None) -> str:
    """Resolve which model weight file to use.

    If a custom model path is given and exists, use it.
    Otherwise, if models/best.pt exists, use it.
    Otherwise, fall back to yolov8n.pt for pipeline testing.
    """
    if model_path and os.path.exists(model_path):
        return model_path

    if os.path.exists(DEFAULT_MODEL_PATH):
        return DEFAULT_MODEL_PATH

    return FALLBACK_MODEL_PATH


def load_detector_model(model_path: Optional[str] = None) -> YOLO:
    """Load a YOLOv8 model instance.

    Args:
        model_path: Path to .pt weights file. If None, resolves automatically.

    Returns:
        YOLO: Loaded Ultralytics YOLO model ready for inference.
    """
    selected_path = resolve_model_path(model_path)
    model = YOLO(selected_path)
    return model


def run_detection(
    image: Union[str, np.ndarray, Image.Image],
    model: Optional[YOLO] = None,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    device: str = "cpu",
) -> List[Dict[str, Any]]:
    """Run YOLOv8 object detection on an image with two-stage confidence filtering.

    1. Executes YOLO inference at a low base confidence (BASE_YOLO_CONFIDENCE = 0.10)
       so that small gear boxes (hardhats, vests) are not discarded prematurely.
    2. Filters raw detections per-class using named constants scaled by the UI slider:
       - Person: 0.30
       - Hardhat: 0.30
       - NO-Hardhat: 0.20
       - Safety Vest: 0.20
       - NO-Safety Vest: 0.20

    Args:
        image: Filepath string, numpy array (OpenCV BGR/RGB), or PIL Image.
        model: Pre-loaded YOLO instance. If None, loads using default resolution.
        conf_threshold: Sensitivity threshold from UI slider (default 0.25 = 1.0x scale).
        iou_threshold: NMS IoU threshold.
        device: Device to run inference on (enforces 'cpu' for our requirements).

    Returns:
        List of detection dictionaries, each with:
            - 'class_name': str
            - 'class_id': int
            - 'confidence': float
            - 'box': [x1, y1, x2, y2] in pixel coordinates
    """
    if model is None:
        model = load_detector_model()

    # Scaling factor from the UI slider (base value 0.25 corresponds to 1.0x scale)
    conf_scale = (conf_threshold / 0.25) if (conf_threshold and conf_threshold > 0) else 1.0

    # Step 1: Run YOLO at low base confidence (0.10) to retain candidate gear boxes
    results = model.predict(
        source=image,
        conf=BASE_YOLO_CONFIDENCE,
        iou=iou_threshold,
        device=device,
        verbose=False,
    )

    detections: List[Dict[str, Any]] = []

    if not results or len(results) == 0:
        return detections

    result = results[0]
    names = result.names  # Class ID to name mapping dictionary

    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

        for box, conf, cls_id in zip(boxes, confs, classes):
            class_name = names.get(cls_id, str(cls_id))
            conf_val = float(conf)

            # Step 2: Per-class threshold filtering using named constants
            effective_thresh = get_class_threshold(class_name, scale=conf_scale)
            if conf_val >= effective_thresh:
                detections.append({
                    "class_name": class_name,
                    "class_id": int(cls_id),
                    "confidence": round(conf_val, 3),
                    "box": [round(float(coord), 1) for coord in box],
                })

    return detections
