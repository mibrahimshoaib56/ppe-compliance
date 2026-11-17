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
    """Run YOLOv8 object detection on an image.

    Args:
        image: Filepath string, numpy array (OpenCV BGR/RGB), or PIL Image.
        model: Pre-loaded YOLO instance. If None, loads using default resolution.
        conf_threshold: Minimum confidence score to keep a detection.
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

    # Run inference explicitly on CPU
    results = model.predict(
        source=image,
        conf=conf_threshold,
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
            detections.append({
                "class_name": class_name,
                "class_id": int(cls_id),
                "confidence": round(float(conf), 3),
                "box": [round(float(coord), 1) for coord in box],
            })

    return detections
