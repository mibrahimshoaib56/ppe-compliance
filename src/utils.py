"""Utility functions for image annotation, metric formatting, and report generation.

Provides color-coded bounding box drawing for workers and PPE gear, table formatting
for UI presentation, and actionable site safety advisories.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image

# Color palette in RGB format
COLOR_COMPLIANT: Tuple[int, int, int] = (40, 200, 40)      # Vivid Green
COLOR_VIOLATION: Tuple[int, int, int] = (220, 30, 30)      # Vivid Red
COLOR_UNKNOWN: Tuple[int, int, int] = (245, 166, 35)       # Amber / Orange
COLOR_GEAR_POS: Tuple[int, int, int] = (0, 180, 240)       # Cyan / Blue
COLOR_GEAR_NEG: Tuple[int, int, int] = (255, 80, 0)        # Orange-Red
COLOR_CONTEXT: Tuple[int, int, int] = (160, 160, 160)      # Faint Grey
COLOR_TEXT_WHITE: Tuple[int, int, int] = (255, 255, 255)
COLOR_TEXT_BLACK: Tuple[int, int, int] = (0, 0, 0)


def ensure_rgb_numpy(image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
    """Ensure the image is a numpy array in RGB channel order."""
    if isinstance(image, str):
        img_bgr = cv2.imread(image)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not load image from path: {image}")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    elif isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image.copy()
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")


def draw_label_tag(
    canvas: np.ndarray,
    text: str,
    top_left: Tuple[int, int],
    bg_color: Tuple[int, int, int],
    text_color: Tuple[int, int, int] = COLOR_TEXT_WHITE,
    font_scale: float = 0.5,
    thickness: int = 1,
) -> None:
    """Draw a text label with a filled background rectangle on canvas in-place."""
    x, y = top_left
    font = cv2.FONT_HERSHEY_SIMPLEX

    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    rect_y1 = max(0, y - text_h - 6)
    rect_y2 = y
    rect_x1 = max(0, x)
    rect_x2 = min(canvas.shape[1], x + text_w + 8)

    # Draw background box for text
    cv2.rectangle(canvas, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, -1)

    # Put text inside
    text_origin = (rect_x1 + 4, rect_y2 - 4)
    cv2.putText(canvas, text, text_origin, font, font_scale, text_color, thickness, cv2.LINE_AA)


def annotate_image(
    image: Union[str, np.ndarray, Image.Image],
    workers: List[Dict[str, Any]],
    unassigned_gear: Optional[List[Dict[str, Any]]] = None,
    show_gear_boxes: bool = True,
    show_zones: bool = False,
) -> np.ndarray:
    """Annotate an image with color-coded bounding boxes and status labels.

    Args:
        image: Original image (RGB numpy array, PIL Image, or file path).
        workers: Evaluated worker list from evaluate_site_compliance().
        unassigned_gear: Optional gear detections not tied to any worker.
        show_gear_boxes: Whether to draw gear boxes (hardhats, vests).
        show_zones: Whether to outline head and torso zones (for debugging).

    Returns:
        np.ndarray: Annotated RGB image.
    """
    canvas = ensure_rgb_numpy(image)

    # 1. Optionally draw head/torso zones faintly
    if show_zones:
        for w in workers:
            hx1, hy1, hx2, hy2 = [int(v) for v in w["head_zone"]]
            tx1, ty1, tx2, ty2 = [int(v) for v in w["torso_zone"]]
            cv2.rectangle(canvas, (hx1, hy1), (hx2, hy2), (100, 100, 255), 1)
            cv2.rectangle(canvas, (tx1, ty1), (tx2, ty2), (100, 255, 100), 1)

    # 2. Draw assigned gear boxes
    if show_gear_boxes:
        for w in workers:
            for gear in w.get("assigned_gear", []):
                gx1, gy1, gx2, gy2 = [int(v) for v in gear["box"]]
                g_name = gear["class_name"]

                if "NO-" in g_name:
                    g_color = COLOR_GEAR_NEG
                else:
                    g_color = COLOR_GEAR_POS

                cv2.rectangle(canvas, (gx1, gy1), (gx2, gy2), g_color, 2)
                draw_label_tag(
                    canvas,
                    f"{g_name} {int(gear.get('confidence', 1.0) * 100)}%",
                    (gx1, gy1),
                    bg_color=g_color,
                    text_color=COLOR_TEXT_WHITE,
                    font_scale=0.4,
                )

        # Draw unassigned gear boxes faintly
        if unassigned_gear:
            for gear in unassigned_gear:
                gx1, gy1, gx2, gy2 = [int(v) for v in gear["box"]]
                cv2.rectangle(canvas, (gx1, gy1), (gx2, gy2), COLOR_CONTEXT, 1)
                draw_label_tag(
                    canvas,
                    f"{gear['class_name']} (loose)",
                    (gx1, gy1),
                    bg_color=COLOR_CONTEXT,
                    text_color=COLOR_TEXT_BLACK,
                    font_scale=0.35,
                )

    # 3. Draw worker bounding boxes on top
    for w in workers:
        x1, y1, x2, y2 = [int(v) for v in w["box"]]
        status = w["status"]

        if status == "Compliant":
            box_color = COLOR_COMPLIANT
        elif status in ("Missing hardhat", "Missing vest", "Missing both"):
            box_color = COLOR_VIOLATION
        else:
            box_color = COLOR_UNKNOWN

        # Thick boundary box for person
        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 3)

        # Worker tag
        worker_id = w.get("worker_id", 1)
        conf_pct = int(w.get("confidence", 1.0) * 100)
        label_text = f"Worker #{worker_id}: {status} ({conf_pct}%)"

        draw_label_tag(
            canvas,
            label_text,
            (x1, y1),
            bg_color=box_color,
            text_color=COLOR_TEXT_WHITE,
            font_scale=0.55,
            thickness=2,
        )

    return canvas


def format_worker_table(workers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format worker list into records suitable for a pandas DataFrame."""
    records = []
    for w in workers:
        records.append({
            "Worker ID": f"Worker #{w['worker_id']}",
            "Overall Status": w["status"],
            "Hardhat": w["hardhat_status"].capitalize(),
            "Safety Vest": w["vest_status"].capitalize(),
            "Confidence": f"{int(w.get('confidence', 1.0) * 100)}%",
        })
    return records


def get_action_recommendation(summary: Dict[str, Any]) -> str:
    """Generate an actionable advisory message based on site metrics."""
    total = summary.get("total_workers", 0)
    violations = summary.get("violation_count", 0)
    rate = summary.get("compliance_rate", 100.0)
    status = summary.get("site_status", "Safe")

    if total == 0:
        return "No workers detected on site. No immediate safety action required."

    if status == "Safe":
        return (
            f"Site safety status is SAFE ({rate}% compliance). All active workers are compliant. "
            "Maintain standard site surveillance."
        )
    elif status == "Warning":
        return (
            f"Site safety status is WARNING ({rate}% compliance, {violations} violation(s)). "
            "Instruct safety supervisors to dispatch missing PPE gear immediately."
        )
    else:
        return (
            f"CRITICAL SAFETY ALERT ({rate}% compliance, {violations} violation(s)). "
            "Severe PPE non-compliance detected. Stop ongoing hazardous tasks until all workers wear required gear."
        )
