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


def compute_box_thickness(frame_width: int) -> int:
    """Compute bounding box border thickness proportional to frame width.

    Scales linearly so boxes look equally fine regardless of resolution:
    - 416px wide  -> 1px  (thin, clean)
    - 640px wide  -> 2px  (standard)
    - 1280px wide -> 3px  (high-res)

    Args:
        frame_width: Width of the frame in pixels.

    Returns:
        int: Border thickness in pixels (minimum 1).
    """
    # 1px per 400px of frame width, clamped between 1 and 4
    return max(1, min(4, round(frame_width / 400)))


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
    frame_w = canvas.shape[1]

    # Proportional thickness: 1px per 400px of frame width, clamped to [1, 4]
    person_thick = compute_box_thickness(frame_w)       # e.g. 2 at 640px
    gear_thick = max(1, person_thick - 1)               # 1 thinner than person box
    # Font scale also proportional so labels stay readable at any resolution
    base_font = max(0.3, frame_w / 1200)

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

                cv2.rectangle(canvas, (gx1, gy1), (gx2, gy2), g_color, gear_thick)
                draw_label_tag(
                    canvas,
                    f"{g_name} {int(gear.get('confidence', 1.0) * 100)}%",
                    (gx1, gy1),
                    bg_color=g_color,
                    text_color=COLOR_TEXT_WHITE,
                    font_scale=max(0.3, base_font - 0.08),
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
                    font_scale=max(0.28, base_font - 0.12),
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

        # Proportional border thickness for person box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, person_thick)

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
            font_scale=max(0.35, base_font),
            thickness=max(1, person_thick - 1),
        )

    return canvas


def format_worker_table(workers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format worker list into records suitable for a pandas DataFrame."""
    records = []
    for w in workers:
        rel_reason = w.get("reliability_reason", "")
        h_reason = w.get("hardhat_reason", "")
        v_reason = w.get("vest_reason", "")

        # Hardhat column: show "Needs review: <reason>" when hardhat has a reason
        # (e.g. conflicting detections or unreliable person); otherwise use capitalised status.
        hardhat_raw = w["hardhat_status"]
        if hardhat_raw == "unknown" and h_reason:
            hardhat_display = f"Needs review: {h_reason}"
        elif hardhat_raw == "unknown" and rel_reason:
            hardhat_display = f"Needs review: {rel_reason}"
        else:
            hardhat_display = hardhat_raw.capitalize()

        # Vest column: "Needs review: <reason>" if vest has a reason or unreliable person;
        # "Not visible — check manually" if reliable but vest undetected;
        # otherwise the capitalised vest status.
        vest_raw = w["vest_status"]
        if vest_raw == "unknown" and v_reason:
            vest_display = f"Needs review: {v_reason}"
        elif vest_raw == "unknown" and rel_reason:
            vest_display = f"Needs review: {rel_reason}"
        elif vest_raw == "unknown":
            vest_display = "Not visible — check manually"
        else:
            vest_display = vest_raw.capitalize()

        records.append({
            "Worker ID": f"Worker #{w['worker_id']}",
            "Overall Status": w["status"],
            "Hardhat": hardhat_display,
            "Safety Vest": vest_display,
            "Confidence": f"{int(w.get('confidence', 1.0) * 100)}%",
        })
    return records


def crop_worker_roi(
    image: Union[str, np.ndarray, Image.Image],
    worker: Dict[str, Any],
    padding_pct: float = 0.08,
    show_zones: bool = True,
) -> np.ndarray:
    """Extract a cropped view of an individual worker with zone and gear overlays.

    Args:
        image: Original RGB image.
        worker: Worker dictionary containing box, head_zone, torso_zone, and assigned_gear.
        padding_pct: Percentage of width/height to pad around the person crop.
        show_zones: Whether to outline the anatomical head and torso zones on the crop.

    Returns:
        np.ndarray: Cropped and annotated RGB image for the selected worker.
    """
    canvas = ensure_rgb_numpy(image)
    h, w, _ = canvas.shape

    px1, py1, px2, py2 = [int(v) for v in worker["box"]]
    pw = max(1, px2 - px1)
    ph = max(1, py2 - py1)

    pad_x = int(pw * padding_pct)
    pad_y = int(ph * padding_pct)

    crop_x1 = max(0, px1 - pad_x)
    crop_y1 = max(0, py1 - pad_y)
    crop_x2 = min(w, px2 + pad_x)
    crop_y2 = min(h, py2 + pad_y)

    crop = canvas[crop_y1:crop_y2, crop_x1:crop_x2].copy()

    # Offset helper to map full-image coordinates to crop coordinates
    def to_crop(box: List[float]) -> Tuple[int, int, int, int]:
        bx1, by1, bx2, by2 = [int(v) for v in box]
        return (
            max(0, bx1 - crop_x1),
            max(0, by1 - crop_y1),
            min(crop.shape[1], bx2 - crop_x1),
            min(crop.shape[0], by2 - crop_y1),
        )

    # Draw anatomical zones
    if show_zones:
        hx1, hy1, hx2, hy2 = to_crop(worker["head_zone"])
        tx1, ty1, tx2, ty2 = to_crop(worker["torso_zone"])

        # Head zone overlay (blue outline)
        cv2.rectangle(crop, (hx1, hy1), (hx2, hy2), (0, 180, 255), 2)
        draw_label_tag(crop, "Head Zone (0-35%)", (hx1, hy1), bg_color=(0, 180, 255), font_scale=0.35)

        # Torso zone overlay (purple outline)
        cv2.rectangle(crop, (tx1, ty1), (tx2, ty2), (180, 100, 255), 2)
        draw_label_tag(crop, "Torso Zone (15-75%)", (tx1, hy1), bg_color=(180, 100, 255), font_scale=0.35)

    # Draw assigned gear inside this crop
    for gear in worker.get("assigned_gear", []):
        gx1, gy1, gx2, gy2 = to_crop(gear["box"])
        g_name = gear["class_name"]
        g_color = COLOR_GEAR_NEG if "NO-" in g_name else COLOR_GEAR_POS

        cv2.rectangle(crop, (gx1, gy1), (gx2, gy2), g_color, 2)
        draw_label_tag(
            crop,
            f"{g_name} ({int(gear.get('confidence', 1.0) * 100)}%)",
            (gx1, gy1),
            bg_color=g_color,
            font_scale=0.35,
        )

    return crop


def generate_csv_report(
    summary: Dict[str, Any],
    workers: List[Dict[str, Any]],
    timestamp_str: str,
) -> str:
    """Generate a CSV safety inspection audit report string."""
    lines = [
        "CONSTRUCTION SITE PPE COMPLIANCE AUDIT REPORT",
        f"Generated At,{timestamp_str}",
        f"Site Status,{summary.get('site_status', 'N/A')}",
        f"Compliance Rate,{summary.get('compliance_rate', 0.0)}%",
        f"Total Workers Detected,{summary.get('total_workers', 0)}",
        f"Compliant Workers,{summary.get('compliant_count', 0)}",
        f"Violations,{summary.get('violation_count', 0)}",
        f"Needs Review (Manual Check),{summary.get('needs_review_count', summary.get('unknown_count', 0))}",
        "",
        "INDIVIDUAL WORKER AUDIT BREAKDOWN",
        "Worker ID,Overall Status,Hardhat Status,Vest Status,Confidence,Assigned Gear",
    ]

    for w in workers:
        gear_names = "; ".join([g["class_name"] for g in w.get("assigned_gear", [])])
        if not gear_names:
            gear_names = "None"
        lines.append(
            f"Worker #{w['worker_id']},{w['status']},{w['hardhat_status'].capitalize()},"
            f"{w['vest_status'].capitalize()},{int(w.get('confidence', 1.0) * 100)}%,\"{gear_names}\""
        )

    return "\n".join(lines)


def encode_image_bytes(image: np.ndarray, format_ext: str = "JPEG") -> bytes:
    """Convert an RGB numpy array to downloadable image bytes."""
    pil_img = Image.fromarray(image)
    import io
    buf = io.BytesIO()
    save_fmt = "PNG" if format_ext.upper() == "PNG" else "JPEG"
    pil_img.save(buf, format=save_fmt, quality=95)
    return buf.getvalue()


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
