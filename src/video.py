"""Pure functions and utilities for CCTV-style Video PPE compliance processing.

Features:
- Sequential frame decoding with OpenCV.
- Proportional frame resizing to MAX_FRAME_WIDTH (640px).
- Detection and compliance evaluation on every PROCESS_EVERY_N-th frame (default: 2).
- Propagation of most recent detections and statuses onto intermediate frames
  so bounding boxes follow the movement without inter-frame tracking.
- H.264 MP4 export using imageio-ffmpeg with yuv420p pixel format, capped at OUTPUT_MAX_FPS (30).
- Executive summary aggregation and CSV reporting computed strictly on analyzed frames.
"""

import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

# ============================================================================
# NAMED CONSTANTS & LIMITS
# ============================================================================

# Maximum video duration in seconds to process (analyze at most the first 30 seconds)
MAX_VIDEO_SECONDS: int = 30

# Fallback frames per second if video metadata reports 0, negative, NaN, or invalid FPS
FALLBACK_FPS: float = 25.0

# Inference stride: run detection + compliance on every N-th frame.
# Set to 1 to analyze every frame for maximum smoothness (slower per frame on CPU,
# but produces the most fluid bounding-box motion in the output video).
PROCESS_EVERY_N: int = 1

# Maximum frame width in pixels before inference (resized proportionally).
# 416px is a standard YOLO inference size that is ~30% faster than 640px on CPU
# with minimal accuracy loss, giving noticeably more fluid output on CPU-only machines.
MAX_FRAME_WIDTH: int = 416

# Maximum output video FPS cap for H.264 mp4 encoding
OUTPUT_MAX_FPS: int = 30

# Number of highest-violation moments to highlight in the risk gallery
TOP_VIOLATION_FRAMES_COUNT: int = 3

# Number of consecutive analyzed frames used for temporal rolling smoothing (default: 7).
# Suppresses transient 1-2 frame posture/occlusion flickers without masking sustained violations.
TEMPORAL_SMOOTHING_WINDOW: int = 7


# ============================================================================
# PURE LOGIC FUNCTIONS
# ============================================================================

def validate_and_get_fps(fps: Any) -> float:
    """Validate raw video FPS and return a safe positive float, falling back to 25.0.

    Args:
        fps: Raw frames-per-second value from video metadata.

    Returns:
        float: Validated positive FPS, or FALLBACK_FPS (25.0) if invalid.
    """
    if fps is None:
        return FALLBACK_FPS
    try:
        val = float(fps)
        if val <= 0.0 or math.isnan(val) or math.isinf(val):
            return FALLBACK_FPS
        return val
    except (ValueError, TypeError):
        return FALLBACK_FPS


def calculate_video_duration(total_frames: int, fps: float) -> float:
    """Calculate video duration in seconds using safe FPS.

    Args:
        total_frames: Total number of frames in the video.
        fps: Raw or validated FPS.

    Returns:
        float: Total duration in seconds (>= 0.0).
    """
    safe_fps = validate_and_get_fps(fps)
    if total_frames <= 0:
        return 0.0
    return float(total_frames) / safe_fps


def compute_max_frames_to_process(
    total_frames: int,
    fps: float,
    max_seconds: int = MAX_VIDEO_SECONDS,
) -> int:
    """Calculate the maximum number of sequential frames to process, capped by max_seconds.

    Args:
        total_frames: Total number of frames available in the video.
        fps: Frames per second.
        max_seconds: Maximum duration in seconds to analyze (default: 30).

    Returns:
        int: Number of frames to read sequentially (min of total_frames and max_seconds * fps).
    """
    if total_frames <= 0 or max_seconds <= 0:
        return 0
    safe_fps = validate_and_get_fps(fps)
    max_cap = int(math.ceil(safe_fps * max_seconds))
    return min(total_frames, max_cap)


def compute_start_frame_index(
    start_seconds: float,
    fps: float,
    total_frames: int,
) -> int:
    """Calculate the 0-based frame index corresponding to a starting timestamp in seconds.

    Pure function that validates FPS, clamps start_seconds, and bounds the frame
    index within [0, max(0, total_frames)].

    Args:
        start_seconds: Desired starting time in seconds (>= 0.0).
        fps: Video frames per second.
        total_frames: Total number of frames in the video.

    Returns:
        int: 0-indexed frame index clamped to [0, max(0, total_frames)].
    """
    if start_seconds is None or start_seconds <= 0.0:
        return 0
    safe_fps = validate_and_get_fps(fps)
    calculated_index = int(round(float(start_seconds) * safe_fps))
    return max(0, min(calculated_index, max(0, total_frames)))


def compute_cctv_frame_range(
    total_frames: int,
    fps: float,
    start_seconds: float = 0.0,
    max_seconds: int = MAX_VIDEO_SECONDS,
) -> Tuple[int, int]:
    """Calculate the [start_frame_index, end_frame_index) range for CCTV processing.

    Pure function: Given total frames, FPS, start time, and max analysis window,
    returns the half-open slice [start_idx, end_idx) to analyze.

    Args:
        total_frames: Total available frames in the video.
        fps: Video frames per second.
        start_seconds: Start offset in seconds.
        max_seconds: Maximum duration to process in seconds from the start.

    Returns:
        Tuple[int, int]: (start_frame_index, end_frame_index) half-open range.
    """
    if total_frames <= 0 or max_seconds <= 0:
        return (0, 0)
    start_idx = compute_start_frame_index(start_seconds, fps, total_frames)
    safe_fps = validate_and_get_fps(fps)
    max_frames_to_read = int(math.ceil(safe_fps * max_seconds))
    end_idx = min(total_frames, start_idx + max_frames_to_read)
    return (start_idx, end_idx)


def is_analyzed_frame(frame_index: int, process_every_n: int = PROCESS_EVERY_N) -> bool:
    """Determine whether a given 0-indexed frame should undergo detection and compliance.

    Args:
        frame_index: 0-indexed sequential frame number.
        process_every_n: Inference stride (default: 2).

    Returns:
        bool: True if detection + compliance should run on this frame, False otherwise.
    """
    stride = max(1, int(process_every_n))
    return (frame_index % stride) == 0


def compute_resized_dimensions(
    orig_w: int,
    orig_h: int,
    max_width: int = MAX_FRAME_WIDTH,
) -> Tuple[int, int]:
    """Compute new (width, height) scaled proportionally so width <= max_width.

    Ensures dimensions are even integers for H.264 video codec compatibility.

    Args:
        orig_w: Original image/frame width.
        orig_h: Original image/frame height.
        max_width: Target maximum width constraint (default: 640).

    Returns:
        Tuple[int, int]: (new_width, new_height) where new_width <= max_width.
    """
    if orig_w <= 0 or orig_h <= 0:
        return (0, 0)

    if orig_w <= max_width:
        target_w = orig_w
        target_h = orig_h
    else:
        scale = float(max_width) / float(orig_w)
        target_w = max_width
        target_h = max(1, int(round(orig_h * scale)))

    # Ensure even dimensions for H.264 encoder compatibility (yuv420p)
    if target_w % 2 != 0:
        target_w -= 1
    if target_h % 2 != 0:
        target_h -= 1

    return (max(2, target_w), max(2, target_h))


def compute_output_fps(source_fps: float, max_fps: int = OUTPUT_MAX_FPS) -> float:
    """Compute the safe output video frame rate capped at OUTPUT_MAX_FPS (30).

    Args:
        source_fps: Native video FPS.
        max_fps: Upper ceiling for output video encoding.

    Returns:
        float: Output encoding FPS.
    """
    safe_fps = validate_and_get_fps(source_fps)
    return min(safe_fps, float(max_fps))


# ============================================================================
# SUMMARY & AGGREGATION LOGIC (COMPUTED ON ANALYZED FRAMES ONLY)
# ============================================================================

def apply_temporal_smoothing(
    analyzed_records: List[Dict[str, Any]],
    window_size: int = TEMPORAL_SMOOTHING_WINDOW,
) -> List[Dict[str, Any]]:
    """Apply rolling median temporal smoothing to CCTV analyzed frame records.

    Filters out transient single-frame detection flickers caused by brief turns,
    occlusions, or posture shifts without masking sustained safety violations.

    Args:
        analyzed_records: List of raw analyzed frame dictionaries.
        window_size: Odd integer defining the temporal rolling window size (default: 7).

    Returns:
        List[Dict[str, Any]]: New list of records with smoothed worker counts and compliance rates.
    """
    n = len(analyzed_records)
    if n == 0:
        return []

    # Ensure window_size is an odd integer >= 1
    w_size = max(1, int(window_size))
    if w_size % 2 == 0:
        w_size += 1

    half_w = w_size // 2
    smoothed_records: List[Dict[str, Any]] = []

    for i in range(n):
        start_idx = max(0, i - half_w)
        end_idx = min(n, i + half_w + 1)
        window = analyzed_records[start_idx:end_idx]

        # Extract window distributions
        worker_counts = [r.get("total_workers", 0) for r in window]
        viol_counts = [r.get("violation_count", 0) for r in window]
        comp_counts = [r.get("compliant_count", 0) for r in window]
        rev_counts = [r.get("needs_review_count", 0) for r in window]

        smoothed_total_w = int(round(float(np.median(worker_counts))))
        smoothed_viol = int(round(float(np.median(viol_counts))))
        smoothed_comp = int(round(float(np.median(comp_counts))))
        smoothed_rev = int(round(float(np.median(rev_counts))))

        # Rate smoothing: if smoothed workers > 0, compute median of available rates
        if smoothed_total_w > 0:
            rates = [
                r["compliance_rate"]
                for r in window
                if r.get("total_workers", 0) > 0 and r.get("compliance_rate") is not None
            ]
            if rates:
                smoothed_rate = round(float(np.median(rates)), 1)
            elif (smoothed_comp + smoothed_viol) > 0:
                smoothed_rate = round((smoothed_comp / (smoothed_comp + smoothed_viol)) * 100.0, 1)
            else:
                smoothed_rate = 0.0
        else:
            smoothed_comp = 0
            smoothed_viol = 0
            smoothed_rev = 0
            smoothed_rate = None

        raw_rec = analyzed_records[i]
        new_rec = dict(raw_rec)
        new_rec.update({
            "total_workers": smoothed_total_w,
            "compliant_count": smoothed_comp,
            "violation_count": smoothed_viol,
            "needs_review_count": smoothed_rev,
            "compliance_rate": smoothed_rate,
            "raw_total_workers": raw_rec.get("total_workers", 0),
            "raw_violation_count": raw_rec.get("violation_count", 0),
            "raw_compliance_rate": raw_rec.get("compliance_rate"),
        })
        smoothed_records.append(new_rec)

    return smoothed_records


def summarize_cctv_results(
    analyzed_records: List[Dict[str, Any]],
    use_smoothing: bool = False,
    smoothing_window: int = TEMPORAL_SMOOTHING_WINDOW,
) -> Dict[str, Any]:
    """Aggregate analyzed frame records into an executive CCTV audit summary.

    Computed strictly on the analyzed frames where detection + compliance ran.
    Frames with zero workers are counted separately and are NEVER treated as 100% compliant.
    Optionally applies temporal rolling smoothing to suppress single-frame posture flickers.

    Args:
        analyzed_records: List of dictionaries for analyzed frames.
        use_smoothing: Whether to apply temporal median smoothing before aggregation.
        smoothing_window: Window size for temporal smoothing if enabled (default: 7).

    Returns:
        Dict[str, Any]: Aggregated summary metrics.
    """
    if use_smoothing:
        effective_records = apply_temporal_smoothing(analyzed_records, window_size=smoothing_window)
    else:
        effective_records = analyzed_records

    frames_analyzed = len(effective_records)
    if frames_analyzed == 0:
        return {
            "frames_analyzed": 0,
            "frames_with_workers": 0,
            "frames_without_workers": 0,
            "frames_with_violations": 0,
            "average_compliance_rate": 0.0,
            "has_worker_frames": False,
            "top_violation_moments": [],
            "smoothing_applied": use_smoothing,
            "smoothing_window": smoothing_window if use_smoothing else 0,
        }

    frames_with_workers = sum(1 for f in effective_records if f.get("total_workers", 0) > 0)
    frames_without_workers = frames_analyzed - frames_with_workers
    frames_with_violations = sum(1 for f in effective_records if f.get("violation_count", 0) > 0)

    # Average compliance rate: computed strictly over frames with detected workers
    worker_rates = [
        f["compliance_rate"]
        for f in effective_records
        if f.get("total_workers", 0) > 0 and f.get("compliance_rate") is not None
    ]

    if worker_rates:
        average_compliance_rate = round(sum(worker_rates) / len(worker_rates), 1)
    else:
        average_compliance_rate = 0.0

    # Top moments (frames) with the most violations
    frames_with_any_violations = [f for f in effective_records if f.get("violation_count", 0) > 0]
    sorted_violation_frames = sorted(
        frames_with_any_violations,
        key=lambda x: x.get("violation_count", 0),
        reverse=True,
    )
    top_violation_moments = sorted_violation_frames[:TOP_VIOLATION_FRAMES_COUNT]

    return {
        "frames_analyzed": frames_analyzed,
        "frames_with_workers": frames_with_workers,
        "frames_without_workers": frames_without_workers,
        "frames_with_violations": frames_with_violations,
        "average_compliance_rate": average_compliance_rate,
        "has_worker_frames": frames_with_workers > 0,
        "top_violation_moments": top_violation_moments,
        "smoothing_applied": use_smoothing,
        "smoothing_window": smoothing_window if use_smoothing else 0,
    }


def generate_cctv_csv_report(
    analyzed_records: List[Dict[str, Any]],
    summary: Dict[str, Any],
    video_filename: str = "Uploaded Video",
) -> str:
    """Generate a structured CSV audit report for CCTV video inspections.

    Args:
        analyzed_records: List of analyzed frame dictionaries.
        summary: Aggregated summary from summarize_cctv_results().
        video_filename: Name of the uploaded video file.

    Returns:
        str: CSV formatted report content.
    """
    lines = [
        "CCTV VIDEO PPE COMPLIANCE AUDIT REPORT",
        f"Source Video,{video_filename}",
        f"Inference Stride,Every {PROCESS_EVERY_N} frames",
        f"Total Frames Analyzed,{summary.get('frames_analyzed', 0)}",
        f"Frames With Workers,{summary.get('frames_with_workers', 0)}",
        f"Frames Without Workers (Empty/Overhead),{summary.get('frames_without_workers', 0)}",
        f"Frames With Violations,{summary.get('frames_with_violations', 0)}",
        f"Average Compliance Rate (Worker Frames Only),{summary.get('average_compliance_rate', 0.0)}%",
        "",
        "ANALYZED FRAMES BREAKDOWN",
        "Analysis #,Frame Index,Time (s),Workers Detected,Compliant,Violations,Needs Review,Compliance Rate (%)",
    ]

    for idx, f in enumerate(analyzed_records, start=1):
        rate = f.get("compliance_rate")
        rate_str = f"{rate:.1f}%" if (rate is not None and f.get("total_workers", 0) > 0) else "N/A (No Workers)"
        lines.append(
            f"{idx},{f.get('frame_number', 0)},{f.get('timestamp_sec', 0.0):.2f},"
            f"{f.get('total_workers', 0)},{f.get('compliant_count', 0)},"
            f"{f.get('violation_count', 0)},{f.get('needs_review_count', 0)},{rate_str}"
        )

    return "\n".join(lines)


# ============================================================================
# VIDEO ENCODING HELPER
# ============================================================================

def write_annotated_h264_video(
    frames_rgb: List[np.ndarray],
    output_path: str,
    fps: float,
) -> None:
    """Write a sequence of RGB numpy frames to an H.264 MP4 file using imageio.

    Uses pixel format yuv420p for universal browser playback in Streamlit.

    Args:
        frames_rgb: List of RGB uint8 numpy image arrays.
        output_path: Destination path for the .mp4 file.
        fps: Target frame rate for video playback.

    Raises:
        ValueError: If frames_rgb is empty.
        RuntimeError: If video writing encounters an encoder error.
    """
    if not frames_rgb:
        raise ValueError("Cannot write an empty list of frames to video.")

    try:
        import imageio.v2 as imageio
    except ImportError:
        import imageio

    safe_fps = compute_output_fps(fps)

    writer = imageio.get_writer(
        output_path,
        format="FFMPEG",
        mode="I",
        fps=safe_fps,
        codec="libx264",
        pixelformat="yuv420p",
    )

    try:
        for frame in frames_rgb:
            writer.append_data(frame)
    finally:
        writer.close()
