"""Pure functions and utilities for video PPE compliance processing.

Handles video metadata reading, frame sampling at fixed intervals,
summary aggregation, report generation, and error checking.
Decoupled from Streamlit to allow pure unit testing.
"""

import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

# ============================================================================
# NAMED CONSTANTS & LIMITS
# ============================================================================

# Maximum video duration in seconds before displaying a duration warning / capping
MAX_VIDEO_SECONDS: int = 60

# Default FPS used as fallback if video metadata reports <= 0, NaN, or invalid frame rate
FALLBACK_FPS: float = 25.0

# Sampling frequency: sample 1 frame every (1 / SAMPLE_FPS) seconds (1 frame per second)
SAMPLE_FPS: float = 1.0

# Hard cap on the maximum number of frames sampled and analyzed per video
MAX_SAMPLED_FRAMES: int = 60

# Number of top violation frames highlighted in the visual risk gallery
TOP_VIOLATION_FRAMES_COUNT: int = 3


# ============================================================================
# PURE SAMPLING & METADATA FUNCTIONS
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


def compute_sample_frame_indices(
    total_frames: int,
    fps: float,
    sample_fps: float = SAMPLE_FPS,
    max_frames: int = MAX_SAMPLED_FRAMES,
) -> List[int]:
    """Compute frame indices to sample at fixed time intervals, up to max_frames.

    Samples one frame every (1 / sample_fps) seconds.

    Args:
        total_frames: Total frame count of the video.
        fps: Video frames per second.
        sample_fps: Sampling frequency in frames per second (default: 1.0).
        max_frames: Hard ceiling on the count of sampled frames (default: 60).

    Returns:
        List[int]: Strictly increasing list of integer frame indices to extract.
    """
    if total_frames <= 0 or max_frames <= 0:
        return []

    safe_fps = validate_and_get_fps(fps)
    safe_sample_fps = max(0.001, float(sample_fps))

    # Step interval in frames (e.g. 25 fps / 1.0 sample_fps = 25 frames between samples)
    step = max(1, int(round(safe_fps / safe_sample_fps)))

    indices: List[int] = []
    current_idx = 0
    while current_idx < total_frames and len(indices) < max_frames:
        indices.append(current_idx)
        current_idx += step

    return indices


# ============================================================================
# SUMMARY & AGGREGATION LOGIC
# ============================================================================

def summarize_video_results(frame_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-frame evaluation records into an executive video audit summary.

    Note on compliance rate: frames with zero workers are counted separately
    and are NOT treated as 100% compliant. Average compliance rate is computed
    strictly over frames that contained at least one worker.

    Args:
        frame_records: List of per-frame dictionaries, each containing:
            - 'frame_number': int
            - 'timestamp_sec': float
            - 'total_workers': int
            - 'compliant_count': int
            - 'violation_count': int
            - 'needs_review_count': int
            - 'compliance_rate': Optional[float]

    Returns:
        Dict[str, Any]: Aggregated summary metrics.
    """
    total_frames_analyzed = len(frame_records)
    if total_frames_analyzed == 0:
        return {
            "total_frames_analyzed": 0,
            "frames_with_workers": 0,
            "frames_without_workers": 0,
            "frames_with_violations": 0,
            "average_compliance_rate": 0.0,
            "has_worker_frames": False,
            "top_violation_frames": [],
        }

    frames_with_workers = sum(1 for f in frame_records if f.get("total_workers", 0) > 0)
    frames_without_workers = total_frames_analyzed - frames_with_workers
    frames_with_violations = sum(1 for f in frame_records if f.get("violation_count", 0) > 0)

    # Average compliance rate: strictly over frames with detected workers
    worker_frame_rates = [
        f["compliance_rate"]
        for f in frame_records
        if f.get("total_workers", 0) > 0 and f.get("compliance_rate") is not None
    ]

    if worker_frame_rates:
        average_compliance_rate = round(sum(worker_frame_rates) / len(worker_frame_rates), 1)
    else:
        average_compliance_rate = 0.0

    # Top violation frames: sorted by violation_count descending (highest risk first)
    frames_with_any_violations = [f for f in frame_records if f.get("violation_count", 0) > 0]
    sorted_violation_frames = sorted(
        frames_with_any_violations,
        key=lambda x: x.get("violation_count", 0),
        reverse=True,
    )
    top_violation_frames = sorted_violation_frames[:TOP_VIOLATION_FRAMES_COUNT]

    return {
        "total_frames_analyzed": total_frames_analyzed,
        "frames_with_workers": frames_with_workers,
        "frames_without_workers": frames_without_workers,
        "frames_with_violations": frames_with_violations,
        "average_compliance_rate": average_compliance_rate,
        "has_worker_frames": frames_with_workers > 0,
        "top_violation_frames": top_violation_frames,
    }


def generate_video_csv_report(
    frame_records: List[Dict[str, Any]],
    summary: Dict[str, Any],
    video_filename: str = "Uploaded Video",
) -> str:
    """Generate a structured CSV audit report string for video inspections.

    Args:
        frame_records: List of evaluated frame dictionaries.
        summary: Aggregated summary dictionary from summarize_video_results().
        video_filename: Label/name of the source video file.

    Returns:
        str: CSV formatted report content.
    """
    lines = [
        "CONSTRUCTION SITE VIDEO PPE COMPLIANCE AUDIT REPORT",
        f"Source Video,{video_filename}",
        f"Total Sampled Frames Analyzed,{summary.get('total_frames_analyzed', 0)}",
        f"Frames With Workers,{summary.get('frames_with_workers', 0)}",
        f"Frames Without Workers (Empty/Overhead),{summary.get('frames_without_workers', 0)}",
        f"Frames With Violations,{summary.get('frames_with_violations', 0)}",
        f"Average Compliance Rate (Worker Frames Only),{summary.get('average_compliance_rate', 0.0)}%",
        "",
        "FRAME-BY-FRAME INSPECTION BREAKDOWN",
        "Sample #,Frame Index,Timestamp (s),Total Workers,Compliant Workers,Violations,Needs Review,Compliance Rate (%)",
    ]

    for idx, f in enumerate(frame_records, start=1):
        rate = f.get("compliance_rate")
        rate_str = f"{rate:.1f}%" if (rate is not None and f.get("total_workers", 0) > 0) else "N/A"
        lines.append(
            f"{idx},{f.get('frame_number', 0)},{f.get('timestamp_sec', 0.0):.1f},"
            f"{f.get('total_workers', 0)},{f.get('compliant_count', 0)},"
            f"{f.get('violation_count', 0)},{f.get('needs_review_count', 0)},{rate_str}"
        )

    return "\n".join(lines)


# ============================================================================
# OPENCV VIDEO IO HELPERS
# ============================================================================

def read_video_metadata(video_path: str) -> Dict[str, Any]:
    """Read video file metadata (FPS, frame count, duration).

    Args:
        video_path: Local filesystem path to video file.

    Returns:
        Dict[str, Any]: Metadata dictionary.

    Raises:
        ValueError: If the file cannot be opened by cv2.VideoCapture.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at path: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"OpenCV could not open video stream for file: {video_path}")

    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    raw_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    safe_fps = validate_and_get_fps(raw_fps)
    total_frames = max(0, raw_frames)
    duration_sec = calculate_video_duration(total_frames, safe_fps)

    cap.release()

    return {
        "raw_fps": raw_fps,
        "safe_fps": safe_fps,
        "total_frames": total_frames,
        "duration_sec": duration_sec,
    }


def extract_rgb_frame_at_index(cap: cv2.VideoCapture, frame_index: int) -> Optional[np.ndarray]:
    """Seek to a specified frame index and read the image in RGB color order.

    Args:
        cap: Active cv2.VideoCapture instance.
        frame_index: Target frame number (0-indexed).

    Returns:
        Optional[np.ndarray]: RGB numpy image array, or None if reading failed.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, bgr_frame = cap.read()
    if not ret or bgr_frame is None or bgr_frame.size == 0:
        return None
    return cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
