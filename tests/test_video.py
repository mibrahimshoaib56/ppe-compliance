"""Unit tests for video PPE compliance processing module (src/video.py).

Verifies frame index sampling across different frame rates, FPS fallback handling,
maximum frame capping, summary aggregation for empty and mixed frame lists,
and end-to-end video decoding with a synthetic OpenCV video.
"""

import math
import os
import tempfile
import cv2
import numpy as np
import pytest

from src.video import (
    FALLBACK_FPS,
    MAX_SAMPLED_FRAMES,
    MAX_VIDEO_SECONDS,
    SAMPLE_FPS,
    TOP_VIOLATION_FRAMES_COUNT,
    calculate_video_duration,
    compute_sample_frame_indices,
    extract_rgb_frame_at_index,
    generate_video_csv_report,
    read_video_metadata,
    summarize_video_results,
    validate_and_get_fps,
)


# ============================================================================
# 1. FPS VALIDATION & FALLBACK TESTS
# ============================================================================

def test_fps_validation_valid():
    """Verify that standard positive FPS values are preserved as floats."""
    assert validate_and_get_fps(30.0) == 30.0
    assert validate_and_get_fps(29.97) == 29.97
    assert validate_and_get_fps(60) == 60.0
    assert validate_and_get_fps("24.0") == 24.0


def test_fps_validation_fallback_cases():
    """Verify that 0, negative, None, NaN, inf, and invalid strings fall back to FALLBACK_FPS (25.0)."""
    assert validate_and_get_fps(0) == FALLBACK_FPS
    assert validate_and_get_fps(0.0) == FALLBACK_FPS
    assert validate_and_get_fps(-15.0) == FALLBACK_FPS
    assert validate_and_get_fps(None) == FALLBACK_FPS
    assert validate_and_get_fps(float("nan")) == FALLBACK_FPS
    assert validate_and_get_fps(float("inf")) == FALLBACK_FPS
    assert validate_and_get_fps("invalid_fps") == FALLBACK_FPS


# ============================================================================
# 2. FRAME SAMPLING & CAP TESTS
# ============================================================================

def test_frame_index_sampling_different_fps():
    """Verify frame index sampling step calculation for 25 FPS, 30 FPS, and 10 FPS."""
    # At 25 fps, sampling 1 frame/sec -> step = 25 -> indices: 0, 25, 50, 75
    indices_25 = compute_sample_frame_indices(total_frames=100, fps=25.0, sample_fps=1.0)
    assert indices_25 == [0, 25, 50, 75]

    # At 30 fps, sampling 1 frame/sec -> step = 30 -> indices: 0, 30, 60, 90
    indices_30 = compute_sample_frame_indices(total_frames=100, fps=30.0, sample_fps=1.0)
    assert indices_30 == [0, 30, 60, 90]

    # At 10 fps, sampling 2 frames/sec (every 0.5s) -> step = 5 -> indices: 0, 5, 10, 15
    indices_10_half = compute_sample_frame_indices(total_frames=20, fps=10.0, sample_fps=2.0)
    assert indices_10_half == [0, 5, 10, 15]


def test_frame_index_sampling_with_fps_fallback():
    """Verify that invalid FPS (e.g. 0) correctly falls back to 25 FPS during sampling."""
    # 0 fps should fall back to 25 fps -> step = 25
    indices = compute_sample_frame_indices(total_frames=60, fps=0, sample_fps=1.0)
    assert indices == [0, 25, 50]


def test_frame_index_sampling_cap():
    """Verify that sampling is strictly capped at max_frames (e.g. MAX_SAMPLED_FRAMES=60)."""
    # A long video with 5000 frames at 25 fps -> without cap would have 200 samples
    indices = compute_sample_frame_indices(total_frames=5000, fps=25.0, max_frames=MAX_SAMPLED_FRAMES)
    assert len(indices) == MAX_SAMPLED_FRAMES
    assert len(indices) == 60
    assert indices[0] == 0
    assert indices[1] == 25
    assert indices[-1] == 25 * 59


def test_frame_index_sampling_empty_or_zero():
    """Verify that 0 or negative total_frames returns an empty list."""
    assert compute_sample_frame_indices(total_frames=0, fps=30.0) == []
    assert compute_sample_frame_indices(total_frames=-10, fps=30.0) == []


def test_calculate_video_duration():
    """Verify video duration calculation with valid and fallback FPS."""
    # 300 frames at 30 fps = 10.0 seconds
    assert calculate_video_duration(300, 30.0) == 10.0
    # 100 frames with invalid 0 fps -> fallback 25 fps = 4.0 seconds
    assert calculate_video_duration(100, 0) == 4.0
    # 0 frames -> 0.0 seconds
    assert calculate_video_duration(0, 25.0) == 0.0


# ============================================================================
# 3. VIDEO SUMMARY AGGREGATION TESTS
# ============================================================================

def test_summary_empty_frame_list():
    """Verify summary output when zero frames are analyzed."""
    summary = summarize_video_results([])
    assert summary["total_frames_analyzed"] == 0
    assert summary["frames_with_workers"] == 0
    assert summary["frames_without_workers"] == 0
    assert summary["frames_with_violations"] == 0
    assert summary["average_compliance_rate"] == 0.0
    assert summary["has_worker_frames"] is False
    assert summary["top_violation_frames"] == []


def test_summary_all_frames_no_workers():
    """Verify that frames with no workers are counted separately and NOT treated as 100% compliant."""
    empty_frames = [
        {"frame_number": 0, "timestamp_sec": 0.0, "total_workers": 0, "compliant_count": 0, "violation_count": 0, "needs_review_count": 0, "compliance_rate": None},
        {"frame_number": 25, "timestamp_sec": 1.0, "total_workers": 0, "compliant_count": 0, "violation_count": 0, "needs_review_count": 0, "compliance_rate": None},
        {"frame_number": 50, "timestamp_sec": 2.0, "total_workers": 0, "compliant_count": 0, "violation_count": 0, "needs_review_count": 0, "compliance_rate": None},
    ]
    summary = summarize_video_results(empty_frames)

    assert summary["total_frames_analyzed"] == 3
    assert summary["frames_with_workers"] == 0
    assert summary["frames_without_workers"] == 3
    assert summary["frames_with_violations"] == 0
    # Average compliance rate must NOT be 100%
    assert summary["average_compliance_rate"] == 0.0
    assert summary["has_worker_frames"] is False
    assert summary["top_violation_frames"] == []


def test_summary_mixed_frames():
    """Verify summary aggregation with mixed frames (some with workers, some without, some with violations)."""
    mixed_frames = [
        # Frame 1: 2 workers, 2 compliant, 0 violations -> 100%
        {"frame_number": 0, "timestamp_sec": 0.0, "total_workers": 2, "compliant_count": 2, "violation_count": 0, "needs_review_count": 0, "compliance_rate": 100.0},
        # Frame 2: 0 workers (empty site / transition) -> ignored in average compliance rate
        {"frame_number": 25, "timestamp_sec": 1.0, "total_workers": 0, "compliant_count": 0, "violation_count": 0, "needs_review_count": 0, "compliance_rate": None},
        # Frame 3: 4 workers, 1 compliant, 2 violations, 1 review -> 25.0%
        {"frame_number": 50, "timestamp_sec": 2.0, "total_workers": 4, "compliant_count": 1, "violation_count": 2, "needs_review_count": 1, "compliance_rate": 25.0},
        # Frame 4: 2 workers, 1 compliant, 1 violation -> 50.0%
        {"frame_number": 75, "timestamp_sec": 3.0, "total_workers": 2, "compliant_count": 1, "violation_count": 1, "needs_review_count": 0, "compliance_rate": 50.0},
        # Frame 5: 3 workers, 0 compliant, 3 violations -> 0.0%
        {"frame_number": 100, "timestamp_sec": 4.0, "total_workers": 3, "compliant_count": 0, "violation_count": 3, "needs_review_count": 0, "compliance_rate": 0.0},
    ]

    summary = summarize_video_results(mixed_frames)

    assert summary["total_frames_analyzed"] == 5
    assert summary["frames_with_workers"] == 4
    assert summary["frames_without_workers"] == 1
    # Frames with violations: Frame 3 (2), Frame 4 (1), Frame 5 (3) -> 3 frames
    assert summary["frames_with_violations"] == 3

    # Average compliance rate: (100.0 + 25.0 + 50.0 + 0.0) / 4 = 175.0 / 4 = 43.8%
    assert summary["average_compliance_rate"] == 43.8
    assert summary["has_worker_frames"] is True

    # Top 3 violation frames sorted by violation count: Frame 5 (3), Frame 3 (2), Frame 4 (1)
    top_frames = summary["top_violation_frames"]
    assert len(top_frames) == 3
    assert top_frames[0]["frame_number"] == 100 and top_frames[0]["violation_count"] == 3
    assert top_frames[1]["frame_number"] == 50 and top_frames[1]["violation_count"] == 2
    assert top_frames[2]["frame_number"] == 75 and top_frames[2]["violation_count"] == 1


def test_generate_video_csv_report():
    """Verify CSV report string generation formatting."""
    records = [
        {"frame_number": 0, "timestamp_sec": 0.0, "total_workers": 2, "compliant_count": 2, "violation_count": 0, "needs_review_count": 0, "compliance_rate": 100.0},
        {"frame_number": 25, "timestamp_sec": 1.0, "total_workers": 0, "compliant_count": 0, "violation_count": 0, "needs_review_count": 0, "compliance_rate": None},
    ]
    summary = summarize_video_results(records)
    csv_text = generate_video_csv_report(records, summary, "test_clip.mp4")

    assert "CONSTRUCTION SITE VIDEO PPE COMPLIANCE AUDIT REPORT" in csv_text
    assert "Source Video,test_clip.mp4" in csv_text
    assert "Total Sampled Frames Analyzed,2" in csv_text
    assert "1,0,0.0,2,2,0,0,100.0%" in csv_text
    assert "2,25,1.0,0,0,0,0,N/A" in csv_text


# ============================================================================
# 4. SYNTHETIC VIDEO IO TEST WITH OPENCV
# ============================================================================

def test_synthetic_video_creation_and_reading():
    """Create a tiny synthetic video with cv2.VideoWriter and verify that reading works."""
    temp_dir = tempfile.gettempdir()
    video_path = os.path.join(temp_dir, "test_synthetic_ppe_video.avi")

    width, height, fps, num_frames = 64, 64, 25.0, 30
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    try:
        # Write 30 frames with varying color patterns
        for i in range(num_frames):
            frame = np.full((height, width, 3), (i * 8) % 256, dtype=np.uint8)
            out.write(frame)
        out.release()

        # Read metadata
        metadata = read_video_metadata(video_path)
        assert metadata["total_frames"] == num_frames
        assert pytest.approx(metadata["safe_fps"], 0.5) == fps
        assert metadata["duration_sec"] > 0.0

        # Read specific frame as RGB
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened()

        frame_0 = extract_rgb_frame_at_index(cap, 0)
        assert frame_0 is not None
        assert frame_0.shape == (height, width, 3)

        frame_last = extract_rgb_frame_at_index(cap, num_frames - 1)
        assert frame_last is not None
        assert frame_last.shape == (height, width, 3)

        # Seek past end -> should return None
        frame_past = extract_rgb_frame_at_index(cap, num_frames + 10)
        assert frame_past is None

        cap.release()

    finally:
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
            except OSError:
                pass
