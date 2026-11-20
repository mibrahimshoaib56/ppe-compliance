"""Unit tests for CCTV-style Video PPE compliance processing (src/video.py).

Verifies:
- Pure logic: which frames are analyzed (inference stride PROCESS_EVERY_N = 2)
- FPS validation and fallback handling (0, negative, None, NaN -> FALLBACK_FPS = 25.0)
- Duration and frame capping for MAX_VIDEO_SECONDS (30s)
- Proportional frame resizing up to MAX_FRAME_WIDTH (640px)
- Summary computation strictly on analyzed frames (empty, worker-free, mixed)
- CSV audit report generation
- Synthetic video creation with cv2.VideoWriter and H.264 MP4 encoding with imageio
"""

import math
import os
import tempfile
import cv2
import numpy as np
import pytest

from src.video import (
    FALLBACK_FPS,
    MAX_FRAME_WIDTH,
    MAX_VIDEO_SECONDS,
    OUTPUT_MAX_FPS,
    PROCESS_EVERY_N,
    TEMPORAL_SMOOTHING_WINDOW,
    TOP_VIOLATION_FRAMES_COUNT,
    apply_temporal_smoothing,
    calculate_video_duration,
    compute_max_frames_to_process,
    compute_output_fps,
    compute_resized_dimensions,
    generate_cctv_csv_report,
    is_analyzed_frame,
    summarize_cctv_results,
    validate_and_get_fps,
    write_annotated_h264_video,
)


# ============================================================================
# 1. FPS VALIDATION & FALLBACK TESTS
# ============================================================================

def test_fps_validation_valid():
    """Verify that valid positive FPS values are returned as floats."""
    assert validate_and_get_fps(30.0) == 30.0
    assert validate_and_get_fps(25) == 25.0
    assert validate_and_get_fps(29.97) == 29.97
    assert validate_and_get_fps("24.0") == 24.0


def test_fps_validation_fallback_cases():
    """Verify that 0, negative, None, NaN, inf, and invalid strings fall back to FALLBACK_FPS (25.0)."""
    assert validate_and_get_fps(0) == FALLBACK_FPS
    assert validate_and_get_fps(0.0) == FALLBACK_FPS
    assert validate_and_get_fps(-10.0) == FALLBACK_FPS
    assert validate_and_get_fps(None) == FALLBACK_FPS
    assert validate_and_get_fps(float("nan")) == FALLBACK_FPS
    assert validate_and_get_fps(float("inf")) == FALLBACK_FPS
    assert validate_and_get_fps("invalid_fps") == FALLBACK_FPS


# ============================================================================
# 2. DURATION & FRAME CAPPING TESTS (MAX_VIDEO_SECONDS = 30)
# ============================================================================

def test_calculate_video_duration():
    """Verify duration calculation with valid and fallback FPS."""
    assert calculate_video_duration(750, 25.0) == 30.0
    assert calculate_video_duration(150, 0) == 6.0  # 0 fps falls back to 25.0
    assert calculate_video_duration(0, 30.0) == 0.0


def test_compute_max_frames_to_process_cap():
    """Verify that frame processing is capped at MAX_VIDEO_SECONDS (30 seconds)."""
    # At 25 fps, 30 seconds = 750 frames.
    # Video with 2000 frames -> should cap at 750 frames.
    assert compute_max_frames_to_process(total_frames=2000, fps=25.0, max_seconds=30) == 750

    # Video shorter than 30 seconds (e.g. 250 frames at 25 fps = 10s) -> returns all 250 frames.
    assert compute_max_frames_to_process(total_frames=250, fps=25.0, max_seconds=30) == 250

    # At 30 fps, 30 seconds = 900 frames. Video with 1200 frames -> caps at 900 frames.
    assert compute_max_frames_to_process(total_frames=1200, fps=30.0, max_seconds=30) == 900

    # Zero or negative inputs
    assert compute_max_frames_to_process(total_frames=0, fps=25.0) == 0
    assert compute_max_frames_to_process(total_frames=100, fps=25.0, max_seconds=0) == 0


# ============================================================================
# 3. INFERENCE STRIDE TESTS (is_analyzed_frame)
# ============================================================================

def test_is_analyzed_frame_default_stride():
    """Verify that with PROCESS_EVERY_N=1 every frame is an analyzed frame."""
    # With stride=1, every frame runs detection
    assert is_analyzed_frame(0, PROCESS_EVERY_N) is True
    assert is_analyzed_frame(1, PROCESS_EVERY_N) is True
    assert is_analyzed_frame(2, PROCESS_EVERY_N) is True
    assert is_analyzed_frame(100, PROCESS_EVERY_N) is True


def test_is_analyzed_frame_custom_stride():
    """Verify is_analyzed_frame works with arbitrary positive stride."""
    assert is_analyzed_frame(3, process_every_n=3) is True
    assert is_analyzed_frame(6, process_every_n=3) is True
    assert is_analyzed_frame(1, process_every_n=3) is False
    assert is_analyzed_frame(2, process_every_n=3) is False


# ============================================================================
# 4. RESIZING & FPS CAP TESTS
# ============================================================================

def test_compute_resized_dimensions():
    """Verify frame resizing to MAX_FRAME_WIDTH (640px) while maintaining aspect ratio."""
    # 1920x1080 (16:9 full HD) -> 640x360
    new_w, new_h = compute_resized_dimensions(1920, 1080, max_width=640)
    assert new_w == 640
    assert new_h == 360
    assert new_w % 2 == 0 and new_h % 2 == 0  # Even dimensions for H.264

    # 1280x720 -> 640x360
    assert compute_resized_dimensions(1280, 720, max_width=640) == (640, 360)

    # Frame already smaller than 640px (e.g. 480x360) -> kept as is (even dimensions)
    assert compute_resized_dimensions(480, 360, max_width=640) == (480, 360)

    # Zero or invalid dimensions
    assert compute_resized_dimensions(0, 0) == (0, 0)


def test_compute_output_fps():
    """Verify output FPS capping at OUTPUT_MAX_FPS (30)."""
    assert compute_output_fps(60.0, max_fps=30) == 30.0
    assert compute_output_fps(25.0, max_fps=30) == 25.0
    assert compute_output_fps(15.0, max_fps=30) == 15.0
    # Fallback when 0
    assert compute_output_fps(0, max_fps=30) == 25.0


# ============================================================================
# 5. SUMMARY AGGREGATION TESTS (ANALYZED FRAMES ONLY)
# ============================================================================

def test_summary_empty_records():
    """Verify summary output when no frames were analyzed."""
    summary = summarize_cctv_results([])
    assert summary["frames_analyzed"] == 0
    assert summary["frames_with_workers"] == 0
    assert summary["frames_without_workers"] == 0
    assert summary["frames_with_violations"] == 0
    assert summary["average_compliance_rate"] == 0.0
    assert summary["has_worker_frames"] is False
    assert summary["top_violation_moments"] == []


def test_summary_all_frames_no_workers():
    """Verify that frames with no workers are NEVER counted as 100% compliant."""
    records = [
        {"frame_number": 0, "timestamp_sec": 0.0, "total_workers": 0, "compliant_count": 0, "violation_count": 0, "needs_review_count": 0, "compliance_rate": None},
        {"frame_number": 2, "timestamp_sec": 0.08, "total_workers": 0, "compliant_count": 0, "violation_count": 0, "needs_review_count": 0, "compliance_rate": None},
    ]
    summary = summarize_cctv_results(records)

    assert summary["frames_analyzed"] == 2
    assert summary["frames_with_workers"] == 0
    assert summary["frames_without_workers"] == 2
    assert summary["frames_with_violations"] == 0
    # Average compliance rate must NOT be 100%
    assert summary["average_compliance_rate"] == 0.0
    assert summary["has_worker_frames"] is False


def test_summary_mixed_frames():
    """Verify summary with mixed analyzed frames (workers, no workers, violations)."""
    records = [
        # Frame 0: 2 workers, 2 compliant, 0 violations -> 100.0%
        {"frame_number": 0, "timestamp_sec": 0.0, "total_workers": 2, "compliant_count": 2, "violation_count": 0, "needs_review_count": 0, "compliance_rate": 100.0},
        # Frame 2: 0 workers -> ignored in average compliance rate
        {"frame_number": 2, "timestamp_sec": 0.08, "total_workers": 0, "compliant_count": 0, "violation_count": 0, "needs_review_count": 0, "compliance_rate": None},
        # Frame 4: 4 workers, 1 compliant, 3 violations -> 25.0%
        {"frame_number": 4, "timestamp_sec": 0.16, "total_workers": 4, "compliant_count": 1, "violation_count": 3, "needs_review_count": 0, "compliance_rate": 25.0},
        # Frame 6: 2 workers, 1 compliant, 1 violation -> 50.0%
        {"frame_number": 6, "timestamp_sec": 0.24, "total_workers": 2, "compliant_count": 1, "violation_count": 1, "needs_review_count": 0, "compliance_rate": 50.0},
    ]
    summary = summarize_cctv_results(records)

    assert summary["frames_analyzed"] == 4
    assert summary["frames_with_workers"] == 3
    assert summary["frames_without_workers"] == 1
    # Frames with violations: Frame 4 (3), Frame 6 (1) -> 2 frames
    assert summary["frames_with_violations"] == 2

    # Average rate: (100.0 + 25.0 + 50.0) / 3 = 175.0 / 3 = 58.3%
    assert summary["average_compliance_rate"] == 58.3
    assert summary["has_worker_frames"] is True

    # Top moments: Frame 4 (3 violations), Frame 6 (1 violation)
    top_moments = summary["top_violation_moments"]
    assert len(top_moments) == 2
    assert top_moments[0]["frame_number"] == 4 and top_moments[0]["violation_count"] == 3
    assert top_moments[1]["frame_number"] == 6 and top_moments[1]["violation_count"] == 1


def test_generate_cctv_csv_report():
    """Verify CSV report generation."""
    records = [
        {"frame_number": 0, "timestamp_sec": 0.0, "total_workers": 2, "compliant_count": 2, "violation_count": 0, "needs_review_count": 0, "compliance_rate": 100.0},
        {"frame_number": 2, "timestamp_sec": 0.08, "total_workers": 0, "compliant_count": 0, "violation_count": 0, "needs_review_count": 0, "compliance_rate": None},
    ]
    summary = summarize_cctv_results(records)
    csv_text = generate_cctv_csv_report(records, summary, "site_cctv.mp4")

    assert "CCTV VIDEO PPE COMPLIANCE AUDIT REPORT" in csv_text
    assert "Source Video,site_cctv.mp4" in csv_text
    assert "Inference Stride,Every 1 frames" in csv_text
    assert "Total Frames Analyzed,2" in csv_text
    assert "1,0,0.00,2,2,0,0,100.0%" in csv_text
    assert "2,2,0.08,0,0,0,0,N/A" in csv_text


# ============================================================================
# 5b. TEMPORAL ANTI-FLICKER SMOOTHING TESTS
# ============================================================================

def test_temporal_smoothing_empty():
    """Verify temporal smoothing on an empty list returns an empty list."""
    assert apply_temporal_smoothing([]) == []


def test_temporal_smoothing_suppresses_transient_flicker():
    """Verify that 1-2 frame posture/detection glitches are smoothed out by a 7-frame window.

    Scenario: 7 workers on site; 5 compliant, 2 violating (true rate = 71.4%).
    In frame 4, a compliant worker turns their head, causing an isolated spike to 3 violations.
    In frame 8, another posture glitch occurs.
    The 7-frame median filter must restore both glitch frames back to 2 violations.
    """
    records = []
    for i in range(12):
        if i == 4 or i == 8:
            # Glitch frame: 1 false-positive violation
            viol = 3
            comp = 4
            rate = round((4 / 7) * 100.0, 1)  # 57.1%
        else:
            # Normal steady state
            viol = 2
            comp = 5
            rate = round((5 / 7) * 100.0, 1)  # 71.4%
        records.append({
            "frame_number": i,
            "timestamp_sec": i * 0.04,
            "total_workers": 7,
            "compliant_count": comp,
            "violation_count": viol,
            "needs_review_count": 0,
            "compliance_rate": rate,
        })

    smoothed = apply_temporal_smoothing(records, window_size=TEMPORAL_SMOOTHING_WINDOW)
    assert len(smoothed) == len(records)

    # Frame 4 and 8 must have their glitch filtered out:
    assert smoothed[4]["violation_count"] == 2
    assert smoothed[4]["compliant_count"] == 5
    assert smoothed[4]["compliance_rate"] == 71.4

    assert smoothed[8]["violation_count"] == 2
    assert smoothed[8]["compliant_count"] == 5
    assert smoothed[8]["compliance_rate"] == 71.4


def test_temporal_smoothing_preserves_sustained_violations():
    """Verify that a genuine, sustained violation spanning many frames is NOT filtered out."""
    records = []
    # 5 compliant frames
    for i in range(5):
        records.append({
            "frame_number": i,
            "timestamp_sec": i * 0.04,
            "total_workers": 2,
            "compliant_count": 2,
            "violation_count": 0,
            "needs_review_count": 0,
            "compliance_rate": 100.0,
        })
    # 8 consecutive violation frames (real sustained violation)
    for i in range(5, 13):
        records.append({
            "frame_number": i,
            "timestamp_sec": i * 0.04,
            "total_workers": 2,
            "compliant_count": 0,
            "violation_count": 2,
            "needs_review_count": 0,
            "compliance_rate": 0.0,
        })

    smoothed = apply_temporal_smoothing(records, window_size=7)
    # The middle of the sustained violation period must remain 2 violations
    assert smoothed[8]["violation_count"] == 2
    assert smoothed[8]["compliance_rate"] == 0.0
    assert smoothed[9]["violation_count"] == 2
    assert smoothed[9]["compliance_rate"] == 0.0


def test_temporal_smoothing_preserves_zero_workers():
    """Verify that frames with 0 workers remain 0 workers and are never marked compliant."""
    records = [
        {"frame_number": i, "timestamp_sec": i * 0.04, "total_workers": 0, "compliant_count": 0, "violation_count": 0, "needs_review_count": 0, "compliance_rate": None}
        for i in range(7)
    ]
    smoothed = apply_temporal_smoothing(records, window_size=7)
    for r in smoothed:
        assert r["total_workers"] == 0
        assert r["compliant_count"] == 0
        assert r["violation_count"] == 0
        assert r["compliance_rate"] is None


def test_summarize_cctv_results_with_smoothing_flag():
    """Verify summarize_cctv_results with use_smoothing=True eliminates glitch frames from KPI counts."""
    # 10 steady frames with 0 violations, and 1 glitch frame at index 5 with 1 violation
    records = []
    for i in range(11):
        if i == 5:
            viol = 1
            comp = 1
            rate = 50.0
        else:
            viol = 0
            comp = 2
            rate = 100.0
        records.append({
            "frame_number": i,
            "timestamp_sec": i * 0.04,
            "total_workers": 2,
            "compliant_count": comp,
            "violation_count": viol,
            "needs_review_count": 0,
            "compliance_rate": rate,
        })

    # Without smoothing: 1 frame with violations, average rate = (10*100 + 50)/11 = 95.5%
    raw_summary = summarize_cctv_results(records, use_smoothing=False)
    assert raw_summary["frames_with_violations"] == 1
    assert raw_summary["average_compliance_rate"] == 95.5

    # With smoothing: glitch frame is eliminated, 0 frames with violations, average rate = 100.0%
    smoothed_summary = summarize_cctv_results(records, use_smoothing=True, smoothing_window=7)
    assert smoothed_summary["frames_with_violations"] == 0
    assert smoothed_summary["average_compliance_rate"] == 100.0
    assert smoothed_summary["smoothing_applied"] is True
    assert smoothed_summary["smoothing_window"] == 7


# ============================================================================
# 6. SYNTHETIC VIDEO & H.264 ENCODING TEST
# ============================================================================

def test_synthetic_video_cv2_writer_and_imageio_h264():
    """Create a synthetic video with cv2.VideoWriter and encode with imageio H.264."""
    temp_dir = tempfile.gettempdir()
    input_video_path = os.path.join(temp_dir, "test_synthetic_cctv_in.avi")
    output_mp4_path = os.path.join(temp_dir, "test_synthetic_cctv_out.mp4")

    w, h, fps, num_frames = 64, 64, 25.0, 10
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(input_video_path, fourcc, fps, (w, h))

    try:
        # 1. Write frames to input video
        for i in range(num_frames):
            frame_bgr = np.full((h, w, 3), (i * 20) % 256, dtype=np.uint8)
            writer.write(frame_bgr)
        writer.release()

        # 2. Read frames with OpenCV
        cap = cv2.VideoCapture(input_video_path)
        assert cap.isOpened()

        read_frames_rgb = []
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            read_frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        assert len(read_frames_rgb) == num_frames

        # 3. Write annotated frames using write_annotated_h264_video (imageio + imageio-ffmpeg)
        write_annotated_h264_video(read_frames_rgb, output_mp4_path, fps=fps)

        # 4. Verify output MP4 exists and has valid size
        assert os.path.exists(output_mp4_path)
        file_size = os.path.getsize(output_mp4_path)
        assert file_size > 100  # Non-trivial MP4 file header + stream data

        # 5. Verify the generated H.264 MP4 can be reopened by OpenCV
        cap_out = cv2.VideoCapture(output_mp4_path)
        assert cap_out.isOpened()
        out_frames_count = int(cap_out.get(cv2.CAP_PROP_FRAME_COUNT))
        assert out_frames_count == num_frames
        cap_out.release()

    finally:
        for p in (input_video_path, output_mp4_path):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
