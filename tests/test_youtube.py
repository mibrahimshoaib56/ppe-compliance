"""Unit tests for YouTube URL validation and start-time seeking logic.

Strictly offline tests: NO network calls are made.
Verifies:
- YouTube URL prefix validation (valid and invalid URLs, edge cases).
- Start-time-to-frame-index calculation (compute_start_frame_index).
- CCTV frame range calculation (compute_cctv_frame_range).
"""

import pytest

from src.youtube import (
    YOUTUBE_ALLOWED_PREFIXES,
    MAX_YOUTUBE_DURATION_SECONDS,
    MAX_YOUTUBE_MB,
    is_valid_youtube_url,
)
from src.video import (
    FALLBACK_FPS,
    MAX_VIDEO_SECONDS,
    compute_cctv_frame_range,
    compute_start_frame_index,
)


# ============================================================================
# 1. YOUTUBE URL VALIDATION TESTS
# ============================================================================

@pytest.mark.parametrize(
    "valid_url",
    [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtube.com/watch?v=dQw4w9WgXcQ",
        "https://m.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://www.youtube.com/shorts/abcdef12345",
        "  https://www.youtube.com/watch?v=test  ",  # leading/trailing spaces
        "https://youtube.com/embed/dQw4w9WgXcQ",
    ],
)
def test_valid_youtube_urls(valid_url: str):
    """Ensure all official YouTube URL prefixes evaluate to True."""
    assert is_valid_youtube_url(valid_url) is True


@pytest.mark.parametrize(
    "invalid_url",
    [
        "http://www.youtube.com/watch?v=dQw4w9WgXcQ",  # insecure http
        "http://youtube.com/watch?v=test",
        "https://vimeo.com/123456789",
        "https://dailymotion.com/video/x12345",
        "https://notyoutube.com/watch?v=123",
        "https://www.youtubee.com/watch?v=123",
        "ftp://youtube.com/file.mp4",
        "",
        "   ",
        None,
        12345,  # type: ignore
        "video.mp4",
    ],
)
def test_invalid_youtube_urls(invalid_url: str):
    """Ensure unapproved, malicious, or non-YouTube URLs evaluate to False."""
    assert is_valid_youtube_url(invalid_url) is False


def test_youtube_constants():
    """Verify defined limits and allowed prefixes."""
    assert MAX_YOUTUBE_DURATION_SECONDS == 900
    assert MAX_YOUTUBE_MB == 100
    assert "https://www.youtube.com/" in YOUTUBE_ALLOWED_PREFIXES
    assert "https://youtu.be/" in YOUTUBE_ALLOWED_PREFIXES


# ============================================================================
# 2. START-TIME TO FRAME INDEX TESTS
# ============================================================================

def test_compute_start_frame_index_standard():
    """Verify normal conversion of timestamp seconds to frame indices."""
    # 5.0 seconds at 30 fps = 150
    assert compute_start_frame_index(start_seconds=5.0, fps=30.0, total_frames=1000) == 150
    # 2.5 seconds at 20 fps = 50
    assert compute_start_frame_index(start_seconds=2.5, fps=20.0, total_frames=500) == 50
    # 0.0 seconds = 0
    assert compute_start_frame_index(start_seconds=0.0, fps=25.0, total_frames=500) == 0


def test_compute_start_frame_index_edge_cases():
    """Verify edge cases: negative, None, invalid FPS, beyond total frames."""
    # Negative seconds clamped to 0
    assert compute_start_frame_index(start_seconds=-5.0, fps=30.0, total_frames=1000) == 0
    # None clamped to 0
    assert compute_start_frame_index(start_seconds=None, fps=30.0, total_frames=1000) == 0
    # Invalid/zero FPS falls back to 25.0 fps: 4.0s * 25 = 100
    assert compute_start_frame_index(start_seconds=4.0, fps=0.0, total_frames=500) == 100
    assert compute_start_frame_index(start_seconds=4.0, fps=-10.0, total_frames=500) == 100
    # Start time beyond total_frames clamped to total_frames
    assert compute_start_frame_index(start_seconds=100.0, fps=30.0, total_frames=500) == 500
    # Video with 0 total frames returns 0
    assert compute_start_frame_index(start_seconds=5.0, fps=30.0, total_frames=0) == 0


# ============================================================================
# 3. CCTV FRAME RANGE TESTS
# ============================================================================

def test_compute_cctv_frame_range_from_start():
    """Verify frame range starting at 0s capped at MAX_VIDEO_SECONDS."""
    # 0s to 30s at 30 fps in a 3000-frame video -> [0, 900)
    start_idx, end_idx = compute_cctv_frame_range(
        total_frames=3000,
        fps=30.0,
        start_seconds=0.0,
        max_seconds=MAX_VIDEO_SECONDS,
    )
    assert start_idx == 0
    assert end_idx == 900


def test_compute_cctv_frame_range_with_offset():
    """Verify frame range starting at an offset timestamp."""
    # 10s at 30 fps = frame 300. Max 30s = 900 frames -> [300, 1200)
    start_idx, end_idx = compute_cctv_frame_range(
        total_frames=3000,
        fps=30.0,
        start_seconds=10.0,
        max_seconds=30,
    )
    assert start_idx == 300
    assert end_idx == 1200


def test_compute_cctv_frame_range_clamped_by_video_end():
    """Verify frame range is capped when start + max_seconds exceeds total_frames."""
    # Start at 50s at 30 fps = frame 1500 in an 1800-frame video.
    # 1500 + 900 = 2400 > 1800, so end_idx is 1800.
    start_idx, end_idx = compute_cctv_frame_range(
        total_frames=1800,
        fps=30.0,
        start_seconds=50.0,
        max_seconds=30,
    )
    assert start_idx == 1500
    assert end_idx == 1800


def test_compute_cctv_frame_range_zero_or_beyond():
    """Verify frame range when total_frames is 0 or start is beyond video length."""
    # Zero total frames
    assert compute_cctv_frame_range(total_frames=0, fps=25.0, start_seconds=0.0) == (0, 0)
    # Start beyond total frames: start_idx == total_frames == end_idx
    assert compute_cctv_frame_range(total_frames=500, fps=25.0, start_seconds=50.0) == (500, 500)
