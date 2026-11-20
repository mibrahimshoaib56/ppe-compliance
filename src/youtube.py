"""YouTube video download and URL validation module.

Provides safe downloading of progressive MP4 video streams via yt-dlp
with strict URL validation, duration caps, filesize limits, and friendly error handling.
"""

import os
from typing import Any, Dict, Optional
import yt_dlp

# ============================================================================
# NAMED CONSTANTS & LIMITS
# ============================================================================

# Allowed URL prefixes for YouTube video links
YOUTUBE_ALLOWED_PREFIXES = (
    "https://www.youtube.com/",
    "https://youtube.com/",
    "https://m.youtube.com/",
    "https://youtu.be/",
)

# Maximum permitted video duration in seconds (15 minutes = 900 seconds)
MAX_YOUTUBE_DURATION_SECONDS: int = 900

# Maximum permitted video download file size in megabytes (100 MB)
MAX_YOUTUBE_MB: int = 100


# ============================================================================
# CUSTOM EXCEPTION
# ============================================================================

class YouTubeDownloadError(Exception):
    """Raised when YouTube URL validation, metadata extraction, or downloading fails."""
    pass


# ============================================================================
# PURE VALIDATION FUNCTIONS
# ============================================================================

def is_valid_youtube_url(url: Optional[str]) -> bool:
    """Validate that the given URL starts with an approved YouTube prefix.

    Args:
        url: Input URL string.

    Returns:
        bool: True if URL starts with an approved YouTube domain prefix, False otherwise.
    """
    if not url or not isinstance(url, str):
        return False
    cleaned_url = url.strip()
    return any(cleaned_url.startswith(prefix) for prefix in YOUTUBE_ALLOWED_PREFIXES)


# ============================================================================
# DOWNLOAD LOGIC
# ============================================================================

def download_youtube(url: str, out_dir: str) -> str:
    """Download a single progressive MP4 video stream from YouTube via yt-dlp.

    Validates URL, checks duration and live-stream status, enforces maximum filesize,
    downloads a single progressive MP4 (height <= 480p), and writes to out_dir.

    Args:
        url: YouTube video URL.
        out_dir: Directory where the downloaded MP4 will be saved.

    Returns:
        str: Absolute filesystem path of the downloaded MP4 file.

    Raises:
        YouTubeDownloadError: If URL is invalid, video exceeds limits, or yt-dlp encounters errors.
    """
    if not is_valid_youtube_url(url):
        raise YouTubeDownloadError(
            "Invalid YouTube URL. Please provide a link starting with "
            "https://www.youtube.com/, https://youtube.com/, https://m.youtube.com/, or https://youtu.be/"
        )

    if not os.path.isdir(out_dir):
        raise YouTubeDownloadError(f"Target download directory does not exist: {out_dir}")

    cleaned_url = url.strip()

    # Step 1: Query metadata (no download) to inspect duration and stream type
    meta_opts: Dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }

    try:
        with yt_dlp.YoutubeDL(meta_opts) as ydl:
            info = ydl.extract_info(cleaned_url, download=False)
    except yt_dlp.utils.DownloadError as e:
        err_msg = str(e).lower()
        if "private" in err_msg or "sign in" in err_msg:
            friendly = "This video is private or requires sign-in."
        elif "age" in err_msg:
            friendly = "This video is age-restricted and cannot be downloaded directly."
        elif "geo" in err_msg or "country" in err_msg or "region" in err_msg:
            friendly = "This video is region-blocked in your location."
        elif "bot" in err_msg or "robot" in err_msg:
            friendly = "YouTube presented a bot verification challenge."
        else:
            friendly = f"Could not retrieve video information: {e}"
        raise YouTubeDownloadError(
            f"{friendly} Please consider downloading the video file yourself and using the 'Upload file' mode."
        ) from e
    except Exception as e:
        raise YouTubeDownloadError(
            f"Network or extraction error: {e}. Suggestion: Download the video manually and use 'Upload file'."
        ) from e

    if not info:
        raise YouTubeDownloadError(
            "Could not read video metadata. Please download the video yourself and use 'Upload file'."
        )

    # Reject live streams
    is_live = info.get("is_live") or info.get("live_status") == "is_live"
    if is_live:
        raise YouTubeDownloadError(
            "Live streams cannot be analyzed. Please provide a link to a recorded, finite YouTube video."
        )

    # Reject videos exceeding duration limit (MAX_YOUTUBE_DURATION_SECONDS = 900)
    duration = info.get("duration")
    if duration is not None and duration > MAX_YOUTUBE_DURATION_SECONDS:
        minutes = int(duration // 60)
        max_min = int(MAX_YOUTUBE_DURATION_SECONDS // 60)
        raise YouTubeDownloadError(
            f"Video duration is {minutes} minutes ({duration}s), which exceeds the {max_min} minute ({MAX_YOUTUBE_DURATION_SECONDS}s) limit. "
            "Please select a shorter video or trim it locally before uploading."
        )

    # Step 2: Download best available MP4 up to 480p.
    # YouTube Shorts (and many modern uploads) only have DASH streams (video-only + audio-only).
    # We use the ffmpeg bundled with imageio-ffmpeg to merge them into a single MP4.
    # Format preference:
    #   1. Best pre-muxed (progressive) mp4 at or under 480p
    #   2. Best video at or under 480p + best audio -> merged to mp4
    #   3. Any best available stream as final fallback
    try:
        import imageio_ffmpeg as _iio_ffmpeg
        _ffmpeg_bin = _iio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        _ffmpeg_bin = None  # yt-dlp will try system ffmpeg

    out_template = os.path.join(out_dir, "%(id)s.%(ext)s")
    dl_opts: Dict[str, Any] = {
        "format": (
            "best[ext=mp4][height<=480]"
            "/bestvideo[ext=mp4][height<=480]+bestaudio[ext=m4a]"
            "/bestvideo[height<=480]+bestaudio"
            "/best[height<=480]"
            "/best"
        ),
        "merge_output_format": "mp4",   # always produce a .mp4 regardless of merge
        "noplaylist": True,
        "max_filesize": MAX_YOUTUBE_MB * 1024 * 1024,
        "outtmpl": out_template,
        "quiet": True,
        "no_warnings": True,
        **({"ffmpeg_location": _ffmpeg_bin} if _ffmpeg_bin else {}),
    }

    try:
        with yt_dlp.YoutubeDL(dl_opts) as ydl:
            ydl.download([cleaned_url])
    except yt_dlp.utils.DownloadError as e:
        raise YouTubeDownloadError(
            f"Download failed: {e}. Suggestion: Download the video yourself and use the 'Upload file' mode."
        ) from e
    except Exception as e:
        raise YouTubeDownloadError(
            f"Unexpected error during download: {e}. Please use 'Upload file' mode instead."
        ) from e

    # Find the downloaded file in out_dir
    files = [os.path.join(out_dir, f) for f in os.listdir(out_dir) if os.path.isfile(os.path.join(out_dir, f))]
    if not files:
        raise YouTubeDownloadError("Download finished but no output file was saved in the temporary folder.")

    # Return the first matching video file
    return files[0]
