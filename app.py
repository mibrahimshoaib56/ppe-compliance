"""SAFETY VISION: Enterprise AI Surveillance & PPE Compliance Monitoring Platform.

Streamlit web application for construction site safety surveillance, combining
an Ultralytics YOLOv8 object detector with pure Python spatial compliance rules
and multi-criteria reliability safeguards.
"""

import json
import os
import shutil
import tempfile
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import altair as alt

from src.compliance import (
    CLASS_HARDHAT,
    CLASS_NO_HARDHAT,
    CLASS_NO_VEST,
    CLASS_PERSON,
    CLASS_VEST,
    HEAD_ZONE_BOTTOM,
    HEAD_ZONE_TOP,
    OVERLAP_THRESHOLD,
    SITE_SAFE_THRESHOLD,
    SITE_WARNING_THRESHOLD,
    TORSO_ZONE_BOTTOM,
    TORSO_ZONE_TOP,
    evaluate_site_compliance,
)
from src.detector import (
    EXPECTED_CLASSES,
    load_detector_model,
    resolve_model_path,
    run_detection,
    verify_model_classes,
)
from src.utils import (
    annotate_image,
    crop_worker_roi,
    encode_image_bytes,
    format_worker_table,
    generate_csv_report,
    get_action_recommendation,
)
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
    compute_cctv_frame_range,
    compute_max_frames_to_process,
    compute_output_fps,
    compute_resized_dimensions,
    compute_start_frame_index,
    generate_cctv_csv_report,
    is_analyzed_frame,
    summarize_cctv_results,
    validate_and_get_fps,
    write_annotated_h264_video,
)
from src.youtube import (
    MAX_YOUTUBE_DURATION_SECONDS,
    MAX_YOUTUBE_MB,
    YouTubeDownloadError,
    download_youtube,
    is_valid_youtube_url,
)

# ============================================================================
# 1. PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="SAFETY VISION — AI Surveillance Platform",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

SAMPLE_DIRS = ["sample_images", os.path.join("eval", "test_photos")]
HISTORY_FILE = os.path.join("eval", "inspection_history.json")


# ============================================================================
# 2. CACHED MODEL LOADER (Ensures model is loaded once on CPU)
# ============================================================================
@st.cache_resource(show_spinner="Initializing YOLOv8 inference engine on CPU...")
def get_detector(model_file_path: str):
    """Load and cache the YOLOv8 model instance on CPU."""
    return load_detector_model(model_file_path)


# ============================================================================
# 3. SESSION STATE INITIALIZATION & HISTORY PERSISTENCE
# ============================================================================

def classify_worker_status(status_str: Any) -> str:
    """Classify raw worker status string into 'compliant', 'warning', or 'violation'."""
    s = str(status_str).strip()
    if s == "Compliant":
        return "compliant"
    if s.startswith("Needs review") or s in ("Vest not visible: manual check", "Unknown", "Needs Review"):
        return "warning"
    return "violation"

def init_session_state():
    """Initialize session state variables for navigation, theme, and inspections."""
    if "theme" not in st.session_state:
        st.session_state.theme = "Dark Theme"

    if "current_page" not in st.session_state:
        st.session_state.current_page = "Dashboard"

    if "nav_radio" not in st.session_state:
        st.session_state.nav_radio = "Dashboard"

    if "inspection_mode" not in st.session_state:
        st.session_state.inspection_mode = "Image"

    if "video_results" not in st.session_state:
        st.session_state.video_results = None

    if "strict_hardhat" not in st.session_state:
        st.session_state.strict_hardhat = True

    if "required_gear" not in st.session_state:
        st.session_state.required_gear = "both"

    if "default_conf_threshold" not in st.session_state:
        st.session_state.default_conf_threshold = 0.25

    if "active_inspection" not in st.session_state:
        st.session_state.active_inspection = None

    if "inspections_store" not in st.session_state:
        st.session_state.inspections_store = {}

    if "inspection_history" not in st.session_state:
        st.session_state.inspection_history = []
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, "r") as f:
                    saved_records = json.load(f)
                    if isinstance(saved_records, list):
                        st.session_state.inspection_history = saved_records
            except Exception:
                pass


def save_inspection_to_history(inspection_record: Dict[str, Any]):
    """Save an inspection record to session state and disk."""
    inspection_id = inspection_record["id"]

    st.session_state.inspections_store[inspection_id] = inspection_record

    history_entry = {
        "id": inspection_id,
        "timestamp": inspection_record["timestamp"],
        "image_name": inspection_record["image_name"],
        "total_workers": inspection_record["summary"]["total_workers"],
        "compliant_count": inspection_record["summary"]["compliant_count"],
        "violation_count": inspection_record["summary"]["violation_count"],
        "unknown_count": inspection_record["summary"]["unknown_count"],
        "compliance_rate": inspection_record["summary"]["compliance_rate"],
        "site_status": inspection_record["summary"]["site_status"],
        "image_path": inspection_record.get("image_path", ""),
        "settings": inspection_record.get("settings", {}),
    }

    st.session_state.inspection_history = [
        h for h in st.session_state.inspection_history if h.get("id") != inspection_id
    ]
    st.session_state.inspection_history.insert(0, history_entry)

    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(st.session_state.inspection_history, f, indent=2)
    except Exception:
        pass


def navigate_to(page_name: str):
    """Programmatically switch active page and trigger a rerun."""
    st.session_state.current_page = page_name
    st.session_state.nav_radio = page_name
    st.rerun()


# ============================================================================
# 4. ENTERPRISE INDUSTRIAL STYLING (Dark & Light Themes)
# ============================================================================
def inject_custom_css(theme: str):
    """Inject responsive, enterprise-grade industrial control center CSS for Phase 1."""
    if theme == "Dark Theme":
        css = """
        <style>
            /* -------------------------------------------------------------
               GLOBAL DESIGN SYSTEM TOKENS - DARK INDUSTRIAL
               ------------------------------------------------------------- */
            :root {
                --bg-main: #0B1118;
                --bg-sidebar: #111B26;
                --surface-card: #162230;
                --surface-elevated: #223244;
                --border-color: #263442;
                --accent-primary: #38BDF8;
                --text-primary: #F1F5F9;
                --text-secondary: #94A3B8;

                --color-safe: #22C55E;
                --color-violation: #EF4444;
                --color-warning: #F59E0B;
            }

            /* Base Document & Typography */
            .stApp {
                background-color: var(--bg-main);
                color: var(--text-primary);
                font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                letter-spacing: -0.01em;
            }

            /* Transparent Streamlit Header */
            header[data-testid="stHeader"] {
                background: transparent !important;
            }

            /* Sidebar Application Shell */
            [data-testid="stSidebar"] {
                background-color: var(--bg-sidebar);
                border-right: 1px solid var(--border-color);
            }
            [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
                gap: 0.5rem;
            }
            [data-testid="stSidebar"] hr {
                border-top: 1px solid var(--border-color) !important;
                margin: 10px 0 !important;
            }

            /* Sidebar Radio Navigation & Active State */
            [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] {
                gap: 4px;
            }
            [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label {
                padding: 7px 12px;
                border-radius: 4px;
                border: 1px solid transparent;
                transition: all 0.15s ease;
                margin-bottom: 2px;
                cursor: pointer;
            }
            [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label:hover {
                background-color: rgba(34, 50, 68, 0.4);
                border-color: var(--border-color);
            }
            [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label p {
                color: var(--text-secondary);
                font-size: 13px;
                font-weight: 500;
            }
            [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label:hover p {
                color: var(--text-primary) !important;
            }
            [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {
                background-color: rgba(56, 189, 248, 0.08) !important;
                border: 1px solid rgba(56, 189, 248, 0.25) !important;
                border-left: 3px solid var(--accent-primary) !important;
            }
            [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) p {
                color: var(--accent-primary) !important;
                font-weight: 600 !important;
            }

            /* Top Navigation Bar */
            .top-nav {
                background-color: var(--surface-card);
                border: 1px solid var(--border-color);
                border-radius: 4px;
                padding: 10px 18px;
                margin-bottom: 20px;
                display: flex;
                align-items: center;
                justify-content: space-between;
            }
            .brand-title {
                color: var(--accent-primary);
                font-weight: 700;
                font-size: 15px;
                letter-spacing: 0.5px;
                display: flex;
                align-items: center;
                gap: 8px;
            }
            .page-crumb {
                color: var(--text-secondary);
                font-size: 13px;
                font-weight: 500;
                letter-spacing: 0.3px;
            }
            .model-pill-ready {
                background-color: rgba(34, 197, 94, 0.1);
                color: var(--color-safe);
                border: 1px solid rgba(34, 197, 94, 0.25);
                padding: 4px 10px;
                border-radius: 4px;
                font-size: 11px;
                font-weight: 600;
                letter-spacing: 0.3px;
            }
            .model-pill-warn {
                background-color: rgba(245, 158, 11, 0.1);
                color: var(--color-warning);
                border: 1px solid rgba(245, 158, 11, 0.25);
                padding: 4px 10px;
                border-radius: 4px;
                font-size: 11px;
                font-weight: 600;
                letter-spacing: 0.3px;
            }

            /* Metric Cards */
            .metric-card {
                background: var(--surface-card);
                border: 1px solid var(--border-color);
                border-radius: 4px;
                padding: 12px 14px;
                text-align: left;
                position: relative;
                overflow: hidden;
            }
            .metric-card-title {
                color: var(--text-secondary);
                font-size: 11px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 4px;
            }
            .metric-card-value {
                color: var(--text-primary);
                font-size: 24px;
                font-weight: 700;
                line-height: 1.2;
            }

            /* Surface Panels */
            .sv-panel {
                background-color: var(--surface-card);
                border: 1px solid var(--border-color);
                border-radius: 4px;
                padding: 16px;
                margin-bottom: 14px;
            }
            .sv-surface-elevated {
                background-color: var(--surface-elevated);
                border: 1px solid var(--border-color);
                border-radius: 4px;
                padding: 12px;
                margin-bottom: 12px;
            }

            /* Semantic Status Badges */
            .badge-safe {
                background-color: rgba(34, 197, 94, 0.12);
                color: var(--color-safe);
                border: 1px solid rgba(34, 197, 94, 0.3);
                padding: 3px 8px;
                border-radius: 4px;
                font-weight: 600;
                font-size: 11px;
                letter-spacing: 0.3px;
                text-transform: uppercase;
            }
            .badge-warning {
                background-color: rgba(245, 158, 11, 0.12);
                color: var(--color-warning);
                border: 1px solid rgba(245, 158, 11, 0.3);
                padding: 3px 8px;
                border-radius: 4px;
                font-weight: 600;
                font-size: 11px;
                letter-spacing: 0.3px;
                text-transform: uppercase;
            }
            .badge-critical {
                background-color: rgba(239, 68, 68, 0.12);
                color: var(--color-violation);
                border: 1px solid rgba(239, 68, 68, 0.3);
                padding: 3px 8px;
                border-radius: 4px;
                font-weight: 600;
                font-size: 11px;
                letter-spacing: 0.3px;
                text-transform: uppercase;
            }

            /* Custom Tab Component */
            .stTabs [data-baseweb="tab-list"] {
                gap: 4px;
                border-bottom: 1px solid var(--border-color);
                padding-bottom: 2px;
            }
            .stTabs [data-baseweb="tab"] {
                border-radius: 4px;
                padding: 6px 14px;
                background-color: transparent;
                color: var(--text-secondary);
                font-size: 13px;
                font-weight: 500;
                border: 1px solid transparent;
            }
            .stTabs [data-baseweb="tab"]:hover {
                color: var(--text-primary);
                background-color: rgba(34, 50, 68, 0.3);
            }
            .stTabs [aria-selected="true"] {
                background-color: var(--surface-card) !important;
                color: var(--accent-primary) !important;
                border: 1px solid var(--border-color) !important;
                border-bottom: 2px solid var(--accent-primary) !important;
            }

            /* Buttons */
            .stButton > button[kind="primary"] {
                background-color: #0284C7;
                color: #FFFFFF;
                border: 1px solid var(--accent-primary);
                border-radius: 4px;
                font-weight: 600;
                font-size: 13px;
                padding: 6px 16px;
                transition: all 0.15s ease;
            }
            .stButton > button[kind="primary"]:hover {
                background-color: #0369A1;
                border-color: #7DD3FC;
            }
            .stButton > button:not([kind="primary"]) {
                background-color: var(--surface-card);
                color: var(--text-primary);
                border: 1px solid var(--border-color);
                border-radius: 4px;
                font-weight: 500;
                font-size: 13px;
                padding: 6px 14px;
                transition: all 0.15s ease;
            }
            .stButton > button:not([kind="primary"]):hover {
                border-color: var(--accent-primary);
                color: var(--accent-primary);
                background-color: var(--surface-elevated);
            }

            /* Inputs & Dropdowns */
            [data-baseweb="input"] {
                background-color: var(--surface-card) !important;
                border: 1px solid var(--border-color) !important;
                border-radius: 4px !important;
            }
            [data-baseweb="input"] input {
                color: var(--text-primary) !important;
            }
            [data-baseweb="select"] > div {
                background-color: var(--surface-card) !important;
                border: 1px solid var(--border-color) !important;
                border-radius: 4px !important;
                color: var(--text-primary) !important;
            }
            [data-testid="stFileUploader"] {
                border: 1px dashed var(--border-color);
                border-radius: 4px;
                padding: 10px;
                background-color: var(--surface-card);
            }
            [data-testid="stExpander"] {
                background-color: var(--surface-card) !important;
                border: 1px solid var(--border-color) !important;
                border-radius: 4px !important;
            }
            [data-testid="stDataFrame"] {
                border: 1px solid var(--border-color);
                border-radius: 4px;
                overflow: hidden;
            }
            div[data-testid="stAlert"] {
                border-radius: 4px;
                border: 1px solid var(--border-color);
                padding: 10px 14px;
            }
        </style>
        """
    else:
        css = """
        <style>
            /* -------------------------------------------------------------
               GLOBAL DESIGN SYSTEM TOKENS - LIGHT INDUSTRIAL
               ------------------------------------------------------------- */
            :root {
                --bg-main: #F8FAFC;
                --bg-sidebar: #F1F5F9;
                --surface-card: #FFFFFF;
                --surface-elevated: #F8FAFC;
                --border-color: #E2E8F0;
                --accent-primary: #0284C7;
                --text-primary: #0F172A;
                --text-secondary: #64748B;

                --color-safe: #16A34A;
                --color-violation: #DC2626;
                --color-warning: #D97706;
            }

            .stApp {
                background-color: var(--bg-main);
                color: var(--text-primary);
                font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                letter-spacing: -0.01em;
            }

            header[data-testid="stHeader"] {
                background: transparent !important;
            }

            [data-testid="stSidebar"] {
                background-color: var(--bg-sidebar);
                border-right: 1px solid var(--border-color);
            }
            [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
                gap: 0.5rem;
            }
            [data-testid="stSidebar"] hr {
                border-top: 1px solid var(--border-color) !important;
                margin: 10px 0 !important;
            }

            [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label {
                padding: 7px 12px;
                border-radius: 4px;
                border: 1px solid transparent;
                transition: all 0.15s ease;
                margin-bottom: 2px;
                cursor: pointer;
            }
            [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label:hover {
                background-color: #E2E8F0;
            }
            [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label p {
                color: var(--text-secondary);
                font-size: 13px;
                font-weight: 500;
            }
            [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label:hover p {
                color: var(--text-primary) !important;
            }
            [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {
                background-color: #E0F2FE !important;
                border: 1px solid #BAE6FD !important;
                border-left: 3px solid var(--accent-primary) !important;
            }
            [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) p {
                color: var(--accent-primary) !important;
                font-weight: 600 !important;
            }

            .top-nav {
                background-color: var(--surface-card);
                border: 1px solid var(--border-color);
                border-radius: 4px;
                padding: 10px 18px;
                margin-bottom: 20px;
                display: flex;
                align-items: center;
                justify-content: space-between;
                box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
            }
            .brand-title {
                color: var(--accent-primary);
                font-weight: 700;
                font-size: 15px;
                letter-spacing: 0.5px;
                display: flex;
                align-items: center;
                gap: 8px;
            }
            .page-crumb {
                color: var(--text-secondary);
                font-size: 13px;
                font-weight: 500;
                letter-spacing: 0.3px;
            }
            .model-pill-ready {
                background-color: rgba(22, 163, 74, 0.1);
                color: var(--color-safe);
                border: 1px solid rgba(22, 163, 74, 0.25);
                padding: 4px 10px;
                border-radius: 4px;
                font-size: 11px;
                font-weight: 600;
                letter-spacing: 0.3px;
            }
            .model-pill-warn {
                background-color: rgba(217, 119, 6, 0.1);
                color: var(--color-warning);
                border: 1px solid rgba(217, 119, 6, 0.25);
                padding: 4px 10px;
                border-radius: 4px;
                font-size: 11px;
                font-weight: 600;
                letter-spacing: 0.3px;
            }

            .metric-card {
                background: var(--surface-card);
                border: 1px solid var(--border-color);
                border-radius: 4px;
                padding: 12px 14px;
                box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
            }
            .metric-card-title {
                color: var(--text-secondary);
                font-size: 11px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 4px;
            }
            .metric-card-value {
                color: var(--text-primary);
                font-size: 24px;
                font-weight: 700;
                line-height: 1.2;
            }

            .sv-panel {
                background-color: var(--surface-card);
                border: 1px solid var(--border-color);
                border-radius: 4px;
                padding: 16px;
                margin-bottom: 14px;
                box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
            }
            .sv-surface-elevated {
                background-color: var(--surface-elevated);
                border: 1px solid var(--border-color);
                border-radius: 4px;
                padding: 12px;
                margin-bottom: 12px;
            }

            .badge-safe {
                background-color: rgba(22, 163, 74, 0.12);
                color: var(--color-safe);
                border: 1px solid rgba(22, 163, 74, 0.3);
                padding: 3px 8px;
                border-radius: 4px;
                font-weight: 600;
                font-size: 11px;
                letter-spacing: 0.3px;
                text-transform: uppercase;
            }
            .badge-warning {
                background-color: rgba(217, 119, 6, 0.12);
                color: var(--color-warning);
                border: 1px solid rgba(217, 119, 6, 0.3);
                padding: 3px 8px;
                border-radius: 4px;
                font-weight: 600;
                font-size: 11px;
                letter-spacing: 0.3px;
                text-transform: uppercase;
            }
            .badge-critical {
                background-color: rgba(220, 38, 38, 0.12);
                color: var(--color-violation);
                border: 1px solid rgba(220, 38, 38, 0.3);
                padding: 3px 8px;
                border-radius: 4px;
                font-weight: 600;
                font-size: 11px;
                letter-spacing: 0.3px;
                text-transform: uppercase;
            }

            .stTabs [data-baseweb="tab-list"] {
                gap: 4px;
                border-bottom: 1px solid var(--border-color);
                padding-bottom: 2px;
            }
            .stTabs [data-baseweb="tab"] {
                border-radius: 4px;
                padding: 6px 14px;
                background-color: transparent;
                color: var(--text-secondary);
                font-size: 13px;
                font-weight: 500;
                border: 1px solid transparent;
            }
            .stTabs [data-baseweb="tab"]:hover {
                color: var(--text-primary);
                background-color: #F1F5F9;
            }
            .stTabs [aria-selected="true"] {
                background-color: #E0F2FE !important;
                color: var(--accent-primary) !important;
                border: 1px solid #BAE6FD !important;
                border-bottom: 2px solid var(--accent-primary) !important;
            }

            .stButton > button[kind="primary"] {
                background-color: var(--accent-primary);
                color: #FFFFFF;
                border: 1px solid var(--accent-primary);
                border-radius: 4px;
                font-weight: 600;
                font-size: 13px;
                padding: 6px 16px;
            }
            .stButton > button:not([kind="primary"]) {
                background-color: var(--surface-card);
                color: var(--text-primary);
                border: 1px solid var(--border-color);
                border-radius: 4px;
                font-weight: 500;
                font-size: 13px;
                padding: 6px 14px;
            }
            .stButton > button:not([kind="primary"]):hover {
                border-color: var(--accent-primary);
                color: var(--accent-primary);
            }

            [data-baseweb="input"] {
                background-color: var(--surface-card) !important;
                border: 1px solid var(--border-color) !important;
                border-radius: 4px !important;
            }
            [data-baseweb="select"] > div {
                background-color: var(--surface-card) !important;
                border: 1px solid var(--border-color) !important;
                border-radius: 4px !important;
            }
            [data-testid="stFileUploader"] {
                border: 1px dashed var(--border-color);
                border-radius: 4px;
                padding: 10px;
                background-color: var(--surface-card);
            }
            [data-testid="stExpander"] {
                background-color: var(--surface-card) !important;
                border: 1px solid var(--border-color) !important;
                border-radius: 4px !important;
            }
            [data-testid="stDataFrame"] {
                border: 1px solid var(--border-color);
                border-radius: 4px;
                overflow: hidden;
            }
            div[data-testid="stAlert"] {
                border-radius: 4px;
                border: 1px solid var(--border-color);
                padding: 10px 14px;
            }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)


# ============================================================================
# 5. TOP NAVIGATION BAR COMPONENT
# ============================================================================
def render_top_navbar(current_page: str, is_model_verified: bool):
    """Render the enterprise top bar with branding, breadcrumb, model status, and theme toggle."""
    model_status_html = (
        "<span class='model-pill-ready'>● YOLOv8n Online (CPU)</span>"
        if is_model_verified
        else "<span class='model-pill-warn'>● YOLOv8 Fallback</span>"
    )

    col_brand, col_status, col_theme = st.columns([5, 3, 2])

    with col_brand:
        st.markdown(
            f"""
            <div style='display: flex; align-items: center; gap: 8px; padding-top: 4px;'>
                <svg width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='#38BDF8' stroke-width='2' stroke-linecap='round' stroke-linejoin='round' style='vertical-align:middle;margin-right:2px;'>
                    <path d='M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z'/>
                </svg>
                <span class='brand-title'>SAFETY VISION</span>
                <span style='color: #64748B;'>/</span>
                <span class='page-crumb'>{current_page.upper()}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_status:
        st.markdown(
            f"<div style='text-align: center; padding-top: 4px;'>{model_status_html}</div>",
            unsafe_allow_html=True,
        )

    with col_theme:
        current_theme = st.session_state.theme
        theme_label = "Light Mode" if current_theme == "Dark Theme" else "Dark Mode"
        if st.button(theme_label, key="top_theme_toggle", use_container_width=True):
            st.session_state.theme = "Light Theme" if current_theme == "Dark Theme" else "Dark Theme"
            st.rerun()

    st.markdown("<hr style='margin: 8px 0 16px 0; border: none; border-top: 1px solid #263442;'>", unsafe_allow_html=True)


# ============================================================================
# 6. PAGE 1: DASHBOARD
# ============================================================================
def render_dashboard(active_inspection: Optional[Dict[str, Any]], history: List[Dict[str, Any]]):
    """Render the executive industrial safety dashboard with KPIs, distribution charts, and recent scans."""
    # A. Page Header with Compact Action Bar
    head_left, head_right = st.columns([4, 2])
    with head_left:
        st.markdown("### Operational Safety Dashboard")
        st.caption("Active job site surveillance, automated PPE compliance verification, and risk mitigation tracking.")
    with head_right:
        st.markdown("<div style='text-align: right; padding-top: 4px;'>", unsafe_allow_html=True)
        h_btn1, h_btn2 = st.columns([1, 1])
        with h_btn1:
            if active_inspection is not None:
                if st.button("Active Results", key="dash_view_active", use_container_width=True):
                    navigate_to("Inspection Results")
        with h_btn2:
            if st.button("New Inspection", key="dash_new_insp", type="primary", use_container_width=True):
                navigate_to("New Inspection")
        st.markdown("</div>", unsafe_allow_html=True)

    has_data = active_inspection is not None or len(history) > 0

    if not has_data:
        st.markdown(
            """
            <div class='sv-panel' style='text-align: center; padding: 48px 24px; margin-top: 10px;'>
                <div style='font-size: 13px; text-transform: uppercase; letter-spacing: 1.5px; color: #38BDF8; font-weight: 600; margin-bottom: 8px;'>[ SURVEILLANCE STANDBY ]</div>
                <h3 style='margin: 12px 0 6px 0; color: #F1F5F9;'>No Site Inspections Recorded Yet</h3>
                <p style='color: #94A3B8; max-width: 540px; margin: 0 auto 24px auto; font-size: 13px; line-height: 1.6;'>
                    No active surveillance data or completed site inspections are available. Initiate an automated safety audit by uploading job site imagery or streaming CCTV footage.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        c_empty1, c_empty2, c_empty3 = st.columns([1, 2, 1])
        with c_empty2:
            if st.button("Start First Site Inspection", key="dash_start_first", type="primary", use_container_width=True):
                navigate_to("New Inspection")
        return

    current = active_inspection if active_inspection is not None else history[0]
    summary = current["summary"] if "summary" in current else current

    total_w = summary.get("total_workers", 0)
    comp_w = summary.get("compliant_count", 0)
    viol_w = summary.get("violation_count", 0)
    unknown_w = summary.get("unknown_count", 0)
    review_w = summary.get("needs_review_count", unknown_w)
    rate = summary.get("compliance_rate", 0.0)
    status = summary.get("site_status", "Safe")

    # Semantic Status & Color Mapping
    if status == "Safe":
        status_color = "#22C55E"
        status_badge_class = "badge-safe"
        status_label = "SAFE"
    elif status == "Warning":
        status_color = "#F59E0B"
        status_badge_class = "badge-warning"
        status_label = "WARNING"
    else:
        status_color = "#EF4444"
        status_badge_class = "badge-critical"
        status_label = "CRITICAL"

    # Semantic Violation Value Styling: non-alarming when 0, vivid red when > 0
    viol_color = "#EF4444" if viol_w > 0 else "#94A3B8"
    review_color = "#F59E0B" if review_w > 0 else "#94A3B8"
    rate_color = "#22C55E" if rate >= 85.0 else ("#F59E0B" if rate >= 60.0 else "#EF4444")

    # B. Safety Overview / KPI Section
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        st.markdown(
            f"<div class='metric-card'><div class='metric-card-title'>Total Workers</div>"
            f"<div class='metric-card-value'>{total_w}</div></div>",
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"<div class='metric-card'><div class='metric-card-title'>Compliant</div>"
            f"<div class='metric-card-value' style='color:#22C55E;'>{comp_w}</div></div>",
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"<div class='metric-card'><div class='metric-card-title'>Violations</div>"
            f"<div class='metric-card-value' style='color:{viol_color};'>{viol_w}</div></div>",
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f"<div class='metric-card'><div class='metric-card-title'>Needs Review</div>"
            f"<div class='metric-card-value' style='color:{review_color};'>{review_w}</div></div>",
            unsafe_allow_html=True,
        )
    with k5:
        st.markdown(
            f"<div class='metric-card'><div class='metric-card-title'>Compliance Rate</div>"
            f"<div class='metric-card-value' style='color:{rate_color};'>{rate:.1f}%</div></div>",
            unsafe_allow_html=True,
        )
    with k6:
        st.markdown(
            f"<div class='metric-card'><div class='metric-card-title'>Site Status</div>"
            f"<div style='margin-top: 4px;'><span class='{status_badge_class}' style='font-size: 13px; padding: 4px 10px;'>{status_label}</span></div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    # C & D. Compliance Visualization and Context-Aware Operational Safety Directive
    c_chart, c_directive = st.columns([3, 2])

    with c_chart:
        st.markdown("#### Worker Compliance Distribution")
        if total_w == 0:
            st.info("No personnel detected in the active scene to plot compliance distribution.")
        else:
            chart_df = pd.DataFrame({
                "Category": ["Compliant", "Violations", "Needs Review"],
                "Count": [comp_w, viol_w, review_w],
            })
            bar_chart = (
                alt.Chart(chart_df)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("Category:N", title=None, axis=alt.Axis(labelAngle=0, labelColor="#94A3B8")),
                    y=alt.Y("Count:Q", title="Workers", axis=alt.Axis(tickMinStep=1, labelColor="#94A3B8", titleColor="#94A3B8")),
                    color=alt.Color(
                        "Category:N",
                        scale=alt.Scale(
                            domain=["Compliant", "Violations", "Needs Review"],
                            range=["#22C55E", "#EF4444", "#F59E0B"],
                        ),
                        legend=None,
                    ),
                    tooltip=["Category", "Count"],
                )
                .properties(height=230)
                .configure_view(strokeWidth=0)
            )
            st.altair_chart(bar_chart, use_container_width=True)

    with c_directive:
        st.markdown("#### Operational Safety Directive")
        active_label = current.get("image_name", "Active Scan")

        if total_w == 0:
            st.warning(
                "No personnel detected in the active scene. Verify camera perspective, angle, and distance "
                "to ensure workers fall within the detection field of view."
            )
        else:
            advisory_msg = get_action_recommendation(summary)
            border_c = status_color
            bg_surface = "rgba(34, 197, 94, 0.05)" if status == "Safe" else ("rgba(245, 158, 11, 0.05)" if status == "Warning" else "rgba(239, 68, 68, 0.05)")

            st.markdown(
                f"""
                <div style='background: {bg_surface}; border: 1px solid #263442; border-left: 4px solid {border_c}; border-radius: 4px; padding: 14px 16px; margin-bottom: 12px;'>
                    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;'>
                        <span style='font-size: 11px; font-weight: 700; text-transform: uppercase; color: {border_c}; letter-spacing: 0.5px;'>
                            {status_label} STATUS ADVISORY
                        </span>
                        <span class='{status_badge_class}'>{status_label}</span>
                    </div>
                    <div style='color: #F1F5F9; font-size: 13px; line-height: 1.5; font-weight: 500;'>
                        {advisory_msg}
                    </div>
                    <div style='margin-top: 10px; font-size: 11px; color: #94A3B8;'>
                        Active Scan: <code>{active_label}</code>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Quick Action Links inside advisory
            c_sub1, c_sub2 = st.columns(2)
            with c_sub1:
                if active_inspection is not None:
                    if st.button("Inspect Active Scene", key="dash_inspect_btn", use_container_width=True):
                        navigate_to("Inspection Results")
            with c_sub2:
                if st.button("Audit Individual Workers", key="dash_audit_btn", use_container_width=True):
                    navigate_to("Worker Audits")

    # E. Recent Inspections Section
    st.markdown("---")
    st.markdown("#### Recent Site Inspections")

    if not history:
        st.info("No recorded inspection history available. Executed inspections will appear here.")
    else:
        records_for_table = []
        for h in history[:6]:
            h_stat = h.get("site_status", "Safe")
            records_for_table.append({
                "Inspection ID": h["id"],
                "Timestamp": h["timestamp"],
                "Source": h["image_name"],
                "Total Workers": h["total_workers"],
                "Compliant": h.get("compliant_count", 0),
                "Violations": h.get("violation_count", 0),
                "Compliance Rate": f"{h['compliance_rate']:.1f}%",
                "Status": h_stat.upper(),
            })

        st.dataframe(pd.DataFrame(records_for_table), use_container_width=True)

        # Quick Review Selector to open any past inspection in the results workstation
        c_sel, c_act = st.columns([4, 1])
        with c_sel:
            hist_options = {
                f"{h['id']} - {h['image_name']} ({h['site_status'].upper()}, {h['compliance_rate']:.1f}% compliance)": h['id']
                for h in history[:10]
            }
            selected_hist_label = st.selectbox(
                "Select Past Inspection to Review in Results Workstation:",
                list(hist_options.keys()),
                key="dash_hist_select",
                label_visibility="collapsed",
            )
        with c_act:
            if st.button("Review Inspection", key="dash_open_hist_btn", use_container_width=True):
                target_id = hist_options[selected_hist_label]
                if target_id in st.session_state.inspections_store:
                    st.session_state.active_inspection = st.session_state.inspections_store[target_id]
                    navigate_to("Inspection Results")
                else:
                    target_entry = next((item for item in history if item["id"] == target_id), None)
                    if target_entry and target_entry.get("image_path") and os.path.exists(target_entry["image_path"]):
                        with st.spinner("Loading inspection..."):
                            reloaded_img = Image.open(target_entry["image_path"]).convert("RGB")
                            req_mode = target_entry.get("settings", {}).get("required_gear", "both")
                            conf_t = target_entry.get("settings", {}).get("conf_threshold", 0.25)
                            detections = run_detection(reloaded_img, model=get_detector(resolve_model_path()), conf_threshold=conf_t, device="cpu")
                            _h2 = float(np.array(reloaded_img).shape[0])
                            evaluation = evaluate_site_compliance(
                                detections=detections,
                                required_gear=req_mode,
                                image_height=_h2,
                                strict_hardhat=st.session_state.strict_hardhat,
                            )
                            ann_img = annotate_image(
                                image=reloaded_img,
                                workers=evaluation["workers"],
                                unassigned_gear=evaluation["unassigned_gear"],
                                show_gear_boxes=True,
                                show_zones=False,
                            )
                            ann_bytes = encode_image_bytes(ann_img, format_ext="JPEG")
                            rec = {
                                "id": target_id,
                                "timestamp": target_entry["timestamp"],
                                "image_name": target_entry["image_name"],
                                "image_path": target_entry["image_path"],
                                "image": reloaded_img,
                                "annotated_image": ann_img,
                                "annotated_bytes": ann_bytes,
                                "detections": detections,
                                "workers": evaluation["workers"],
                                "summary": evaluation["summary"],
                                "unassigned_gear": evaluation["unassigned_gear"],
                                "settings": target_entry.get("settings", {}),
                            }
                            st.session_state.active_inspection = rec
                            st.session_state.inspections_store[target_id] = rec
                            navigate_to("Inspection Results")


# ============================================================================
# 7. PAGE 2: NEW INSPECTION
# ============================================================================
def render_new_inspection(model: Any):
    """Render the inspection creation and inference workflow page."""
    # A. Page Header
    st.markdown("### New Safety Inspection")
    st.caption("Configure job site imagery and compliance parameters for automated YOLOv8 PPE analysis.")

    # Find sample images
    sample_files = []
    sample_base = ""
    for s_dir in SAMPLE_DIRS:
        if os.path.exists(s_dir):
            found = sorted([f for f in os.listdir(s_dir) if f.endswith((".jpg", ".png", ".jpeg"))])
            if found:
                sample_files = found
                sample_base = s_dir
                break

    # 2-Column Structured Workflow Layout
    col_source, col_config = st.columns([7, 5])

    # Left Column: Image Source Selection & Preview
    with col_source:
        st.markdown("#### Image Source & Inspection Preview")

        image_mode = st.radio(
            "Source Type",
            ["Benchmark Sample Images", "Upload Custom Photo"],
            horizontal=True,
            label_visibility="collapsed",
            key="new_insp_source_mode",
        )

        input_image = None
        source_label = ""
        source_path = ""

        if image_mode == "Benchmark Sample Images" and sample_files:
            selected_file = st.selectbox(
                "Select Benchmark Construction Sample:",
                sample_files,
                index=0,
                key="new_insp_sample_select",
            )
            source_label = selected_file
            source_path = os.path.join(sample_base, selected_file)
            try:
                input_image = Image.open(source_path)
                if input_image.mode != "RGB":
                    input_image = input_image.convert("RGB")
            except Exception as e:
                st.error(f"Failed to load sample image: {e}")
        else:
            uploaded_file = st.file_uploader(
                "Upload Site Photograph (JPG, PNG, JPEG)",
                type=["jpg", "jpeg", "png"],
                help="High-resolution site photograph showing workers, hardhats, and safety vests.",
                key="new_insp_file_upload",
            )
            if uploaded_file is not None:
                source_label = uploaded_file.name
                try:
                    input_image = Image.open(uploaded_file)
                    if input_image.mode != "RGB":
                        input_image = input_image.convert("RGB")
                except Exception as e:
                    st.error(f"Failed to parse uploaded image: {e}")

        # Image Preview Display
        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        if input_image is not None:
            st.image(input_image, caption=f"Inspection Source: {source_label}", use_container_width=True)
            # Compact Specifications Bar
            st.markdown(
                f"""
                <div class='sv-surface-elevated' style='padding: 8px 14px; margin-top: 6px; display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; font-size: 12px;'>
                    <div><strong>File:</strong> <code>{source_label}</code></div>
                    <div><strong>Resolution:</strong> <code>{input_image.size[0]} &times; {input_image.size[1]} px</code> &nbsp;|&nbsp; <strong>Format:</strong> <code>{input_image.mode}</code></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class='sv-panel' style='text-align: center; padding: 48px 20px; border: 1px dashed #263442; margin-top: 6px;'>
                    <div style='color: #38BDF8; font-size: 12px; text-transform: uppercase; letter-spacing: 1.2px; font-weight: 600; margin-bottom: 8px;'>[ NO MEDIA SELECTED ]</div>
                    <div style='color: #94A3B8; font-size: 13px; max-width: 360px; margin: 0 auto; line-height: 1.5;'>
                        Select a verified benchmark image from the dropdown or upload an on-site photo above to preview and analyze.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Right Column: Compliance Configuration & Inspection Execution
    with col_config:
        st.markdown("#### Compliance Configuration")

        # 1. PPE Requirements Card
        st.markdown(
            """
            <div style='font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; color: #94A3B8; font-weight: 600; margin-bottom: 6px;'>
                Mandatory PPE Requirements
            </div>
            """,
            unsafe_allow_html=True,
        )

        cur_req = st.session_state.get("required_gear", "both")
        default_gear_idx = 1 if cur_req == "hardhat_only" else (2 if cur_req == "vest_only" else 0)
        gear_choice = st.radio(
            "Enforced Safety Equipment",
            ["Hardhat + Safety Vest", "Hardhat Only", "Safety Vest Only"],
            index=default_gear_idx,
            key="new_insp_gear_choice",
            label_visibility="collapsed",
            help="Select PPE standards enforced for this site inspection.",
        )
        if gear_choice == "Hardhat Only":
            required_gear_mode = "hardhat_only"
        elif gear_choice == "Safety Vest Only":
            required_gear_mode = "vest_only"
        else:
            required_gear_mode = "both"

        st.markdown(
            f"""
            <div class='sv-panel' style='padding: 10px 14px; margin: 8px 0 16px 0; font-size: 12px; line-height: 1.5; color: #94A3B8;'>
                Enforcing: <strong style='color: #F1F5F9;'>{gear_choice}</strong><br>
                <span>Personnel missing any enforced equipment will be flagged as safety violations.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 2. Advanced Detection Settings (Collapsible)
        with st.expander("Advanced Detection Settings", expanded=False):
            conf_threshold = st.slider(
                "Sensitivity Slider (Scales Per-Class Thresholds)",
                min_value=0.10,
                max_value=0.90,
                value=float(st.session_state.get("default_conf_threshold", 0.25)),
                step=0.05,
                key="new_insp_conf_slider",
                help="Scales per-class thresholds (Base: Person 0.30, Hardhat 0.30, NO-Hardhat 0.20, Safety Vest 0.20). Slider at 0.25 represents 1.0x baseline scale.",
            )
            show_gear_boxes = st.checkbox("Draw Gear Bounding Boxes", value=True, key="new_insp_gear_boxes_cb")
            show_zones = st.checkbox("Outline Body Zones (Head 35% / Torso 15-75%)", value=False, key="new_insp_zones_cb")

        st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)

        # 3. Primary Execution Action
        run_disabled = input_image is None
        btn_label = "Run Compliance Check" if not run_disabled else "Select an Image to Run Check"

        if st.button(btn_label, key="new_insp_run_btn", type="primary", use_container_width=True, disabled=run_disabled):
            with st.spinner("Executing YOLOv8 inference & spatial compliance logic on CPU..."):
                detections = run_detection(
                    image=input_image,
                    model=model,
                    conf_threshold=conf_threshold,
                    device="cpu",
                )

                _img_h = float(np.array(input_image).shape[0])
                evaluation = evaluate_site_compliance(
                    detections=detections,
                    required_gear=required_gear_mode,
                    image_height=_img_h,
                    strict_hardhat=st.session_state.strict_hardhat,
                )

                annotated_img = annotate_image(
                    image=input_image,
                    workers=evaluation["workers"],
                    unassigned_gear=evaluation["unassigned_gear"],
                    show_gear_boxes=show_gear_boxes,
                    show_zones=show_zones,
                )
                annotated_bytes = encode_image_bytes(annotated_img, format_ext="JPEG")

                inspection_id = f"INSP-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                inspection_record = {
                    "id": inspection_id,
                    "timestamp": timestamp_str,
                    "image_name": source_label,
                    "image_path": source_path,
                    "image": input_image,
                    "annotated_image": annotated_img,
                    "annotated_bytes": annotated_bytes,
                    "detections": detections,
                    "workers": evaluation["workers"],
                    "summary": evaluation["summary"],
                    "unassigned_gear": evaluation["unassigned_gear"],
                    "settings": {
                        "conf_threshold": conf_threshold,
                        "required_gear": required_gear_mode,
                        "show_gear_boxes": show_gear_boxes,
                        "show_zones": show_zones,
                        "strict_hardhat": st.session_state.strict_hardhat,
                    },
                }

                st.session_state.active_inspection = inspection_record
                save_inspection_to_history(inspection_record)
                st.success(f"Inspection `{inspection_id}` completed successfully.")
                navigate_to("Inspection Results")


# ============================================================================
# 8. PAGE 3: INSPECTION RESULTS
# ============================================================================
def render_inspection_results(active_inspection: Optional[Dict[str, Any]]):
    """Render the AI inspection workstation: summary metrics, high-res viewer, and worker analysis."""
    # A. Page Header
    st.markdown("### Inspection Results & Spatial Evidence")
    st.caption("AI-annotated site inspection overlays, compliance ratios, and actionable safety advisories.")

    # Empty State Handling
    if active_inspection is None:
        st.markdown(
            """
            <div class='sv-panel' style='text-align: center; padding: 48px 24px; margin-top: 10px;'>
                <div style='font-size: 13px; text-transform: uppercase; letter-spacing: 1.5px; color: #38BDF8; font-weight: 600; margin-bottom: 8px;'>[ NO ACTIVE INSPECTION ]</div>
                <h3 style='margin: 12px 0 6px 0; color: #F1F5F9;'>No Inspection Results Available</h3>
                <p style='color: #94A3B8; max-width: 520px; margin: 0 auto 24px auto; font-size: 13px; line-height: 1.6;'>
                    No active inspection data is currently loaded in the analysis workspace. Run a new site inspection or load a previous audit from the inspection history.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        c_emp1, c_emp2, c_emp3 = st.columns([1, 2, 1])
        with c_emp2:
            if st.button("Start New Site Inspection", key="results_empty_new_btn", type="primary", use_container_width=True):
                navigate_to("New Inspection")
        return

    summary = active_inspection["summary"]
    workers = active_inspection["workers"]
    image_input = active_inspection["image"]
    annotated_img = active_inspection["annotated_image"]

    # 1. Compact Inspection Summary Header & Metadata Bar
    status = summary.get("site_status", "Safe")
    if status == "Safe":
        status_color = "#22C55E"
        status_badge_class = "badge-safe"
        status_label = "SAFE"
    elif status == "Warning":
        status_color = "#F59E0B"
        status_badge_class = "badge-warning"
        status_label = "WARNING"
    else:
        status_color = "#EF4444"
        status_badge_class = "badge-critical"
        status_label = "CRITICAL"

    st.markdown(
        f"""
        <div class='sv-panel' style='padding: 10px 16px; margin-bottom: 12px; display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center;'>
            <div>
                <strong>Inspection ID:</strong> <code>{active_inspection['id']}</code> &nbsp;|&nbsp;
                <strong>Source:</strong> <code>{active_inspection['image_name']}</code> &nbsp;|&nbsp;
                <strong>Timestamp:</strong> <code>{active_inspection['timestamp']}</code>
            </div>
            <div>
                <span style='font-size: 11px; text-transform: uppercase; color: #94A3B8; font-weight: 600; margin-right: 8px;'>Site Evaluation:</span>
                <span class='{status_badge_class}' style='font-size: 12px; padding: 4px 10px;'>{status_label}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Compact Inspection Metrics Row (KPIs)
    total_w = summary.get("total_workers", 0)
    comp_w = summary.get("compliant_count", 0)
    viol_w = summary.get("violation_count", 0)
    review_w = summary.get("needs_review_count", summary.get("unknown_count", 0))
    rate = summary.get("compliance_rate", 0.0)

    viol_color = "#EF4444" if viol_w > 0 else "#94A3B8"
    review_color = "#F59E0B" if review_w > 0 else "#94A3B8"
    rate_color = "#22C55E" if rate >= 85.0 else ("#F59E0B" if rate >= 60.0 else "#EF4444")

    r1, r2, r3, r4, r5, r6 = st.columns(6)
    with r1:
        st.markdown(
            f"<div class='metric-card'><div class='metric-card-title'>Total Workers</div>"
            f"<div class='metric-card-value'>{total_w}</div></div>",
            unsafe_allow_html=True,
        )
    with r2:
        st.markdown(
            f"<div class='metric-card'><div class='metric-card-title'>Compliant</div>"
            f"<div class='metric-card-value' style='color:#22C55E;'>{comp_w}</div></div>",
            unsafe_allow_html=True,
        )
    with r3:
        st.markdown(
            f"<div class='metric-card'><div class='metric-card-title'>Violations</div>"
            f"<div class='metric-card-value' style='color:{viol_color};'>{viol_w}</div></div>",
            unsafe_allow_html=True,
        )
    with r4:
        st.markdown(
            f"<div class='metric-card'><div class='metric-card-title'>Needs Review</div>"
            f"<div class='metric-card-value' style='color:{review_color};'>{review_w}</div></div>",
            unsafe_allow_html=True,
        )
    with r5:
        st.markdown(
            f"<div class='metric-card'><div class='metric-card-title'>Compliance Rate</div>"
            f"<div class='metric-card-value' style='color:{rate_color};'>{rate:.1f}%</div></div>",
            unsafe_allow_html=True,
        )
    with r6:
        st.markdown(
            f"<div class='metric-card'><div class='metric-card-title'>Site Status</div>"
            f"<div style='margin-top: 4px;'><span class='{status_badge_class}' style='font-size: 13px; padding: 4px 10px;'>{status_label}</span></div></div>",
            unsafe_allow_html=True,
        )

    # 3. Compact Contextual Safety Advisory
    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
    if total_w == 0:
        st.warning(
            "No personnel detected in the inspection image. Ensure the camera angle captures workers clearly "
            "from an eye-level or elevated vantage point."
        )
    else:
        advisory_msg = get_action_recommendation(summary)
        bg_adv = "rgba(34, 197, 94, 0.05)" if status == "Safe" else ("rgba(245, 158, 11, 0.05)" if status == "Warning" else "rgba(239, 68, 68, 0.05)")
        st.markdown(
            f"""
            <div style='background: {bg_adv}; border: 1px solid #263442; border-left: 4px solid {status_color}; border-radius: 4px; padding: 10px 14px; margin-bottom: 12px; font-size: 13px;'>
                <strong style='color: {status_color}; text-transform: uppercase;'>Site Directive:</strong>
                <span style='color: #F1F5F9; margin-left: 6px;'>{advisory_msg}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 4. Visual Evidence Viewer (Centerpiece)
    st.markdown("#### Visual Evidence Viewer")

    # Legend Bar directly above viewer
    st.markdown(
        """
        <div style='display:flex; flex-wrap:wrap; gap:16px; align-items:center; background:#111B26; border:1px solid #263442; border-radius:4px; padding:6px 14px; margin-bottom:10px; font-size:12px;'>
            <span style='color:#94A3B8; font-weight:600; text-transform:uppercase;'>Detection Legend:</span>
            <span><span style='color:#22C55E; font-weight:bold;'>&#9679;</span> Compliant Worker (Hardhat + Vest)</span>
            <span><span style='color:#EF4444; font-weight:bold;'>&#9679;</span> PPE Violation (Missing Gear)</span>
            <span><span style='color:#F59E0B; font-weight:bold;'>&#9679;</span> Needs Review / Ambiguous</span>
            <span><span style='color:#38BDF8; font-weight:bold;'>&#9679;</span> Detected PPE Gear Box</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_annotated, tab_side_by_side, tab_raw, tab_raw_boxes = st.tabs([
        "Annotated Detection Overlay",
        "Side-by-Side Comparison",
        "Original Site Photograph",
        "Raw Detection Coordinates",
    ])

    with tab_annotated:
        img_np = np.array(image_input)
        h_px, w_px = img_np.shape[:2]
        st.image(
            annotated_img,
            caption=f"AI Annotated Detection Overlay | Resolution: {w_px}x{h_px} px | Detected Workers: {len(workers)}",
            use_container_width=True,
        )

    with tab_side_by_side:
        c_left, c_right = st.columns(2)
        with c_left:
            st.image(image_input, caption="Original Site Photograph", use_container_width=True)
        with c_right:
            st.image(annotated_img, caption="AI Spatial Detection & Compliance Overlay", use_container_width=True)

    with tab_raw:
        st.image(image_input, caption="Original Unprocessed Site Photograph", use_container_width=True)

    with tab_raw_boxes:
        all_dets = active_inspection.get("detections", [])
        if all_dets:
            det_rows = [
                {
                    "Class": d["class_name"],
                    "Confidence": f"{int(d.get('confidence', 1.0) * 100)}%",
                    "BBox [x1, y1, x2, y2]": [round(float(c), 1) for c in (d.get("box") or d.get("bbox") or [])],
                }
                for d in all_dets
            ]
            st.dataframe(pd.DataFrame(det_rows), use_container_width=True)
        else:
            st.info("No raw detection items recorded.")

    # 5. Worker Analysis & Selected Worker Spatial Evidence
    st.markdown("---")
    st.markdown("#### Worker Analysis & Spatial Evidence")

    if not workers:
        st.info("No workers detected in this scene for individual spatial analysis.")
    else:
        col_roster, col_drilldown = st.columns([7, 5])

        with col_roster:
            st.markdown("##### Detected Worker Roster")

            filter_choice = st.radio(
                "Filter Workers by Status:",
                ["All", "Violations Only", "Compliant Only", "Needs Review"],
                horizontal=True,
                key="results_worker_filter",
            )

            filtered_workers = workers
            if filter_choice == "Violations Only":
                filtered_workers = [w for w in workers if classify_worker_status(w.get("status", "")) == "violation"]
            elif filter_choice == "Compliant Only":
                filtered_workers = [w for w in workers if classify_worker_status(w.get("status", "")) == "compliant"]
            elif filter_choice == "Needs Review":
                filtered_workers = [w for w in workers if classify_worker_status(w.get("status", "")) == "warning"]

            if not filtered_workers:
                st.info(f"No workers match the '{filter_choice}' filter.")
            else:
                table_records = format_worker_table(filtered_workers)
                st.dataframe(pd.DataFrame(table_records), use_container_width=True)

        with col_drilldown:
            st.markdown("##### Selected Worker Spatial Evidence")

            worker_options = {
                f"Worker #{w['worker_id']} - {w['status']} ({int(w.get('confidence', 1.0)*100)}% conf)": w
                for w in workers
            }
            selected_key = st.selectbox(
                "Select Worker to Inspect Body Zones & Assigned PPE:",
                list(worker_options.keys()),
                key="results_worker_select",
                label_visibility="collapsed",
            )
            selected_worker = worker_options[selected_key]

            # Generate ROI Crop with zones
            worker_crop = crop_worker_roi(image_input, selected_worker, padding_pct=0.10, show_zones=True)

            sub_crop, sub_info = st.columns([1, 1])
            with sub_crop:
                st.image(
                    worker_crop,
                    caption=f"Worker #{selected_worker['worker_id']} ROI (Head / Torso Zones)",
                    use_container_width=True,
                )

            with sub_info:
                w_status = selected_worker["status"]
                c_stat = classify_worker_status(w_status)
                stat_badge_class = "badge-safe" if c_stat == "compliant" else ("badge-warning" if c_stat == "warning" else "badge-critical")

                # Head and Torso Status formatting with semantic colors
                h_stat = selected_worker.get("hardhat_status", "missing")
                v_stat = selected_worker.get("vest_status", "missing")
                h_color = "#22C55E" if h_stat == "present" else ("#F59E0B" if h_stat == "unknown" else "#EF4444")
                v_color = "#22C55E" if v_stat == "present" else ("#F59E0B" if v_stat == "unknown" else "#EF4444")

                st.markdown(
                    f"""
                    <div class='sv-panel' style='padding: 12px; margin-bottom: 8px;'>
                        <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;'>
                            <strong>Worker #{selected_worker['worker_id']}</strong>
                            <span class='{stat_badge_class}'>{w_status.upper()}</span>
                        </div>
                        <div style='font-size:12px; color:#94A3B8; line-height:1.6;'>
                            <strong>Confidence:</strong> {int(selected_worker.get('confidence', 1.0) * 100)}%<br>
                            <strong>Head Zone:</strong> <span style='color:{h_color}; font-weight:600;'>{h_stat.capitalize()}</span><br>
                            <strong>Torso Zone:</strong> <span style='color:{v_color}; font-weight:600;'>{v_stat.capitalize()}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                assigned_gear = selected_worker.get("assigned_gear", [])
                if assigned_gear:
                    gear_data = [
                        {
                            "Item": g["class_name"],
                            "Conf": f"{int(g.get('confidence', 1.0)*100)}%",
                            "IoG": f"{int(g.get('overlap', 0.0)*100)}%",
                        }
                        for g in assigned_gear
                    ]
                    st.dataframe(pd.DataFrame(gear_data), use_container_width=True, height=120)
                else:
                    st.caption("No safety gear detected in assigned zones.")

    # 6. Action Toolbar
    st.markdown("---")
    c_btn1, c_btn2, c_btn3 = st.columns(3)
    with c_btn1:
        if st.button("Conduct New Inspection", key="results_action_new", use_container_width=True):
            navigate_to("New Inspection")
    with c_btn2:
        if st.button("Worker Audits Inspector", key="results_action_audits", use_container_width=True):
            navigate_to("Worker Audits")
    with c_btn3:
        if st.button("Export Compliance Report", key="results_action_export", use_container_width=True):
            navigate_to("Reports & Exports")


# ============================================================================
# 9. PAGE 4: WORKER AUDITS
# ============================================================================
def render_worker_audits(active_inspection: Optional[Dict[str, Any]]):
    """Render granular worker compliance audit table and individual crop drill-down inspector."""
    # A. Page Header
    st.markdown("### Worker Compliance Audits & Inspector")
    st.caption("Granular worker identification, spatial body-zone attribution, and PPE compliance evidence verification.")

    # Empty State Handling
    if active_inspection is None:
        st.markdown(
            """
            <div class='sv-panel' style='text-align: center; padding: 48px 24px; margin-top: 10px;'>
                <div style='font-size: 13px; text-transform: uppercase; letter-spacing: 1.5px; color: #38BDF8; font-weight: 600; margin-bottom: 8px;'>[ NO ACTIVE AUDIT DATA ]</div>
                <h3 style='margin: 12px 0 6px 0; color: #F1F5F9;'>No Worker Audit Records Available</h3>
                <p style='color: #94A3B8; max-width: 520px; margin: 0 auto 24px auto; font-size: 13px; line-height: 1.6;'>
                    Worker audit records are generated following automated site inspection. Initiate an inspection or load historical data to inspect worker-level spatial evidence.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        c_emp1, c_emp2, c_emp3 = st.columns([1, 2, 1])
        with c_emp2:
            if st.button("Start New Site Inspection", key="audits_empty_new_btn", type="primary", use_container_width=True):
                navigate_to("New Inspection")
        return

    workers = active_inspection.get("workers", [])
    input_image = active_inspection["image"]
    summary = active_inspection["summary"]

    # 1. Inspection Context Header
    st.markdown(
        f"""
        <div class='sv-panel' style='padding: 10px 16px; margin-bottom: 12px; display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center;'>
            <div>
                <strong>Active Inspection:</strong> <code>{active_inspection['id']}</code> &nbsp;|&nbsp;
                <strong>Source:</strong> <code>{active_inspection['image_name']}</code> &nbsp;|&nbsp;
                <strong>Timestamp:</strong> <code>{active_inspection['timestamp']}</code>
            </div>
            <div>
                <span style='font-size: 11px; text-transform: uppercase; color: #94A3B8; font-weight: 600; margin-right: 8px;'>Total Workers:</span>
                <span class='badge-safe' style='font-size: 12px; padding: 3px 8px;'>{len(workers)} DETECTED</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not workers:
        st.warning("No workers were detected in this inspection scene. Try an image with clearer worker visibility.")
        if "annotated_image" in active_inspection:
            st.image(active_inspection["annotated_image"], caption="AI Annotated Inspection", use_container_width=True)
        if st.button("Back to New Inspection", key="audits_no_workers_back_btn"):
            navigate_to("New Inspection")
        return

    # 2. Worker Summary Overview (5 Compact Metrics)
    comp_w = summary.get("compliant_count", 0)
    viol_w = summary.get("violation_count", 0)
    review_w = summary.get("needs_review_count", summary.get("unknown_count", 0))
    rate = summary.get("compliance_rate", 0.0)

    viol_color = "#EF4444" if viol_w > 0 else "#94A3B8"
    review_color = "#F59E0B" if review_w > 0 else "#94A3B8"
    rate_color = "#22C55E" if rate >= 85.0 else ("#F59E0B" if rate >= 60.0 else "#EF4444")

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(
            f"<div class='metric-card'><div class='metric-card-title'>Total Workers</div>"
            f"<div class='metric-card-value'>{len(workers)}</div></div>",
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"<div class='metric-card'><div class='metric-card-title'>Compliant</div>"
            f"<div class='metric-card-value' style='color:#22C55E;'>{comp_w}</div></div>",
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"<div class='metric-card'><div class='metric-card-title'>Violations</div>"
            f"<div class='metric-card-value' style='color:{viol_color};'>{viol_w}</div></div>",
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f"<div class='metric-card'><div class='metric-card-title'>Needs Review</div>"
            f"<div class='metric-card-value' style='color:{review_color};'>{review_w}</div></div>",
            unsafe_allow_html=True,
        )
    with k5:
        st.markdown(
            f"<div class='metric-card'><div class='metric-card-title'>Compliance Rate</div>"
            f"<div class='metric-card-value' style='color:{rate_color};'>{rate:.1f}%</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    # 3. Worker Roster with Status Filtering
    st.markdown("#### Worker Compliance Roster")

    filter_choice = st.radio(
        "Filter Workers by Status:",
        ["All Workers", "Violations Only", "Compliant Only", "Needs Review"],
        horizontal=True,
        key="audits_roster_filter",
    )

    filtered_workers = workers
    if filter_choice == "Violations Only":
        filtered_workers = [w for w in workers if classify_worker_status(w.get("status", "")) == "violation"]
    elif filter_choice == "Compliant Only":
        filtered_workers = [w for w in workers if classify_worker_status(w.get("status", "")) == "compliant"]
    elif filter_choice == "Needs Review":
        filtered_workers = [w for w in workers if classify_worker_status(w.get("status", "")) == "warning"]

    if not filtered_workers:
        st.info(f"No workers match the '{filter_choice}' filter.")
    else:
        records = format_worker_table(filtered_workers)
        st.dataframe(pd.DataFrame(records), use_container_width=True)

    # 4. Individual Worker Spatial Evidence Drill-Down
    st.markdown("---")
    st.markdown("#### Individual Worker Spatial Evidence & ROI Zoom")

    worker_labels = [
        f"Worker #{w['worker_id']} — Status: {w['status'].upper()} (Confidence: {int(w.get('confidence', 1.0) * 100)}%)"
        for w in workers
    ]
    selected_label = st.selectbox("Select Worker for Granular Evidence Inspection:", worker_labels, key="audits_worker_select")
    selected_idx = worker_labels.index(selected_label)
    selected_worker = workers[selected_idx]

    worker_crop = crop_worker_roi(input_image, selected_worker, padding_pct=0.10, show_zones=True)

    col_crop, col_details = st.columns([5, 7])

    with col_crop:
        st.image(
            worker_crop,
            caption=f"Worker #{selected_worker['worker_id']} ROI (Head 0-35% | Torso 15-75% Boundaries)",
            use_container_width=True,
        )

    with col_details:
        w_status = selected_worker["status"]
        c_stat = classify_worker_status(w_status)
        stat_badge_class = "badge-safe" if c_stat == "compliant" else ("badge-warning" if c_stat == "warning" else "badge-critical")

        h_stat = selected_worker.get("hardhat_status", "missing")
        v_stat = selected_worker.get("vest_status", "missing")
        h_color = "#22C55E" if h_stat == "present" else ("#F59E0B" if h_stat == "unknown" else "#EF4444")
        v_color = "#22C55E" if v_stat == "present" else ("#F59E0B" if v_stat == "unknown" else "#EF4444")

        # Explain why worker is in this state
        if w_status == "Compliant":
            reason_text = "Verified compliant: Approved head protection (hardhat) and high-visibility safety vest detected within spatial zones."
        elif "Hardhat" in w_status and "Vest" in w_status:
            reason_text = "Critical violation: Both head protection (hardhat) and torso protection (safety vest) are missing."
        elif "Hardhat" in w_status:
            reason_text = "PPE violation: Hardhat missing or not detected within the designated head zone (0-35% of worker height)."
        elif "Vest" in w_status:
            reason_text = "PPE violation: High-visibility safety vest missing or not detected within torso zone (15-75% of worker height)."
        else:
            reason_text = "Attention required: Ambiguous detection geometry or low person confidence requiring supervisor verification."

        st.markdown(
            f"""
            <div class='sv-panel' style='padding: 16px; margin-bottom: 12px;'>
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;'>
                    <span style='font-size: 16px; font-weight: 700; color: #F1F5F9;'>Worker #{selected_worker['worker_id']} Profile</span>
                    <span class='{stat_badge_class}' style='font-size: 12px; padding: 3px 10px;'>{w_status.upper()}</span>
                </div>
                <div style='color: #94A3B8; font-size: 13px; line-height: 1.7;'>
                    <strong>Person Detection Confidence:</strong> <span style='color: #F1F5F9;'>{int(selected_worker.get('confidence', 1.0) * 100)}%</span><br>
                    <strong>Head Protection (Hardhat):</strong> <span style='color: {h_color}; font-weight: 600;'>{h_stat.capitalize()}</span><br>
                    <strong>Torso Protection (Safety Vest):</strong> <span style='color: {v_color}; font-weight: 600;'>{v_stat.capitalize()}</span><br>
                    <div style='margin-top: 8px; padding: 8px 10px; background: rgba(38, 52, 66, 0.4); border-radius: 4px; border-left: 3px solid {h_color if h_stat != "present" else v_color};'>
                        <span style='color: #F1F5F9;'>{reason_text}</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        assigned_gear = selected_worker.get("assigned_gear", [])
        if assigned_gear:
            st.markdown("<strong style='font-size: 13px; color: #F1F5F9;'>Assigned PPE Evidence Items:</strong>", unsafe_allow_html=True)
            gear_rows = [
                {
                    "Detected Item": g["class_name"],
                    "Confidence": f"{int(g.get('confidence', 1.0) * 100)}%",
                    "Zone Overlap (IoG)": f"{int(g.get('overlap', 0.0) * 100)}%",
                }
                for g in assigned_gear
            ]
            st.dataframe(pd.DataFrame(gear_rows), use_container_width=True, height=120)
        else:
            st.info("No safety gear detected inside this worker's head or torso zones.")

    # 5. Action Toolbar
    st.markdown("---")
    a_btn1, a_btn2, a_btn3 = st.columns(3)
    with a_btn1:
        if st.button("Return to Inspection Results", key="audits_back_to_results", use_container_width=True):
            navigate_to("Inspection Results")
    with a_btn2:
        if st.button("Conduct New Inspection", key="audits_new_insp_btn", use_container_width=True):
            navigate_to("New Inspection")
    with a_btn3:
        if st.button("Export Compliance Report", key="audits_export_btn", use_container_width=True):
            navigate_to("Reports & Exports")


# ============================================================================
# 10. PAGE 5: INSPECTION HISTORY
# ============================================================================
def render_inspection_history(history: List[Dict[str, Any]], model: Any):
    """Render persistent history of completed site inspections with review actions."""
    st.markdown(
        """
        <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom: 1.25rem;">
            <div>
                <h2 style="margin:0 0 4px 0; font-size:1.5rem; font-weight:700; color:#F1F5F9; letter-spacing:-0.01em;">Site Inspection History</h2>
                <div style="font-size:0.875rem; color:#94A3B8;">Review, track, and audit historical safety inspections across monitored site assets.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not history:
        st.markdown(
            """
            <div style="background:#162230; border:1px solid #263442; border-radius:8px; padding:2.75rem 1.5rem; text-align:center; margin-top:0.5rem;">
                <div style="display:inline-flex; align-items:center; justify-content:center; width:48px; height:48px; border-radius:8px; background:#223244; margin-bottom:1rem;">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#94A3B8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                        <polyline points="14 2 14 8 20 8"></polyline>
                        <line x1="16" y1="13" x2="8" y2="13"></line>
                        <line x1="16" y1="17" x2="8" y2="17"></line>
                        <polyline points="10 9 9 9 8 9"></polyline>
                    </svg>
                </div>
                <div style="font-size:1.05rem; font-weight:600; color:#F1F5F9; margin-bottom:0.35rem;">No Historical Inspections Found</div>
                <div style="font-size:0.875rem; color:#94A3B8; max-width:440px; margin:0 auto 1.5rem auto;">
                    Completed site inspections, compliance statistics, and spatial evidence records will be cataloged here.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_c, _ = st.columns([3, 7])
        with col_c:
            if st.button("Start New Inspection", type="primary", use_container_width=True):
                navigate_to("New Inspection")
        return

    # Overview KPI Section
    total_inspections = len(history)
    total_workers_audited = sum(int(h.get("total_workers", 0)) for h in history)
    total_violations = sum(int(h.get("violation_count", 0)) for h in history)
    avg_compliance = (
        round(sum(float(h.get("compliance_rate", 0.0)) for h in history) / total_inspections, 1)
        if total_inspections > 0
        else 0.0
    )

    avg_color = "#22C55E" if avg_compliance >= 90.0 else ("#F59E0B" if avg_compliance >= 75.0 else "#EF4444")
    viol_color = "#22C55E" if total_violations == 0 else "#EF4444"

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f"""
            <div style="background:#162230; border:1px solid #263442; border-radius:8px; padding:0.9rem 1rem; border-top:3px solid #38BDF8;">
                <div style="font-size:0.75rem; font-weight:600; color:#94A3B8; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.25rem;">Total Inspections</div>
                <div style="font-size:1.6rem; font-weight:700; color:#F1F5F9; line-height:1.2;">{total_inspections}</div>
                <div style="font-size:0.75rem; color:#94A3B8; margin-top:0.25rem;">Logged audit events</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""
            <div style="background:#162230; border:1px solid #263442; border-radius:8px; padding:0.9rem 1rem; border-top:3px solid #38BDF8;">
                <div style="font-size:0.75rem; font-weight:600; color:#94A3B8; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.25rem;">Workers Audited</div>
                <div style="font-size:1.6rem; font-weight:700; color:#F1F5F9; line-height:1.2;">{total_workers_audited}</div>
                <div style="font-size:0.75rem; color:#94A3B8; margin-top:0.25rem;">Cumulative detections</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"""
            <div style="background:#162230; border:1px solid #263442; border-radius:8px; padding:0.9rem 1rem; border-top:3px solid {avg_color};">
                <div style="font-size:0.75rem; font-weight:600; color:#94A3B8; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.25rem;">Avg Compliance</div>
                <div style="font-size:1.6rem; font-weight:700; color:{avg_color}; line-height:1.2;">{avg_compliance}%</div>
                <div style="font-size:0.75rem; color:#94A3B8; margin-top:0.25rem;">Site-wide performance</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f"""
            <div style="background:#162230; border:1px solid #263442; border-radius:8px; padding:0.9rem 1rem; border-top:3px solid {viol_color};">
                <div style="font-size:0.75rem; font-weight:600; color:#94A3B8; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.25rem;">Total Violations</div>
                <div style="font-size:1.6rem; font-weight:700; color:{viol_color}; line-height:1.2;">{total_violations}</div>
                <div style="font-size:0.75rem; color:#94A3B8; margin-top:0.25rem;">Flagged PPE breaches</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-bottom:1.25rem;'></div>", unsafe_allow_html=True)

    # Search, Filter and Sort Controls
    f_search, f_status, f_sort = st.columns([5, 4, 3])
    with f_search:
        search_query = st.text_input(
            "Search Records",
            placeholder="Search by ID, filename, or date...",
            label_visibility="collapsed",
            key="hist_search_query",
        ).strip().lower()
    with f_status:
        status_filter = st.selectbox(
            "Status Filter",
            options=["All Statuses", "Safe Only", "Warning Only", "Violation Only"],
            label_visibility="collapsed",
            key="hist_status_filter",
        )
    with f_sort:
        sort_order = st.selectbox(
            "Sort Order",
            options=["Newest First", "Oldest First", "Lowest Compliance", "Highest Compliance", "Most Violations"],
            label_visibility="collapsed",
            key="hist_sort_order",
        )

    # Filter records
    filtered = list(history)
    if search_query:
        filtered = [
            h for h in filtered
            if search_query in str(h.get("id", "")).lower()
            or search_query in str(h.get("image_name", "")).lower()
            or search_query in str(h.get("timestamp", "")).lower()
        ]

    if status_filter == "Safe Only":
        filtered = [h for h in filtered if str(h.get("site_status", "")).lower() in ["safe", "compliant"]]
    elif status_filter == "Warning Only":
        filtered = [h for h in filtered if str(h.get("site_status", "")).lower() in ["warning", "caution"]]
    elif status_filter == "Violation Only":
        filtered = [h for h in filtered if str(h.get("site_status", "")).lower() in ["violation", "unsafe", "critical"]]

    # Sort records
    if sort_order == "Newest First":
        pass  # History is already stored newest first
    elif sort_order == "Oldest First":
        filtered = list(reversed(filtered))
    elif sort_order == "Lowest Compliance":
        filtered = sorted(filtered, key=lambda x: float(x.get("compliance_rate", 0.0)))
    elif sort_order == "Highest Compliance":
        filtered = sorted(filtered, key=lambda x: float(x.get("compliance_rate", 0.0)), reverse=True)
    elif sort_order == "Most Violations":
        filtered = sorted(filtered, key=lambda x: int(x.get("violation_count", 0)), reverse=True)

    st.markdown(
        f"""
        <div style="display:flex; justify-content:space-between; align-items:center; margin:0.75rem 0 0.5rem 0; padding-bottom:0.4rem; border-bottom:1px solid #263442;">
            <div style="font-size:0.8rem; font-weight:600; color:#94A3B8; text-transform:uppercase; letter-spacing:0.05em;">
                Showing {len(filtered)} of {len(history)} Inspection Records
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not filtered:
        st.markdown(
            """
            <div style="background:#162230; border:1px solid #263442; border-radius:8px; padding:2rem 1.5rem; text-align:center; margin-top:0.5rem;">
                <div style="font-size:0.95rem; font-weight:600; color:#F1F5F9; margin-bottom:0.25rem;">No Matching Records Found</div>
                <div style="font-size:0.85rem; color:#94A3B8;">Adjust or clear your search query and filters to display historical inspections.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # List of Inspection Records
    for idx, item in enumerate(filtered):
        status = str(item.get("site_status", "Safe")).capitalize()
        if status in ["Safe", "Compliant"]:
            badge_bg = "rgba(34, 197, 94, 0.12)"
            badge_border = "rgba(34, 197, 94, 0.35)"
            badge_color = "#22C55E"
            status_label = "SAFE"
        elif status in ["Warning", "Caution"]:
            badge_bg = "rgba(245, 158, 11, 0.12)"
            badge_border = "rgba(245, 158, 11, 0.35)"
            badge_color = "#F59E0B"
            status_label = "WARNING"
        else:
            badge_bg = "rgba(239, 68, 68, 0.12)"
            badge_border = "rgba(239, 68, 68, 0.35)"
            badge_color = "#EF4444"
            status_label = "VIOLATION"

        item_id = item.get("id", f"INSP-{idx}")
        comp_rate = item.get("compliance_rate", 0.0)
        compliant_n = item.get("compliant_count", 0)
        total_w = item.get("total_workers", 0)
        viol_n = item.get("violation_count", 0)
        img_name = item.get("image_name", "Unknown Source")
        timestamp = item.get("timestamp", "Unknown Date")

        with st.container():
            c_meta, c_metrics, c_action = st.columns([5, 4, 3])

            with c_meta:
                st.markdown(
                    f"""
                    <div style="display:flex; flex-direction:column; gap:3px;">
                        <div style="display:flex; align-items:center; gap:8px;">
                            <span style="font-family:monospace; font-size:0.95rem; font-weight:700; color:#38BDF8;">{item_id}</span>
                            <span style="background:{badge_bg}; border:1px solid {badge_border}; color:{badge_color}; font-size:0.7rem; font-weight:700; padding:2px 7px; border-radius:4px; letter-spacing:0.04em;">{status_label}</span>
                        </div>
                        <div style="font-size:0.8rem; color:#94A3B8;">
                            Source: <code style="color:#CBD5E1; background:#0B1118; padding:1px 5px; border-radius:3px; border:1px solid #263442;">{img_name}</code>
                        </div>
                        <div style="font-size:0.75rem; color:#64748B;">
                            Logged: {timestamp}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with c_metrics:
                rate_color = "#22C55E" if comp_rate >= 90.0 else ("#F59E0B" if comp_rate >= 75.0 else "#EF4444")
                st.markdown(
                    f"""
                    <div style="display:flex; flex-direction:column; gap:3px;">
                        <div style="font-size:0.8rem; color:#94A3B8;">
                            Compliance Rate: <strong style="color:{rate_color}; font-size:0.95rem;">{comp_rate}%</strong>
                        </div>
                        <div style="font-size:0.8rem; color:#94A3B8;">
                            Personnel: <strong style="color:#F1F5F9;">{compliant_n}/{total_w}</strong> compliant
                        </div>
                        <div style="font-size:0.75rem; color:#94A3B8;">
                            Violations: <strong style="color:{'#EF4444' if viol_n > 0 else '#94A3B8'};">{viol_n}</strong> detected
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with c_action:
                st.markdown("<div style='padding-top:6px;'></div>", unsafe_allow_html=True)
                if st.button("Review Inspection", key=f"view_hist_{item_id}_{idx}", use_container_width=True):
                    if item_id in st.session_state.inspections_store:
                        st.session_state.active_inspection = st.session_state.inspections_store[item_id]
                        navigate_to("Inspection Results")
                    elif item.get("image_path") and os.path.exists(item["image_path"]):
                        with st.spinner(f"Reloading inspection for {img_name}..."):
                            reloaded_img = Image.open(item["image_path"]).convert("RGB")
                            req_mode = item.get("settings", {}).get("required_gear", "both")
                            conf_t = item.get("settings", {}).get("conf_threshold", 0.25)

                            detections = run_detection(reloaded_img, model=model, conf_threshold=conf_t, device="cpu")
                            _h2 = float(np.array(reloaded_img).shape[0])
                            evaluation = evaluate_site_compliance(
                                detections,
                                required_gear=req_mode,
                                image_height=_h2,
                                strict_hardhat=st.session_state.strict_hardhat,
                            )
                            annotated_img = annotate_image(reloaded_img, evaluation["workers"], evaluation["unassigned_gear"])
                            annotated_bytes = encode_image_bytes(annotated_img, format_ext="JPEG")

                            reloaded_record = {
                                "id": item_id,
                                "timestamp": item["timestamp"],
                                "image_name": item["image_name"],
                                "image_path": item["image_path"],
                                "image": reloaded_img,
                                "annotated_image": annotated_img,
                                "annotated_bytes": annotated_bytes,
                                "detections": detections,
                                "workers": evaluation["workers"],
                                "summary": evaluation["summary"],
                                "unassigned_gear": evaluation["unassigned_gear"],
                                "settings": item.get("settings", {}),
                            }
                            st.session_state.inspections_store[item_id] = reloaded_record
                            st.session_state.active_inspection = reloaded_record
                            navigate_to("Inspection Results")
                    else:
                        st.warning("Original source image file not located on disk. Cannot reconstruct session.")

            st.markdown("<div style='margin: 0.6rem 0; border-top: 1px solid #263442;'></div>", unsafe_allow_html=True)
# ============================================================================
# 11. PAGE 6: REPORTS & EXPORTS
# ============================================================================
def render_reports_exports(active_inspection: Optional[Dict[str, Any]]):
    """Render export portal for generating compliance audit reports and downloading evidence artifacts."""
    st.markdown(
        """
        <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom: 1.25rem;">
            <div>
                <h2 style="margin:0 0 4px 0; font-size:1.5rem; font-weight:700; color:var(--text-primary); letter-spacing:-0.01em;">Safety Reports & Exports</h2>
                <div style="font-size:0.875rem; color:var(--text-secondary);">Generate, review and export actionable safety inspection reports and photographic evidence.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    history = st.session_state.get("inspection_history", [])
    store = st.session_state.get("inspections_store", {})

    # If active_inspection is None but store has items, select the most recent one
    if active_inspection is None and store:
        active_inspection = next(iter(store.values()))

    # Empty State: No inspection data available at all
    if active_inspection is None:
        st.markdown(
            """
            <div style="background:var(--surface-card); border:1px solid var(--border-color); border-radius:8px; padding:2.75rem 1.5rem; text-align:center; margin-top:0.5rem;">
                <div style="display:inline-flex; align-items:center; justify-content:center; width:48px; height:48px; border-radius:8px; background:var(--surface-elevated); margin-bottom:1rem;">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--text-secondary)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                        <polyline points="14 2 14 8 20 8"></polyline>
                        <line x1="16" y1="13" x2="8" y2="13"></line>
                        <line x1="16" y1="17" x2="8" y2="17"></line>
                        <polyline points="10 9 9 9 8 9"></polyline>
                    </svg>
                </div>
                <div style="font-size:1.05rem; font-weight:600; color:var(--text-primary); margin-bottom:0.35rem;">No Inspection Data Available</div>
                <div style="font-size:0.875rem; color:var(--text-secondary); max-width:440px; margin:0 auto 1.5rem auto;">
                    Execute a site safety inspection or select a recorded audit from history to generate compliance reports and download evidence files.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_c, _ = st.columns([3, 7])
        with col_c:
            if st.button("Start New Inspection", type="primary", use_container_width=True):
                navigate_to("New Inspection")
        return

    # Overview KPI Section (using existing history or active inspection metrics)
    if history:
        ov_total_inspections = len(history)
        ov_total_workers = sum(int(h.get("total_workers", 0)) for h in history)
        ov_avg_compliance = round(sum(float(h.get("compliance_rate", 0.0)) for h in history) / ov_total_inspections, 1)
        ov_total_violations = sum(int(h.get("violation_count", 0)) for h in history)
    else:
        ov_total_inspections = 1
        ov_total_workers = int(active_inspection.get("summary", {}).get("total_workers", 0))
        ov_avg_compliance = float(active_inspection.get("summary", {}).get("compliance_rate", 0.0))
        ov_total_violations = int(active_inspection.get("summary", {}).get("violation_count", 0))

    avg_color = "#22C55E" if ov_avg_compliance >= 90.0 else ("#F59E0B" if ov_avg_compliance >= 75.0 else "#EF4444")
    viol_color = "#22C55E" if ov_total_violations == 0 else "#EF4444"

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f"""
            <div style="background:var(--surface-card); border:1px solid var(--border-color); border-radius:8px; padding:0.9rem 1rem; border-top:3px solid var(--accent-primary);">
                <div style="font-size:0.75rem; font-weight:600; color:var(--text-secondary); text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.25rem;">Total Inspections</div>
                <div style="font-size:1.6rem; font-weight:700; color:var(--text-primary); line-height:1.2;">{ov_total_inspections}</div>
                <div style="font-size:0.75rem; color:var(--text-secondary); margin-top:0.25rem;">Logged audit events</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""
            <div style="background:var(--surface-card); border:1px solid var(--border-color); border-radius:8px; padding:0.9rem 1rem; border-top:3px solid var(--accent-primary);">
                <div style="font-size:0.75rem; font-weight:600; color:var(--text-secondary); text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.25rem;">Workers Analyzed</div>
                <div style="font-size:1.6rem; font-weight:700; color:var(--text-primary); line-height:1.2;">{ov_total_workers}</div>
                <div style="font-size:0.75rem; color:var(--text-secondary); margin-top:0.25rem;">Cumulative detections</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"""
            <div style="background:var(--surface-card); border:1px solid var(--border-color); border-radius:8px; padding:0.9rem 1rem; border-top:3px solid {avg_color};">
                <div style="font-size:0.75rem; font-weight:600; color:var(--text-secondary); text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.25rem;">Avg Compliance</div>
                <div style="font-size:1.6rem; font-weight:700; color:{avg_color}; line-height:1.2;">{ov_avg_compliance}%</div>
                <div style="font-size:0.75rem; color:var(--text-secondary); margin-top:0.25rem;">Site-wide rate</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f"""
            <div style="background:var(--surface-card); border:1px solid var(--border-color); border-radius:8px; padding:0.9rem 1rem; border-top:3px solid {viol_color};">
                <div style="font-size:0.75rem; font-weight:600; color:var(--text-secondary); text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.25rem;">Total Violations</div>
                <div style="font-size:1.6rem; font-weight:700; color:{viol_color}; line-height:1.2;">{ov_total_violations}</div>
                <div style="font-size:0.75rem; color:var(--text-secondary); margin-top:0.25rem;">Recorded breaches</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-bottom:1.25rem;'></div>", unsafe_allow_html=True)

    # Optional Multi-Inspection Selector if multiple sessions are in memory
    if len(store) > 1:
        store_keys = list(store.keys())
        current_key = active_inspection.get("id")
        default_idx = store_keys.index(current_key) if current_key in store_keys else 0
        selected_target = st.selectbox(
            "Target Inspection Record",
            options=store_keys,
            index=default_idx,
            format_func=lambda k: f"{k} — {store[k].get('image_name', 'Image')} ({store[k].get('timestamp', '')})",
            key="rep_target_selector",
        )
        active_inspection = store[selected_target]

    summary = active_inspection["summary"]
    raw_workers = active_inspection["workers"]
    timestamp_slug = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Report Generation Controls
    st.markdown(
        """
        <div style="font-size:0.8rem; font-weight:700; color:var(--text-secondary); text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.4rem;">
            Report Generation & Filter Parameters
        </div>
        """,
        unsafe_allow_html=True,
    )

    c_scope, c_filter = st.columns(2)
    with c_scope:
        report_scope = st.selectbox(
            "Report Scope",
            options=[
                "Full Regulatory Compliance Audit (All Sections)",
                "Executive Safety Summary Only",
                "Exception Log: Violations & Breaches Only",
            ],
            key="rep_scope_select",
        )
    with c_filter:
        worker_filter = st.selectbox(
            "Worker Status Filter",
            options=["All Detected Workers", "Violations Only", "Compliant Workers Only", "Needs Review Only"],
            key="rep_worker_filter",
        )

    with st.expander("Advanced Regulatory Metadata & Audit Headers", expanded=False):
        meta_c1, meta_c2 = st.columns(2)
        with meta_c1:
            auditor_id = st.text_input("Auditor / Operator ID", value="SV-INSPECTOR-01", key="rep_auditor_id")
            facility_id = st.text_input("Site / Facility Location", value="Zone 4 - Construction Bay", key="rep_facility_id")
        with meta_c2:
            reg_standard = st.text_input("Regulatory Standard Reference", value="OSHA 1926.100 & ANSI/ISEA 107-2020", key="rep_reg_std")
            notes = st.text_area("Audit Notes & Compliance Directives", value="Automated spatial PPE audit executed via YOLOv8 computer vision surveillance.", height=68, key="rep_notes")

    # Filter workers based on controls
    if worker_filter == "Violations Only" or "Violations & Breaches Only" in report_scope:
        filtered_workers = [w for w in raw_workers if classify_worker_status(w.get("status", "")) == "violation"]
    elif worker_filter == "Compliant Workers Only":
        filtered_workers = [w for w in raw_workers if classify_worker_status(w.get("status", "")) == "compliant"]
    elif worker_filter == "Needs Review Only":
        filtered_workers = [w for w in raw_workers if classify_worker_status(w.get("status", "")) == "warning"]
    else:
        filtered_workers = raw_workers

    # Executive Inspection Summary Card
    status = str(summary.get("site_status", "Safe")).capitalize()
    if status in ["Safe", "Compliant"]:
        badge_bg = "rgba(34, 197, 94, 0.12)"
        badge_border = "rgba(34, 197, 94, 0.35)"
        badge_color = "#22C55E"
        status_label = "SAFE"
    elif status in ["Warning", "Caution"]:
        badge_bg = "rgba(245, 158, 11, 0.12)"
        badge_border = "rgba(245, 158, 11, 0.35)"
        badge_color = "#F59E0B"
        status_label = "WARNING"
    else:
        badge_bg = "rgba(239, 68, 68, 0.12)"
        badge_border = "rgba(239, 68, 68, 0.35)"
        badge_color = "#EF4444"
        status_label = "VIOLATION"

    st.markdown(
        f"""
        <div style="background:var(--surface-card); border:1px solid var(--border-color); border-radius:8px; padding:1.2rem 1.35rem; margin-top:0.75rem; margin-bottom:1.25rem;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.75rem; padding-bottom:0.6rem; border-bottom:1px solid var(--border-color);">
                <div>
                    <span style="font-size:0.75rem; font-weight:700; color:var(--text-secondary); text-transform:uppercase; letter-spacing:0.05em;">Audit Target Record</span>
                    <div style="font-size:1.15rem; font-weight:700; color:var(--text-primary); margin-top:2px; font-family:monospace;">
                        {active_inspection['id']}
                    </div>
                </div>
                <div style="background:{badge_bg}; border:1px solid {badge_border}; color:{badge_color}; font-size:0.75rem; font-weight:700; padding:4px 10px; border-radius:4px; letter-spacing:0.04em;">
                    SITE STATUS: {status_label}
                </div>
            </div>
            <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap:12px; font-size:0.85rem; color:var(--text-secondary);">
                <div><strong style="color:var(--text-primary);">Source Asset:</strong> <code style="color:var(--accent-primary); background:var(--bg-main); padding:1px 5px; border-radius:3px; border:1px solid var(--border-color);">{active_inspection['image_name']}</code></div>
                <div><strong style="color:var(--text-primary);">Execution Time:</strong> {active_inspection['timestamp']}</div>
                <div><strong style="color:var(--text-primary);">Compliance Grade:</strong> <span style="color:{badge_color}; font-weight:700;">{summary['compliance_rate']}%</span></div>
                <div><strong style="color:var(--text-primary);">Personnel Breakdown:</strong> {summary['total_workers']} Detected | {summary['compliant_count']} Compliant | {summary['violation_count']} Violations</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Worker Record Table Preview
    st.markdown(
        f"""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
            <div style="font-size:0.8rem; font-weight:700; color:var(--text-secondary); text-transform:uppercase; letter-spacing:0.05em;">
                Worker Compliance Records ({len(filtered_workers)} of {len(raw_workers)} personnel)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if filtered_workers:
        st.dataframe(pd.DataFrame(format_worker_table(filtered_workers)), use_container_width=True)
    else:
        st.markdown(
            """
            <div style="background:var(--surface-card); border:1px solid var(--border-color); border-radius:8px; padding:1.5rem; text-align:center; margin-bottom:1rem;">
                <div style="font-size:0.9rem; font-weight:600; color:var(--text-primary); margin-bottom:0.25rem;">No Workers Match Filter</div>
                <div style="font-size:0.8rem; color:var(--text-secondary);">There are no worker audit records matching the currently selected status filter.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Export Artifacts Section
    st.markdown(
        """
        <div style="font-size:0.8rem; font-weight:700; color:var(--text-secondary); text-transform:uppercase; letter-spacing:0.05em; margin-top:1.25rem; margin-bottom:0.6rem;">
            Export Compliance Artifacts
        </div>
        """,
        unsafe_allow_html=True,
    )

    csv_data = generate_csv_report(summary, filtered_workers, active_inspection["timestamp"])
    annotated_bytes = active_inspection.get("annotated_bytes")
    if annotated_bytes is None and "annotated_image" in active_inspection:
        annotated_bytes = encode_image_bytes(active_inspection["annotated_image"])

    json_data = json.dumps(
        {
            "inspection_id": active_inspection.get("id"),
            "timestamp": active_inspection.get("timestamp"),
            "image_name": active_inspection.get("image_name"),
            "auditor_id": auditor_id if 'auditor_id' in locals() else "SV-INSPECTOR-01",
            "facility_id": facility_id if 'facility_id' in locals() else "Zone 4",
            "regulatory_standard": reg_standard if 'reg_standard' in locals() else "OSHA 1926.100 & ANSI/ISEA 107",
            "summary": summary,
            "workers": filtered_workers,
            "settings": active_inspection.get("settings", {}),
        },
        indent=2,
    )

    exp_c1, exp_c2, exp_c3 = st.columns(3)

    with exp_c1:
        st.markdown(
            """
            <div style="background:var(--surface-card); border:1px solid var(--border-color); border-radius:8px; padding:1rem; min-height:120px; display:flex; flex-direction:column; justify-content:space-between; margin-bottom:8px;">
                <div>
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                        <span style="font-size:0.9rem; font-weight:700; color:var(--text-primary);">CSV Audit Report</span>
                        <span style="font-size:0.68rem; font-weight:700; background:rgba(56, 189, 248, 0.15); border:1px solid rgba(56, 189, 248, 0.35); color:#38BDF8; padding:2px 6px; border-radius:3px;">.CSV</span>
                    </div>
                    <div style="font-size:0.75rem; color:var(--text-secondary); line-height:1.4;">
                        Official audit spreadsheet with worker-by-worker gear breakdown and compliance metrics.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.download_button(
            label="Download Audit Report (.csv)",
            data=csv_data,
            file_name=f"PPE_Safety_Inspection_{timestamp_slug}.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True,
            key="btn_download_csv",
        )

    with exp_c2:
        st.markdown(
            """
            <div style="background:var(--surface-card); border:1px solid var(--border-color); border-radius:8px; padding:1rem; min-height:120px; display:flex; flex-direction:column; justify-content:space-between; margin-bottom:8px;">
                <div>
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                        <span style="font-size:0.9rem; font-weight:700; color:var(--text-primary);">Annotated Evidence</span>
                        <span style="font-size:0.68rem; font-weight:700; background:rgba(34, 197, 94, 0.15); border:1px solid rgba(34, 197, 94, 0.35); color:#22C55E; padding:2px 6px; border-radius:3px;">.JPG</span>
                    </div>
                    <div style="font-size:0.75rem; color:var(--text-secondary); line-height:1.4;">
                        Photographic proof with localized bounding boxes, anatomical zones, and compliance tags.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if annotated_bytes:
            st.download_button(
                label="Download Evidence Photo (.jpg)",
                data=annotated_bytes,
                file_name=f"Annotated_Evidence_{timestamp_slug}.jpg",
                mime="image/jpeg",
                use_container_width=True,
                key="btn_download_jpg",
            )
        else:
            st.button("Evidence Unavailable", disabled=True, use_container_width=True, key="btn_download_jpg_dis")

    with exp_c3:
        st.markdown(
            """
            <div style="background:var(--surface-card); border:1px solid var(--border-color); border-radius:8px; padding:1rem; min-height:120px; display:flex; flex-direction:column; justify-content:space-between; margin-bottom:8px;">
                <div>
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                        <span style="font-size:0.9rem; font-weight:700; color:var(--text-primary);">JSON Audit Record</span>
                        <span style="font-size:0.68rem; font-weight:700; background:rgba(245, 158, 11, 0.15); border:1px solid rgba(245, 158, 11, 0.35); color:#F59E0B; padding:2px 6px; border-radius:3px;">.JSON</span>
                    </div>
                    <div style="font-size:0.75rem; color:var(--text-secondary); line-height:1.4;">
                        Machine-readable JSON schema containing complete spatial detection telemetry.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.download_button(
            label="Download Audit Data (.json)",
            data=json_data,
            file_name=f"Safety_Audit_Data_{timestamp_slug}.json",
            mime="application/json",
            use_container_width=True,
            key="btn_download_json",
        )
# ============================================================================
# 12. PAGE 7: SETTINGS & MODEL SPECIFICATIONS
# ============================================================================
def render_settings(model: Any, is_model_verified: bool):
    """Render system settings, model information, and application preferences."""
    import platform
    import sys
    try:
        import torch
        torch_version = torch.__version__
    except ImportError:
        torch_version = "2.15.0+cpu"

    st.markdown(
        """
        <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom: 1.25rem;">
            <div>
                <h2 style="margin:0 0 4px 0; font-size:1.5rem; font-weight:700; color:var(--text-primary); letter-spacing:-0.01em;">System Settings</h2>
                <div style="font-size:0.875rem; color:var(--text-secondary);">Configuration, model information and system preferences.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_det, tab_model, tab_pref, tab_sys = st.tabs([
        "Detection Configuration",
        "Model Information",
        "Application Preferences",
        "System Information",
    ])

    # =========================================================================
    # TAB 1: DETECTION CONFIGURATION
    # =========================================================================
    with tab_det:
        st.markdown(
            """
            <div style="font-size:0.85rem; font-weight:700; color:var(--text-secondary); text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.75rem;">
                Surveillance Detection Policy & Confidence Controls
            </div>
            """,
            unsafe_allow_html=True,
        )

        c_pol1, c_pol2 = st.columns(2)
        with c_pol1:
            strict_val = st.toggle(
                "Enforce Strict Hardhat Protocol",
                value=st.session_state.get("strict_hardhat", True),
                help="When enabled, workers lacking positive hardhat detection in the cranial ROI are classified as non-compliant.",
            )
            if strict_val != st.session_state.get("strict_hardhat", True):
                st.session_state.strict_hardhat = strict_val

            st.markdown(
                """
                <div style="font-size:0.75rem; color:var(--text-secondary); line-height:1.4; margin-top:-0.4rem; margin-bottom:0.8rem;">
                    Strict protocol requires positive hardhat evidence in the top cranial zone. Unverified workers are flagged as non-compliant.
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c_pol2:
            current_gear = st.session_state.get("required_gear", "both")
            gear_options = ["both", "hardhat_only", "vest_only"]
            gear_labels = {
                "both": "Hardhat and Safety Vest (Full PPE)",
                "hardhat_only": "Hardhat Only (Cranial Protection)",
                "vest_only": "Safety Vest Only (High-Visibility)",
            }
            selected_gear = st.selectbox(
                "Default Required PPE Policy",
                options=gear_options,
                index=gear_options.index(current_gear) if current_gear in gear_options else 0,
                format_func=lambda x: gear_labels.get(x, x),
            )
            if selected_gear != current_gear:
                st.session_state.required_gear = selected_gear

        current_conf = float(st.session_state.get("default_conf_threshold", 0.25))
        new_conf = st.slider(
            "Default Inference Confidence Threshold",
            min_value=0.05,
            max_value=0.90,
            value=current_conf,
            step=0.05,
            help="Global default confidence threshold for candidate detection boxes before spatial zone reasoning.",
        )
        if new_conf != current_conf:
            st.session_state.default_conf_threshold = new_conf

        st.markdown("<div style='margin-top:1.25rem;'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="font-size:0.85rem; font-weight:700; color:var(--text-secondary); text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.75rem;">
                Spatial Reasoning Engine Architecture & Fixed Parameters
            </div>
            """,
            unsafe_allow_html=True,
        )

        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.markdown(
                f"""
                <div style="background:var(--surface-card); border:1px solid var(--border-color); border-radius:8px; padding:0.85rem 1rem; border-top:3px solid var(--accent-primary);">
                    <div style="font-size:0.72rem; font-weight:600; color:var(--text-secondary); text-transform:uppercase; letter-spacing:0.04em;">Head ROI Zone</div>
                    <div style="font-size:1.3rem; font-weight:700; color:var(--text-primary); margin:0.2rem 0;">0.0 – {HEAD_ZONE_BOTTOM}</div>
                    <div style="font-size:0.72rem; color:var(--text-secondary);">Top {int(HEAD_ZONE_BOTTOM * 100)}% of worker bbox</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with s2:
            st.markdown(
                f"""
                <div style="background:var(--surface-card); border:1px solid var(--border-color); border-radius:8px; padding:0.85rem 1rem; border-top:3px solid var(--accent-primary);">
                    <div style="font-size:0.72rem; font-weight:600; color:var(--text-secondary); text-transform:uppercase; letter-spacing:0.04em;">Torso ROI Zone</div>
                    <div style="font-size:1.3rem; font-weight:700; color:var(--text-primary); margin:0.2rem 0;">{TORSO_ZONE_TOP} – {TORSO_ZONE_BOTTOM}</div>
                    <div style="font-size:0.72rem; color:var(--text-secondary);">{int(TORSO_ZONE_TOP * 100)}% to {int(TORSO_ZONE_BOTTOM * 100)}% bbox range</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with s3:
            st.markdown(
                f"""
                <div style="background:var(--surface-card); border:1px solid var(--border-color); border-radius:8px; padding:0.85rem 1rem; border-top:3px solid var(--accent-primary);">
                    <div style="font-size:0.72rem; font-weight:600; color:var(--text-secondary); text-transform:uppercase; letter-spacing:0.04em;">Spatial Overlap</div>
                    <div style="font-size:1.3rem; font-weight:700; color:var(--text-primary); margin:0.2rem 0;">&ge; {int(OVERLAP_THRESHOLD * 100)}%</div>
                    <div style="font-size:0.72rem; color:var(--text-secondary);">IoG (Intersection over Gear)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with s4:
            st.markdown(
                f"""
                <div style="background:var(--surface-card); border:1px solid var(--border-color); border-radius:8px; padding:0.85rem 1rem; border-top:3px solid #22C55E;">
                    <div style="font-size:0.72rem; font-weight:600; color:var(--text-secondary); text-transform:uppercase; letter-spacing:0.04em;">Safety Thresholds</div>
                    <div style="font-size:1.3rem; font-weight:700; color:#22C55E; margin:0.2rem 0;">&ge; {SITE_SAFE_THRESHOLD}%</div>
                    <div style="font-size:0.72rem; color:var(--text-secondary);">Warning tier: &ge; {SITE_WARNING_THRESHOLD}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # =========================================================================
    # TAB 2: MODEL INFORMATION
    # =========================================================================
    with tab_model:
        st.markdown(
            """
            <div style="font-size:0.85rem; font-weight:700; color:var(--text-secondary); text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.75rem;">
                YOLOv8 Detection Engine Architecture & Status
            </div>
            """,
            unsafe_allow_html=True,
        )

        active_path = resolve_model_path()
        status_text = "Verified (10/10 Classes)" if is_model_verified else "Fallback Mode"
        status_color = "#22C55E" if is_model_verified else "#F59E0B"

        m1, m2 = st.columns(2)
        with m1:
            st.markdown(
                f"""
                <div style="background:var(--surface-card); border:1px solid var(--border-color); border-radius:8px; padding:1.1rem; margin-bottom:0.75rem;">
                    <div style="font-size:0.8rem; font-weight:600; color:var(--text-secondary); margin-bottom:4px;">MODEL WEIGHTS FILE</div>
                    <div style="font-family:monospace; font-size:0.95rem; font-weight:700; color:var(--accent-primary); word-break:break-all;">{active_path}</div>
                    <div style="font-size:0.75rem; color:var(--text-secondary); margin-top:6px;">
                        Architecture: <code>YOLOv8n (Nano)</code> | Runtime: <code>CPU Mode (PyTorch)</code>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f"""
                <div style="background:var(--surface-card); border:1px solid var(--border-color); border-radius:8px; padding:1.1rem; margin-bottom:0.75rem;">
                    <div style="font-size:0.8rem; font-weight:600; color:var(--text-secondary); margin-bottom:4px;">CLASS TAXONOMY INTEGRITY</div>
                    <div style="font-size:1.05rem; font-weight:700; color:{status_color};">
                        {status_text}
                    </div>
                    <div style="font-size:0.75rem; color:var(--text-secondary); margin-top:6px;">
                        Dataset: <code>Roboflow Construction Site Safety (v28)</code> | Caching: <code>Active</code>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            """
            <div style="font-size:0.85rem; font-weight:700; color:var(--text-secondary); text-transform:uppercase; letter-spacing:0.05em; margin:0.75rem 0 0.4rem 0;">
                Verified Class Taxonomy & Spatial Roles
            </div>
            """,
            unsafe_allow_html=True,
        )

        class_table_data = [
            {
                "Class ID": cid,
                "Class Name": cname,
                "Application Role": (
                    "Base Worker Bounding Box" if cid == 5
                    else ("Head Protection Evidence" if cid in (0, 2)
                    else ("Torso Protection Evidence" if cid in (7, 4)
                    else "Contextual Scene Class (Ignored in logic)"))
                ),
            }
            for cid, cname in EXPECTED_CLASSES.items()
        ]
        st.dataframe(pd.DataFrame(class_table_data), use_container_width=True)

    # =========================================================================
    # TAB 3: APPLICATION PREFERENCES
    # =========================================================================
    with tab_pref:
        st.markdown(
            """
            <div style="font-size:0.85rem; font-weight:700; color:var(--text-secondary); text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.75rem;">
                Interface Theme & Display Preferences
            </div>
            """,
            unsafe_allow_html=True,
        )

        pref_col1, pref_col2 = st.columns(2)
        with pref_col1:
            theme_choice = st.radio(
                "Interface Theme Mode",
                ["Dark Theme", "Light Theme"],
                index=0 if st.session_state.theme == "Dark Theme" else 1,
                horizontal=True,
                key="settings_theme_radio",
            )
            if theme_choice != st.session_state.theme:
                st.session_state.theme = theme_choice
                st.rerun()

            st.markdown(
                """
                <div style="font-size:0.75rem; color:var(--text-secondary); line-height:1.4; margin-top:-0.4rem;">
                    Dark Theme optimized for low-glare industrial control centers. Light Theme engineered for high-contrast day operations.
                </div>
                """,
                unsafe_allow_html=True,
            )

        with pref_col2:
            current_mode = st.session_state.get("inspection_mode", "Image")
            mode_choice = st.radio(
                "Default Surveillance Workflow",
                ["Image Inspection", "CCTV Video Inspection"],
                index=0 if current_mode == "Image" else 1,
                horizontal=True,
                key="settings_mode_radio",
            )
            mapped_mode = "Image" if "Image" in mode_choice else "Video"
            if mapped_mode != st.session_state.get("inspection_mode", "Image"):
                st.session_state.inspection_mode = mapped_mode

        st.markdown("<div style='margin-top:1.25rem;'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="font-size:0.85rem; font-weight:700; color:var(--text-secondary); text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.75rem;">
                Design System Palette Verification
            </div>
            """,
            unsafe_allow_html=True,
        )

        p1, p2, p3, p4 = st.columns(4)
        with p1:
            st.markdown(
                """
                <div style="background:#162230; border:1px solid #263442; border-radius:6px; padding:0.6rem 0.8rem;">
                    <div style="font-size:0.7rem; font-weight:700; color:#94A3B8;">PANEL SURFACE</div>
                    <div style="font-family:monospace; font-size:0.85rem; color:#F1F5F9;">#162230</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with p2:
            st.markdown(
                """
                <div style="background:#162230; border:1px solid #38BDF8; border-radius:6px; padding:0.6rem 0.8rem;">
                    <div style="font-size:0.7rem; font-weight:700; color:#38BDF8;">ACCENT PRIMARY</div>
                    <div style="font-family:monospace; font-size:0.85rem; color:#38BDF8;">#38BDF8</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with p3:
            st.markdown(
                """
                <div style="background:#162230; border:1px solid #22C55E; border-radius:6px; padding:0.6rem 0.8rem;">
                    <div style="font-size:0.7rem; font-weight:700; color:#22C55E;">SAFE / COMPLIANT</div>
                    <div style="font-family:monospace; font-size:0.85rem; color:#22C55E;">#22C55E</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with p4:
            st.markdown(
                """
                <div style="background:#162230; border:1px solid #EF4444; border-radius:6px; padding:0.6rem 0.8rem;">
                    <div style="font-size:0.7rem; font-weight:700; color:#EF4444;">VIOLATION / UNSAFE</div>
                    <div style="font-family:monospace; font-size:0.85rem; color:#EF4444;">#EF4444</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # =========================================================================
    # TAB 4: SYSTEM INFORMATION
    # =========================================================================
    with tab_sys:
        st.markdown(
            """
            <div style="font-size:0.85rem; font-weight:700; color:var(--text-secondary); text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.75rem;">
                Platform Environment & Runtime Telemetry
            </div>
            """,
            unsafe_allow_html=True,
        )

        sys1, sys2, sys3, sys4 = st.columns(4)
        with sys1:
            st.markdown(
                f"""
                <div style="background:var(--surface-card); border:1px solid var(--border-color); border-radius:8px; padding:0.85rem 1rem; border-top:3px solid var(--accent-primary);">
                    <div style="font-size:0.72rem; font-weight:600; color:var(--text-secondary); text-transform:uppercase; letter-spacing:0.04em;">Platform Engine</div>
                    <div style="font-size:1.1rem; font-weight:700; color:var(--text-primary); margin:0.2rem 0;">Safety Vision</div>
                    <div style="font-size:0.72rem; color:var(--text-secondary);">Enterprise v2.4</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with sys2:
            st.markdown(
                f"""
                <div style="background:var(--surface-card); border:1px solid var(--border-color); border-radius:8px; padding:0.85rem 1rem; border-top:3px solid var(--accent-primary);">
                    <div style="font-size:0.72rem; font-weight:600; color:var(--text-secondary); text-transform:uppercase; letter-spacing:0.04em;">Python Core</div>
                    <div style="font-size:1.1rem; font-weight:700; color:var(--text-primary); margin:0.2rem 0;">v{sys.version.split()[0]}</div>
                    <div style="font-size:0.72rem; color:var(--text-secondary);">{platform.system()} {platform.release()}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with sys3:
            st.markdown(
                f"""
                <div style="background:var(--surface-card); border:1px solid var(--border-color); border-radius:8px; padding:0.85rem 1rem; border-top:3px solid var(--accent-primary);">
                    <div style="font-size:0.72rem; font-weight:600; color:var(--text-secondary); text-transform:uppercase; letter-spacing:0.04em;">PyTorch Engine</div>
                    <div style="font-size:1.1rem; font-weight:700; color:var(--text-primary); margin:0.2rem 0;">{torch_version}</div>
                    <div style="font-size:0.72rem; color:var(--text-secondary);">CPU Compute Device</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with sys4:
            st.markdown(
                f"""
                <div style="background:var(--surface-card); border:1px solid var(--border-color); border-radius:8px; padding:0.85rem 1rem; border-top:3px solid #22C55E;">
                    <div style="font-size:0.72rem; font-weight:600; color:var(--text-secondary); text-transform:uppercase; letter-spacing:0.04em;">System Status</div>
                    <div style="font-size:1.1rem; font-weight:700; color:#22C55E; margin:0.2rem 0;">OPERATIONAL</div>
                    <div style="font-size:0.72rem; color:var(--text-secondary);">Streamlit {st.__version__}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='margin-top:1.25rem;'></div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="background:var(--surface-card); border:1px solid var(--border-color); border-radius:8px; padding:1.1rem 1.25rem;">
                <div style="font-size:0.85rem; font-weight:700; color:var(--text-primary); margin-bottom:0.4rem;">Data Storage & Persistence Directives</div>
                <div style="font-size:0.8rem; color:var(--text-secondary); line-height:1.5;">
                    Historical safety inspection sessions are serialized to <code>eval/inspection_history.json</code>.<br>
                    Active image inferences and spatial evidence arrays reside in in-memory session state (<code>st.session_state.inspections_store</code>).
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
# ============================================================================
# 13. PAGE: CCTV-STYLE VIDEO PPE INSPECTION
# ============================================================================
def render_video_inspection(model: Any):
    """Render the CCTV video PPE compliance surveillance interface."""
    # A. Page Header
    st.markdown("### Video Safety Inspection")
    st.caption("Automated sequential video analysis, interval inference stride, and temporal compliance tracking.")

    cached_results = st.session_state.get("video_results")
    if cached_results is not None:
        st.markdown(
            """
            <div class='sv-panel' style='padding: 10px 16px; margin-bottom: 14px; display: flex; justify-content: space-between; align-items: center;'>
                <div>
                    <strong style='color: #38BDF8;'>Surveillance Cache:</strong>
                    <span style='color: #F1F5F9; margin-left: 6px;'>Completed video inspection results are available for review.</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        c_v1, c_v2, c_v3 = st.columns([3, 1, 1])
        with c_v2:
            if st.button("View Cached Results", key="vid_view_cached_btn", use_container_width=True):
                navigate_to("Inspection Results")

    # 2-Column Workflow Layout
    col_source, col_config = st.columns([7, 5])

    with col_source:
        st.markdown("#### Video Source Selection")
        video_source = st.radio(
            "Source Selection",
            ["Upload Video File", "YouTube Stream Link"],
            horizontal=True,
            key="cctv_video_source",
            label_visibility="collapsed",
        )

        uploaded_video = None
        youtube_url = ""

        if video_source == "Upload Video File":
            uploaded_video = st.file_uploader(
                "Upload Site Surveillance Video (.mp4, .avi, .mov)",
                type=["mp4", "avi", "mov"],
                help=f"Upload site surveillance footage. Processing is capped at the first {MAX_VIDEO_SECONDS}s from start.",
                key="cctv_file_uploader",
            )

            if uploaded_video is not None:
                file_mb = round(uploaded_video.size / (1024 * 1024), 2)
                st.markdown(
                    f"""
                    <div class='sv-surface-elevated' style='padding: 10px 14px; margin-top: 10px; font-size: 12px; line-height: 1.6;'>
                        <div><strong>Selected Video:</strong> <code>{uploaded_video.name}</code></div>
                        <div style='color: #94A3B8;'><strong>File Size:</strong> <code>{file_mb} MB</code> &nbsp;|&nbsp; <strong>Processing Limit:</strong> First {MAX_VIDEO_SECONDS}s (Width &le; {MAX_FRAME_WIDTH}px)</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class='sv-panel' style='text-align: center; padding: 36px 20px; border: 1px dashed #263442; margin-top: 10px;'>
                        <div style='color: #38BDF8; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; margin-bottom: 6px;'>[ NO VIDEO LOADED ]</div>
                        <div style='color: #94A3B8; font-size: 13px;'>Drop or select an MP4, AVI, or MOV surveillance video above.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                """
                <div class='sv-panel' style='padding: 10px 14px; margin-bottom: 10px; font-size: 12px; color: #94A3B8;'>
                    Notice: Ensure you possess authorization to analyze target footage. Video downloads from streaming sources are subject to platform terms.
                </div>
                """,
                unsafe_allow_html=True,
            )
            youtube_url = st.text_input(
                "YouTube Video URL",
                placeholder="https://www.youtube.com/watch?v=...",
                key="cctv_youtube_url",
                help="Paste a valid YouTube link starting with https://www.youtube.com/, https://youtube.com/, https://m.youtube.com/, or https://youtu.be/",
            )
            st.caption(
                f"Limits: max duration {MAX_YOUTUBE_DURATION_SECONDS // 60} mins ({MAX_YOUTUBE_DURATION_SECONDS}s), "
                f"max download size {MAX_YOUTUBE_MB} MB. Analyzes up to {MAX_VIDEO_SECONDS}s from start time."
            )

        st.markdown("<div style='margin-top: 12px;'></div>", unsafe_allow_html=True)
        start_seconds = st.number_input(
            "Start Analysis at Timestamp (seconds)",
            min_value=0.0,
            value=0.0,
            step=1.0,
            help="Start CCTV inspection from this timestamp in seconds (optional).",
            key="cctv_start_seconds",
        )

    with col_config:
        st.markdown("#### Processing Configuration")

        st.markdown(
            """
            <div style='font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; color: #94A3B8; font-weight: 600; margin-bottom: 6px;'>
                Mandatory PPE Rules
            </div>
            """,
            unsafe_allow_html=True,
        )

        gear_choice = st.radio(
            "Mandatory Site Equipment",
            ["Hardhat + Safety Vest", "Hardhat Only", "Safety Vest Only"],
            index=0,
            key="cctv_gear_choice",
            label_visibility="collapsed",
            help="Select PPE requirements enforced for this surveillance inspection.",
        )
        if gear_choice == "Hardhat Only":
            required_gear_mode = "hardhat_only"
        elif gear_choice == "Safety Vest Only":
            required_gear_mode = "vest_only"
        else:
            required_gear_mode = "both"

        st.markdown(
            f"""
            <div class='sv-panel' style='padding: 10px 14px; margin: 8px 0 16px 0; font-size: 12px; line-height: 1.5; color: #94A3B8;'>
                Enforcing: <strong style='color: #F1F5F9;'>{gear_choice}</strong><br>
                <span>Inference stride analyzes every {PROCESS_EVERY_N} frames on CPU.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Technical Processing Controls (Collapsible)
        with st.expander("Technical Processing Controls", expanded=False):
            conf_threshold = st.slider(
                "Sensitivity Slider (Confidence Threshold)",
                min_value=0.10,
                max_value=0.90,
                value=0.25,
                step=0.05,
                key="cctv_conf_slider",
                help="Scales per-class thresholds (Base: Person 0.30, Hardhat 0.30, NO-Hardhat 0.20, Safety Vest 0.20).",
            )
            show_gear_boxes = st.checkbox("Draw Gear Bounding Boxes", value=True, key="cctv_gear_box_cb")
            show_zones = st.checkbox("Outline Body Zones (Head / Torso)", value=False, key="cctv_zones_cb")
            temporal_smoothing = st.checkbox(
                "Temporal Summary Smoothing",
                value=True,
                key="cctv_temporal_smoothing_cb",
                help=f"Applies a {TEMPORAL_SMOOTHING_WINDOW}-frame rolling filter to smooth aggregate counts and rates across video frames.",
            )

        st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)

        if video_source == "Upload Video File":
            run_disabled = uploaded_video is None
            btn_label = "Run Video Compliance Check" if not run_disabled else "Select a Video to Run Check"
        else:
            run_disabled = not bool(youtube_url and youtube_url.strip())
            btn_label = "Run Video Compliance Check" if not run_disabled else "Provide YouTube URL to Run Check"

        if st.button(btn_label, key="cctv_run_btn", type="primary", use_container_width=True, disabled=run_disabled):
            if video_source == "Upload Video File":
                process_cctv_video_upload(
                    uploaded_video=uploaded_video,
                    model=model,
                    required_gear=required_gear_mode,
                    conf_threshold=conf_threshold,
                    show_gear_boxes=show_gear_boxes,
                    show_zones=show_zones,
                    start_seconds=start_seconds,
                    temporal_smoothing=temporal_smoothing,
                )
            else:
                process_cctv_youtube(
                    youtube_url=youtube_url,
                    model=model,
                    required_gear=required_gear_mode,
                    conf_threshold=conf_threshold,
                    show_gear_boxes=show_gear_boxes,
                    show_zones=show_zones,
                    start_seconds=start_seconds,
                    temporal_smoothing=temporal_smoothing,
                )


def process_cctv_video_upload(
    uploaded_video: Any,
    model: Any,
    required_gear: str,
    conf_threshold: float,
    show_gear_boxes: bool,
    show_zones: bool,
    start_seconds: float,
    temporal_smoothing: bool,
):
    """Save uploaded video file to a temporary location and run the CCTV inspection pipeline."""
    temp_dir = tempfile.mkdtemp()
    try:
        temp_input_path = os.path.join(temp_dir, uploaded_video.name)
        with open(temp_input_path, "wb") as f:
            f.write(uploaded_video.getbuffer())

        execute_cctv_pipeline(
            video_path=temp_input_path,
            video_display_name=uploaded_video.name,
            model=model,
            required_gear=required_gear,
            conf_threshold=conf_threshold,
            show_gear_boxes=show_gear_boxes,
            show_zones=show_zones,
            start_seconds=start_seconds,
            temporal_smoothing=temporal_smoothing,
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def process_cctv_youtube_link(
    youtube_url: str,
    model: Any,
    required_gear: str,
    conf_threshold: float,
    show_gear_boxes: bool,
    show_zones: bool,
    start_seconds: float,
    temporal_smoothing: bool,
):
    """Download YouTube video via yt-dlp into a temp dir and run the CCTV inspection pipeline."""
    if not is_valid_youtube_url(youtube_url):
        st.error(
            "Invalid YouTube URL. Please provide a link starting with "
            "https://www.youtube.com/, https://youtube.com/, https://m.youtube.com/, or https://youtu.be/"
        )
        return

    temp_dir = tempfile.mkdtemp()
    try:
        with st.spinner("Extracting metadata and downloading YouTube video stream..."):
            try:
                downloaded_file = download_youtube(youtube_url, temp_dir)
            except YouTubeDownloadError as e:
                st.error(f"YouTube Download Error: {e}")
                return
            except Exception as e:
                st.error(
                    f"Unexpected error downloading YouTube video: {e}. "
                    "Please download the video file manually and use 'Upload file' mode."
                )
                return

        display_name = f"YouTube: {os.path.basename(downloaded_file)}"
        execute_cctv_pipeline(
            video_path=downloaded_file,
            video_display_name=display_name,
            model=model,
            required_gear=required_gear,
            conf_threshold=conf_threshold,
            show_gear_boxes=show_gear_boxes,
            show_zones=show_zones,
            start_seconds=start_seconds,
            temporal_smoothing=temporal_smoothing,
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def execute_cctv_pipeline(
    video_path: str,
    video_display_name: str,
    model: Any,
    required_gear: str,
    conf_threshold: float,
    show_gear_boxes: bool,
    show_zones: bool,
    start_seconds: float,
    temporal_smoothing: bool,
):
    """Execute the core CCTV video processing, per-frame inference, temporal smoothing, and MP4 generation."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Failed to open video source. Please check the file format.")
        return

    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    safe_fps = compute_output_fps(raw_fps, max_fps=OUTPUT_MAX_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = calculate_video_duration(total_frames, raw_fps)

    start_frame_idx, max_frames = compute_cctv_frame_range(
        total_frames=total_frames,
        fps=raw_fps,
        start_seconds=start_seconds,
        max_seconds=MAX_VIDEO_SECONDS,
    )

    if total_frames > max_frames:
        st.warning(
            f"Video duration is {duration_sec:.1f}s. Inspection is capped at the first "
            f"{MAX_VIDEO_SECONDS}s ({max_frames} frames) starting from {start_seconds:.1f}s."
        )

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)

    progress_bar = st.progress(0, text="Initializing CCTV surveillance analysis on CPU...")
    status_text = st.empty()

    analyzed_records: List[Dict[str, Any]] = []
    annotated_frames_rgb: List[np.ndarray] = []

    last_workers: List[Dict[str, Any]] = []
    last_unassigned: List[Dict[str, Any]] = []

    out_mp4_fd, out_mp4_path = tempfile.mkstemp(suffix=".mp4")
    os.close(out_mp4_fd)

    try:
        for step_idx in range(max_frames):
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                break

            orig_h, orig_w = frame_bgr.shape[:2]
            target_w, target_h = compute_resized_dimensions(orig_w, orig_h, max_width=MAX_FRAME_WIDTH)
            if target_w != orig_h or target_h != orig_h:
                frame_bgr = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)

            rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(rgb_frame)

            current_frame_idx = start_frame_idx + step_idx
            timestamp_sec = round(current_frame_idx / safe_fps, 2)

            if is_analyzed_frame(step_idx, process_every_n=PROCESS_EVERY_N):
                detections = run_detection(
                    image=rgb_frame,
                    model=model,
                    conf_threshold=conf_threshold,
                    device="cpu",
                )
                evaluation = evaluate_site_compliance(
                    detections=detections,
                    required_gear=required_gear,
                    image_height=float(rgb_frame.shape[0]),
                    strict_hardhat=st.session_state.strict_hardhat,
                )
                last_workers = evaluation["workers"]
                last_unassigned = evaluation["unassigned_gear"]

                total_w = evaluation["summary"]["total_workers"]
                comp_w = evaluation["summary"]["compliant_count"]
                viol_w = evaluation["summary"]["violation_count"]
                rev_w = evaluation["summary"].get("needs_review_count", evaluation["summary"].get("unknown_count", 0))
                comp_rate = evaluation["summary"]["compliance_rate"]

                annotated_pil = annotate_image(
                    image=pil_frame,
                    workers=last_workers,
                    unassigned_gear=last_unassigned,
                    show_gear_boxes=show_gear_boxes,
                    show_zones=show_zones,
                )
                annotated_rgb = np.array(annotated_pil)

                analyzed_records.append({
                    "frame_number": current_frame_idx,
                    "timestamp_sec": timestamp_sec,
                    "total_workers": total_w,
                    "compliant_count": comp_w,
                    "violation_count": viol_w,
                    "needs_review_count": rev_w,
                    "compliance_rate": comp_rate,
                    "workers": last_workers,
                    "annotated_image": annotated_pil,
                })
            else:
                annotated_pil = annotate_image(
                    image=pil_frame,
                    workers=last_workers,
                    unassigned_gear=last_unassigned,
                    show_gear_boxes=show_gear_boxes,
                    show_zones=show_zones,
                )
                annotated_rgb = np.array(annotated_pil)

            annotated_frames_rgb.append(annotated_rgb)

            pct = int(((step_idx + 1) / max_frames) * 100)
            progress_bar.progress(
                pct,
                text=f"Analyzing CCTV Video: Frame {step_idx + 1}/{max_frames} (t={timestamp_sec:.1f}s)...",
            )
            status_text.caption(
                f"Progress: {pct}% | Analyzed: {len(analyzed_records)} frames | Buffer: {len(annotated_frames_rgb)} rendered"
            )

        progress_bar.empty()
        status_text.empty()

        if not analyzed_records:
            st.error("No frames could be decoded from this video. Please check the file format.")
            return

        video_bytes = None
        write_error = None
        try:
            with st.spinner("Encoding annotated H.264 MP4 playback video..."):
                write_annotated_h264_video(annotated_frames_rgb, out_mp4_path, fps=safe_fps)
                with open(out_mp4_path, "rb") as vf:
                    video_bytes = vf.read()
        except Exception as e:
            write_error = str(e)

        if temporal_smoothing:
            smoothed_records = apply_temporal_smoothing(
                analyzed_records,
                window_size=TEMPORAL_SMOOTHING_WINDOW,
            )
            summary = summarize_cctv_results(smoothed_records, use_smoothing=False)
            summary["smoothing_applied"] = True
            summary["smoothing_window"] = TEMPORAL_SMOOTHING_WINDOW
            effective_records = smoothed_records
        else:
            summary = summarize_cctv_results(analyzed_records, use_smoothing=False)
            effective_records = analyzed_records

        metadata_dict = {
            "raw_fps": raw_fps,
            "safe_fps": safe_fps,
            "total_frames": total_frames,
            "duration_sec": duration_sec,
            "processed_frames": len(annotated_frames_rgb),
            "start_seconds": start_seconds,
            "temporal_smoothing": temporal_smoothing,
        }

        st.session_state.video_results = {
            "video_filename": video_display_name,
            "metadata": metadata_dict,
            "summary": summary,
            "analyzed_records": effective_records,
            "raw_records": analyzed_records,
            "video_bytes": video_bytes,
            "write_error": write_error,
        }

        st.success(f"Video processing complete! Analyzed {len(analyzed_records)} frames ({len(annotated_frames_rgb)} total frames rendered).")
        navigate_to("Inspection Results")

    finally:
        if cap.isOpened():
            cap.release()
        if os.path.exists(out_mp4_path):
            try:
                os.remove(out_mp4_path)
            except OSError:
                pass


def render_video_results_view(video_results: Dict[str, Any]):
    """Render CCTV video results: video player, metrics, charts, table, and moments gallery."""
    # A. Page Header
    st.markdown("### Video Surveillance Results & Temporal Analysis")
    st.caption("Sequential video inspection playback, temporal compliance trends, and high-risk moments.")

    summary = video_results["summary"]
    analyzed = video_results["analyzed_records"]
    meta = video_results["metadata"]
    filename = video_results.get("video_filename", "Uploaded Footage")
    video_bytes = video_results.get("video_bytes")
    write_error = video_results.get("write_error")

    smoothing_badge = (
        f" &nbsp;|&nbsp; <strong>Anti-Flicker:</strong> <code>{TEMPORAL_SMOOTHING_WINDOW}-frame Window</code>"
        if meta.get("temporal_smoothing")
        else ""
    )

    # 1. Compact Metadata Panel
    st.markdown(
        f"""
        <div class='sv-panel' style='padding: 10px 16px; margin-bottom: 14px; display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center;'>
            <div>
                <strong>Source:</strong> <code>{filename}</code> &nbsp;|&nbsp;
                <strong>Duration:</strong> <code>{meta['duration_sec']:.1f}s</code> &nbsp;|&nbsp;
                <strong>Stride:</strong> <code>Every {PROCESS_EVERY_N} frames</code> &nbsp;|&nbsp;
                <strong>Analyzed:</strong> <code>{summary['frames_analyzed']} frames</code>{smoothing_badge}
            </div>
            <div>
                <span class='badge-safe' style='font-size: 11px; padding: 3px 8px;'>H.264 ENCODED</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Annotated Video Playback (Centerpiece)
    st.markdown("#### Annotated CCTV Playback")
    if write_error:
        st.error(f"Video file encoding error: {write_error}. Inspection statistics remain available below.")

    if video_bytes:
        st.video(video_bytes, format="video/mp4")
        st.download_button(
            "Download Annotated CCTV Video (.mp4)",
            data=video_bytes,
            file_name=f"annotated_cctv_{filename}.mp4",
            mime="video/mp4",
            use_container_width=True,
            key="cctv_download_video_btn",
        )
    else:
        st.info("Video playback buffer unavailable; temporal metrics and frame analysis are presented below.")

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    if meta.get("temporal_smoothing"):
        st.markdown(
            f"""
            <div class='sv-surface-elevated' style='padding: 8px 12px; margin-bottom: 12px; font-size: 12px; color: #94A3B8;'>
                <strong style='color: #38BDF8;'>Temporal Anti-Flicker Active:</strong>
                <span>Summary metrics utilize a {TEMPORAL_SMOOTHING_WINDOW}-frame rolling median filter across surrounding frames to stabilize aggregate counts.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 3. KPI Metrics Row (5 Compact Cards)
    viol_color = "#EF4444" if summary["frames_with_violations"] > 0 else "#94A3B8"
    rate_display = f"{summary['average_compliance_rate']:.1f}%" if summary["has_worker_frames"] else "N/A"
    rate_color = "#22C55E" if (summary["has_worker_frames"] and summary["average_compliance_rate"] >= 85.0) else ("#F59E0B" if summary["has_worker_frames"] and summary["average_compliance_rate"] >= 60.0 else "#EF4444")

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(
            f"<div class='metric-card'><div class='metric-card-title'>Frames Analyzed</div>"
            f"<div class='metric-card-value'>{summary['frames_analyzed']}</div></div>",
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"<div class='metric-card'><div class='metric-card-title'>Frames With Workers</div>"
            f"<div class='metric-card-value' style='color:#38BDF8;'>{summary['frames_with_workers']}</div></div>",
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"<div class='metric-card'><div class='metric-card-title'>Frames No Workers</div>"
            f"<div class='metric-card-value' style='color:#94A3B8;'>{summary['frames_without_workers']}</div></div>",
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f"<div class='metric-card'><div class='metric-card-title'>Frames With Violations</div>"
            f"<div class='metric-card-value' style='color:{viol_color};'>{summary['frames_with_violations']}</div></div>",
            unsafe_allow_html=True,
        )
    with k5:
        st.markdown(
            f"<div class='metric-card'><div class='metric-card-title'>Avg Compliance Rate</div>"
            f"<div class='metric-card-value' style='color:{rate_color};'>{rate_display}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    # 4. Compliance Rate Timeline
    st.markdown("#### Compliance Rate Over Time (By Second)")
    chart_rows = []
    for f in analyzed:
        r_val = f["compliance_rate"] if f["total_workers"] > 0 else None
        chart_rows.append({
            "Time (s)": f"{f['timestamp_sec']:.1f}s",
            "Compliance Rate (%)": r_val,
        })
    df_chart = pd.DataFrame(chart_rows).set_index("Time (s)")
    if summary["has_worker_frames"]:
        st.line_chart(df_chart, color="#0284C7", height=230)
    else:
        st.info("No personnel detected in the analyzed video frames to plot compliance timeline.")

    # 5. Top Risk Moments
    st.markdown("---")
    st.markdown("#### Top Risk Moments (High Violations)")
    top_moments = summary.get("top_violation_moments", [])
    if top_moments:
        st.caption(f"Displaying up to {TOP_VIOLATION_FRAMES_COUNT} sampled moments with the highest non-compliance counts:")
        g_cols = st.columns(len(top_moments))
        for col, f in zip(g_cols, top_moments):
            with col:
                st.image(
                    f["annotated_image"],
                    caption=f"Frame #{f['frame_number']} (t={f['timestamp_sec']:.2f}s) | Violations: {f['violation_count']} | Workers: {f['total_workers']}",
                    use_container_width=True,
                )
    else:
        st.markdown(
            """
            <div class='sv-panel' style='padding: 12px 16px; border-left: 4px solid #22C55E; color: #22C55E; font-size: 13px;'>
                <strong>Clean Surveillance Record:</strong>
                <span style='color: #F1F5F9; margin-left: 6px;'>Zero PPE violations were detected across all analyzed video frames.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 6. Analyzed Frames Breakdown
    st.markdown("---")
    st.markdown("#### Analyzed Frames Breakdown")
    table_rows = []
    for idx, f in enumerate(analyzed, start=1):
        rate_cell = f"{f['compliance_rate']:.1f}%" if f["total_workers"] > 0 and f["compliance_rate"] is not None else "N/A (No Workers)"
        table_rows.append({
            "Analysis #": idx,
            "Frame #": f["frame_number"],
            "Time (s)": f"{f['timestamp_sec']:.2f}s",
            "Workers": f["total_workers"],
            "Compliant": f["compliant_count"],
            "Violations": f["violation_count"],
            "Needs Review": f["needs_review_count"],
            "Compliance Rate": rate_cell,
        })

    df_table = pd.DataFrame(table_rows)
    st.dataframe(df_table, use_container_width=True)

    # 7. Action Toolbar
    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
    c_csv, c_reset = st.columns([3, 1])
    with c_csv:
        csv_content = generate_cctv_csv_report(analyzed, summary, filename)
        st.download_button(
            "Download CCTV Inspection CSV (.csv)",
            data=csv_content,
            file_name=f"cctv_inspection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
            key="cctv_download_csv_btn",
        )
    with c_reset:
        if st.button("New Video Inspection", key="cctv_new_inspection_btn", use_container_width=True):
            navigate_to("New Inspection")


# ============================================================================
# 14. MAIN APPLICATION ORCHESTRATOR
# ============================================================================
def main():
    init_session_state()
    inject_custom_css(st.session_state.theme)

    active_model_path = resolve_model_path()
    try:
        model = get_detector(active_model_path)
        is_model_verified = verify_model_classes(model)
    except Exception as e:
        st.error(f"Error loading YOLO model from `{active_model_path}`: {e}")
        st.stop()

    with st.sidebar:
        st.markdown(
            """
            <div style='display: flex; align-items: center; gap: 10px; padding: 8px 4px 14px 4px;'>
                <div style='width: 36px; height: 36px; border-radius: 4px; background: #162230; border: 1px solid #263442; display: flex; align-items: center; justify-content: center;'>
                    <svg width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='#38BDF8' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z'/></svg>
                </div>
                <div>
                    <div style='font-size: 15px; font-weight: 700; letter-spacing: 0.5px; color: #38BDF8;'>SAFETY VISION</div>
                    <div style='font-size: 10px; color: #94A3B8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px;'>Industrial AI Surveillance</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<hr style='margin: 4px 0 10px 0;'>", unsafe_allow_html=True)

        st.markdown(
            "<span style='font-size: 11px; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600;'>"
            "Surveillance Mode</span>",
            unsafe_allow_html=True,
        )
        mode_options = ["Image", "Video"]
        curr_mode = st.session_state.get("inspection_mode", "Image")
        selected_mode = st.radio(
            "Mode Selector",
            options=mode_options,
            index=mode_options.index(curr_mode) if curr_mode in mode_options else 0,
            horizontal=True,
            label_visibility="collapsed",
        )
        if selected_mode != st.session_state.inspection_mode:
            st.session_state.inspection_mode = selected_mode
            st.rerun()

        st.markdown("<hr style='margin: 8px 0 12px 0;'>", unsafe_allow_html=True)

        st.markdown(
            "<span style='font-size: 11px; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600;'>"
            "Navigation</span>",
            unsafe_allow_html=True,
        )

        pages = [
            "Dashboard",
            "New Inspection",
            "Inspection Results",
            "Worker Audits",
            "Inspection History",
            "Reports & Exports",
            "Settings",
        ]

        selected_nav = st.radio(
            "Navigation Menu",
            options=pages,
            index=pages.index(st.session_state.nav_radio) if st.session_state.nav_radio in pages else 0,
            label_visibility="collapsed",
        )

        if selected_nav != st.session_state.current_page:
            st.session_state.current_page = selected_nav
            st.session_state.nav_radio = selected_nav
            st.rerun()

        st.markdown("<hr style='margin: 14px 0 12px 0;'>", unsafe_allow_html=True)

        st.markdown(
            "<span style='font-size: 11px; color: #94A3B8; text-transform: uppercase; "
            "letter-spacing: 0.5px; font-weight: 600;'>Hardhat Policy</span>",
            unsafe_allow_html=True,
        )
        st.session_state.strict_hardhat = st.checkbox(
            "Treat undetected hardhat as violation",
            value=st.session_state.strict_hardhat,
            help=(
                "When ON, workers whose head zone has no detectable hardhat box are "
                "marked as violations — but only if their person detection is reliable "
                "(confidence ≥ 0.40, box height ≥ 12 %% of frame, upright posture). "
                "Unreliable persons are always flagged as Needs Review regardless."
            ),
        )

        st.markdown("<hr style='margin: 12px 0 12px 0;'>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class='sv-surface-elevated' style='padding: 10px; font-size: 11px; line-height: 1.5;'>
                <div style='color: #94A3B8; font-weight: 600; text-transform: uppercase;'>Inference Engine</div>
                <div style='color: #F1F5F9; font-weight: 500;'>YOLOv8 Nano (CPU Mode)</div>
                <div style='color: #94A3B8; margin-top: 4px;'>Weights: <code>{os.path.basename(active_model_path)}</code></div>
                <div style='margin-top: 4px; color: #22C55E; font-weight: 600;'>● 10 Classes Verified</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


        render_top_navbar(st.session_state.current_page, is_model_verified)

    active_insp = st.session_state.active_inspection
    history_list = st.session_state.inspection_history

    if st.session_state.inspection_mode == "Video":
        if st.session_state.current_page == "New Inspection":
            render_video_inspection(model)
        elif st.session_state.current_page == "Inspection Results":
            video_res = st.session_state.get("video_results")
            if video_res is not None:
                render_video_results_view(video_res)
            else:
                st.warning("No active video inspection results available. Please run a video inspection first.")
                if st.button("Go to Video Inspection", type="primary"):
                    navigate_to("New Inspection")
        elif st.session_state.current_page == "Dashboard":
            render_dashboard(active_insp, history_list)
        elif st.session_state.current_page == "Worker Audits":
            render_worker_audits(active_insp)
        elif st.session_state.current_page == "Inspection History":
            render_inspection_history(history_list, model)
        elif st.session_state.current_page == "Reports & Exports":
            render_reports_exports(active_insp)
        elif st.session_state.current_page == "Settings":
            render_settings(model, is_model_verified)
    else:
        if st.session_state.current_page == "Dashboard":
            render_dashboard(active_insp, history_list)
        elif st.session_state.current_page == "New Inspection":
            render_new_inspection(model)
        elif st.session_state.current_page == "Inspection Results":
            render_inspection_results(active_insp)
        elif st.session_state.current_page == "Worker Audits":
            render_worker_audits(active_insp)
        elif st.session_state.current_page == "Inspection History":
            render_inspection_history(history_list, model)
        elif st.session_state.current_page == "Reports & Exports":
            render_reports_exports(active_insp)
        elif st.session_state.current_page == "Settings":
            render_settings(model, is_model_verified)


if __name__ == "__main__":
    main()
