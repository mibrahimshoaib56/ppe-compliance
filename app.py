"""SAFETY VISION — Construction PPE Safety Management Portal.

Enterprise-grade safety inspection application using YOLOv8 object detection
and rule-based spatial compliance logic on CPU.

Organized into 7 dedicated operational views:
1. Dashboard: Executive safety metrics, compliance distribution, recent records, alerts.
2. New Inspection: Image selection, preview, configuration, and compliance detection.
3. Inspection Results: Visual inspection tabs, KPI summary, and safety advisories.
4. Worker Audits: Individual worker compliance breakdown and drill-down ROI inspector.
5. Inspection History: Audit log of past inspections with one-click review.
6. Reports & Exports: Standardized CSV audit reports and annotated image downloads.
7. Settings: Engine status, class verification, default thresholds, and viva reference.
"""

from datetime import datetime
import json
import os
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

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
import shutil
import tempfile
import time
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
from src.utils import (
    annotate_image,
    crop_worker_roi,
    encode_image_bytes,
    format_worker_table,
    generate_csv_report,
    get_action_recommendation,
)

# ============================================================================
# 1. PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="SAFETY VISION | PPE Portal",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Constants & Paths
SAMPLE_DIR = "sample_images"
HISTORY_FILE = os.path.join("eval", "inspection_history.json")


# ============================================================================
# 2. CACHED MODEL LOADER (Ensures model is loaded once on CPU)
# ============================================================================
@st.cache_resource(show_spinner="Loading YOLOv8 model into memory...")
def get_detector(model_file_path: str):
    """Load and cache the YOLOv8 model instance on CPU."""
    return load_detector_model(model_file_path)


# ============================================================================
# 3. SESSION STATE INITIALIZATION & HISTORY PERSISTENCE
# ============================================================================
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
        st.session_state.strict_hardhat = True  # mirrors STRICT_HARDHAT_POLICY default

    if "active_inspection" not in st.session_state:
        st.session_state.active_inspection = None

    if "inspections_store" not in st.session_state:
        st.session_state.inspections_store = {}

    if "inspection_history" not in st.session_state:
        st.session_state.inspection_history = []
        # Attempt to load persistent inspection records from disk
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

    # Store full in-memory object (images + detections + evaluation)
    st.session_state.inspections_store[inspection_id] = inspection_record

    # Build clean metadata record for history persistence
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

    # Remove existing duplicate ID if present, then insert newest first
    st.session_state.inspection_history = [
        h for h in st.session_state.inspection_history if h.get("id") != inspection_id
    ]
    st.session_state.inspection_history.insert(0, history_entry)

    # Persist metadata to disk
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
# 4. MODERN INDUSTRIAL STYLING (Dark & Light Themes)
# ============================================================================
def inject_custom_css(theme: str):
    """Inject responsive, enterprise-grade CSS based on theme selection."""
    if theme == "Dark Theme":
        css = """
        <style>
            /* Base Theme - Dark Slate */
            .stApp {
                background-color: #0b0f17;
                color: #e2e8f0;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            }
            [data-testid="stSidebar"] {
                background-color: #111726;
                border-right: 1px solid #1e293b;
            }
            /* Top Navigation Bar */
            .top-nav {
                background: linear-gradient(90deg, #131b2e, #182238);
                border: 1px solid #23314d;
                border-radius: 12px;
                padding: 12px 22px;
                margin-bottom: 22px;
                display: flex;
                align-items: center;
                justify-content: space-between;
                box-shadow: 0 4px 16px rgba(0,0,0,0.35);
            }
            .brand-title {
                color: #38bdf8;
                font-weight: 800;
                font-size: 20px;
                letter-spacing: 0.5px;
                margin: 0;
            }
            .page-crumb {
                color: #94a3b8;
                font-size: 14px;
                font-weight: 600;
            }
            .model-pill-ready {
                background-color: rgba(34, 197, 94, 0.15);
                color: #4ade80;
                border: 1px solid #22c55e;
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: 700;
                letter-spacing: 0.4px;
            }
            /* KPI Metric Cards */
            .metric-card {
                background: linear-gradient(145deg, #131a29, #182236);
                border: 1px solid #23314d;
                border-radius: 10px;
                padding: 16px;
                text-align: center;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
            }
            .metric-card-title {
                color: #94a3b8;
                font-size: 12px;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.6px;
                margin-bottom: 6px;
            }
            .metric-card-value {
                color: #f8fafc;
                font-size: 28px;
                font-weight: 800;
            }
            /* Enterprise Panel / Box */
            .sv-panel {
                background-color: #131b2e;
                border: 1px solid #23314d;
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 18px;
            }
            /* Status Pills */
            .status-safe { color: #4ade80; font-weight: 800; }
            .status-warning { color: #fbbf24; font-weight: 800; }
            .status-critical { color: #f87171; font-weight: 800; }
            /* Custom Tab Styling */
            .stTabs [data-baseweb="tab-list"] {
                gap: 8px;
            }
            .stTabs [data-baseweb="tab"] {
                border-radius: 6px;
                padding: 8px 16px;
                background-color: #162035;
                color: #94a3b8;
            }
            .stTabs [aria-selected="true"] {
                background-color: #2563eb !important;
                color: #ffffff !important;
            }
        </style>
        """
    else:
        css = """
        <style>
            /* Base Theme - Light Industrial */
            .stApp {
                background-color: #f8fafc;
                color: #0f172a;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            }
            [data-testid="stSidebar"] {
                background-color: #ffffff;
                border-right: 1px solid #e2e8f0;
            }
            /* Top Navigation Bar */
            .top-nav {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 12px;
                padding: 12px 22px;
                margin-bottom: 22px;
                display: flex;
                align-items: center;
                justify-content: space-between;
                box-shadow: 0 2px 6px rgba(0,0,0,0.04);
            }
            .brand-title {
                color: #0284c7;
                font-weight: 800;
                font-size: 20px;
                letter-spacing: 0.5px;
                margin: 0;
            }
            .page-crumb {
                color: #64748b;
                font-size: 14px;
                font-weight: 600;
            }
            .model-pill-ready {
                background-color: rgba(34, 197, 94, 0.12);
                color: #16a34a;
                border: 1px solid #22c55e;
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: 700;
                letter-spacing: 0.4px;
            }
            /* KPI Metric Cards */
            .metric-card {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 10px;
                padding: 16px;
                text-align: center;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
            }
            .metric-card-title {
                color: #64748b;
                font-size: 12px;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.6px;
                margin-bottom: 6px;
            }
            .metric-card-value {
                color: #0f172a;
                font-size: 28px;
                font-weight: 800;
            }
            /* Enterprise Panel / Box */
            .sv-panel {
                background-color: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 18px;
                box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
            }
            /* Status Pills */
            .status-safe { color: #16a34a; font-weight: 800; }
            .status-warning { color: #d97706; font-weight: 800; }
            .status-critical { color: #dc2626; font-weight: 800; }
            /* Custom Tab Styling */
            .stTabs [data-baseweb="tab-list"] {
                gap: 8px;
            }
            .stTabs [data-baseweb="tab"] {
                border-radius: 6px;
                padding: 8px 16px;
                background-color: #f1f5f9;
                color: #64748b;
            }
            .stTabs [aria-selected="true"] {
                background-color: #2563eb !important;
                color: #ffffff !important;
            }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)


# ============================================================================
# 5. TOP NAVIGATION BAR COMPONENT
# ============================================================================
def render_top_navbar(current_page: str, is_model_verified: bool):
    """Render the top enterprise bar with branding, breadcrumb, model status, and theme toggle."""
    model_status_html = (
        "<span class='model-pill-ready'>🟢 YOLOv8 Ready (CPU)</span>"
        if is_model_verified
        else "<span class='model-pill-ready' style='color:#f59e0b; border-color:#f59e0b;'>⚠️ YOLOv8 Fallback</span>"
    )

    col_brand, col_status, col_theme = st.columns([5, 3, 2])

    with col_brand:
        st.markdown(
            f"<div style='display:flex; align-items:center; gap:12px; padding-top:4px;'>"
            f"<span style='font-size:26px;'>🛡️</span>"
            f"<div><span class='brand-title'>SAFETY VISION</span><br>"
            f"<span class='page-crumb'>Enterprise Safety Management Portal &bull; <strong>{current_page}</strong></span>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

    with col_status:
        st.markdown(
            f"<div style='text-align:center; padding-top:10px;'>{model_status_html}</div>",
            unsafe_allow_html=True,
        )

    with col_theme:
        current_theme = st.session_state.theme
        theme_icon = "☀️ Light" if current_theme == "Dark Theme" else "🌙 Dark"
        if st.button(f"{theme_icon} Mode", key="top_theme_toggle", use_container_width=True):
            st.session_state.theme = "Light Theme" if current_theme == "Dark Theme" else "Dark Theme"
            st.rerun()

    st.markdown("<hr style='margin: 12px 0 20px 0; border: none; border-top: 1px solid rgba(148,163,184,0.15);'>", unsafe_allow_html=True)


# ============================================================================
# 6. PAGE 1: DASHBOARD
# ============================================================================
def render_dashboard(active_inspection: Optional[Dict[str, Any]], history: List[Dict[str, Any]]):
    """Render the executive safety dashboard with KPIs, charts, and recent audit logs."""
    st.markdown("### 📊 Safety Operations Dashboard")
    st.caption("Real-time construction safety overview, compliance trends, and incident alerts.")

    # Check if we have inspection data
    has_data = active_inspection is not None or len(history) > 0

    if not has_data:
        st.markdown(
            """
            <div class='sv-panel' style='text-align: center; padding: 48px 24px;'>
                <span style='font-size: 54px;'>🏗️</span>
                <h3 style='margin: 12px 0 6px 0;'>No Safety Inspections Conducted Yet</h3>
                <p style='color: #94a3b8; max-width: 520px; margin: 0 auto 20px auto;'>
                    The portal has not performed any PPE compliance scans in this session.
                    Upload a site photo or select a test sample to evaluate workers, hardhats, and vests.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("🚀 Start First Inspection", type="primary"):
            navigate_to("New Inspection")
        return

    # Derive statistics from active inspection (or latest history entry)
    current = active_inspection if active_inspection is not None else history[0]
    summary = current["summary"] if "summary" in current else current

    total_w = summary.get("total_workers", 0)
    comp_w = summary.get("compliant_count", 0)
    viol_w = summary.get("violation_count", 0)
    unknown_w = summary.get("unknown_count", 0)
    review_w = summary.get("needs_review_count", unknown_w)
    rate = summary.get("compliance_rate", 0.0)
    status = summary.get("site_status", "Safe")

    if status == "Safe":
        status_color = "#22c55e"
        status_badge = "✅ SAFE"
    elif status == "Warning":
        status_color = "#f59e0b"
        status_badge = "⚠️ WARNING"
    else:
        status_color = "#ef4444"
        status_badge = "🚨 CRITICAL"

    # KPI Row
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
            f"<div class='metric-card-value' style='color:#22c55e;'>{comp_w}</div></div>",
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"<div class='metric-card'><div class='metric-card-title'>Violations</div>"
            f"<div class='metric-card-value' style='color:#ef4444;'>{viol_w}</div></div>",
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f"<div class='metric-card'><div class='metric-card-title'>Needs Review</div>"
            f"<div class='metric-card-value' style='color:#f59e0b;'>{review_w}</div></div>",
            unsafe_allow_html=True,
        )
    with k5:
        st.markdown(
            f"<div class='metric-card'><div class='metric-card-title'>Compliance Rate</div>"
            f"<div class='metric-card-value'>{rate}%</div></div>",
            unsafe_allow_html=True,
        )
    with k6:
        st.markdown(
            f"<div class='metric-card'><div class='metric-card-title'>Site Status</div>"
            f"<div class='metric-card-value' style='color:{status_color}; font-size:20px;'>{status_badge}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

    # Main Dashboard Visuals: Distribution Chart + Alerts
    c_chart, c_alerts = st.columns([3, 2])

    with c_chart:
        st.markdown("#### 📈 Compliance Distribution")
        chart_data = pd.DataFrame({
            "Status Category": ["Compliant Workers", "PPE Violations", "Needs Review"],
            "Worker Count": [comp_w, viol_w, review_w],
        }).set_index("Status Category")
        st.bar_chart(chart_data, color="#2563eb", height=240)

    with c_alerts:
        st.markdown("#### 🚨 Latest Safety Directive")
        if total_w == 0:
            st.warning(
                "No workers detected. The image may be from an unsupported angle (for example overhead) "
                "or workers are too small. Try a front or side view."
            )
        else:
            advisory_msg = get_action_recommendation(summary)

            if status == "Safe":
                st.success(f"**Routine Surveillance Active:**\n\n{advisory_msg}")
            elif status == "Warning":
                st.warning(f"**Immediate Attention Required:**\n\n{advisory_msg}")
            else:
                st.error(f"**Severe Safety Violation:**\n\n{advisory_msg}")

        active_label = current.get("image_name", "Live Scan")
        st.caption(f"Inspection source: `{active_label}`")

        c_act1, c_act2 = st.columns(2)
        with c_act1:
            if st.button("🔍 Conduct New Scan", use_container_width=True):
                navigate_to("New Inspection")
        with c_act2:
            if active_inspection is not None:
                if st.button("👉 View Active Results", type="primary", use_container_width=True):
                    navigate_to("Inspection Results")

    # Recent Inspection Table
    st.markdown("---")
    st.markdown("#### 📜 Recent Site Inspections")

    if history:
        history_df = pd.DataFrame([
            {
                "Inspection ID": h["id"],
                "Date & Time": h["timestamp"],
                "Site Image": h["image_name"],
                "Workers": h["total_workers"],
                "Compliance Rate": f"{h['compliance_rate']}%",
                "Status": h["site_status"],
            }
            for h in history[:6]
        ])
        st.dataframe(history_df, use_container_width=True)
    else:
        st.info("No recorded inspection history.")


# ============================================================================
# 7. PAGE 2: NEW INSPECTION
# ============================================================================
def render_new_inspection(model: Any):
    """Render the dedicated inspection creation and inference workflow page."""
    st.markdown("### 🔍 Conduct New Safety Inspection")
    st.caption("Upload an on-site photo or choose a verified Roboflow dataset sample for YOLOv8 inference.")

    col_input, col_config = st.columns([3, 2])

    sample_files = (
        sorted([f for f in os.listdir(SAMPLE_DIR) if f.endswith((".jpg", ".png", ".jpeg"))])
        if os.path.exists(SAMPLE_DIR)
        else []
    )

    with col_input:
        st.markdown("#### 1. Select Construction Site Image")

        image_mode = st.radio(
            "Image Source",
            ["Sample Dataset Image", "Upload Custom Photo"],
            horizontal=True,
        )

        input_image = None
        source_label = ""
        source_path = ""

        if image_mode == "Sample Dataset Image" and sample_files:
            selected_file = st.selectbox(
                "Choose Roboflow Construction Site Safety Sample:",
                sample_files,
                index=4 if len(sample_files) > 4 else 0,
            )
            source_label = selected_file
            source_path = os.path.join(SAMPLE_DIR, selected_file)
            try:
                input_image = Image.open(source_path)
                if input_image.mode != "RGB":
                    input_image = input_image.convert("RGB")
            except Exception as e:
                st.error(f"Failed to load sample image: {e}")
        else:
            uploaded_file = st.file_uploader(
                "Upload Site Photo (JPG, PNG)",
                type=["jpg", "jpeg", "png"],
                help="High-resolution site image showing workers with hardhats and safety vests.",
            )
            if uploaded_file is not None:
                source_label = uploaded_file.name
                try:
                    input_image = Image.open(uploaded_file)
                    if input_image.mode != "RGB":
                        input_image = input_image.convert("RGB")
                except Exception as e:
                    st.error(f"Failed to parse uploaded image: {e}")

        # Image Preview Card
        if input_image is not None:
            st.image(input_image, caption=f"Selected: {source_label}", use_container_width=True)

    with col_config:
        st.markdown("#### 2. Inspection Parameters")

        if input_image is not None:
            st.markdown(
                f"""
                <div class='sv-panel' style='padding: 14px;'>
                    <strong>Image Information:</strong><br>
                    &bull; Resolution: <code>{input_image.size[0]} x {input_image.size[1]} px</code><br>
                    &bull; Color Channels: <code>{input_image.mode} (3 Channels)</code><br>
                    &bull; Source File: <code>{source_label}</code>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Mandatory Gear Selection
        gear_choice = st.radio(
            "Mandatory Site Equipment",
            ["Hardhat + Safety Vest", "Hardhat Only", "Safety Vest Only"],
            index=0,
            help="Select PPE requirements enforced for this specific construction zone.",
        )
        if gear_choice == "Hardhat Only":
            required_gear_mode = "hardhat_only"
        elif gear_choice == "Safety Vest Only":
            required_gear_mode = "vest_only"
        else:
            required_gear_mode = "both"

        # Expandable Advanced Configuration
        with st.expander("⚙️ Advanced Detection Settings", expanded=False):
            conf_threshold = st.slider(
                "Confidence Threshold (Scales Per-Class Minimums)",
                min_value=0.10,
                max_value=0.90,
                value=0.25,
                step=0.05,
                help="Scales per-class thresholds (Base: Person 0.30, Hardhat 0.30, NO-Hardhat 0.20, Safety Vest 0.20, NO-Safety Vest 0.20). Slider at 0.25 represents 1.0x baseline scale.",
            )
            show_gear_boxes = st.checkbox("Draw Gear Bounding Boxes", value=True)
            show_zones = st.checkbox("Outline Body Zones (Head 35% / Torso 15-75%)", value=False)

        st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)

        # Primary Run Action
        run_disabled = input_image is None
        if st.button("🔍 Run Compliance Check", type="primary", use_container_width=True, disabled=run_disabled):
            with st.spinner("Analyzing site imagery with YOLOv8 & calculating zone compliance on CPU..."):
                # 1. Run YOLOv8 CPU Inference
                detections = run_detection(
                    image=input_image,
                    model=model,
                    conf_threshold=conf_threshold,
                    device="cpu",
                )

                # 2. Evaluate Spatial PPE Compliance
                # Pass image height for person-size reliability check and
                # the sidebar strict_hardhat toggle.
                import numpy as _np
                _img_h = float(_np.array(input_image).shape[0])
                evaluation = evaluate_site_compliance(
                    detections=detections,
                    required_gear=required_gear_mode,
                    image_height=_img_h,
                    strict_hardhat=st.session_state.strict_hardhat,
                )

                # 3. Generate Annotated Visualization
                annotated_img = annotate_image(
                    image=input_image,
                    workers=evaluation["workers"],
                    unassigned_gear=evaluation["unassigned_gear"],
                    show_gear_boxes=show_gear_boxes,
                    show_zones=show_zones,
                )
                annotated_bytes = encode_image_bytes(annotated_img, format_ext="JPEG")

                # 4. Construct Inspection Object
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

                # Update Active Inspection & Persistent History
                st.session_state.active_inspection = inspection_record
                save_inspection_to_history(inspection_record)

                st.success(f"✅ Inspection `{inspection_id}` completed successfully!")

                # Immediate navigation to Inspection Results
                navigate_to("Inspection Results")


# ============================================================================
# 8. PAGE 3: INSPECTION RESULTS
# ============================================================================
def render_inspection_results(active_inspection: Optional[Dict[str, Any]]):
    """Render dedicated inspection outcome views with tabbed visualizers and KPIs."""
    st.markdown("### 🎯 Inspection Results & Visual Evidence")
    st.caption("AI-annotated site inspection overlays, compliance ratios, and actionable safety advisories.")

    if active_inspection is None:
        st.warning("⚠️ No active inspection available to display. Conduct an inspection first.")
        if st.button("Go to New Inspection"):
            navigate_to("New Inspection")
        return

    summary = active_inspection["summary"]
    workers = active_inspection["workers"]
    unassigned = active_inspection["unassigned_gear"]

    # Header Card
    st.markdown(
        f"""
        <div class='sv-panel' style='padding: 12px 18px; margin-bottom: 16px;'>
            <strong>Inspection ID:</strong> <code>{active_inspection['id']}</code> &nbsp;|&nbsp;
            <strong>Analyzed Image:</strong> <code>{active_inspection['image_name']}</code> &nbsp;|&nbsp;
            <strong>Timestamp:</strong> <code>{active_inspection['timestamp']}</code>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # KPI Banner
    status = summary["site_status"]
    status_color = "#22c55e" if status == "Safe" else ("#f59e0b" if status == "Warning" else "#ef4444")
    status_badge = f"✅ {status.upper()}" if status == "Safe" else (f"⚠️ {status.upper()}" if status == "Warning" else f"🚨 {status.upper()}")

    r1, r2, r3, r4, r5, r6 = st.columns(6)
    with r1:
        st.markdown(
            f"<div class='metric-card'><div class='metric-card-title'>Workers Detected</div>"
            f"<div class='metric-card-value'>{summary['total_workers']}</div></div>",
            unsafe_allow_html=True,
        )
    with r2:
        st.markdown(
            f"<div class='metric-card'><div class='metric-card-title'>Compliant</div>"
            f"<div class='metric-card-value' style='color:#22c55e;'>{summary['compliant_count']}</div></div>",
            unsafe_allow_html=True,
        )
    with r3:
        st.markdown(
            f"<div class='metric-card'><div class='metric-card-title'>Violations</div>"
            f"<div class='metric-card-value' style='color:#ef4444;'>{summary['violation_count']}</div></div>",
            unsafe_allow_html=True,
        )
    with r4:
        st.markdown(
            f"<div class='metric-card'><div class='metric-card-title'>Needs Review</div>"
            f"<div class='metric-card-value' style='color:#f59e0b;'>{summary.get('needs_review_count', 0)}</div></div>",
            unsafe_allow_html=True,
        )
    with r5:
        st.markdown(
            f"<div class='metric-card'><div class='metric-card-title'>Compliance Rate</div>"
            f"<div class='metric-card-value'>{summary['compliance_rate']}%</div></div>",
            unsafe_allow_html=True,
        )
    with r6:
        st.markdown(
            f"<div class='metric-card'><div class='metric-card-title'>Site Status</div>"
            f"<div class='metric-card-value' style='color:{status_color}; font-size:20px;'>{status_badge}</div></div>",
            unsafe_allow_html=True,
        )

    # Advisory Banner
    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
    if summary.get("total_workers", 0) == 0:
        st.warning(
            "No workers detected. The image may be from an unsupported angle (for example overhead) "
            "or workers are too small. Try a front or side view."
        )
    else:
        advisory = get_action_recommendation(summary)
        if status == "Safe":
            st.success(f"📋 **Site Safety Directive:** {advisory}")
        elif status == "Warning":
            st.warning(f"📋 **Site Safety Directive:** {advisory}")
        else:
            st.error(f"📋 **Site Safety Directive:** {advisory}")

    # Visual Inspection Tabs
    st.markdown("---")
    st.markdown("#### Visual Evidence Tabs")

    tab_annotated, tab_side_by_side, tab_raw = st.tabs([
        "🎯 AI Annotated Image",
        "⚖️ Side-by-Side Comparison",
        "📷 Original Image",
    ])

    with tab_annotated:
        st.image(
            active_inspection["annotated_image"],
            caption="AI Annotated Inspection (Green: Compliant, Red: Violation, Orange: Unknown)",
            use_container_width=True,
        )

    with tab_side_by_side:
        c_left, c_right = st.columns(2)
        with c_left:
            st.image(active_inspection["image"], caption="Original Unprocessed Photo", use_container_width=True)
        with c_right:
            st.image(active_inspection["annotated_image"], caption="AI Detection & Compliance", use_container_width=True)

    with tab_raw:
        st.image(active_inspection["image"], caption="Original High-Resolution Image", use_container_width=True)

    # Next Steps Navigation
    st.markdown("---")
    c_btn1, c_btn2 = st.columns(2)
    with c_btn1:
        if st.button("👷 Inspect Individual Workers (Worker Audits)", use_container_width=True):
            navigate_to("Worker Audits")
    with c_btn2:
        if st.button("📑 Export PDF/CSV Compliance Report", use_container_width=True):
            navigate_to("Reports & Exports")


# ============================================================================
# 9. PAGE 4: WORKER AUDITS
# ============================================================================
def render_worker_audits(active_inspection: Optional[Dict[str, Any]]):
    """Render the granular worker audit table and individual crop drill-down inspector."""
    st.markdown("### 👷 Granular Worker Audits & Drill-Down Inspector")
    st.caption("Inspect individual worker detections, body zone alignment, and gear evidence.")

    if active_inspection is None:
        st.info("ℹ️ No active inspection available. Conduct an inspection to evaluate workers.")
        if st.button("Go to New Inspection"):
            navigate_to("New Inspection")
        return

    workers = active_inspection.get("workers", [])
    input_image = active_inspection["image"]

    if not workers:
        st.warning(
            "No workers detected. The image may be from an unsupported angle (for example overhead) "
            "or workers are too small. Try a front or side view."
        )
        if "annotated_image" in active_inspection:
            st.image(
                active_inspection["annotated_image"],
                caption="AI Annotated Inspection (Gear Detections)",
                use_container_width=True,
            )
        if st.button("Back to New Inspection"):
            navigate_to("New Inspection")
        return

    # Overview Table
    st.markdown("#### 1. Worker Compliance Summary Table")
    records = format_worker_table(workers)
    st.dataframe(pd.DataFrame(records), use_container_width=True)

    # Drill-Down Inspector
    st.markdown("---")
    st.markdown("#### 2. Individual Worker Drill-Down Inspector (Zoom Magnifier)")
    st.caption("Select a worker from the dropdown to inspect their cropped ROI and zone evidence:")

    worker_labels = [
        f"Worker #{w['worker_id']} — Status: {w['status']} (Confidence: {int(w.get('confidence', 1.0) * 100)}%)"
        for w in workers
    ]
    selected_label = st.selectbox("Select Worker to Inspect:", worker_labels)
    selected_idx = worker_labels.index(selected_label)
    selected_worker = workers[selected_idx]

    # Crop worker bounding box with zone overlays
    worker_crop = crop_worker_roi(input_image, selected_worker, padding_pct=0.10, show_zones=True)

    col_crop, col_details = st.columns([1, 2])

    with col_crop:
        st.image(
            worker_crop,
            caption=f"Worker #{selected_worker['worker_id']} Cropped ROI",
            use_container_width=True,
        )

    with col_details:
        w_status = selected_worker["status"]
        stat_color = "#22c55e" if w_status == "Compliant" else ("#ef4444" if "Missing" in w_status else "#f59e0b")

        st.markdown(
            f"""
            <div class='sv-panel' style='padding: 16px; margin-bottom: 12px;'>
                <h4 style='margin:0 0 10px 0;'>Worker #{selected_worker['worker_id']} Profile</h4>
                <p style='margin:4px 0;'><strong>Overall Status:</strong> <span style='color:{stat_color}; font-weight:bold;'>{w_status}</span></p>
                <p style='margin:4px 0;'><strong>Person Confidence:</strong> {int(selected_worker.get('confidence', 1.0) * 100)}%</p>
                <p style='margin:4px 0;'><strong>Head Protection (Hardhat):</strong> {selected_worker['hardhat_status'].capitalize()}</p>
                <p style='margin:4px 0;'><strong>Torso Protection (Safety Vest):</strong> {selected_worker['vest_status'].capitalize()}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        assigned_gear = selected_worker.get("assigned_gear", [])
        if assigned_gear:
            st.markdown("**Assigned PPE Evidence Items:**")
            gear_rows = []
            for g in assigned_gear:
                gear_rows.append({
                    "Detected Item": g["class_name"],
                    "Detection Confidence": f"{int(g.get('confidence', 1.0) * 100)}%",
                    "Zone Overlap (IoG)": f"{int(g.get('overlap', 0.0) * 100)}%",
                })
            st.table(pd.DataFrame(gear_rows))
        else:
            st.info("No safety gear detected inside this worker's head or torso zones.")


# ============================================================================
# 10. PAGE 5: INSPECTION HISTORY
# ============================================================================
def render_inspection_history(history: List[Dict[str, Any]], model: Any):
    """Render the persistent history of all completed site inspections with review actions."""
    st.markdown("### 📜 Site Inspection History")
    st.caption("Audit log of all previously completed construction site safety inspections. Select any record to load its full results.")

    if not history:
        st.info("ℹ️ No historical inspections recorded yet. Once an inspection is executed, it will be logged here.")
        if st.button("Start First Inspection"):
            navigate_to("New Inspection")
        return

    # Render History Cards with interactive View Results actions
    for idx, item in enumerate(history):
        status = item["site_status"]
        status_color = "#22c55e" if status == "Safe" else ("#f59e0b" if status == "Warning" else "#ef4444")
        item_id = item["id"]

        with st.container():
            c_info, c_stats, c_btn = st.columns([4, 4, 2])

            with c_info:
                st.markdown(
                    f"<span style='font-size:16px; font-weight:700; color:#38bdf8;'>{item_id}</span><br>"
                    f"<span style='font-size:13px; color:#94a3b8;'>Date: {item['timestamp']}</span><br>"
                    f"<span style='font-size:13px; color:#94a3b8;'>Image: <code>{item['image_name']}</code></span>",
                    unsafe_allow_html=True,
                )

            with c_stats:
                st.markdown(
                    f"<span style='font-size:18px; font-weight:800; color:{status_color};'>{status.upper()}</span><br>"
                    f"<span style='font-size:13px; color:#94a3b8;'>Compliance: <strong>{item['compliance_rate']}%</strong></span><br>"
                    f"<span style='font-size:13px; color:#94a3b8;'>Workers: <strong>{item['compliant_count']}/{item['total_workers']} compliant</strong></span>",
                    unsafe_allow_html=True,
                )

            with c_btn:
                st.markdown("<div style='padding-top:14px;'></div>", unsafe_allow_html=True)
                if st.button("🔍 View Results", key=f"view_hist_{item_id}_{idx}", use_container_width=True):
                    # 1. Check if full object is in session store
                    if item_id in st.session_state.inspections_store:
                        st.session_state.active_inspection = st.session_state.inspections_store[item_id]
                        navigate_to("Inspection Results")
                    # 2. Re-construct from image path if available on disk
                    elif item.get("image_path") and os.path.exists(item["image_path"]):
                        with st.spinner(f"Reloading inspection for {item['image_name']}..."):
                            reloaded_img = Image.open(item["image_path"]).convert("RGB")
                            req_mode = item.get("settings", {}).get("required_gear", "both")
                            conf_t = item.get("settings", {}).get("conf_threshold", 0.25)

                            detections = run_detection(reloaded_img, model=model, conf_threshold=conf_t, device="cpu")
                            import numpy as _np2
                            _h2 = float(_np2.array(reloaded_img).shape[0])
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
                        st.info("Full image not cached in memory for this session.")

            st.markdown("<hr style='margin: 8px 0 12px 0; border: none; border-top: 1px solid rgba(148,163,184,0.1);'>", unsafe_allow_html=True)


# ============================================================================
# 11. PAGE 6: REPORTS & EXPORTS
# ============================================================================
def render_reports_exports(active_inspection: Optional[Dict[str, Any]]):
    """Render the export portal for generating CSV reports and downloading annotated images."""
    st.markdown("### 📑 Regulatory Compliance Reports & Exports")
    st.caption("Generate official safety audit logs and export annotated photographic evidence.")

    if active_inspection is None:
        st.warning("⚠️ No active inspection data available to export. Conduct an inspection first.")
        if st.button("Go to New Inspection"):
            navigate_to("New Inspection")
        return

    summary = active_inspection["summary"]
    workers = active_inspection["workers"]
    timestamp_slug = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Audit Overview Card
    st.markdown(
        f"""
        <div class='sv-panel'>
            <h4>Executive Inspection Summary</h4>
            &bull; <strong>Inspection ID:</strong> <code>{active_inspection['id']}</code><br>
            &bull; <strong>Source Image:</strong> <code>{active_inspection['image_name']}</code><br>
            &bull; <strong>Execution Date:</strong> <code>{active_inspection['timestamp']}</code><br>
            &bull; <strong>Site Safety Grade:</strong> <strong>{summary['site_status'].upper()}</strong> ({summary['compliance_rate']}% Compliance)<br>
            &bull; <strong>Worker Breakdown:</strong> {summary['total_workers']} Detected | {summary['compliant_count']} Compliant | {summary['violation_count']} Violations | {summary.get('needs_review_count', 0)} Needs Review
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Worker Audit Table Preview
    st.markdown("#### Worker Record Table")
    st.dataframe(pd.DataFrame(format_worker_table(workers)), use_container_width=True)

    # Download Buttons
    st.markdown("---")
    st.markdown("#### Export Artifacts")

    csv_data = generate_csv_report(summary, workers, active_inspection["timestamp"])
    annotated_bytes = active_inspection["annotated_bytes"]

    col_csv, col_img = st.columns(2)
    with col_csv:
        st.download_button(
            label="📥 Download Official CSV Audit Report",
            data=csv_data,
            file_name=f"PPE_Safety_Audit_{timestamp_slug}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_img:
        st.download_button(
            label="🖼️ Download Annotated Evidence Photo (JPG)",
            data=annotated_bytes,
            file_name=f"Annotated_Evidence_{timestamp_slug}.jpg",
            mime="image/jpeg",
            use_container_width=True,
        )


# ============================================================================
# 12. PAGE 7: SETTINGS & MODEL SPECIFICATIONS
# ============================================================================
def render_settings(model: Any, is_model_verified: bool):
    """Render portal configuration, model verification details, and viva reference specs."""
    st.markdown("### ⚙️ Portal Settings & Model Specifications")
    st.caption("Engine runtime configuration, class taxonomy verification, and academic viva reference.")

    # 1. Interface Theme
    st.markdown("#### 1. Interface Appearance")
    theme_choice = st.radio(
        "Select Portal Theme:",
        ["Dark Theme", "Light Theme"],
        index=0 if st.session_state.theme == "Dark Theme" else 1,
        horizontal=True,
    )
    if theme_choice != st.session_state.theme:
        st.session_state.theme = theme_choice
        st.rerun()

    st.markdown("---")

    # 2. YOLOv8 Model Specifications
    st.markdown("#### 2. YOLOv8 Detection Engine Specifications")
    active_path = resolve_model_path()

    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.markdown(
            f"""
            <div class='sv-panel' style='padding: 16px;'>
                <strong>Model File:</strong> <code>{active_path}</code><br>
                <strong>Architecture:</strong> <code>YOLOv8n (Nano)</code><br>
                <strong>Execution Hardware:</strong> <code>CPU Inference (Zero GPU required)</code><br>
                <strong>Dataset:</strong> <code>Roboflow Construction Site Safety (v28)</code>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col2:
        status_text = "✅ Fully Verified (10/10 Classes)" if is_model_verified else "⚠️ Fallback Mode"
        st.markdown(
            f"""
            <div class='sv-panel' style='padding: 16px;'>
                <strong>Model Class Integrity:</strong> <code>{status_text}</code><br>
                <strong>Class Count:</strong> <code>{len(EXPECTED_CLASSES)} Classes</code><br>
                <strong>Compliance Engine:</strong> <code>Pure Python Spatial Reasoner</code><br>
                <strong>Caching:</strong> <code>Enabled via @st.cache_resource</code>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Class Taxonomy Table
    st.markdown("##### Model Class Taxonomy (Verified against PROJECT_CONTEXT.md)")
    class_table_data = [
        {"Class ID": cid, "Class Name": cname, "Application Role": "Base Worker Box" if cid == 5 else ("Head Protection Evidence" if cid in (0, 2) else ("Torso Protection Evidence" if cid in (7, 4) else "Contextual Class (Ignored in logic)"))}
        for cid, cname in EXPECTED_CLASSES.items()
    ]
    st.dataframe(pd.DataFrame(class_table_data), use_container_width=True)

    st.markdown("---")

    # 3. Compliance Rule Reference (Viva Sheet)
    st.markdown("#### 3. Compliance Logic & Viva Quick Reference")
    st.markdown(
        f"""
        * **Base YOLO Inference Confidence:** <code>0.10</code> (low threshold pass to retain small candidate gear boxes).
        * **Per-Class Confidence Filtering:** Person: <code>0.30</code> | Hardhat: <code>0.30</code> | NO-Hardhat: <code>0.20</code> | Safety Vest: <code>0.20</code> | NO-Safety Vest: <code>0.20</code>.
        * **Head Zone Range:** Top <code>{int(HEAD_ZONE_BOTTOM * 100)}%</code> of worker bounding box (0.0 to {HEAD_ZONE_BOTTOM}).
        * **Torso Zone Range:** <code>{int(TORSO_ZONE_TOP * 100)}%</code> to <code>{int(TORSO_ZONE_BOTTOM * 100)}%</code> of worker bounding box.
        * **Overlap Metric:** Intersection-over-Gear ($\\\\text{{IoG}} = \\\\frac{{\\\\text{{Area}}(\\\\text{{Gear}} \\\\cap \\\\text{{Zone}})}}{{\\\\text{{Area}}(\\\\text{{Gear}})}}$).
        * **Overlap Threshold:** Minimum <code>{int(OVERLAP_THRESHOLD * 100)}%</code> (&ge; {OVERLAP_THRESHOLD}) required for zone association.
        * **Safe Site Threshold:** &ge; {SITE_SAFE_THRESHOLD}% compliance rate.
        * **Warning Site Threshold:** &ge; {SITE_WARNING_THRESHOLD}% and &lt; {SITE_SAFE_THRESHOLD}%.
        """
    )


# ============================================================================
# 13. PAGE: CCTV-STYLE VIDEO PPE INSPECTION
# ============================================================================
def render_video_inspection(model: Any):
    """Render the CCTV-style continuous video PPE compliance inspection interface."""
    st.markdown("### 📹 CCTV Surveillance PPE Inspection")
    st.caption("Sequential CCTV-style analysis: continuous playback, interval inference, and H.264 video export on CPU.")

    # Show banner if previous inspection results exist
    cached_results = st.session_state.get("video_results")
    if cached_results is not None:
        c_banner, c_bbtn = st.columns([4, 1])
        with c_banner:
            st.info("ℹ️ Completed video inspection results are available in **Inspection Results**.")
        with c_bbtn:
            if st.button("👉 View Results", use_container_width=True):
                navigate_to("Inspection Results")

    # Upload and Configuration Columns
    c_upload, c_config = st.columns([3, 2])

    with c_upload:
        st.markdown("#### 1. Video Source")
        video_source = st.radio(
            "Source Selection",
            ["Upload file", "YouTube link"],
            horizontal=True,
            key="cctv_video_source",
        )

        uploaded_video = None
        youtube_url = ""

        if video_source == "Upload file":
            uploaded_video = st.file_uploader(
                "Select Video File (.mp4, .avi, .mov)",
                type=["mp4", "avi", "mov"],
                help=f"Upload site surveillance footage. Videos longer than {MAX_VIDEO_SECONDS}s will be warned/capped at {MAX_VIDEO_SECONDS}s from start.",
            )

            if uploaded_video is not None:
                file_mb = round(uploaded_video.size / (1024 * 1024), 2)
                st.markdown(
                    f"""
                    <div class='sv-panel' style='padding: 12px; margin-top: 8px;'>
                        <strong>Selected Footage:</strong> <code>{uploaded_video.name}</code><br>
                        <strong>File Size:</strong> <code>{file_mb} MB</code><br>
                        <strong>Processing Cap:</strong> First {MAX_VIDEO_SECONDS} seconds from start | Width &le; {MAX_FRAME_WIDTH}px
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("ℹ️ Only use videos you are allowed to analyze. Downloading from YouTube may be restricted by its terms.")
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

        start_seconds = st.number_input(
            "Start at (seconds)",
            min_value=0.0,
            value=0.0,
            step=1.0,
            help="Start CCTV inspection from this timestamp in seconds (optional).",
            key="cctv_start_seconds",
        )

    with c_config:
        st.markdown("#### 2. Detection & PPE Rules")
        gear_choice = st.radio(
            "Mandatory Site Equipment",
            ["Hardhat + Safety Vest", "Hardhat Only", "Safety Vest Only"],
            index=0,
            key="cctv_gear_choice",
            help="Select PPE requirements enforced for this surveillance inspection.",
        )
        if gear_choice == "Hardhat Only":
            required_gear_mode = "hardhat_only"
        elif gear_choice == "Safety Vest Only":
            required_gear_mode = "vest_only"
        else:
            required_gear_mode = "both"

        conf_threshold = st.slider(
            "Confidence Threshold",
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
            "Temporal Anti-Flicker Smoothing",
            value=True,
            key="cctv_temporal_smoothing_cb",
            help=f"Applies a {TEMPORAL_SMOOTHING_WINDOW}-frame rolling filter to suppress transient posture and angle flickers from skewing audit statistics.",
        )

    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

    # Run Action
    if video_source == "Upload file":
        run_disabled = uploaded_video is None
    else:
        run_disabled = not bool(youtube_url and youtube_url.strip())

    if st.button("🎬 Run CCTV Compliance Inspection", type="primary", use_container_width=True, disabled=run_disabled):
        if video_source == "Upload file":
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
            process_cctv_youtube_link(
                youtube_url=youtube_url.strip(),
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
    start_seconds: float = 0.0,
    temporal_smoothing: bool = True,
):
    """Process uploaded video sequentially: resize, interval inference, annotate, encode H.264 MP4."""
    suffix = os.path.splitext(uploaded_video.name)[1]
    if not suffix:
        suffix = ".mp4"

    # NamedTemporaryFile closed before cv2.VideoCapture opens it, deleted in finally block
    in_temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    in_temp_path = in_temp.name

    try:
        in_temp.write(uploaded_video.getbuffer())
        in_temp.close()

        execute_cctv_pipeline(
            video_path=in_temp_path,
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
        if os.path.exists(in_temp_path):
            try:
                os.remove(in_temp_path)
            except OSError:
                pass


def process_cctv_youtube_link(
    youtube_url: str,
    model: Any,
    required_gear: str,
    conf_threshold: float,
    show_gear_boxes: bool,
    show_zones: bool,
    start_seconds: float = 0.0,
    temporal_smoothing: bool = True,
):
    """Download YouTube video via yt-dlp into a temp dir and run the CCTV inspection pipeline."""
    if not is_valid_youtube_url(youtube_url):
        st.error(
            "❌ Invalid YouTube URL. Please provide a link starting with "
            "https://www.youtube.com/, https://youtube.com/, https://m.youtube.com/, or https://youtu.be/"
        )
        return

    temp_dir = tempfile.mkdtemp(prefix="safety_vision_yt_")
    try:
        with st.spinner("📥 Extracting metadata and downloading YouTube video stream..."):
            try:
                downloaded_file = download_youtube(youtube_url, temp_dir)
            except YouTubeDownloadError as e:
                st.error(f"❌ {e}")
                return
            except Exception as e:
                st.error(
                    f"❌ Unexpected error downloading YouTube video: {e}. "
                    "Please consider downloading the video file yourself and using the 'Upload file' mode."
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
    start_seconds: float = 0.0,
    temporal_smoothing: bool = True,
):
    """Core CCTV processing pipeline: seek, interval inference, annotation, and H.264 MP4 export."""
    out_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out_mp4_path = out_temp.name
    out_temp.close()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("❌ OpenCV could not open or decode the video stream.")
        if os.path.exists(out_mp4_path):
            try:
                os.remove(out_mp4_path)
            except OSError:
                pass
        return

    try:
        raw_fps = cap.get(cv2.CAP_PROP_FPS)
        raw_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        safe_fps = validate_and_get_fps(raw_fps)
        total_frames = max(0, raw_frames)

        if total_frames <= 0:
            st.error("❌ Video file contains zero readable frames.")
            cap.release()
            return

        duration_sec = calculate_video_duration(total_frames, safe_fps)

        # Validate start time
        if start_seconds >= duration_sec:
            st.error(
                f"❌ Start time ({start_seconds:.1f}s) is beyond the total video duration ({duration_sec:.1f}s). "
                "Please specify a start time within the video duration."
            )
            cap.release()
            return

        # Calculate CCTV frame range from start_seconds
        start_frame_idx, end_frame_idx = compute_cctv_frame_range(
            total_frames=total_frames,
            fps=safe_fps,
            start_seconds=start_seconds,
            max_seconds=MAX_VIDEO_SECONDS,
        )
        total_to_process = end_frame_idx - start_frame_idx

        if total_to_process <= 0:
            st.error("❌ Zero frames eligible for processing.")
            cap.release()
            return

        # Check duration cap warning (MAX_VIDEO_SECONDS = 30 from start)
        remaining_sec = max(0.0, duration_sec - start_seconds)
        if remaining_sec > MAX_VIDEO_SECONDS:
            st.warning(
                f"⚠️ Remaining video duration from {start_seconds:.1f}s ({remaining_sec:.1f}s) exceeds MAX_VIDEO_SECONDS ({MAX_VIDEO_SECONDS}s). "
                f"Only the first {MAX_VIDEO_SECONDS} seconds from {start_seconds:.1f}s will be analyzed."
            )

        # Seek to starting frame
        if start_frame_idx > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)

        # Progress UI
        st.markdown("---")
        st.markdown("#### ⏳ Analyzing CCTV Video Stream...")
        prog_bar = st.progress(0.0)
        status_line = st.empty()
        preview_container = st.empty()

        annotated_frames_rgb: List[np.ndarray] = []
        analyzed_records: List[Dict[str, Any]] = []

        last_workers: List[Dict[str, Any]] = []
        last_unassigned: List[Dict[str, Any]] = []

        start_time = time.time()

        for step_idx in range(total_to_process):
            ret, bgr_frame = cap.read()
            if not ret or bgr_frame is None or bgr_frame.size == 0:
                break

            orig_h, orig_w = bgr_frame.shape[:2]
            new_w, new_h = compute_resized_dimensions(orig_w, orig_h, max_width=MAX_FRAME_WIDTH)
            if (new_w, new_h) != (orig_w, orig_h):
                bgr_frame = cv2.resize(bgr_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            current_frame_idx = start_frame_idx + step_idx
            timestamp_sec = round(current_frame_idx / safe_fps, 2)

            # Check if this frame undergoes inference (every PROCESS_EVERY_N-th frame)
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
                review_w = evaluation["summary"]["needs_review_count"]
                rate = evaluation["summary"]["compliance_rate"] if total_w > 0 else None

                annotated_frame = annotate_image(
                    image=rgb_frame,
                    workers=last_workers,
                    unassigned_gear=last_unassigned,
                    show_gear_boxes=show_gear_boxes,
                    show_zones=show_zones,
                )

                analyzed_records.append({
                    "frame_number": current_frame_idx,
                    "timestamp_sec": timestamp_sec,
                    "total_workers": total_w,
                    "compliant_count": comp_w,
                    "violation_count": viol_w,
                    "needs_review_count": review_w,
                    "compliance_rate": rate,
                    "annotated_image": annotated_frame,
                })
            else:
                # Frames in between: draw the most recent detections and statuses on the actual frame
                annotated_frame = annotate_image(
                    image=rgb_frame,
                    workers=last_workers,
                    unassigned_gear=last_unassigned,
                    show_gear_boxes=show_gear_boxes,
                    show_zones=show_zones,
                )

            annotated_frames_rgb.append(annotated_frame)

            # Update progress, elapsed time, and preview
            elapsed = round(time.time() - start_time, 1)
            pct = (step_idx + 1) / total_to_process
            prog_bar.progress(pct)
            status_line.caption(
                f"Processing frame {step_idx + 1}/{total_to_process} ({int(pct * 100)}%) | "
                f"Video time: {timestamp_sec:.2f}s | Elapsed: {elapsed}s"
            )
            preview_container.image(
                annotated_frame,
                caption=f"CCTV Stream Preview — Frame #{current_frame_idx} (t={timestamp_sec:.2f}s)",
                use_container_width=True,
            )

        cap.release()

        if not analyzed_records:
            st.error("❌ No frames could be analyzed.")
            return

        # Write annotated video to H.264 MP4 using imageio and imageio-ffmpeg
        video_bytes: Optional[bytes] = None
        write_error: Optional[str] = None
        try:
            write_annotated_h264_video(
                frames_rgb=annotated_frames_rgb,
                output_path=out_mp4_path,
                fps=safe_fps,
            )
            with open(out_mp4_path, "rb") as vf:
                video_bytes = vf.read()
        except Exception as e:
            write_error = str(e)

        # Summary computed on the ANALYZED frames (with anti-flicker smoothing if enabled)
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

        # Save to session state so results survive reruns
        st.session_state.video_results = {
            "video_filename": video_display_name,
            "metadata": metadata_dict,
            "summary": summary,
            "analyzed_records": effective_records,
            "raw_records": analyzed_records,
            "video_bytes": video_bytes,
            "write_error": write_error,
        }

        st.success(f"✅ Video processing complete! Analyzed {len(analyzed_records)} frames ({len(annotated_frames_rgb)} total frames rendered).")
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
    """Render the CCTV video results: video player, metrics, charts, table, and moments gallery."""
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

    # Header Panel
    st.markdown(
        f"""
        <div class='sv-panel' style='padding: 12px 18px; margin-bottom: 16px;'>
            <strong>Source Video:</strong> <code>{filename}</code> &nbsp;|&nbsp;
            <strong>Duration:</strong> <code>{meta['duration_sec']:.1f}s</code> &nbsp;|&nbsp;
            <strong>Inference Stride:</strong> <code>Every {PROCESS_EVERY_N} frames</code> &nbsp;|&nbsp;
            <strong>Analyzed Frames:</strong> <code>{summary['frames_analyzed']} frames</code>{smoothing_badge}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Video Player & Download
    st.markdown("#### 🎬 Annotated CCTV Playback")
    if write_error:
        st.error(f"⚠️ Video file encoding error: {write_error}. Audit statistics remain available below.")

    if video_bytes:
        st.video(video_bytes, format="video/mp4")
        st.download_button(
            "📥 Download Annotated CCTV Video (.mp4)",
            data=video_bytes,
            file_name=f"annotated_cctv_{filename}.mp4",
            mime="video/mp4",
            use_container_width=True,
        )
    else:
        st.info("Video player unavailable, but all analytical audit metrics are presented below.")

    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

    # Anti-flicker banner if active
    if meta.get("temporal_smoothing"):
        st.info(
            f"🛡️ **Temporal Anti-Flicker Active:** Summary metrics are calculated using a {TEMPORAL_SMOOTHING_WINDOW}-frame "
            "rolling median filter. Transient detection flickers caused by rapid head-turns, occlusions, and body-angle shifts "
            "are smoothed out so temporary postures do not unfairly degrade your site compliance rate."
        )

    # 2. Executive KPI Metric Row (Computed on analyzed frames only)
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
            f"<div class='metric-card-value' style='color:#38bdf8;'>{summary['frames_with_workers']}</div></div>",
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"<div class='metric-card'><div class='metric-card-title'>Frames No Workers</div>"
            f"<div class='metric-card-value' style='color:#94a3b8;'>{summary['frames_without_workers']}</div></div>",
            unsafe_allow_html=True,
        )
    with k4:
        viol_color = "#ef4444" if summary["frames_with_violations"] > 0 else "#22c55e"
        st.markdown(
            f"<div class='metric-card'><div class='metric-card-title'>Frames With Violations</div>"
            f"<div class='metric-card-value' style='color:{viol_color};'>{summary['frames_with_violations']}</div></div>",
            unsafe_allow_html=True,
        )
    with k5:
        rate_display = f"{summary['average_compliance_rate']}%" if summary["has_worker_frames"] else "N/A"
        rate_color = "#22c55e" if (summary["has_worker_frames"] and summary["average_compliance_rate"] >= 85.0) else ("#f59e0b" if summary["has_worker_frames"] and summary["average_compliance_rate"] >= 60.0 else "#ef4444")
        st.markdown(
            f"<div class='metric-card'><div class='metric-card-title'>Avg Compliance Rate</div>"
            f"<div class='metric-card-value' style='color:{rate_color};'>{rate_display}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

    # 3. Compliance Rate Over Time Line Chart (Grouped by second)
    st.markdown("#### 📈 Compliance Rate Over Time (By Second)")
    chart_rows = []
    for f in analyzed:
        r_val = f["compliance_rate"] if f["total_workers"] > 0 else None
        chart_rows.append({
            "Time (s)": f"{f['timestamp_sec']:.1f}s",
            "Compliance Rate (%)": r_val,
        })
    df_chart = pd.DataFrame(chart_rows).set_index("Time (s)")
    if summary["has_worker_frames"]:
        st.line_chart(df_chart, color="#2563eb", height=240)
    else:
        st.info("ℹ️ No workers were detected in the analyzed frames to plot compliance trends.")

    # 4. Top Moments with the Most Violations Gallery (Top 3)
    st.markdown("---")
    st.markdown("#### 🚨 Top Risk Moments (Most Violations)")
    top_moments = summary.get("top_violation_moments", [])
    if top_moments:
        st.caption(f"Displaying up to {TOP_VIOLATION_FRAMES_COUNT} moments with the highest count of non-compliant workers:")
        g_cols = st.columns(len(top_moments))
        for col, f in zip(g_cols, top_moments):
            with col:
                st.image(
                    f["annotated_image"],
                    caption=f"Frame #{f['frame_number']} (t={f['timestamp_sec']:.2f}s) | Violations: {f['violation_count']} | Workers: {f['total_workers']}",
                    use_container_width=True,
                )
    else:
        st.success("✅ Clean Video Surveillance: Zero PPE violations detected across all analyzed frames!")

    # 5. Per-Frame Table & CSV Download
    st.markdown("---")
    st.markdown("#### 📋 Analyzed Frames Breakdown")
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

    # Actions
    c_csv, c_reset = st.columns([2, 1])
    with c_csv:
        csv_content = generate_cctv_csv_report(analyzed, summary, filename)
        st.download_button(
            "📥 Download CCTV Audit CSV Report",
            data=csv_content,
            file_name=f"cctv_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c_reset:
        if st.button("🔄 Analyze Another Video", use_container_width=True):
            st.session_state.video_results = None
            navigate_to("New Inspection")


# ============================================================================
# 13. MAIN APPLICATION ORCHESTRATOR
# ============================================================================
def main():
    # Initialize session state
    init_session_state()

    # Apply CSS matching current theme
    inject_custom_css(st.session_state.theme)

    # Load cached YOLOv8 Model on CPU
    active_model_path = resolve_model_path()
    try:
        model = get_detector(active_model_path)
        is_model_verified = verify_model_classes(model)
    except Exception as e:
        st.error(f"❌ Error loading YOLO model from `{active_model_path}`: {e}")
        st.stop()

    # ------------------------------------------------------------------------
    # SIDEBAR NAVIGATION
    # ------------------------------------------------------------------------
    with st.sidebar:
        st.markdown(
            """
            <div style='text-align: center; padding: 12px 0 16px 0;'>
                <span style='font-size: 38px;'>🛡️</span>
                <h3 style='margin: 4px 0 0 0; color: #38bdf8;'>SAFETY VISION</h3>
                <span style='font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px;'>Construction Portal</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<hr style='margin: 8px 0 12px 0; border: none; border-top: 1px solid rgba(148,163,184,0.2);'>", unsafe_allow_html=True)

        # ── Inspection Mode Selector (Image vs Video) ────────────────────
        st.markdown(
            "<span style='font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px;'>"
            "Inspection Mode</span>",
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

        st.markdown("<hr style='margin: 8px 0 16px 0; border: none; border-top: 1px solid rgba(148,163,184,0.2);'>", unsafe_allow_html=True)

        pages = [
            "Dashboard",
            "New Inspection",
            "Inspection Results",
            "Worker Audits",
            "Inspection History",
            "Reports & Exports",
            "Settings",
        ]

        # Use st.radio for navigation with synced session state
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

        st.markdown("<hr style='margin: 20px 0 16px 0; border: none; border-top: 1px solid rgba(148,163,184,0.2);'>", unsafe_allow_html=True)

        # ── Hardhat Enforcement Policy ────────────────────────────────────
        st.markdown(
            "<span style='font-size:11px; color:#94a3b8; text-transform:uppercase; "
            "letter-spacing:0.5px;'>Hardhat Policy</span>",
            unsafe_allow_html=True,
        )
        st.session_state.strict_hardhat = st.checkbox(
            "Treat undetected hardhat as violation",
            value=st.session_state.strict_hardhat,
            help=(
                "When ON, workers whose head zone has no detectable hardhat box are "
                "marked as violations \u2014 but only if their person detection is reliable "
                "(confidence \u2265 0.40, box height \u2265 12\u202f%% of frame, upright posture). "
                "Unreliable persons are always flagged as Needs Review regardless."
            ),
        )
        st.markdown(
            "<hr style='margin: 12px 0 16px 0; border: none; border-top: 1px solid rgba(148,163,184,0.2);'>",
            unsafe_allow_html=True,
        )

        # Sidebar Footer
        st.caption(f"**Engine:** YOLOv8 (CPU Mode)\n\n**Weights:** `{os.path.basename(active_model_path)}`")
        if is_model_verified:
            st.caption("✅ 10 Classes Verified")

    # ------------------------------------------------------------------------
    # TOP NAVBAR & ROUTING
    # ------------------------------------------------------------------------
    render_top_navbar(st.session_state.current_page, is_model_verified)

    active_insp = st.session_state.active_inspection
    history_list = st.session_state.inspection_history

    # Route to selected page based on mode and navigation
    if st.session_state.inspection_mode == "Video":
        if st.session_state.current_page == "New Inspection":
            render_video_inspection(model)
        elif st.session_state.current_page == "Inspection Results":
            video_res = st.session_state.get("video_results")
            if video_res is not None:
                render_video_results_view(video_res)
            else:
                st.warning("⚠️ No active video inspection results available. Please run a video inspection first.")
                if st.button("🎬 Go to Video Inspection", type="primary"):
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
        # Image Mode: route to selected page (stays exactly as it was)
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
