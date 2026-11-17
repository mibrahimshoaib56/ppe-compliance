"""Streamlit web application for Construction Worker Hardhat & Safety-Vest Compliance.

AI2002 Course Project: Software-Based PPE Compliance Monitoring using YOLOv8.
Runs 100% on CPU, caches the detection model, and applies spatial rule-based
logic to classify worker safety status.
"""

import os
from typing import Optional
from PIL import Image
import pandas as pd
import streamlit as st

from src.compliance import evaluate_site_compliance
from src.detector import load_detector_model, resolve_model_path, run_detection
from src.utils import (
    annotate_image,
    format_worker_table,
    get_action_recommendation,
)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="PPE Safety Compliance Detector",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# CACHED MODEL LOADER (Ensures model is loaded only once on CPU)
# ============================================================================
@st.cache_resource(show_spinner="Loading YOLOv8 detection model...")
def get_detector(model_file_path: str):
    """Load and cache the YOLOv8 model instance on CPU."""
    return load_detector_model(model_file_path)


# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    st.title("🦺 Hardhat & Safety-Vest Compliance Detector")
    st.markdown(
        "Automated construction site monitoring using **YOLOv8** object detection "
        "and rule-based spatial compliance logic."
    )

    # ------------------------------------------------------------------------
    # SIDEBAR CONTROLS
    # ------------------------------------------------------------------------
    st.sidebar.header("⚙️ Configuration")

    # 1. Model Resolution & Status Check
    active_model_path = resolve_model_path()
    using_custom_weights = os.path.exists(os.path.join("models", "best.pt"))

    if using_custom_weights:
        st.sidebar.success(f"Loaded trained model: `{active_model_path}`")
    else:
        st.sidebar.warning(
            "⚠️ Custom model `models/best.pt` not found.\n\n"
            "Using fallback `yolov8n.pt` (Demo/pipeline mode). "
            "Note that COCO weights detect people, but not custom PPE gear."
        )

    # 2. Confidence Threshold Slider
    conf_threshold = st.sidebar.slider(
        "Detection Confidence Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
        help="Minimum confidence score required for an object detection to be considered.",
    )

    # 3. Required Gear Selector
    gear_option = st.sidebar.radio(
        "Mandatory Safety Equipment",
        options=["Hardhat + Safety Vest", "Hardhat Only"],
        index=0,
        help="Select which PPE equipment is legally mandatory on this construction site.",
    )
    required_gear_mode = "both" if gear_option == "Hardhat + Safety Vest" else "hardhat_only"

    # 4. Visualization Options
    show_gear_boxes = st.sidebar.checkbox("Draw Gear Boxes", value=True)
    show_zones = st.sidebar.checkbox("Show Anatomical Body Zones", value=False)

    st.sidebar.markdown("---")
    st.sidebar.caption("AI2002 Project | CPU Inference Mode")

    # Load cached model
    try:
        model = get_detector(active_model_path)
    except Exception as e:
        st.error(f"❌ Error loading YOLO model from `{active_model_path}`: {e}")
        st.stop()

    # ------------------------------------------------------------------------
    # IMAGE UPLOAD SECTION
    # ------------------------------------------------------------------------
    st.subheader("1. Upload Construction Site Image")
    uploaded_file = st.file_uploader(
        "Choose a JPG or PNG site image",
        type=["jpg", "jpeg", "png"],
        help="Upload clear site photos showing construction personnel.",
    )

    if uploaded_file is None:
        st.info("👆 Please upload an image file to begin safety inspection.")
        st.stop()

    # Edge case: Verify image integrity
    try:
        input_image = Image.open(uploaded_file)
        # Convert to RGB if needed (handles RGBA / Greyscale uploads)
        if input_image.mode != "RGB":
            input_image = input_image.convert("RGB")
    except Exception as err:
        st.error(f"❌ Invalid or corrupted image file: {err}")
        st.stop()

    # Display preview and action button
    col_input, col_action = st.columns([2, 1])
    with col_input:
        st.image(input_image, caption="Uploaded Site Image", use_container_width=True)
    with col_action:
        st.write("### Ready to Analyze")
        st.write(f"- **Resolution:** {input_image.size[0]} x {input_image.size[1]} px")
        st.write(f"- **Confidence:** `{conf_threshold}`")
        st.write(f"- **Requirements:** `{gear_option}`")
        detect_button = st.button("🔍 Run Compliance Check", type="primary", use_container_width=True)

    if not detect_button:
        st.stop()

    # ------------------------------------------------------------------------
    # INFERENCE & COMPLIANCE EVALUATION
    # ------------------------------------------------------------------------
    with st.spinner("Running YOLOv8 inference and evaluating worker compliance..."):
        detections = run_detection(
            image=input_image,
            model=model,
            conf_threshold=conf_threshold,
            device="cpu",
        )

        evaluation = evaluate_site_compliance(
            detections=detections,
            required_gear=required_gear_mode,
        )

    workers = evaluation["workers"]
    summary = evaluation["summary"]
    unassigned_gear = evaluation["unassigned_gear"]

    # ------------------------------------------------------------------------
    # SITE METRICS & SUMMARY
    # ------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("2. Site Compliance Summary")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Workers Detected", summary["total_workers"])
    m2.metric("Compliant", summary["compliant_count"])
    m3.metric("Violations", summary["violation_count"])
    m4.metric("Compliance Rate", f"{summary['compliance_rate']}%")

    site_status = summary["site_status"]
    status_color = "green" if site_status == "Safe" else ("orange" if site_status == "Warning" else "red")
    m5.markdown(
        f"**Site Status**<br><span style='font-size: 22px; font-weight: bold; color: {status_color};'>"
        f"{site_status.upper()}</span>",
        unsafe_allow_html=True,
    )

    # Actionable Advisory Message
    advisory = get_action_recommendation(summary)
    if site_status == "Safe":
        st.success(f"📋 **Action Advisory:** {advisory}")
    elif site_status == "Warning":
        st.warning(f"📋 **Action Advisory:** {advisory}")
    else:
        st.error(f"📋 **Action Advisory:** {advisory}")

    # Edge Case: Zero workers detected
    if summary["total_workers"] == 0:
        st.info("ℹ️ No personnel detected in this image. Adjust the confidence threshold if workers are visible.")
        if len(unassigned_gear) > 0:
            st.warning(f"Found {len(unassigned_gear)} loose gear item(s) not associated with any worker.")

    # ------------------------------------------------------------------------
    # ANNOTATED VISUALIZATION
    # ------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("3. Detection & Compliance Visual Inspection")

    annotated_img = annotate_image(
        image=input_image,
        workers=workers,
        unassigned_gear=unassigned_gear,
        show_gear_boxes=show_gear_boxes,
        show_zones=show_zones,
    )

    st.image(
        annotated_img,
        caption="Annotated Inspection Image (Green: Compliant, Red: Violation, Orange: Unknown)",
        use_container_width=True,
    )

    # ------------------------------------------------------------------------
    # PER-WORKER BREAKDOWN TABLE
    # ------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("4. Detailed Worker Breakdown")

    if workers:
        records = format_worker_table(workers)
        df_workers = pd.DataFrame(records)
        st.dataframe(df_workers, use_container_width=True)
    else:
        st.write("No worker records to display.")


if __name__ == "__main__":
    main()
