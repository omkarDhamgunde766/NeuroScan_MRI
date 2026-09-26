import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import sys
import time
import io
from datetime import datetime

# Ensure project root is in Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from neuroscan.inference import NeuroScanPipeline
from neuroscan.visualizer import plot_confidence_bar


# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="NeuroScan Analytics",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =====================================================
# CUSTOM STYLING
# =====================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family:'Inter',sans-serif; }
#MainMenu, footer { visibility:hidden; }
.stApp { background:#f6f9fc; color:#172033; }
/* Keep the Streamlit chrome/header white. */
header[data-testid="stHeader"] {
    background:#ffffff !important;
    border-bottom:1px solid #edf1f5 !important;
    box-shadow:none !important;
}
header[data-testid="stHeader"] * { color:#667487 !important; }
div[data-testid="stToolbar"] { background:#ffffff !important; }

.block-container { padding:1.2rem 2.2rem 2.5rem 2.2rem; max-width:1500px; }

section[data-testid="stSidebar"] {
    width:188px !important;
    min-width:188px !important;
    background:#ffffff !important;
    border-right:1px solid #e7edf4 !important;
    box-shadow:4px 0 18px rgba(31,55,86,.025) !important;
}
section[data-testid="stSidebar"] > div { padding:1rem .85rem 1.25rem; }
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#6f7c8f; }
.neuro-sidebar-brand { display:flex; align-items:center; gap:11px; margin:4px 5px 25px; }
.neuro-logo {
    width:46px; height:46px; flex:0 0 46px; border-radius:15px;
    display:flex; align-items:center; justify-content:center;
    background:linear-gradient(135deg,#dff9ff,#b9eff9);
    color:#16b8d6; font-size:24px;
    box-shadow:0 8px 20px rgba(38,191,221,.16);
}
.neuro-brand-title { display:block; color:#273143; font-size:14px; font-weight:800; }
.neuro-brand-subtitle { display:block; color:#9aa5b3; font-size:10px; margin-top:2px; }
.sidebar-status {
    display:block; color:#8b98a8; font-size:9px; letter-spacing:.05em;
    margin:0 7px 14px;
}

/* clean navigation */
div[data-testid="stRadio"] > label { display:none; }
div[data-testid="stRadio"] > div { gap:7px !important; }
div[data-testid="stRadio"] > div > label {
    min-height:45px !important; padding:9px 11px !important;
    border-radius:13px !important; border:1px solid transparent !important;
    background:transparent !important; color:#667487 !important;
    justify-content:flex-start !important;
}
div[data-testid="stRadio"] > div > label:hover {
    background:#f1fbfd !important; color:#10abc5 !important;
    border-color:#e0f5f8 !important;
}
div[data-testid="stRadio"] > div > label[data-checked="true"] {
    background:#e9faff !important; color:#0daac5 !important;
    border-color:#c9f3fa !important; box-shadow:0 4px 12px rgba(38,191,221,.08) !important;
}
div[data-testid="stRadio"] > div > label p {
    font-size:12px !important; font-weight:600 !important; margin:0 !important;
    color:inherit !important;
}
div[data-testid="stRadio"] > div > label p::before {
    display:inline-block; width:27px; font-size:17px; vertical-align:-2px; color:inherit;
}
div[data-testid="stRadio"] > div > label:nth-child(1) p::before { content:'◉'; }
div[data-testid="stRadio"] > div > label:nth-child(2) p::before { content:'◷'; }
div[data-testid="stRadio"] > div > label:nth-child(3) p::before { content:'▣'; }
div[data-testid="stRadio"] > div > label:nth-child(4) p::before { content:'⚙'; }
section[data-testid="stSidebar"] hr { border-color:#edf1f5; margin:18px 0; }

.dashboard-topbar { display:flex; align-items:center; justify-content:space-between; gap:18px; margin-bottom:28px; }
.search-box { background:#fff; border:1px solid #e9eef4; border-radius:13px; padding:11px 18px; color:#9aa6b6; width:330px; box-shadow:0 5px 18px rgba(31,55,86,.04); }
.top-actions { display:flex; align-items:center; gap:12px; color:#738095; font-size:12px; }
.date-pill,.status-pill { background:#fff; border:1px solid #e8edf3; border-radius:12px; padding:10px 13px; }
.status-pill { color:#12afc9; font-weight:700; }
.dashboard-title { font-size:36px; line-height:1.1; font-weight:700; color:#1d2533; margin:0; letter-spacing:-1px; }
.dashboard-subtitle { color:#8793a4; font-size:13px; margin-top:7px; }

.metric-card { background:#fff; border:1px solid #edf1f5; border-radius:18px; padding:20px; min-height:132px; box-shadow:0 7px 24px rgba(31,55,86,.05); }
.metric-label { color:#778497; font-size:12px; font-weight:600; margin-bottom:13px; }
.metric-value { color:#172033; font-size:30px; font-weight:700; letter-spacing:-.6px; }
.metric-note { color:#9aa5b3; font-size:11px; margin-top:7px; }
.metric-icon { float:right; width:34px; height:34px; border-radius:11px; background:#e8faff; color:#15b6d4; display:flex; align-items:center; justify-content:center; font-size:17px; }

.section-label { font-size:15px; font-weight:700; color:#273143; margin:24px 0 12px; }
.glass-panel { background:#fff; border:1px solid #edf1f5; border-radius:20px; padding:20px; box-shadow:0 7px 24px rgba(31,55,86,.05); }
.pathology-card,.confidence-card { min-height:185px; height:auto; text-align:left; display:block; }
.pathology-label,.confidence-label,.tumor-area-title { color:#7a8797; font-size:11px; letter-spacing:.06em; font-weight:700; text-transform:uppercase; }
.pathology-value { color:#1c2738; font-size:32px; font-weight:700; margin:14px 0 12px; text-shadow:none; }
.confidence-value { color:#172033; font-size:34px; font-weight:700; text-shadow:none; margin-top:12px; }
.confidence-ai-label { color:#12b8d6; font-size:10px; margin-top:5px; }
.badge-critical,.badge-stable { padding:6px 11px; font-size:10px; border-radius:999px; }
.badge-critical { background:#fff1f0; border:1px solid #ffd8d4; color:#e35b51; }
.badge-stable { background:#eafaff; border:1px solid #c8f2fa; color:#10abc5; }


.confidence-analysis-grid { display:grid; grid-template-columns:1.15fr .85fr; gap:18px; margin-top:12px; }
.uncertainty-card { background:#fff; border:1px solid #edf1f5; border-radius:20px; padding:22px; box-shadow:0 7px 24px rgba(31,55,86,.05); }
.uncertainty-head { display:flex; justify-content:space-between; align-items:center; gap:12px; margin-bottom:18px; }
.uncertainty-title { color:#273143; font-size:16px; font-weight:700; }
.uncertainty-subtitle { color:#98a3b1; font-size:11px; margin-top:4px; }
.uncertainty-pill { background:#eafaff; color:#0eabc6; border:1px solid #c8f2fa; border-radius:999px; padding:6px 10px; font-size:10px; font-weight:700; }
.uncertainty-main { display:flex; align-items:center; gap:18px; }
.uncertainty-score { font-size:34px; font-weight:700; color:#172033; line-height:1; }
.uncertainty-label { color:#7a8797; font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:.06em; margin-top:6px; }
.uncertainty-bar { height:9px; background:#edf3f6; border-radius:999px; overflow:hidden; margin-top:12px; }
.uncertainty-fill { height:100%; border-radius:999px; background:linear-gradient(90deg,#5ad7e8,#12b8d6); }
.uncertainty-metric { padding:12px 0; border-bottom:1px solid #eef2f5; }
.uncertainty-metric:last-child { border-bottom:none; }
.uncertainty-metric-row { display:flex; justify-content:space-between; align-items:center; gap:10px; }
.uncertainty-metric-label { color:#748195; font-size:11px; font-weight:600; }
.uncertainty-metric-value { color:#202b3c; font-size:14px; font-weight:700; }
.uncertainty-note { color:#9aa5b3; font-size:10px; line-height:1.5; margin-top:14px; }
@media (max-width: 900px) { .confidence-analysis-grid { grid-template-columns:1fr; } }

.modern-topbar { background:#fff; border:1px solid #edf1f5; border-radius:18px; padding:15px 18px; box-shadow:0 7px 24px rgba(31,55,86,.04); display:flex; justify-content:space-between; align-items:center; }
.topbar-title { color:#273143; font-size:14px; font-weight:700; }
.topbar-subtitle { color:#929dad; font-size:11px; margin-top:3px; }
.online-pill { background:#eafaff; border:1px solid #caf2f9; color:#0eabc6; border-radius:999px; padding:7px 12px; font-size:10px; font-weight:700; }
.status-dot { width:7px; height:7px; display:inline-block; border-radius:50%; background:#20c5df; margin-right:5px; box-shadow:none; }

.stFileUploader { margin-bottom:12px; }
div[data-testid="stFileUploader"] > section {
    background:#ffffff !important;
    border:2px dashed #cfeaf1 !important;
    border-radius:18px !important;
    padding:1.25rem 1.4rem !important;
    box-shadow:0 5px 18px rgba(31,55,86,.03) !important;
}
div[data-testid="stFileUploader"] section,
div[data-testid="stFileUploader"] section * {
    color:#526174 !important;
}
div[data-testid="stFileUploader"] section button {
    background:#e9faff !important;
    color:#079db9 !important;
    border:1px solid #bfeaf2 !important;
    border-radius:10px !important;
    font-weight:700 !important;
    box-shadow:none !important;
}
div[data-testid="stFileUploader"] section button:hover {
    background:#d9f7fc !important;
    color:#078fa9 !important;
    border-color:#9fe1ec !important;
}
div[data-testid="stFileUploader"] section button:focus {
    background:#e9faff !important;
    color:#079db9 !important;
    border-color:#9fe1ec !important;
    box-shadow:0 0 0 2px rgba(22,183,212,.12) !important;
}
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
    background:#f7fbfd !important;
    border:1px solid #dfeef2 !important;
    border-radius:10px !important;
    color:#344154 !important;
}
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] * {
    color:#344154 !important;
}
div[data-testid="stFileUploader"] small {
    color:#8a97a7 !important;
}
div[data-testid="stFileUploader"] svg {
    color:#10abc5 !important;
    fill:#10abc5 !important;
}
/* File uploader: force a light theme for the button and selected-file chip. */
div[data-testid="stFileUploader"] button,
div[data-testid="stFileUploader"] button[data-testid="stBaseButton-secondary"] {
    opacity:1 !important;
    visibility:visible !important;
    display:inline-flex !important;
    background:#e9faff !important;
    color:#079db9 !important;
    border:1px solid #bfeaf2 !important;
}
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
    background:#f7fbfd !important;
    color:#344154 !important;
    border:1px solid #dfeef2 !important;
    box-shadow:none !important;
}
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] button {
    background:transparent !important;
    color:#7c8999 !important;
    border:0 !important;
}
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] svg {
    color:#10abc5 !important;
    fill:#10abc5 !important;
}


.viewport-header { display:flex; align-items:center; gap:8px; margin:22px 0 12px; color:#283345; }
.viewport-header .material-symbols-outlined { color:#13b7d5; font-size:22px; }
.viewport-header span:last-child { font-size:16px; font-weight:700; }
.image-container { border-radius:16px; overflow:hidden; border:1px solid #e9eef4; background:#fff; box-shadow:0 7px 20px rgba(31,55,86,.05); margin-bottom:8px; }
.image-caption { text-align:left; padding:10px 13px; background:#fff; color:#5f6d80; font-weight:600; font-size:11px; border-top:1px solid #eef2f5; }

.summary-card { gap:1rem; }
.summary-title { font-size:17px; font-weight:700; color:#273143; }
.summary-row { padding:11px 0; border-bottom:1px solid #edf1f5; }
.summary-field-label { color:#9aa5b3; font-size:9px; letter-spacing:.08em; font-weight:700; }
.summary-field-value { color:#384558; font-size:12px; font-weight:600; margin-top:3px; }

.ood-alert { background:#fff5f3; border:1px solid #ffd7d0; color:#b95145; border-radius:16px; padding:16px; display:flex; gap:14px; }
.ood-alert-title { color:#d9574c; font-size:14px; font-weight:700; }
.ood-alert-body { color:#8c6863; font-size:12px; line-height:1.6; }
.ood-check-fail { background:#fff0ed; border:1px solid #ffd9d2; color:#c85d51; border-radius:8px; padding:4px 8px; font-size:10px; margin:3px; display:inline-flex; }
.ood-score-bar,.tumor-area-bar { height:8px; border-radius:999px; background:#edf2f6; overflow:hidden; margin-top:12px; }
.ood-score-fill { height:100%; border-radius:999px; background:linear-gradient(90deg,#ff8b77,#ffc05b); }
.conf-warning { background:#fff9e9; border:1px solid #f8e3a5; color:#8d742d; border-radius:14px; padding:13px 15px; margin-bottom:15px; font-size:12px; }

.tumor-area-card { background:#fff; border:1px solid #edf1f5; border-radius:18px; padding:18px; margin-top:8px; min-height:125px; box-shadow:0 7px 24px rgba(31,55,86,.04); }
.tumor-area-title { margin-bottom:8px; }
.tumor-area-value { font-size:30px; font-weight:700; color:#16b7d4; }
.tumor-area-subtitle { font-size:10px; color:#98a3b0; margin-top:4px; }
.tumor-area-fill { height:100%; border-radius:999px; background:linear-gradient(90deg,#72ddec,#16b7d4); }

div[data-testid="stPlotlyChart"] { background:#fff; border:1px solid #edf1f5; border-radius:18px; padding:8px; box-shadow:0 7px 24px rgba(31,55,86,.04); }
.stButton > button, .stDownloadButton > button { border-radius:11px !important; border:1px solid #cfeaf1 !important; background:#e9faff !important; color:#079db9 !important; font-weight:700 !important; }
.stProgress > div > div > div { background:#19b9d6 !important; }
.stAlert { border-radius:14px !important; }

@media (max-width:900px) {
    .block-container { padding:1rem; }
    section[data-testid="stSidebar"] { width:188px !important; min-width:188px !important; }
    .search-box { width:190px; }
    .dashboard-title { font-size:28px; }
}

.xai-panel { background:#fff; border:1px solid #e7eef3; border-radius:22px; padding:22px; margin-top:22px; box-shadow:0 8px 26px rgba(31,55,86,.05); }
.xai-head { display:flex; justify-content:space-between; align-items:center; gap:12px; margin-bottom:16px; }
.xai-title { font-size:18px; font-weight:800; color:#273143; }
.xai-subtitle { font-size:11px; color:#96a1ae; margin-top:4px; }
.xai-badge { background:#eafaff; color:#10abc5; border:1px solid #c8f2fa; border-radius:999px; padding:7px 11px; font-size:10px; font-weight:800; letter-spacing:.04em; }
.xai-grid { display:grid; grid-template-columns:1.35fr .65fr; gap:18px; align-items:stretch; }
.xai-info { background:#f8fcfd; border:1px solid #e7f3f6; border-radius:16px; padding:18px; }
.xai-metric { font-size:28px; font-weight:800; color:#14aec8; margin:8px 0 4px; }
.xai-label { font-size:10px; color:#8793a2; font-weight:700; letter-spacing:.08em; text-transform:uppercase; }
.xai-copy { font-size:12px; line-height:1.65; color:#697585; margin-top:12px; }
.xai-legend { display:flex; align-items:center; gap:10px; margin-top:16px; }
.xai-gradient { height:8px; flex:1; border-radius:99px; background:linear-gradient(90deg,#2146d0,#22b9e1,#f5d94a,#e63d35); }
.xai-legend-text { display:flex; justify-content:space-between; font-size:9px; color:#8b96a4; margin-top:5px; }
.xai-note { margin-top:14px; padding:11px 13px; border-radius:12px; background:#fffaf0; border:1px solid #f5e6bd; color:#806c3d; font-size:10px; line-height:1.5; }
@media(max-width:900px){ .xai-grid{grid-template-columns:1fr;} }


.report-actions { display:flex; gap:10px; flex-wrap:wrap; }
.feature-card { background:#fff; border:1px solid #edf1f5; border-radius:20px; padding:20px; box-shadow:0 7px 24px rgba(31,55,86,.05); }
.feature-card-title { color:#273143; font-size:16px; font-weight:800; }
.feature-card-subtitle { color:#98a3b1; font-size:11px; margin-top:4px; }
.performance-table { width:100%; border-collapse:collapse; margin-top:12px; font-size:11px; }
.performance-table th { text-align:left; color:#718096; background:#f7fafc; padding:10px; border-bottom:1px solid #e8eef3; }
.performance-table td { padding:10px; color:#253246; border-bottom:1px solid #eef2f5; }
.performance-table td.num { font-weight:700; }
.history-item { background:#fff; border:1px solid #edf1f5; border-radius:16px; padding:16px; margin-bottom:10px; box-shadow:0 5px 16px rgba(31,55,86,.03); }
.history-class { font-weight:800; color:#273143; font-size:15px; }
.history-meta { color:#8793a4; font-size:11px; margin-top:5px; }
.patient-intake-wrap { background:#ffffff; border:1px solid #e4edf3; border-radius:24px; padding:28px 30px; margin:18px auto 22px; max-width:980px; box-shadow:0 10px 32px rgba(31,55,86,.06); }
.patient-intake-badge { display:inline-block; color:#10abc5; background:#e9fbff; border:1px solid #c9f2f8; border-radius:999px; padding:6px 11px; font-size:9px; font-weight:800; letter-spacing:.12em; }
.patient-intake-title { color:#172033; font-size:28px; font-weight:800; margin-top:13px; }
.patient-intake-subtitle { color:#7d8998; font-size:12px; margin-top:5px; }

</style>
""",
    unsafe_allow_html=True,
)


# =====================================================
# PATIENT INTAKE / SESSION PROFILE
# =====================================================
if "patient_profile" not in st.session_state:
    st.session_state.patient_profile = None

if st.session_state.patient_profile is None:
    st.markdown(
        """
        <div class="patient-intake-wrap">
            <div class="patient-intake-badge">PATIENT REGISTRATION</div>
            <div class="patient-intake-title">Start a New MRI Analysis</div>
            <div class="patient-intake-subtitle">Enter the patient's basic details before uploading the MRI scan.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("patient_intake_form", clear_on_submit=False):
        st.markdown("### Patient Details")
        col1, col2 = st.columns(2, gap="large")
        with col1:
            patient_name = st.text_input("Patient Name *", placeholder="Enter full name")
            patient_age = st.number_input("Patient Age *", min_value=0, max_value=120, value=18, step=1)
        with col2:
            patient_location = st.text_input("Village / City *", placeholder="Enter village, town or city")
            patient_district = st.text_input("District / State (optional)", placeholder="e.g. Latur, Maharashtra")

        submitted = st.form_submit_button("Continue to MRI Upload →", type="primary", width="stretch")
        if submitted:
            if not patient_name.strip():
                st.error("Please enter the patient's name.")
            elif not patient_location.strip():
                st.error("Please enter the patient's village or city.")
            else:
                st.session_state.patient_profile = {
                    "name": patient_name.strip(),
                    "age": int(patient_age),
                    "location": patient_location.strip(),
                    "district_state": patient_district.strip(),
                }
                st.session_state.scan_history = []
                st.rerun()

    st.caption("Patient details are kept only in this active Streamlit session and are added to generated reports.")
    st.stop()


# =====================================================
# INITIALIZE PIPELINE
# =====================================================
@st.cache_resource
def load_pipeline():
    return NeuroScanPipeline()


pipeline = load_pipeline()


# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.markdown(
        """
        <div class="neuro-sidebar-brand">
            <div class="neuro-logo">🧠</div>
            <div>
                <div class="neuro-brand-title">NeuroScan</div>
                <div class="neuro-brand-subtitle">MRI Intelligence</div>
            </div>
        </div>
        <div class="sidebar-status">
            <span class="status-dot"></span>MODEL SYSTEM ONLINE
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        ["🔬 Live Scan", "🕒 History", "📂 Patient Records", "⚙️ Settings"],
        label_visibility="collapsed",
    )

    patient = st.session_state.get("patient_profile")
    if patient:
        st.markdown(
            f"""<div style=\"margin:10px 6px 14px;padding:11px 12px;border-radius:14px;background:#f5fbfd;border:1px solid #dceff4;\">
                <div style=\"font-size:9px;color:#8a97a6;letter-spacing:.08em;font-weight:800;\">CURRENT PATIENT</div>
                <div style=\"font-size:13px;color:#273143;font-weight:800;margin-top:4px;\">{patient["name"]}</div>
                <div style=\"font-size:10px;color:#7c8998;margin-top:2px;\">{patient["age"]} yrs • {patient["location"]}</div>
            </div>""",
            unsafe_allow_html=True,
        )
        if st.button("↻ Change Patient", width="stretch"):
            st.session_state.patient_profile = None
            st.session_state.scan_history = []
            st.rerun()

    st.markdown("---")
    st.markdown("**System Status**")

    if pipeline.seg_model is not None:
        st.success("✅ Segmentation Core Online")
    else:
        st.error("❌ Segmentation Core Offline")

    if pipeline.cls_model is not None:
        st.success("✅ Classification Node Online")
    else:
        st.error("❌ Classification Node Offline")


# =====================================================
# HELPER: OOD / MRI-LIKENESS VALIDATOR
# =====================================================
def validate_mri_image(image_bgr):
    """
    Lightweight MRI-likeness validation.
    This is a heuristic input check, not a clinical OOD detector.
    """
    failed_checks = []
    scores = []

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mean_saturation = float(np.mean(hsv[:, :, 1]))
    sat_ok = mean_saturation < 30
    scores.append(
        1.0
        if sat_ok
        else max(0.0, 1.0 - (mean_saturation - 30) / 120)
    )

    if not sat_ok:
        failed_checks.append(
            f"High color saturation ({mean_saturation:.0f}/255) — MRI scans are grayscale"
        )

    b, g, r = cv2.split(image_bgr.astype(np.float32))
    channel_diff = (
        float(np.mean(np.abs(r - g)))
        + float(np.mean(np.abs(g - b)))
    ) / 2

    gray_ok = channel_diff < 15
    scores.append(
        1.0
        if gray_ok
        else max(0.0, 1.0 - (channel_diff - 15) / 60)
    )

    if not gray_ok:
        failed_checks.append(
            f"Non-uniform color channels (avg diff={channel_diff:.1f})"
        )

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    dark_pixel_ratio = float(np.sum(gray < 30) / gray.size)
    dark_ok = dark_pixel_ratio > 0.20
    scores.append(
        1.0 if dark_ok else dark_pixel_ratio / 0.20
    )

    if not dark_ok:
        failed_checks.append(
            f"Insufficient dark background ({dark_pixel_ratio*100:.1f}%)"
        )

    mean_brightness = float(np.mean(gray))
    std_brightness = float(np.std(gray))

    brightness_ok = mean_brightness < 160 and std_brightness > 15
    scores.append(1.0 if brightness_ok else 0.3)

    if not brightness_ok:
        if mean_brightness >= 160:
            failed_checks.append(
                f"Image too bright (mean={mean_brightness:.0f})"
            )
        if std_brightness <= 15:
            failed_checks.append(
                f"Low contrast (std={std_brightness:.1f})"
            )

    ood_score = float(np.mean(scores))
    is_valid = len(failed_checks) <= 1 and ood_score >= 0.55

    return {
        "is_valid_mri": is_valid,
        "ood_score": ood_score,
        "failed_checks": failed_checks,
    }


def calculate_uncertainty_metrics(probabilities, confidence):
    """Compute probability-based uncertainty indicators for the classifier output.

    These are model-output indicators, not calibrated clinical uncertainty estimates.
    """
    probs = np.asarray(list((probabilities or {}).values()), dtype=np.float64)
    probs = probs[np.isfinite(probs)]
    probs = np.clip(probs, 0.0, 1.0)
    total = float(probs.sum())
    if probs.size == 0 or total <= 0:
        return {
            "uncertainty_pct": max(0.0, 100.0 - float(confidence) * 100.0),
            "margin_pct": 0.0,
            "entropy_pct": 0.0,
        }
    probs = probs / total
    sorted_probs = np.sort(probs)[::-1]
    top1 = float(sorted_probs[0])
    top2 = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
    margin_pct = max(0.0, (top1 - top2) * 100.0)
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
    max_entropy = float(np.log(len(probs))) if len(probs) > 1 else 1.0
    entropy_pct = (entropy / max_entropy * 100.0) if max_entropy > 0 else 0.0
    uncertainty_pct = max(0.0, min(100.0, 100.0 - float(confidence) * 100.0))
    return {
        "uncertainty_pct": uncertainty_pct,
        "margin_pct": margin_pct,
        "entropy_pct": entropy_pct,
    }


# =====================================================
# ADDITIONAL FEATURES 7–17
# =====================================================
def calculate_bbox_and_centroid(mask):
    """Return a bounding box and centroid for the predicted binary mask."""
    if mask is None:
        return None, None
    m = np.asarray(mask)
    if m.ndim == 3:
        m = np.squeeze(m)
    binary = (m > 0).astype(np.uint8)
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return None, None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    bbox = {"x": x1, "y": y1, "width": x2-x1+1, "height": y2-y1+1}
    centroid = {"x": float(xs.mean()), "y": float(ys.mean())}
    return bbox, centroid


def create_bbox_centroid_overlay(image_rgb, mask):
    """Draw approximate segmentation bounding box and centroid on an RGB image."""
    canvas = np.asarray(image_rgb).copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
    bbox, centroid = calculate_bbox_and_centroid(mask)
    if bbox is None:
        return canvas, None, None
    x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
    cv2.rectangle(canvas, (x, y), (x+w-1, y+h-1), (22, 183, 212), 2)
    cx, cy = int(round(centroid["x"])), int(round(centroid["y"]))
    cv2.circle(canvas, (cx, cy), 5, (235, 83, 73), -1)
    return canvas, bbox, centroid


def calculate_image_quality_metrics(image_bgr, ood_result):
    """Project-specific technical image quality indicators, not clinical quality assessment."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    saturation = float(np.mean(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)[:, :, 1]))
    resolution = int(image_bgr.shape[0] * image_bgr.shape[1])
    resolution_score = min(100.0, resolution / 65536.0 * 100.0)
    contrast_score = min(100.0, max(0.0, (contrast / 60.0) * 100.0))
    brightness_score = 100.0 - min(100.0, abs(brightness - 90.0) / 90.0 * 100.0)
    saturation_score = max(0.0, 100.0 - saturation * 2.0)
    ood_score = float(ood_result.get("ood_score", 0.0)) * 100.0
    overall = float(np.mean([resolution_score, contrast_score, brightness_score, saturation_score, ood_score]))
    return {
        "overall": max(0.0, min(100.0, overall)),
        "resolution": resolution,
        "resolution_score": resolution_score,
        "brightness": brightness,
        "brightness_score": brightness_score,
        "contrast": contrast,
        "contrast_score": contrast_score,
        "saturation": saturation,
        "saturation_score": saturation_score,
        "mri_likeness": ood_score,
    }


def make_findings_summary(patient, pred_class, confidence_pct, tumor_percentage, uncertainty, bbox, centroid):
    """Create a structured model-generated findings summary without anatomical/clinical claims."""
    finding = f"The classifier predicted {str(pred_class).upper()} with {confidence_pct:.1f}% model confidence."
    seg = f"The segmentation mask covers approximately {tumor_percentage:.2f}% of the image frame."
    if bbox:
        loc = f"The predicted region bounding box is approximately {bbox['width']} × {bbox['height']} pixels."
    else:
        loc = "No positive segmentation region was detected."
    uncertainty_text = f"Probability-based uncertainty indicator is {uncertainty['uncertainty_pct']:.1f}% with a top-1 margin of {uncertainty['margin_pct']:.1f}%."
    return finding + " " + seg + " " + loc + " " + uncertainty_text


def process_uploaded_mri(uploaded, pipeline):
    """Convert an uploaded file and run the same pipeline used by Live Scan."""
    img = Image.open(uploaded).convert("RGB")
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    validation = validate_mri_image(bgr)
    if not validation["is_valid_mri"]:
        return None, validation, None
    start = time.time()
    result = pipeline.process_image(bgr)
    latency = int((time.time() - start) * 1000)
    return result, validation, latency


# =====================================================
# HELPER: CONFIDENCE CARD
# =====================================================
# =====================================================
# ALL-FEATURE SUPPORT
# =====================================================

def add_scan_to_history(filename, pred_class, confidence_pct, tumor_percentage, latency_ms):
    if "scan_history" not in st.session_state:
        st.session_state.scan_history = []
    patient = st.session_state.get("patient_profile") or {}
    entry = {
        "time": datetime.now().strftime("%d %b %Y, %I:%M %p"),
        "patient": patient.get("name", "Unknown patient"),
        "age": patient.get("age", "—"),
        "file": filename or "Uploaded MRI",
        "class": pred_class.upper(),
        "confidence": float(confidence_pct),
        "tumor_area": float(tumor_percentage),
        "latency": float(latency_ms),
    }
    key = (entry["file"], entry["class"], round(entry["confidence"], 3), round(entry["tumor_area"], 3))
    if not any((x["file"], x["class"], round(x["confidence"], 3), round(x["tumor_area"], 3)) == key for x in st.session_state.scan_history):
        st.session_state.scan_history.insert(0, entry)
        st.session_state.scan_history = st.session_state.scan_history[:25]

def normalize_probability_distribution(probabilities):
    """Normalize classifier probability keys for dashboard/report display.

    The inference pipeline may use labels such as NoTumor, no_tumor, or
    lowercase class names. The UI must not treat a label-format difference
    as a missing probability. Values are kept exactly as model outputs.
    """
    source = probabilities or {}
    aliases = {
        "glioma": "Glioma",
        "meningioma": "Meningioma",
        "pituitary": "Pituitary",
        "notumor": "No Tumor",
        "no tumor": "No Tumor",
        "no_tumor": "No Tumor",
        "no-tumor": "No Tumor",
    }
    normalized = {"Glioma": 0.0, "Meningioma": 0.0, "Pituitary": 0.0, "No Tumor": 0.0}
    for raw_key, raw_value in source.items():
        key = str(raw_key).strip().lower().replace("_", " ")
        display_key = aliases.get(key, str(raw_key).strip())
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if display_key in normalized:
            normalized[display_key] = value
    return normalized


def build_pdf_report(patient, pred_class, confidence_pct, threshold_pct, below_threshold, summary, latency, tumor_area, uncertainty, xai_metrics, probabilities=None, original_image=None, segmentation_image=None, gradcam_image=None, roi_image=None):
    """Build a clean A4 NeuroScan MRI report with patient, model and visual findings."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether
    except ImportError:
        return None

    def image_flowable(image, width=230, height=165):
        if image is None:
            return None
        try:
            arr = np.asarray(image)
            if arr.ndim == 2:
                arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            elif arr.ndim == 3 and arr.shape[-1] == 3:
                arr = arr.astype(np.uint8)
            pil = Image.fromarray(arr)
            bio = io.BytesIO()
            pil.save(bio, format="PNG")
            bio.seek(0)
            return RLImage(bio, width=width, height=height, kind="proportional")
        except Exception:
            return None

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=34, bottomMargin=34)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("NeuroTitle", parent=styles["Title"], alignment=TA_CENTER, fontSize=21, leading=25, textColor=colors.HexColor("#123047"), spaceAfter=4)
    subtitle = ParagraphStyle("NeuroSub", parent=styles["BodyText"], alignment=TA_CENTER, fontSize=9, textColor=colors.HexColor("#6B7B8C"), spaceAfter=12)
    h2 = ParagraphStyle("NeuroH2", parent=styles["Heading2"], fontSize=13, leading=16, textColor=colors.HexColor("#0B7285"), spaceBefore=8, spaceAfter=7)
    small = ParagraphStyle("NeuroSmall", parent=styles["BodyText"], fontSize=8.5, leading=12, textColor=colors.HexColor("#536577"))
    note = ParagraphStyle("NeuroNote", parent=small, fontSize=7.8, textColor=colors.HexColor("#687889"))

    story = [
        Paragraph("NeuroScan MRI Report", title),
        Paragraph("Brain MRI Classification & Tumor Segmentation Analysis", subtitle),
    ]

    patient_rows = [
        ["Patient Name", str(patient.get("name", "—"))],
        ["Age", f"{patient.get('age', '—')} years"],
        ["Village / City", str(patient.get("location", "—"))],
    ]
    if patient.get("district_state"):
        patient_rows.append(["District / State", str(patient["district_state"])])
    patient_rows.append(["Report Generated", datetime.now().strftime("%d %b %Y, %I:%M %p")])
    story += [Paragraph("Patient Details", h2)]
    pt = Table(patient_rows, colWidths=[145, 335])
    pt.setStyle(TableStyle([("BACKGROUND",(0,0),(0,-1),colors.HexColor("#EAF8FC")),("TEXTCOLOR",(0,0),(0,-1),colors.HexColor("#0B7285")),("FONTNAME",(0,0),(0,-1),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),9),("GRID",(0,0),(-1,-1),.35,colors.HexColor("#D8E7ED")),("PADDING",(0,0),(-1,-1),7)]))
    story += [pt, Spacer(1, 10), Paragraph("Analysis Result", h2)]

    result_rows = [
        ["Detected Class", pred_class.upper()],
        ["Model Confidence", f"{confidence_pct:.1f}%"],
        ["Confidence Threshold", f"{threshold_pct:.0f}%"],
        ["Below Threshold", "Yes" if below_threshold else "No"],
        ["Tumor Area", f"{tumor_area['area_percentage']:.2f}% pixel-area proportion"],
        ["Tumor Pixels", f"{tumor_area['tumor_pixels']:,}"],
        ["Processing Time", f"{latency:.0f} ms"],
    ]
    rt = Table(result_rows, colWidths=[165,315])
    rt.setStyle(TableStyle([("BACKGROUND",(0,0),(0,-1),colors.HexColor("#F4FAFC")),("FONTNAME",(0,0),(0,-1),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),9),("GRID",(0,0),(-1,-1),.35,colors.HexColor("#DCE8EE")),("PADDING",(0,0),(-1,-1),7)]))
    story += [rt, Spacer(1, 8)]

    probabilities = normalize_probability_distribution(probabilities)
    story += [Paragraph("Probability Distribution", h2)]
    prob_rows = [["Class", "Probability"]] + [[str(k), f"{float(v)*100:.2f}%"] for k,v in probabilities.items()]
    ptab = Table(prob_rows, colWidths=[300,180])
    ptab.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#0B7285")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),9),("GRID",(0,0),(-1,-1),.35,colors.HexColor("#DCE8EE")),("PADDING",(0,0),(-1,-1),6)]))
    story += [ptab, Spacer(1, 8)]

    story += [Paragraph("Tumor Segmentation Analysis", h2)]
    seg_rows = [["Metric", "Value"], ["Tumor Pixels", f"{tumor_area['tumor_pixels']:,}"], ["Total Mask Pixels", f"{tumor_area['total_pixels']:,}"], ["Tumor Area", f"{tumor_area['area_percentage']:.2f}%"], ["Segmentation Model", "Attention U-Net"]]
    stab = Table(seg_rows, colWidths=[300,180])
    stab.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#10ABC5")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),9),("GRID",(0,0),(-1,-1),.35,colors.HexColor("#DCE8EE")),("PADDING",(0,0),(-1,-1),6)]))
    story += [stab, Spacer(1, 8)]

    # Visual Analysis: include every available MRI visualization in a clean 2x2 grid.
    visual_items = [
        (original_image, "Original MRI Scan"),
        (segmentation_image, "Segmentation Mask Overlay"),
        (roi_image, "Isolated Region of Interest"),
        (gradcam_image, "Grad-CAM Heatmap / Model Focus"),
    ]
    visual_cells = []
    for img, label in visual_items:
        flow = image_flowable(img, 210, 145)
        if flow:
            visual_cells.append([flow, Paragraph(f"<b>{label}</b>", small)])

    if visual_cells:
        story += [Paragraph("Visual Analysis", h2)]
        rows = []
        for i in range(0, len(visual_cells), 2):
            left = visual_cells[i]
            right = visual_cells[i + 1] if i + 1 < len(visual_cells) else ["", ""]
            rows.append([left, right])
        vt = Table(rows, colWidths=[240, 240], hAlign="CENTER")
        vt.setStyle(TableStyle([
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("BOX", (0,0), (-1,-1), .5, colors.HexColor("#DCE8EE")),
            ("INNERGRID", (0,0), (-1,-1), .4, colors.HexColor("#EDF2F5")),
            ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#FAFCFD")),
            ("ALIGN", (0,0), (-1,-1), "CENTER"),
            ("PADDING", (0,0), (-1,-1), 9),
        ]))
        story += [vt, Spacer(1, 10)]

    story += [Paragraph("Model & Explainability", h2)]
    story += [Paragraph(f"Classification model: <b>EfficientNetV2B0</b><br/>Segmentation model: <b>Attention U-Net</b><br/>Explainability: <b>Grad-CAM</b><br/>Model uncertainty: {uncertainty.get('uncertainty_pct', 0):.1f}% &nbsp;|&nbsp; Top-1 margin: {uncertainty.get('margin_pct', 0):.1f}% &nbsp;|&nbsp; Mean Grad-CAM activation: {xai_metrics.get('mean_activation', 0):.1f}%", small)]
    story += [Spacer(1, 10), Paragraph("Important: This report is generated by a research prototype. Probability, segmentation area and Grad-CAM are model outputs/analysis aids and are not a clinical diagnosis or a physical tumor-volume measurement.", note)]

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()

def build_confidence_card(confidence_pct, latency_ms):
    radius = 45
    circumference = 2 * 3.14159 * radius
    safe_confidence = min(max(float(confidence_pct), 0.0), 100.0)
    offset = circumference * (1 - safe_confidence / 100)

    return (
        '<div class="glass-panel confidence-card">'
        '<span class="confidence-label">Diagnostic Confidence</span>'
        '<div style="position:relative;width:200px;height:200px;margin:0 auto;">'
        '<svg viewBox="0 0 100 100" style="width:100%;height:100%;transform:rotate(-90deg);">'
        f'<circle cx="50" cy="50" r="{radius}" fill="transparent" '
        'stroke="#edf2f6" stroke-width="8"/>'
        f'<circle cx="50" cy="50" r="{radius}" fill="transparent" '
        'stroke="url(#ng)" stroke-width="8" '
        f'stroke-dasharray="{circumference:.1f}" '
        f'stroke-dashoffset="{offset:.1f}" stroke-linecap="round"/>'
        '<defs><linearGradient id="ng" x1="0%" y1="0%" x2="100%" y2="100%">'
        '<stop offset="0%" style="stop-color:#54339c"/>'
        '<stop offset="50%" style="stop-color:#2355cc"/>'
        '<stop offset="100%" style="stop-color:#28d9f3"/>'
        '</linearGradient></defs>'
        '</svg>'
        '<div style="position:absolute;inset:0;display:flex;'
        'flex-direction:column;align-items:center;justify-content:center;">'
        f'<span class="confidence-value">{safe_confidence:.1f}%</span>'
        '<span class="confidence-ai-label">MODEL OUTPUT</span>'
        '</div>'
        '</div>'
        '<div style="margin-top:1rem;display:flex;gap:1.5rem;'
        'justify-content:center;font-size:10px;letter-spacing:0.1em;'
        'font-weight:600;color:#8d98a8;">'
        f'<span>MODEL: EFFNET-V2B0</span>'
        f'<span>LATENCY: {latency_ms}MS</span>'
        '</div>'
        '</div>'
    )


# =====================================================
# HELPER: SEVERITY BADGE
# =====================================================
def get_severity_badge(pred_class):
    critical_classes = ["glioma", "meningioma"]

    if str(pred_class).lower() in critical_classes:
        return (
            '<div class="badge-critical">'
            '<span class="material-symbols-outlined" style="font-size:14px;">'
            'warning</span>MODEL FLAGGED TUMOR CLASS'
            '</div>'
        )

    return (
        '<div class="badge-stable">'
        '<span class="material-symbols-outlined" style="font-size:14px;">'
        'check_circle</span>MODEL OUTPUT'
        '</div>'
    )


# =====================================================
# HELPER: ANALYSIS SUMMARY
# =====================================================
def get_analysis_summary(pred_class):
    """
    Only reports what the current model actually provides.
    It does not infer morphology, location, or vascularity.
    """
    key = str(pred_class).lower().replace(" ", "")

    if key == "notumor":
        return {
            "morphology": "Not provided by model",
            "localization": "Not provided by model",
            "vascularity": "Not provided by model",
        }

    return {
        "morphology": "Not directly measured",
        "localization": "Not directly measured",
        "vascularity": "Not directly measured",
    }


# =====================================================
# FEATURE 1: TUMOR AREA ANALYSIS
# =====================================================
def calculate_tumor_area(mask):
    """
    Calculate segmented tumor pixel area and percentage.

    Important:
    This is a pixel-area proportion only.
    It is NOT a physical measurement in cm² and does NOT represent volume.
    """
    if mask is None:
        return {
            "tumor_pixels": 0,
            "total_pixels": 0,
            "area_percentage": 0.0,
        }

    mask_array = np.asarray(mask)

    if mask_array.ndim == 3:
        mask_array = np.squeeze(mask_array)

    binary_mask = mask_array > 0

    tumor_pixels = int(np.sum(binary_mask))
    total_pixels = int(binary_mask.size)

    area_percentage = (
        (tumor_pixels / total_pixels) * 100.0
        if total_pixels > 0
        else 0.0
    )

    return {
        "tumor_pixels": tumor_pixels,
        "total_pixels": total_pixels,
        "area_percentage": area_percentage,
    }



def calculate_xai_metrics(heatmap):
    """Summarize Grad-CAM activation for display only.

    This is an explanation aid, not a tumor-size or clinical measurement.
    """
    if heatmap is None:
        return {"mean_activation": 0.0, "high_activation_pct": 0.0}
    arr = np.asarray(heatmap)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY) if arr.shape[-1] == 3 else np.squeeze(arr)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return {"mean_activation": 0.0, "high_activation_pct": 0.0}
    arr_min, arr_max = float(np.nanmin(arr)), float(np.nanmax(arr))
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min)
    else:
        arr = np.zeros_like(arr)
    return {
        "mean_activation": float(np.mean(arr) * 100.0),
        "high_activation_pct": float(np.mean(arr >= 0.70) * 100.0),
    }

# =====================================================
# LIVE SCAN PAGE
# =====================================================
if page == "🔬 Live Scan":

    st.markdown(
        """
        <div class="dashboard-topbar">
            <div class="search-box">⌕&nbsp;&nbsp; Search NeuroScan...</div>
            <div class="top-actions">
                <div class="date-pill">◷&nbsp; Live Analysis</div>
                <div class="status-pill">●&nbsp; Models Online</div>
                <div style="font-size:22px;color:#17b8d5;">↻</div>
            </div>
        </div>
        <div style="margin-bottom:24px;">
            <div class="dashboard-title">Dashboard Overview</div>
            <div class="dashboard-subtitle">Brain MRI analysis, tumor segmentation and explainable AI in one workspace.</div>
        </div>
        <div class="modern-topbar">
            <div>
                <div class="topbar-title">NeuroScan MRI Intelligence</div>
                <div class="topbar-subtitle">Attention U-Net • EfficientNetV2B0 • Grad-CAM</div>
            </div>
            <div class="online-pill"><span class="status-dot"></span>ANALYSIS READY</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if pipeline.seg_model is None or pipeline.cls_model is None:
        st.warning(
            "⚠️ Models not loaded. Check the .keras checkpoints in checkpoints/."
        )

    st.markdown(
        '<p class="section-label">Insert Medical Imaging Data (JPG/PNG)</p>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload MRI Scan",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if (
        uploaded_file is not None
        and pipeline.seg_model is not None
        and pipeline.cls_model is not None
    ):

        try:
            pil_img = Image.open(uploaded_file).convert("RGB")
            open_cv_image = cv2.cvtColor(
                np.array(pil_img),
                cv2.COLOR_RGB2BGR,
            )
        except Exception as exc:
            st.error(f"Could not read the uploaded image: {exc}")
            st.stop()

        # -------------------------------------------------
        # Step 0: MRI-LIKENESS VALIDATION
        # -------------------------------------------------
        ood_result = validate_mri_image(open_cv_image)
        ood_score_pct = int(ood_result["ood_score"] * 100)

        if not ood_result["is_valid_mri"]:

            fail_tags = "".join(
                f'<span class="ood-check-fail">'
                f'<span class="material-symbols-outlined" '
                f'style="font-size:13px;">cancel</span>'
                f'{check}</span>'
                for check in ood_result["failed_checks"]
            )

            st.markdown(
                f"""
                <div class="ood-alert">
                    <span class="material-symbols-outlined ood-alert-icon">
                        warning
                    </span>
                    <div>
                        <div class="ood-alert-title">
                            ⚠ Invalid Input — Image Does Not Appear to Be an MRI Scan
                        </div>
                        <div class="ood-alert-body">
                            The uploaded image failed
                            {len(ood_result["failed_checks"])}
                            MRI-likeness check(s). This model was trained on brain
                            MRI images, so non-MRI images should not be used for inference.
                            <br><br>
                            <b>Failed checks:</b><br>
                            {fail_tags}
                            <div class="ood-score-bar">
                                <div class="ood-score-fill"
                                     style="width:{ood_score_pct}%"></div>
                            </div>
                            <div style="font-size:11px;
                                color:#8d98a8;
                                margin-top:4px;">
                                MRI Likeness Score: {ood_score_pct}%
                                (threshold: 55%)
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.stop()

        # -------------------------------------------------
        # INFERENCE
        # -------------------------------------------------
        progress_bar = st.progress(
            0,
            text="Initializing Neural Pipeline...",
        )

        progress_bar.progress(
            25,
            text="Segmenting Anomaly Regions (Attention U-Net)...",
        )

        start_time = time.time()

        try:
            results = pipeline.process_image(open_cv_image)
        except Exception as exc:
            progress_bar.empty()
            st.error(f"Model inference failed: {exc}")
            st.stop()

        latency = int((time.time() - start_time) * 1000)

        progress_bar.progress(
            70,
            text="Classifying Pathology (EfficientNetV2B0)...",
        )

        progress_bar.progress(
            90,
            text="Generating Grad-CAM Explanation...",
        )

        progress_bar.progress(
            100,
            text="Analysis complete.",
        )

        time.sleep(0.15)
        progress_bar.empty()

        st.toast("Neural processing complete!", icon="🧠")

        confidence_pct = float(results["confidence"]) * 100
        pred_class = str(results["pred_class"])
        below_threshold = bool(results.get("below_threshold", False))
        threshold_pct = int(
            float(results.get("confidence_threshold", 0.70)) * 100
        )

        # Dashboard KPI values
        kpi_tumor = calculate_tumor_area(results.get("mask"))
        kpi_area = kpi_tumor["area_percentage"]

        st.markdown('<div class="section-label">Analysis Snapshot</div>', unsafe_allow_html=True)
        k1, k2, k3, k4 = st.columns(4, gap="medium")
        with k1:
            st.markdown(f"<div class=\"metric-card\"><div class=\"metric-icon\">⌁</div><div class=\"metric-label\">DETECTED CLASS</div><div class=\"metric-value\" style=\"font-size:23px;\">{pred_class.upper()}</div><div class=\"metric-note\">4-class MRI classifier</div></div>", unsafe_allow_html=True)
        with k2:
            st.markdown(f"<div class=\"metric-card\"><div class=\"metric-icon\">✓</div><div class=\"metric-label\">MODEL CONFIDENCE</div><div class=\"metric-value\">{confidence_pct:.1f}%</div><div class=\"metric-note\">Threshold: {threshold_pct}%</div></div>", unsafe_allow_html=True)
        with k3:
            st.markdown(f"<div class=\"metric-card\"><div class=\"metric-icon\">◉</div><div class=\"metric-label\">TUMOR AREA</div><div class=\"metric-value\">{kpi_area:.2f}%</div><div class=\"metric-note\">Segmentation pixel proportion</div></div>", unsafe_allow_html=True)
        with k4:
            st.markdown(f"<div class=\"metric-card\"><div class=\"metric-icon\">⚡</div><div class=\"metric-label\">PROCESSING TIME</div><div class=\"metric-value\">{latency}ms</div><div class=\"metric-note\">End-to-end inference</div></div>", unsafe_allow_html=True)

        # -------------------------------------------------
        # LOW CONFIDENCE WARNING
        # -------------------------------------------------
        if below_threshold:
            st.markdown(
                f"""
                <div class="conf-warning">
                    <span class="material-symbols-outlined">info</span>
                    <span>
                        <b>Low Model Confidence ({confidence_pct:.1f}%)</b>
                        — the best class prediction is below the configured
                        {threshold_pct}% threshold. Treat this as a research
                        prototype output, not a clinical diagnosis.
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # -------------------------------------------------
        # FEATURE 2: CONFIDENCE & UNCERTAINTY ANALYSIS
        # -------------------------------------------------
        probabilities = normalize_probability_distribution(results.get("probabilities", {}))
        uncertainty = calculate_uncertainty_metrics(
            probabilities, confidence_pct / 100.0
        )
        uncertainty_pct = uncertainty["uncertainty_pct"]
        margin_pct = uncertainty["margin_pct"]
        entropy_pct = uncertainty["entropy_pct"]

        st.markdown(
            '<div class="section-label">Confidence &amp; Uncertainty Analysis</div>',
            unsafe_allow_html=True,
        )
        st.html(
            f"""
            <div class="uncertainty-card">
                <div class="uncertainty-head">
                    <div>
                        <div class="uncertainty-title">Model Confidence &amp; Uncertainty</div>
                        <div class="uncertainty-subtitle">Probability-based indicators from the 4-class classifier</div>
                    </div>
                    <div class="uncertainty-pill">RESEARCH INDICATOR</div>
                </div>

                <div class="uncertainty-main">
                    <div style="min-width:105px;">
                        <div class="uncertainty-score">{confidence_pct:.1f}%</div>
                        <div class="uncertainty-label">Confidence</div>
                    </div>
                    <div style="flex:1;">
                        <div class="uncertainty-metric-row">
                            <span class="uncertainty-metric-label">Model confidence</span>
                            <span class="uncertainty-metric-value">{confidence_pct:.1f}%</span>
                        </div>
                        <div class="uncertainty-bar"><div class="uncertainty-fill" style="width:{confidence_pct:.1f}%;"></div></div>
                    </div>
                </div>

                <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:18px;">
                    <div class="uncertainty-metric">
                        <div class="uncertainty-metric-row"><span class="uncertainty-metric-label">Uncertainty</span><span class="uncertainty-metric-value">{uncertainty_pct:.1f}%</span></div>
                    </div>
                    <div class="uncertainty-metric">
                        <div class="uncertainty-metric-row"><span class="uncertainty-metric-label">Top-1 margin</span><span class="uncertainty-metric-value">{margin_pct:.1f}%</span></div>
                    </div>
                    <div class="uncertainty-metric">
                        <div class="uncertainty-metric-row"><span class="uncertainty-metric-label">Entropy</span><span class="uncertainty-metric-value">{entropy_pct:.1f}%</span></div>
                    </div>
                </div>

                <div class="uncertainty-note">
                    Uncertainty is estimated from the classifier probability distribution. It is not a calibrated probability of disease and should not be interpreted as a clinical diagnosis.
                </div>
            </div>
            """
        )

        # -------------------------------------------------
        # RESULT CARDS
        # -------------------------------------------------
        col_path, col_conf = st.columns(2, gap="large")

        with col_path:
            badge_html = get_severity_badge(pred_class)

            st.markdown(
                f"""
                <div class="glass-panel pathology-card">
                    <span class="pathology-label">
                        Detected Pathology Class
                    </span>
                    <div class="pathology-value">
                        {pred_class.upper()}
                    </div>
                    {badge_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_conf:
            st.markdown(
                build_confidence_card(
                    confidence_pct,
                    latency,
                ),
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # -------------------------------------------------
        # IMAGE RESULTS
        # -------------------------------------------------
        st.markdown(
            """
            <div class="viewport-header">
                <span class="material-symbols-outlined">visibility</span>
                <span>Axial MRI Layer Scan</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        row1_col1, row1_col2 = st.columns(2, gap="medium")
        row2_col1, row2_col2 = st.columns(2, gap="medium")

        raw_rgb = cv2.cvtColor(
            open_cv_image,
            cv2.COLOR_BGR2RGB,
        )

        target_dim = (
            raw_rgb.shape[1],
            raw_rgb.shape[0],
        )

        with row1_col1:
            st.markdown(
                '<div class="image-container">',
                unsafe_allow_html=True,
            )
            st.image(
                raw_rgb,
                width="stretch",
            )
            st.markdown(
                '<div class="image-caption">Raw MRI Scan</div></div>',
                unsafe_allow_html=True,
            )

        with row1_col2:
            overlay = results.get("overlay")

            if overlay is not None:
                overlay_resized = cv2.resize(
                    overlay,
                    target_dim,
                    interpolation=cv2.INTER_CUBIC,
                )
            else:
                overlay_resized = raw_rgb

            st.markdown(
                '<div class="image-container">',
                unsafe_allow_html=True,
            )
            st.image(
                overlay_resized,
                width="stretch",
            )
            st.markdown(
                '<div class="image-caption">'
                'Segmentation Mask Overlay'
                '</div></div>',
                unsafe_allow_html=True,
            )

        with row2_col1:
            roi_display = results.get("cropped_roi")

            if roi_display is None:
                roi_display = open_cv_image.copy()

            if len(roi_display.shape) == 2:
                roi_display = cv2.cvtColor(
                    roi_display,
                    cv2.COLOR_GRAY2RGB,
                )
            else:
                roi_display = cv2.cvtColor(
                    roi_display,
                    cv2.COLOR_BGR2RGB,
                )

            roi_resized = cv2.resize(
                roi_display,
                target_dim,
                interpolation=cv2.INTER_CUBIC,
            )

            st.markdown(
                '<div class="image-container">',
                unsafe_allow_html=True,
            )
            st.image(
                roi_resized,
                width="stretch",
            )
            st.markdown(
                '<div class="image-caption">'
                'Isolated Region of Interest'
                '</div></div>',
                unsafe_allow_html=True,
            )

        with row2_col2:
            gradcam = results.get("gradcam_overlay")

            if gradcam is not None:
                gradcam_resized = cv2.resize(
                    gradcam,
                    target_dim,
                    interpolation=cv2.INTER_CUBIC,
                )
                display_image = gradcam_resized
            else:
                display_image = raw_rgb

            st.markdown(
                '<div class="image-container">',
                unsafe_allow_html=True,
            )
            st.image(
                display_image,
                width="stretch",
            )
            st.markdown(
                '<div class="image-caption">'
                'Grad-CAM (Model Focus)'
                '</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br><br>", unsafe_allow_html=True)



        # -------------------------------------------------
        # FEATURE 3: EXPLAINABLE AI / GRAD-CAM
        # -------------------------------------------------
        xai_metrics = calculate_xai_metrics(results.get("gradcam_heatmap"))
        mean_activation = xai_metrics["mean_activation"]
        high_activation = xai_metrics["high_activation_pct"]

        st.markdown(
            f"""
            <div class="xai-panel">
                <div class="xai-head">
                    <div>
                        <div class="xai-title">Explainable AI — Grad-CAM</div>
                        <div class="xai-subtitle">Visual explanation of image regions that influenced the classifier output</div>
                    </div>
                    <div class="xai-badge">MODEL FOCUS</div>
                </div>
                <div class="xai-grid">
                    <div class="xai-info">
                        <div class="xai-label">Mean activation</div>
                        <div class="xai-metric">{mean_activation:.1f}%</div>
                        <div class="xai-copy">
                            Grad-CAM highlights regions receiving stronger activation from the
                            classification network for the predicted class. Warmer colors indicate
                            stronger relative activation in the displayed heatmap.
                        </div>
                        <div class="xai-legend"><div class="xai-gradient"></div></div>
                        <div class="xai-legend-text"><span>Lower focus</span><span>Higher focus</span></div>
                    </div>
                    <div class="xai-info">
                        <div class="xai-label">High-activation region</div>
                        <div class="xai-metric">{high_activation:.1f}%</div>
                        <div class="xai-copy">
                            Percentage of heatmap pixels above the relative 0.70 activation level.
                            This describes model attention, not tumor size or tumor volume.
                        </div>
                    </div>
                </div>
                <div class="xai-note">
                    <b>Important:</b> Grad-CAM is an explainability aid. It shows where the
                    classifier was responding, but it does not prove that the highlighted region
                    is a tumor boundary and should not be used as a clinical diagnosis.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # -------------------------------------------------
        # PROBABILITY DISTRIBUTION + MODEL SUMMARY
        # -------------------------------------------------
        col_prob, col_summary = st.columns(
            [3, 1],
            gap="large",
        )

        with col_prob:
            st.markdown(
                '<p class="section-label">Probability Distribution</p>',
                unsafe_allow_html=True,
            )

            probabilities = results.get("probabilities", {})

            if probabilities:
                fig = plot_confidence_bar(probabilities)
                st.plotly_chart(
                    fig,
                    width="stretch",
                )
            else:
                st.info("Probability data is not available.")

        with col_summary:
            st.markdown(
                '<p class="section-label">Model Summary</p>',
                unsafe_allow_html=True,
            )

            summary = get_analysis_summary(pred_class)

            st.html(
                f"""
                <div class="glass-panel summary-card" style="min-height:380px;">
                    <div class="summary-title">Analysis Summary</div>

                    <div class="summary-row">
                        <div>
                            <div class="summary-field-label">MORPHOLOGY</div>
                            <div class="summary-field-value">{summary["morphology"]}</div>
                        </div>
                        <span class="material-symbols-outlined" style="color:#28d9f3;">info</span>
                    </div>

                    <div class="summary-row">
                        <div>
                            <div class="summary-field-label">LOCALIZATION</div>
                            <div class="summary-field-value">{summary["localization"]}</div>
                        </div>
                        <span class="material-symbols-outlined" style="color:#28d9f3;">info</span>
                    </div>

                    <div class="summary-row" style="border-bottom:none;">
                        <div>
                            <div class="summary-field-label">VASCULARITY</div>
                            <div class="summary-field-value">{summary["vascularity"]}</div>
                        </div>
                        <span class="material-symbols-outlined" style="color:#8d98a8;">info</span>
                    </div>
                </div>
                """
            )

        # -------------------------------------------------
        # ANALYSIS BRIDGE — fills the visual gap between probability and segmentation
        # -------------------------------------------------
        st.markdown('<div class="section-label">MRI Analysis Bridge</div>', unsafe_allow_html=True)
        bridge_left, bridge_mid, bridge_right = st.columns([1.25, 1.0, 1.25], gap="large")

        with bridge_left:
            st.image(raw_rgb, caption="Uploaded Brain MRI", width="stretch")

        with bridge_mid:
            bridge_area = calculate_tumor_area(results.get("mask"))
            st.markdown(
                f"""
                <div class="glass-panel" style="min-height:235px;padding:22px;text-align:center;display:flex;flex-direction:column;justify-content:center;">
                    <div style="font-size:12px;letter-spacing:.12em;color:#7b8897;font-weight:700;">ANALYSIS FLOW</div>
                    <div style="font-size:27px;color:#0b7285;font-weight:800;margin:12px 0 4px;">{pred_class.upper()}</div>
                    <div style="font-size:13px;color:#617083;">Classifier → Segmentation</div>
                    <div style="margin-top:16px;padding:10px 12px;border-radius:12px;background:#eaf8fc;color:#0b7285;font-weight:700;">
                        Segmented area: {bridge_area['area_percentage']:.2f}%
                    </div>
                    <div style="font-size:10px;color:#8b98a7;margin-top:10px;line-height:1.5;">The segmentation mask is an independent Attention U-Net output.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with bridge_right:
            st.image(overlay_resized, caption="Tumor Segmentation Overlay", width="stretch")

        st.markdown('<div style="height:18px;"></div>', unsafe_allow_html=True)


        # -------------------------------------------------
        # FEATURE 1: TUMOR AREA ANALYSIS
        # -------------------------------------------------
        tumor_area = calculate_tumor_area(
            results.get("mask")
        )

        tumor_pixels = tumor_area["tumor_pixels"]
        total_pixels = tumor_area["total_pixels"]
        tumor_percentage = tumor_area["area_percentage"]

        st.markdown(
            "<br><br>",
            unsafe_allow_html=True,
        )

        st.markdown(
            '<p class="section-label">'
            'Tumor Segmentation Analysis'
            '</p>',
            unsafe_allow_html=True,
        )

        area_col1, area_col2, area_col3 = st.columns(
            [1.1, 1.1, 2],
            gap="large",
        )

        with area_col1:
            st.markdown(
                f"""
                <div class="tumor-area-card">
                    <div class="tumor-area-title">
                        Tumor Pixels
                    </div>
                    <div class="tumor-area-value">
                        {tumor_pixels:,}
                    </div>
                    <div class="tumor-area-subtitle">
                        Segmented pixels
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with area_col2:
            st.markdown(
                f"""
                <div class="tumor-area-card">
                    <div class="tumor-area-title">
                        Tumor Area
                    </div>
                    <div class="tumor-area-value">
                        {tumor_percentage:.2f}%
                    </div>
                    <div class="tumor-area-subtitle">
                        Of segmentation frame
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with area_col3:
            bar_width = min(
                max(tumor_percentage, 0.0),
                100.0,
            )

            st.html(
                f"""
                <div class="tumor-area-card">
                    <div class="tumor-area-title">
                        Segmentation Coverage
                    </div>
                    <div class="tumor-area-subtitle">
                        Tumor pixels / total mask pixels
                    </div>
                    <div class="tumor-area-bar">
                        <div class="tumor-area-fill" style="width:{bar_width:.2f}%;"></div>
                    </div>
                    <div style="margin-top:10px;font-size:11px;color:rgba(195,198,214,0.55);">
                        {tumor_pixels:,} / {total_pixels:,} pixels
                    </div>
                </div>
                """
            )

        st.caption(
            "Tumor area is calculated from the predicted segmentation mask. "
            "It represents pixel-area proportion only and is not a physical "
            "measurement in cm² or tumor volume."
        )

        if tumor_pixels == 0:
            st.info(
                "ℹ️ No positive segmentation pixels were detected. "
                "The ROI therefore defaults to the full frame."
            )

        # -------------------------------------------------
        # REPORT GENERATION
        # -------------------------------------------------
        st.markdown('<div class="section-label">Final Analysis Report</div>', unsafe_allow_html=True)
        report_button_col1, report_button_col2, report_button_col3 = st.columns([1, 2, 1])
        with report_button_col2:
            generate_report = st.button(
                "📄  Generate NeuroScan MRI Report",
                width="stretch",
                type="primary",
                key="generate_neuroscan_report",
            )

        if generate_report:
            report_probabilities = normalize_probability_distribution(results.get("probabilities", {}))
            current_uncertainty = calculate_uncertainty_metrics(
                report_probabilities,
                float(results.get("confidence", 0.0)),
            )
            current_xai = calculate_xai_metrics(results.get("gradcam_heatmap"))
            generated_pdf = build_pdf_report(
                st.session_state.patient_profile,
                pred_class,
                confidence_pct,
                threshold_pct,
                below_threshold,
                summary,
                latency,
                tumor_area,
                current_uncertainty,
                current_xai,
                probabilities=report_probabilities,
                original_image=raw_rgb,
                segmentation_image=overlay_resized,
                gradcam_image=gradcam_resized if results.get("gradcam_overlay") is not None else None,
                roi_image=roi_resized,
            )
            st.session_state["neuroscan_generated_pdf"] = generated_pdf
            st.session_state["neuroscan_generated_pdf_name"] = f"neuroscan_mri_report_{pred_class.lower()}.pdf"
            if generated_pdf is None:
                st.error("PDF तयार झाला नाही. तुमच्या project venv मध्ये ReportLab install करा: python -m pip install reportlab")

        if st.session_state.get("neuroscan_generated_pdf"):
            st.success("Report तयार आहे. खाली Download PDF Report वर क्लिक करा.")
            st.download_button(
                "⬇️  Download PDF Report",
                data=st.session_state["neuroscan_generated_pdf"],
                file_name=st.session_state.get("neuroscan_generated_pdf_name", "neuroscan_mri_report.pdf"),
                mime="application/pdf",
                width="stretch",
                key="download_generated_neuroscan_report",
            )

        add_scan_to_history(getattr(uploaded_file, "name", "Uploaded MRI"), pred_class, confidence_pct, tumor_percentage, latency)

        # FEATURE 8: SIDE-BY-SIDE ANALYSIS
        st.markdown('<div class="section-label">Side-by-Side MRI Analysis</div>', unsafe_allow_html=True)
        bbox_overlay, bbox_info, centroid_info = create_bbox_centroid_overlay(raw_rgb, results.get("mask"))
        compare_cols = st.columns(4, gap="medium")
        side_images = [
            (raw_rgb, "Original MRI"),
            (overlay_resized, "Segmentation Overlay"),
            (gradcam_resized if results.get("gradcam_overlay") is not None else roi_resized, "Grad-CAM Focus"),
            (bbox_overlay, "Region Box + Centroid"),
        ]
        for col, (im, cap) in zip(compare_cols, side_images):
            with col:
                st.image(im, caption=cap, width="stretch")

        # FEATURE 9 + 10: BOUNDING BOX AND CENTROID
        st.markdown('<div class="section-label">Tumor Region Geometry</div>', unsafe_allow_html=True)
        g1,g2,g3 = st.columns(3, gap="large")
        if bbox_info:
            with g1: st.markdown(f'<div class="metric-card"><div class="metric-label">Bounding Box Width</div><div class="metric-value">{bbox_info["width"]} px</div><div class="metric-note">Approximate segmented region</div></div>',unsafe_allow_html=True)
            with g2: st.markdown(f'<div class="metric-card"><div class="metric-label">Bounding Box Height</div><div class="metric-value">{bbox_info["height"]} px</div><div class="metric-note">Approximate segmented region</div></div>',unsafe_allow_html=True)
            with g3: st.markdown(f'<div class="metric-card"><div class="metric-label">Centroid</div><div class="metric-value" style="font-size:22px;">({centroid_info["x"]:.0f}, {centroid_info["y"]:.0f})</div><div class="metric-note">Image-relative coordinates</div></div>',unsafe_allow_html=True)
            st.caption("Bounding-box and centroid values are image-relative geometry from the predicted segmentation mask. They are not anatomical coordinates or physical tumor measurements.")
        else:
            st.info("No positive segmentation region was detected, so bounding-box and centroid values are unavailable.")

        # FEATURE 11: CLASS PROBABILITY COMPARISON
        st.markdown('<div class="section-label">Class Probability Comparison</div>', unsafe_allow_html=True)
        probs = normalize_probability_distribution(results.get("probabilities", {}))
        probability_colors = {
            "Glioma": ("#0b7285", "#e6faff"),
            "Meningioma": ("#6f42c1", "#f3edff"),
            "Pituitary": ("#157347", "#eaf8ef"),
            "No Tumor": ("#b54708", "#fff4e5"),
        }
        prob_cols = st.columns(4, gap="medium")
        for col, cls_name in zip(prob_cols, ["Glioma", "Meningioma", "Pituitary", "No Tumor"]):
            prob = float(probs.get(cls_name, 0.0)) * 100.0
            text_color, bg_color = probability_colors[cls_name]
            with col:
                st.markdown(
                    f"""
                    <div style="
                        background:{bg_color};
                        border:1px solid rgba(30,55,80,.10);
                        border-radius:16px;
                        padding:18px 16px;
                        min-height:118px;
                        box-shadow:0 5px 18px rgba(31,55,86,.05);
                    ">
                        <div style="color:#526174;font-size:12px;font-weight:700;margin-bottom:8px;">
                            {cls_name}
                        </div>
                        <div style="color:{text_color};font-size:30px;font-weight:800;line-height:1.1;">
                            {prob:.1f}%
                        </div>
                        <div style="height:7px;background:#ffffff;border-radius:99px;margin-top:13px;overflow:hidden;">
                            <div style="height:100%;width:{min(max(prob,0),100):.1f}%;background:{text_color};border-radius:99px;"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.caption("These are the classifier's output probabilities for the four project classes; they are not calibrated clinical probabilities.")

        # FEATURE 14: PATIENT REPORT HISTORY
        st.markdown('<div class="section-label">Patient Report History</div>', unsafe_allow_html=True)
        patient_history = [x for x in st.session_state.get("scan_history", []) if x.get("patient") == st.session_state.patient_profile.get("name")]
        if patient_history:
            tumor_messages = {
                "GLIOMA": "A Glioma classification was detected. Please consult a qualified doctor/radiologist for clinical evaluation and confirmation.",
                "MENINGIOMA": "A Meningioma classification was detected. Please consult a qualified doctor/radiologist for clinical evaluation and confirmation.",
                "PITUITARY": "A Pituitary tumor classification was detected. Please consult a qualified doctor/radiologist for clinical evaluation and confirmation.",
                "NOTUMOR": "No tumor classification was detected by the model. No tumor-specific information is shown for this result.",
                "NO TUMOR": "No tumor classification was detected by the model. No tumor-specific information is shown for this result.",
            }
            for item in patient_history[:8]:
                cls_name = str(item.get("class", "Unknown"))
                advice = tumor_messages.get(
                    cls_name.upper(),
                    "The scan result is model-generated. Please consult a qualified doctor/radiologist for clinical interpretation.",
                )
                patient_name = item.get("patient", "Current patient")
                age = item.get("age", "—")
                html = (
                    '<div class="history-item">'
                    f'<div class="history-class">{cls_name}'
                    f'<span style="float:right;color:#10abc5;">{item["confidence"]:.1f}%</span></div>'
                    f'<div class="history-meta"><b>Patient:</b> {patient_name} • '
                    f'<b>Age:</b> {age} • <b>Date & Time:</b> {item["time"]}<br/>'
                    f'<b>Report:</b> {item["file"]} • Tumor area {item["tumor_area"]:.2f}% • '
                    f'Processing {item["latency"]:.0f} ms</div>'
                    '<div style="margin-top:10px;padding:11px 13px;border-radius:10px;'
                    'background:#f5fbfd;border-left:4px solid #10abc5;color:#405466;'
                    'font-size:12px;line-height:1.5;">'
                    f'<b>Information:</b> {advice}</div></div>'
                )
                st.markdown(html, unsafe_allow_html=True)
        else:
            st.info("No previous report exists for this patient in the current session.")

        # Update history once per analyzed scan
        add_scan_to_history(getattr(uploaded_file, "name", "Uploaded MRI"), pred_class, confidence_pct, tumor_percentage, latency)



# =====================================================
# HISTORY PAGE
# =====================================================
elif page == "🕒 History":
    st.markdown('<div class="hero-title">Analysis History</div>', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-subtitle">Session-only scan history for the current patient. Patient details remain in this active session only.</div>', unsafe_allow_html=True)
    history=st.session_state.get("scan_history",[])
    if history:
        if st.button("🗑️ Clear Session History"):
            st.session_state.scan_history=[]
            st.rerun()
        for item in history:
            st.markdown(f'<div class="history-item"><div class="history-class">{item["class"]}<span style="float:right;color:#10abc5;">{item["confidence"]:.1f}%</span></div><div class="history-meta">Patient: {item.get("patient", "Current patient")} • {item["file"]} • {item["time"]}<br/>Tumor area: {item["tumor_area"]:.2f}% • Processing: {item["latency"]:.0f} ms</div></div>',unsafe_allow_html=True)
    else:
        st.info("No scan history yet. Upload and analyze an MRI from Live Scan.")
    st.caption("History and patient details are stored only in the active Streamlit session.")


# =====================================================
# PATIENT RECORDS PAGE
# =====================================================
elif page == "📂 Patient Records":

    st.markdown(
        '<div class="hero-title">Patient Records</div>',
        unsafe_allow_html=True,
    )

    patient = st.session_state.get("patient_profile")
    if patient:
        st.markdown(
            f"""
            <div class="feature-card">
                <div class="feature-card-title">Current Patient</div>
                <div class="feature-card-subtitle">Details entered at the start of this analysis session</div>
                <br>
                <b>Name:</b> {patient['name']}<br><br>
                <b>Age:</b> {patient['age']} years<br><br>
                <b>Village / City:</b> {patient['location']}<br><br>
                <b>District / State:</b> {patient['district_state'] or 'Not provided'}
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.info(
        "Persistent patient-record storage is not connected. "
        "Patient details are kept only for the active Streamlit session and generated reports."
    )


# =====================================================
# SETTINGS PAGE
# =====================================================
elif page == "⚙️ Settings":

    st.markdown(
        '<div class="hero-title">System Settings</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="glass-panel">
            <div class="summary-title">Current Configuration</div>
            <br>
            <b>Segmentation model:</b> Attention U-Net<br><br>
            <b>Classification model:</b> EfficientNetV2B0<br><br>
            <b>Classification confidence threshold:</b> 70%<br><br>
            <b>Explainability:</b> Grad-CAM<br><br>
            <b>Tumor area:</b> Segmentation pixel-area proportion<br><br>
            <b>Confidence & uncertainty:</b> Probability-based indicators<br><br>
            <b>Analysis report:</b> TXT + PDF export<br><br>
            <b>Model performance:</b> Verified test-set metrics<br><br>
            <b>Patient intake:</b> Name, age, village/city and optional district/state<br><br>
            <b>History:</b> Session-only patient-linked scan history
        </div>
        """,
        unsafe_allow_html=True,
    )
