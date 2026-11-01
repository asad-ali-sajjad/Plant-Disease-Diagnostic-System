"""
app.py — Main Streamlit application for FloraScan AI.
"""
import io
from html import escape
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import streamlit as st

# config.py imports
from config import (
    APP_NAME, APP_TAGLINE, CROP_EMOJI,
    CLASS_NAMES, NUM_CLASSES, CONFIDENCE_THRESHOLD, INPUT_SIZE,
    PATH_1, PATH_2,
    ENGINE_1_LABEL, ENGINE_2_LABEL, DUAL_LABEL, ENGINE_1_SHORT, ENGINE_2_SHORT,
    UNKNOWN_TONE, get_guide, get_meta,
    SAMPLE_IMAGES, STUDENT_NAME, STUDENT_ID, CLASS_TERM, COURSE_INSTRUCTOR,
)

# ---------------------------------------------------------
# Page Setup (must be the first Streamlit command)
# ---------------------------------------------------------
st.set_page_config(
    page_title=f"{APP_NAME} — Smart Phytopathology System",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

from model import load_all_models, model_stamp, predict_cached
from ui import (
    render, inject_css, init_state, log_scan,
    verdict_card, probability_bars, build_report,
)

inject_css()
init_state()

# ---------------------------------------------------------
# Selection state (engine + input source)
# ---------------------------------------------------------
UPLOAD_KEY = "__upload__"

ENGINE_OPTIONS = [
    (ENGINE_1_LABEL, "🔹", f"{ENGINE_1_SHORT}: lightweight and fast"),
    (ENGINE_2_LABEL, "🔸", f"{ENGINE_2_SHORT}: deeper transfer-learning model"),
    (DUAL_LABEL, "⚖️", "Run both engines and compare them side by side"),
]

if "engine_choice" not in st.session_state:
    st.session_state.engine_choice = ENGINE_1_LABEL
if "source_choice" not in st.session_state:
    st.session_state.source_choice = UPLOAD_KEY


def set_engine(label):
    st.session_state.engine_choice = label


def set_source(key):
    st.session_state.source_choice = key


# ---------------------------------------------------------
# Modern UI styling: Sidebar, Engine Picker, Sample Gallery, Footer
# (one shared emerald palette so everything feels consistent)
# ---------------------------------------------------------
st.markdown("""
<style>
:root {
    --fs-emerald: #10b981;
    --fs-emerald-dark: #059669;
    --fs-emerald-soft: #ecfdf5;
    --fs-emerald-line: #a7f3d0;
    --fs-ink: #0f172a;
    --fs-text: #334155;
    --fs-muted: #64748b;
    --fs-border: #e2e8f0;
    --fs-card: #ffffff;
}

/* =========================================================
   SIDEBAR — light, clean, card based
   ========================================================= */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f0fdf4 0%, #ecfeff 55%, #f8fafc 100%) !important;
    border-right: 1px solid var(--fs-emerald-line) !important;
}
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] summary p,
[data-testid="stSidebar"] summary span {
    color: var(--fs-text) !important;
}
[data-testid="stSidebar"] h3 {
    color: var(--fs-ink) !important;
    font-weight: 800 !important;
    letter-spacing: -0.01em;
    padding: 6px 0 2px;
}
[data-testid="stSidebar"] hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, var(--fs-emerald-line), transparent) !important;
    margin: 1rem 0 !important;
}

/* Metric cards */
[data-testid="stSidebar"] [data-testid="stMetric"] {
    background: var(--fs-card);
    border: 1px solid var(--fs-border);
    border-top: 3px solid var(--fs-emerald);
    border-radius: 12px;
    padding: 10px 12px;
    box-shadow: 0 2px 10px rgba(16, 185, 129, 0.08);
    transition: transform .15s ease, box-shadow .15s ease;
}
[data-testid="stSidebar"] [data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 18px rgba(16, 185, 129, 0.18);
}
[data-testid="stSidebar"] [data-testid="stMetricLabel"] p {
    color: var(--fs-muted) !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
[data-testid="stSidebar"] [data-testid="stMetricValue"] {
    color: var(--fs-ink) !important;
    font-weight: 800 !important;
}

/* Expanders */
[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: var(--fs-card) !important;
    border: 1px solid var(--fs-border) !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    overflow: hidden;
    margin-bottom: 10px;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    font-weight: 700 !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {
    background: var(--fs-emerald-soft) !important;
}

/* =========================================================
   ENGINE PICKER — big tab-style buttons in the sidebar
   ========================================================= */
.fs-side-label {
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--fs-emerald-dark);
    margin: 4px 0 8px;
}
[class*="st-key-fs_engine_"],
[class*="st-key-fs_engine_"] [data-testid="stButton"],
[class*="st-key-fs_engine_"] button {
    width: 100% !important;
}
[class*="st-key-fs_engine_"] button {
    min-height: 56px !important;
    padding: 12px 16px !important;
    border-radius: 14px !important;
    border: 1.5px solid var(--fs-border) !important;
    background: var(--fs-card) !important;
    color: var(--fs-text) !important;
    justify-content: flex-start !important;
    text-align: left !important;
    box-shadow: 0 2px 6px rgba(15, 23, 42, 0.05) !important;
    transition: all .18s ease !important;
}
[class*="st-key-fs_engine_"] button p {
    color: inherit !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    margin: 0 !important;
}
[class*="st-key-fs_engine_"] button:hover {
    border-color: var(--fs-emerald) !important;
    background: var(--fs-emerald-soft) !important;
    color: var(--fs-emerald-dark) !important;
    transform: translateY(-2px);
    box-shadow: 0 8px 16px rgba(16, 185, 129, 0.18) !important;
}
[class*="st-key-fs_engine_"] button[kind="primary"],
[class*="st-key-fs_engine_"] button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #10b981, #059669) !important;
    border-color: #059669 !important;
    color: #ffffff !important;
    box-shadow: 0 8px 18px rgba(5, 150, 105, 0.35) !important;
}
[class*="st-key-fs_engine_"] button[kind="primary"] p,
[class*="st-key-fs_engine_"] button[data-testid="stBaseButton-primary"] p {
    color: #ffffff !important;
    font-weight: 800 !important;
}

/* =========================================================
   SAMPLE GALLERY — thumbnail cards
   ========================================================= */
.source-hint {
    font-size: 0.85rem;
    color: var(--fs-muted);
    margin: -2px 0 12px;
    line-height: 1.5;
}
.source-hint b { color: var(--fs-emerald-dark); }

[class*="st-key-fs_card_"] {
    background: var(--fs-card);
    border: 1.5px solid var(--fs-border);
    border-radius: 18px;
    padding: 10px;
    gap: 0.5rem;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
    transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease, background .18s ease;
}
[class*="st-key-fs_card_"]:hover {
    transform: translateY(-3px);
    border-color: var(--fs-emerald);
    box-shadow: 0 12px 24px rgba(16, 185, 129, 0.18);
}
/* selected card = the one whose button is primary */
[class*="st-key-fs_card_"]:has(button[kind="primary"]) {
    border-color: var(--fs-emerald-dark);
    background: linear-gradient(160deg, #ecfdf5, #ffffff);
    box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.25), 0 12px 26px rgba(16, 185, 129, 0.22);
}
[class*="st-key-fs_card_"]:has([data-testid="stBaseButton-primary"]) {
    border-color: var(--fs-emerald-dark);
    background: linear-gradient(160deg, #ecfdf5, #ffffff);
    box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.25), 0 12px 26px rgba(16, 185, 129, 0.22);
}
[class*="st-key-fs_card_"] img {
    border-radius: 12px !important;
}
.fs-card-title {
    font-weight: 800;
    font-size: 0.92rem;
    color: var(--fs-ink);
    line-height: 1.25;
}
.fs-card-sub {
    font-size: 0.76rem;
    color: var(--fs-muted);
    margin-top: 2px;
}
.fs-thumb-missing {
    height: 120px;
    border-radius: 12px;
    border: 1.5px dashed var(--fs-emerald-line);
    background: var(--fs-emerald-soft);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    color: var(--fs-muted);
    font-size: 0.78rem;
    gap: 4px;
}
.fs-upload-body {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 4px 4px 0;
}
.fs-upload-icon {
    width: 54px;
    height: 54px;
    flex: 0 0 54px;
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.7rem;
    background: linear-gradient(135deg, #d1fae5, #cffafe);
    border: 1.5px dashed var(--fs-emerald);
}

/* card buttons */
[class*="st-key-fs_card_"] [data-testid="stButton"],
[class*="st-key-fs_card_"] button {
    width: 100% !important;
}
[class*="st-key-fs_card_"] button {
    border-radius: 10px !important;
    border: 1.5px solid var(--fs-emerald-line) !important;
    background: var(--fs-emerald-soft) !important;
    color: var(--fs-emerald-dark) !important;
    min-height: 38px !important;
    transition: all .15s ease !important;
}
[class*="st-key-fs_card_"] button p {
    color: inherit !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    margin: 0 !important;
}
[class*="st-key-fs_card_"] button:hover:not(:disabled) {
    background: var(--fs-emerald) !important;
    border-color: var(--fs-emerald) !important;
    color: #ffffff !important;
}
[class*="st-key-fs_card_"] button[kind="primary"],
[class*="st-key-fs_card_"] button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #10b981, #059669) !important;
    border-color: #059669 !important;
    color: #ffffff !important;
}
[class*="st-key-fs_card_"] button:disabled {
    opacity: 0.55 !important;
}

/* =========================================================
   FOOTER — light card, same palette as sidebar & selector
   ========================================================= */
.fs-footer {
    position: relative;
    margin-top: 2.5rem;
    padding: 28px 26px 18px;
    border-radius: 20px;
    background: linear-gradient(135deg, #f0fdf4 0%, #ffffff 55%, #ecfeff 100%);
    border: 1px solid var(--fs-emerald-line);
    box-shadow: 0 10px 30px rgba(16, 185, 129, 0.12);
    overflow: hidden;
}
.fs-footer::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 5px;
    background: linear-gradient(90deg, #10b981, #06b6d4, #10b981);
}
.fs-footer-grid {
    display: grid;
    grid-template-columns: 1fr 1.3fr;
    gap: 28px;
    margin-bottom: 22px;
}
.fs-brand-title {
    font-size: 1.35rem;
    font-weight: 800;
    color: var(--fs-ink);
    letter-spacing: -0.01em;
    margin-bottom: 6px;
}
.fs-brand-desc {
    margin: 0 0 16px;
    font-size: 0.9rem;
    color: var(--fs-muted);
    line-height: 1.55;
}
.fs-disclaimer {
    display: inline-block;
    background: #fffbeb;
    border: 1px solid #fde68a;
    color: #92400e;
    padding: 9px 14px;
    border-radius: 12px;
    font-size: 0.82rem;
    line-height: 1.45;
}
.fs-meta {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
}
.fs-meta-card {
    background: var(--fs-card);
    border: 1px solid var(--fs-border);
    border-left: 4px solid var(--fs-emerald);
    border-radius: 12px;
    padding: 11px 14px;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease;
}
.fs-meta-card:hover {
    transform: translateY(-3px);
    border-color: var(--fs-emerald);
    box-shadow: 0 10px 20px rgba(16, 185, 129, 0.18);
}
.fs-meta-label {
    display: block;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 800;
    color: var(--fs-emerald-dark);
    margin-bottom: 3px;
}
.fs-meta-val {
    display: block;
    font-weight: 700;
    color: var(--fs-ink);
    font-size: 0.92rem;
    line-height: 1.3;
}
.fs-footer-bottom {
    border-top: 1px solid var(--fs-emerald-line);
    padding-top: 14px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 10px;
    font-size: 0.82rem;
    color: var(--fs-muted);
}
.fs-top-pill {
    background: var(--fs-emerald-soft);
    color: var(--fs-emerald-dark);
    border: 1px solid var(--fs-emerald-line);
    padding: 4px 12px;
    border-radius: 999px;
    font-weight: 700;
}
@media (max-width: 768px) {
    .fs-footer-grid { grid-template-columns: 1fr; gap: 20px; }
    .fs-meta { grid-template-columns: 1fr; }
}
</style>
""", unsafe_allow_html=True)


def show_image(img):
    """Full-width image renderer for both new and older Streamlit versions."""
    try:
        st.image(img, width="stretch")
    except Exception:
        try:
            st.image(img, use_container_width=True)
        except Exception:
            st.image(img, use_column_width=True)


def clear_history():
    st.session_state.history = []


@st.cache_data(show_spinner=False)
def load_thumbnail(path_str, mtime):
    """Square-ish cropped thumbnail for the sample gallery (cached)."""
    img = Image.open(path_str).convert("RGB")
    return ImageOps.fit(img, (360, 270), Image.LANCZOS)


def sample_card(idx, name, path):
    """One gallery card: thumbnail, title and a select button."""
    path = Path(path)
    exists = path.exists()
    selected = st.session_state.source_choice == name

    with st.container(key=f"fs_card_{idx}"):
        if exists:
            try:
                show_image(load_thumbnail(str(path), path.stat().st_mtime))
            except Exception:
                render('<div class="fs-thumb-missing"><span>🖼️</span><span>Preview unavailable</span></div>')
        else:
            render('<div class="fs-thumb-missing"><span>🖼️</span><span>Image not found</span></div>')

        render(f'<div class="fs-card-title">{escape(name)}</div>')
        st.button(
            "✓ Selected" if selected else "Select",
            key=f"fs_pick_{idx}",
            type="primary" if selected else "secondary",
            disabled=not exists,
            on_click=set_source,
            args=(name,),
        )


# ---------------------------------------------------------
# Sidebar Control Center
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎛️ Control Center")
    render('<div class="fs-side-label">Inference engine</div>')

    for i, (label, icon, tip) in enumerate(ENGINE_OPTIONS):
        active = st.session_state.engine_choice == label
        st.button(
            f"{icon}  {label}" + ("   ✓" if active else ""),
            key=f"fs_engine_{i}",
            type="primary" if active else "secondary",
            help=tip,
            on_click=set_engine,
            args=(label,),
        )

    selected_engine = st.session_state.engine_choice

    st.markdown("---")
    st.markdown("### 📡 System Status")
    c1, c2 = st.columns(2)
    c1.metric("Threshold", f"{CONFIDENCE_THRESHOLD:.0f}%", "Active")
    c2.metric("Compute", "CPU", "Ready")

    with st.expander("🛠️ Engine Specifications"):
        st.write("**Dataset:** Balanced 5-Class Set (4 Tomato + 1 Not_Leaf)")
        st.write(f"**Resolution:** {INPUT_SIZE}x{INPUT_SIZE} RGB")
        st.write(f"**Outputs:** {NUM_CLASSES} Health States")

    with st.expander("📸 Photo Tips", expanded=True):
        st.write("• Use natural daylight")
        st.write("• Fill the frame with one leaf")
        st.write("• Keep the leaf flat and in focus")

# ---------------------------------------------------------
# Model Loader Verification
# ---------------------------------------------------------
model_1, model_2, load_errors = load_all_models(model_stamp())
if load_errors or model_1 is None or model_2 is None:
    st.error(
        "⚠️ **System Offline:** the model checkpoints could not be loaded. "
        f"Please place `{PATH_1.name}` and `{PATH_2.name}` next to `app.py`."
    )
    for message in load_errors:
        st.warning(message)
    st.stop()

# ---------------------------------------------------------
# Hero Banner
# ---------------------------------------------------------
render(f"""
<div class="hero">
    <h1>{escape(APP_NAME)}</h1>
    <p>{escape(APP_TAGLINE)}</p>
    <span class="chip">{CROP_EMOJI} {NUM_CLASSES} Health States</span>
    <span class="chip">⚡ Real-time Inference</span>
    <span class="chip">🧠 Dual Engine Validation</span>
    <span class="chip">🛡️ {CONFIDENCE_THRESHOLD:.0f}% Safety Threshold</span>
</div>
""")

tab_scan, tab_lib, tab_hist = st.tabs(["🔬  Scan Leaf", "📚  Disease Library", "🕘  Scan History"])

# ---------------------------------------------------------
# Scan Tab
# ---------------------------------------------------------
with tab_scan:
    left, right = st.columns([1, 1.35], gap="large")

    sample = None   # {"name": str, "bytes": bytes}
    image = None

    with left:
        render('<div class="section-title">📥 Select Input Source</div>')
        render('<div class="source-hint">Upload your own photo, or tap a <b>test leaf</b> below. The selected card is highlighted in green.</div>')

        # ---- Upload card (full width) ----
        upload_selected = st.session_state.source_choice == UPLOAD_KEY
        with st.container(key="fs_card_upload"):
            render("""
            <div class="fs-upload-body">
                <div class="fs-upload-icon">📁</div>
                <div>
                    <div class="fs-card-title">Upload your own leaf</div>
                    <div class="fs-card-sub">JPG · PNG · WEBP</div>
                </div>
            </div>
            """)
            st.button(
                "✓ Selected" if upload_selected else "Use upload",
                key="fs_pick_upload",
                type="primary" if upload_selected else "secondary",
                on_click=set_source,
                args=(UPLOAD_KEY,),
            )

        # ---- Sample gallery (2 per row) ----
        render('<div class="fs-side-label" style="margin-top:14px;">Test samples</div>')
        sample_items = list(SAMPLE_IMAGES.items())
        for row_start in range(0, len(sample_items), 2):
            row_cols = st.columns(2, gap="small")
            for offset, col in enumerate(row_cols):
                idx = row_start + offset
                if idx < len(sample_items):
                    with col:
                        sample_card(idx, sample_items[idx][0], sample_items[idx][1])

        # ---- Resolve the chosen source ----
        sample_choice = st.session_state.source_choice
        if sample_choice != UPLOAD_KEY and sample_choice not in SAMPLE_IMAGES:
            sample_choice = UPLOAD_KEY

        if sample_choice == UPLOAD_KEY:
            uploaded_file = st.file_uploader(
                "Drop a leaf sample here for instant analysis",
                type=["jpg", "jpeg", "png", "webp"],
                label_visibility="collapsed",
                key="leaf_uploader",
            )
            if uploaded_file is not None:
                try:
                    sample = {"name": uploaded_file.name, "bytes": uploaded_file.getvalue()}
                    image = Image.open(io.BytesIO(sample["bytes"])).convert("RGB")
                except Exception:
                    sample, image = None, None
                    st.error("😕 We couldn't read that image. Please upload a valid JPG, PNG, or WEBP file.")
        else:
            sample_path = Path(SAMPLE_IMAGES[sample_choice])
            if sample_path.exists():
                try:
                    with open(sample_path, "rb") as f:
                        img_bytes = f.read()
                    sample = {"name": sample_path.name, "bytes": img_bytes}
                    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                except Exception as e:
                    sample, image = None, None
                    st.error(f"😕 Error loading sample image: {e}")
            else:
                st.warning(f"Sample image `{sample_path}` not found in project directory.")

        if image is not None:
            render('<div class="section-title" style="margin-top:14px;">🖼️ Selected Leaf</div>')
            show_image(image)
            render(f'<p class="muted" style="text-align:center;margin-top:8px;">📄 {escape(sample["name"])} · {image.size[0]}×{image.size[1]} px</p>')

    with right:
        if image is None:
            render("""
            <div class="glass">
                <div class="section-title">✨ How it works</div>
                <div class="step"><div class="num">1</div><div><b>Upload or select a leaf photo</b><br><span class="muted">Upload a leaf image or click any test preset on the left.</span></div></div>
                <div class="step"><div class="num">2</div><div><b>AI analyzes the leaf</b><br><span class="muted">Your chosen engine reads color, texture and lesion patterns.</span></div></div>
                <div class="step"><div class="num">3</div><div><b>Get a clear action plan</b><br><span class="muted">See the diagnosis, confidence, and what to do next.</span></div></div>
            </div>
            <div class="glass" style="background:linear-gradient(135deg,#ecfdf5,#e0f2fe);">
                <div class="section-title">🌤️ Waiting for your leaf…</div>
                <p class="muted">Tip: switch to <b>Dual Benchmark Mode</b> in the sidebar to compare both engines side by side.</p>
            </div>
            """)
        else:
            file_name = sample["name"]
            file_id = f"{file_name}-{len(sample['bytes'])}"
            stamp = model_stamp()
            new_scan = False

            if selected_engine != DUAL_LABEL:
                # ---------------- SINGLE ENGINE ----------------
                use_engine_1 = selected_engine == ENGINE_1_LABEL
                engine_label = ENGINE_1_LABEL if use_engine_1 else ENGINE_2_LABEL

                with st.spinner("Analyzing leaf…"):
                    probabilities = predict_cached(sample["bytes"], 1 if use_engine_1 else 2, stamp)

                pred_class = CLASS_NAMES[int(np.argmax(probabilities))]
                confidence = float(np.max(probabilities) * 100)
                is_confident = confidence >= CONFIDENCE_THRESHOLD

                new_scan = log_scan(
                    f"{file_id}-{engine_label}", file_name, engine_label,
                    pred_class if is_confident else "Unrecognized", confidence,
                )

                verdict_card(pred_class, confidence)

                if not is_confident:
                    with st.expander("📊 See raw probability distribution"):
                        probability_bars(probabilities)
                else:
                    guide = get_guide(pred_class)
                    t_diag, t_metrics = st.tabs(["🩺 Diagnosis & Action Plan", "📊 Confidence Metrics"])

                    with t_diag:
                        if pred_class == "Non Leaf":
                            st.warning(f"**Notice:** {guide['description']}")
                            st.info(f"**Recommendation:** {guide['action']}")
                        else:
                            st.info(f"**Description:** {guide['description']}")
                            st.success(f"**Action Plan:** {guide['action']}")

                        st.download_button(
                            "⬇️ Download diagnosis report",
                            data=build_report(file_name, engine_label, pred_class, confidence),
                            file_name=f"florascan_report_{pred_class.replace(' ', '_').lower()}.txt",
                            mime="text/plain",
                            key=f"dl-{file_id}-{engine_label}",
                        )

                    with t_metrics:
                        render('<div class="section-title">Probability Distribution</div>')
                        probability_bars(probabilities)
            else:
                # ---------------- DUAL BENCHMARK ----------------
                with st.spinner("Running both engines…"):
                    prob_1 = predict_cached(sample["bytes"], 1, stamp)
                    prob_2 = predict_cached(sample["bytes"], 2, stamp)

                conf_1 = float(np.max(prob_1) * 100)
                conf_2 = float(np.max(prob_2) * 100)
                cls_1 = CLASS_NAMES[int(np.argmax(prob_1))]
                cls_2 = CLASS_NAMES[int(np.argmax(prob_2))]

                n1 = log_scan(f"{file_id}-engine1", file_name, ENGINE_1_SHORT,
                              cls_1 if conf_1 >= CONFIDENCE_THRESHOLD else "Unrecognized", conf_1)
                n2 = log_scan(f"{file_id}-engine2", file_name, ENGINE_2_SHORT,
                              cls_2 if conf_2 >= CONFIDENCE_THRESHOLD else "Unrecognized", conf_2)
                new_scan = n1 or n2

                both_sure = conf_1 >= CONFIDENCE_THRESHOLD and conf_2 >= CONFIDENCE_THRESHOLD
                if both_sure and cls_1 == cls_2:
                    if cls_1 == "Non Leaf":
                        banner_bg, banner_color = "#f1f5f9", "#334155"
                        banner_text = "📷 Both engines agree: <b>Non-leaf object detected</b>"
                    else:
                        banner_bg, banner_color = "#dcfce7", "#14532d"
                        banner_text = f"🤝 Both engines agree: <b>{escape(cls_1)}</b>"
                elif both_sure:
                    banner_bg, banner_color = "#fef2f2", "#7f1d1d"
                    banner_text = "⚖️ Engines disagree. Please review the sample manually."
                else:
                    banner_bg, banner_color = "#fffbeb", "#92400e"
                    banner_text = "🔎 At least one engine has low confidence. Try a clearer photo."

                render(f"""
                <div class="glass" style="background:{banner_bg}; color:{banner_color}; font-weight:600; padding:1rem 1.3rem;">
                    {banner_text}
                </div>
                """)

                col_a, col_b = st.columns(2, gap="medium")
                with col_a:
                    render(f'<div class="section-title">🔹 {escape(ENGINE_1_SHORT)} Engine</div>')
                    verdict_card(cls_1, conf_1, compact=True)
                    with st.expander(f"View {ENGINE_1_SHORT} Probabilities"):
                        probability_bars(prob_1)
                with col_b:
                    render(f'<div class="section-title">🔸 {escape(ENGINE_2_SHORT)} Engine</div>')
                    verdict_card(cls_2, conf_2, compact=True)
                    with st.expander(f"View {ENGINE_2_SHORT} Probabilities"):
                        probability_bars(prob_2)

            if new_scan:
                st.toast("Analysis complete — saved to history", icon="✅")

# ---------------------------------------------------------
# Disease Library Tab
# ---------------------------------------------------------
with tab_lib:
    render("""
    <div class="glass">
        <div class="section-title">📚 Know your leaf conditions</div>
        <p class="muted">The health states FloraScan can recognize, with telltale signs and recommended care.</p>
    </div>
    """)
    lib_cols = st.columns(2, gap="large")
    for idx, name in enumerate(CLASS_NAMES):
        meta = get_meta(name)
        guide = get_guide(name)
        signs_html = "".join(f'<span class="sig">{escape(s)}</span>' for s in meta["signs"])
        with lib_cols[idx % 2]:
            render(f"""
            <div class="lib" style="border-top:5px solid {meta['tone']};">
                <div class="icon">{meta['icon']}</div>
                <h3>{escape(name)}</h3>
                <span class="tag" style="background:{meta['tone']}22; color:{meta['tone']};">{escape(meta['kind']).upper()} · SEVERITY: {escape(meta['severity']).upper()}</span>
                <p class="muted" style="margin-top:8px;">{meta['desc_html']}</p>
                <div>{signs_html}</div>
                <div class="act"><b>Recommended action:</b> {escape(guide['action'])}</div>
            </div>
            """)

# ---------------------------------------------------------
# Scan History Tab
# ---------------------------------------------------------
with tab_hist:
    top_l, top_r = st.columns([4, 1])
    with top_l:
        render('<div class="section-title">🕘 Scans this session</div>')
    with top_r:
        if st.session_state.history:
            st.button("🧹 Clear history", key="clear_history", on_click=clear_history)

    if not st.session_state.history:
        render("""
        <div class="glass" style="text-align:center;">
            <div style="font-size:2.5rem;">🍃</div>
            <p class="muted">No scans yet. Upload a leaf in the Scan tab and your results will appear here.</p>
        </div>
        """)
    else:
        for item in reversed(st.session_state.history):
            if item["label"] in CLASS_NAMES:
                tone = get_meta(item["label"])["tone"]
            else:
                tone = UNKNOWN_TONE
            render(f"""
            <div class="hist">
                <div class="dot" style="background:{tone};"></div>
                <div style="flex:1;">
                    <b style="color:#0f172a;">{escape(item['label'])}</b><br>
                    <span class="muted">{escape(item['file'])} · {escape(item['engine'])}</span>
                </div>
                <div style="text-align:right;">
                    <b style="color:{tone};">{item['conf']:.1f}%</b><br>
                    <span class="muted">{item['time']}</span>
                </div>
            </div>
            """)

# ---------------------------------------------------------
# Academic Context & Footer
# ---------------------------------------------------------
render(f"""
<div class="fs-footer">
<div class="fs-footer-grid">
<div>
<div class="fs-brand-title">🌿 {escape(APP_NAME)}</div>
<p class="fs-brand-desc">Dual-engine computer vision system for real-time tomato leaf disease diagnosis.</p>
<div class="fs-disclaimer">⚠️ <strong>AI Guidance Only</strong> · Confirm critical crop decisions with a qualified agronomist.</div>
</div>
<div class="fs-meta">
<div class="fs-meta-card"><span class="fs-meta-label">Author / Student</span><span class="fs-meta-val">{escape(STUDENT_NAME)}</span></div>
<div class="fs-meta-card"><span class="fs-meta-label">Student ID</span><span class="fs-meta-val">{escape(STUDENT_ID)}</span></div>
<div class="fs-meta-card"><span class="fs-meta-label">Class &amp; Term</span><span class="fs-meta-val">{escape(CLASS_TERM)}</span></div>
<div class="fs-meta-card"><span class="fs-meta-label">Course &amp; Instructor</span><span class="fs-meta-val">{escape(COURSE_INSTRUCTOR)}</span></div>
</div>
</div>
<div class="fs-footer-bottom">
<span>© 2026 {escape(APP_NAME)} · Deep Learning Semester Project</span>
<span class="fs-top-pill">Back to top ↑</span>
</div>
</div>
""")

render(f'<div class="footnote">{escape(APP_NAME)} provides AI-assisted guidance. Confirm critical decisions with an agronomist.</div>')