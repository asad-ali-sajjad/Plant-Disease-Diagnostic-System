import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import streamlit as st

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION & VISUAL THEME
# ---------------------------------------------------------
st.set_page_config(
    page_title="FloraScan AI — Smart Phytopathology System",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Deep UI/UX Overhaul & Fixing Image Styles
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', sans-serif;
        }
        
        /* Global Background */
        .stApp {
            background-color: #f8fafc;
        }
        
        /* Magic Image Targeting (Fixes the TypeError) */
        [data-testid="stImage"] img {
            border-radius: 16px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.08);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        [data-testid="stImage"] img:hover {
            transform: translateY(-5px) scale(1.02);
            box-shadow: 0 20px 30px rgba(0, 0, 0, 0.12);
        }

        /* Animated Hero Banner */
        .hero-banner {
            background: linear-gradient(120deg, #064e3b 0%, #047857 50%, #10b981 100%);
            padding: 2.5rem 3rem;
            border-radius: 20px;
            color: white;
            margin-bottom: 2rem;
            box-shadow: 0 15px 30px rgba(4, 120, 87, 0.2);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .hero-banner h1 { margin: 0; font-weight: 800; font-size: 2.8rem; letter-spacing: -1px; }
        .hero-banner p { margin: 5px 0 0 0; font-size: 1.1rem; opacity: 0.9; }
        
        /* Streamlit Tabs Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
            background-color: white;
            padding: 10px 20px;
            border-radius: 16px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.02);
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: transparent;
            border-radius: 8px;
            color: #64748b;
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            color: #047857 !important;
            background-color: #ecfdf5 !important;
        }

        /* Status Cards */
        .status-healthy {
            background: #dcfce7;
            border-left: 6px solid #22c55e;
            color: #14532d;
            padding: 1.5rem;
            border-radius: 12px;
            font-size: 1.2rem;
            font-weight: 700;
        }
        .status-warning {
            background: #fef2f2;
            border-left: 6px solid #ef4444;
            color: #7f1d1d;
            padding: 1.5rem;
            border-radius: 12px;
            font-size: 1.2rem;
            font-weight: 700;
        }
        .status-error {
            background: #fffbeb;
            border-left: 6px solid #f59e0b;
            color: #92400e;
            padding: 1.5rem;
            border-radius: 12px;
            font-size: 1.1rem;
        }
        
        /* File Uploader Hover */
        [data-testid="stFileUploadDropzone"] {
            border: 2px dashed #6ee7b7;
            border-radius: 20px;
            background-color: #ffffff;
            transition: all 0.3s ease;
        }
        [data-testid="stFileUploadDropzone"]:hover {
            border-color: #10b981;
            background-color: #ecfdf5;
        }
    </style>
""", unsafe_allow_html=True)

# Custom Progress Bar HTML
def render_progress_bar(label, percentage, is_healthy=False):
    color = "#10b981" if is_healthy else "#ef4444"
    if percentage < 15: color = "#cbd5e1"
    
    return f"""
    <div style="margin-bottom: 1rem;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
            <span style="font-weight: 600; color: #334155;">{label}</span>
            <span style="font-weight: 700; color: {color};">{percentage:.1f}%</span>
        </div>
        <div style="width: 100%; background-color: #e2e8f0; border-radius: 99px; height: 10px;">
            <div style="background-color: {color}; height: 100%; border-radius: 99px; width: {percentage}%;"></div>
        </div>
    </div>
    """

# ---------------------------------------------------------
# 2. CONSTANTS & METADATA (Logic Untouched)
# ---------------------------------------------------------
CONFIDENCE_THRESHOLD = 65.0

CLASS_NAMES = [
    "Tomato Bacterial Spot",
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Tomato Healthy"
]

DIAGNOSTIC_GUIDE = {
    "Tomato Bacterial Spot": {
        "description": "Triggered by *Xanthomonas* bacteria. Symptoms include small, water-soaked dark spots on foliage that quickly enlarge and yellow surrounding tissue.",
        "action": "Avoid overhead irrigation, apply copper-based bactericides early, and discard infected foliage to prevent spores spreading via water splash."
    },
    "Tomato Early Blight": {
        "description": "Fungal infection caused by *Alternaria solani*. Characterized by distinct concentric target-like rings on mature leaf surfaces.",
        "action": "Ensure adequate plant spacing for airflow, stake foliage off soil, and treat proactively with suitable organic or chemical fungicides."
    },
    "Tomato Late Blight": {
        "description": "Devastating water-mold infection caused by *Phytophthora infestans*. Causes rapidly spreading grayish-green dark foliage decay in cool, wet environments.",
        "action": "Immediately prune and dispose of infected leaves. Maintain low ambient humidity in greenhouses and apply preventive systemic fungicides."
    },
    "Tomato Healthy": {
        "description": "Foliage exhibits rich chlorophyll saturation, standard vascular structure, and no signs of bacterial or fungal pathogen colonization.",
        "action": "Maintain optimal irrigation cycles, routine nitrogen balance, and continue periodic pest monitoring."
    }
}

PREDICT_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------------------------------------------------
# 3. MODEL ARCHITECTURE & LOADERS (Logic Untouched)
# ---------------------------------------------------------
def get_resnet18_model(num_classes=4):
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

@st.cache_resource
def load_all_models():
    device = torch.device("cpu")
    model_1 = get_resnet18_model(num_classes=4)
    path_1 = "best_custom_cnn.pth"
    if os.path.exists(path_1):
        model_1.load_state_dict(torch.load(path_1, map_location=device))
        model_1.eval()
    else: model_1 = None

    model_2 = get_resnet18_model(num_classes=4)
    path_2 = "best_resnet18.pth"
    if os.path.exists(path_2):
        model_2.load_state_dict(torch.load(path_2, map_location=device))
        model_2.eval()
    else: model_2 = None

    return model_1, model_2

# ---------------------------------------------------------
# 4. INTERACTIVE SIDEBAR DASHBOARD
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎛️ Control Center")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Styled Radio instead of Selectbox for better UX
    selected_model_name = st.radio(
        "Select Inference Engine",
        ["Primary Engine (CNN)", "Secondary Engine (ResNet)", "Dual Benchmark Mode"],
        index=0,
        help="Switching engines changes the background AI model in real-time."
    )
    
    st.markdown("---")
    st.markdown("### 📡 System Status")
    
    # Interactive Dashboard Metrics
    col1, col2 = st.columns(2)
    col1.metric("Safety Threshold", f"{CONFIDENCE_THRESHOLD}%", "Active")
    col2.metric("Compute Node", "CPU", "Ready")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Hide details in an expander for clean UI
    with st.expander("🛠️ View Engine Specifications"):
        st.write("**Dataset:** Balanced Tomato Sets")
        st.write("**Resolution:** 224x224 RGB")
        st.write("**Outputs:** 4 Health States")

model_1, model_2 = load_all_models()
if model_1 is None or model_2 is None:
    st.error("⚠️ **System Offline:** Model checkpoints not found. Please place both `.pth` files next to `app.py`.")
    st.stop()

# ---------------------------------------------------------
# 5. MAIN DASHBOARD UI
# ---------------------------------------------------------
st.markdown("""
    <div class="hero-banner">
        <div>
            <h1>FloraScan AI</h1>
            <p>Smart Phytopathology & Diagnostic Engine</p>
        </div>
        <div style="font-size: 3rem; background: rgba(255,255,255,0.2); padding: 15px; border-radius: 50%;">🌿</div>
    </div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📥 Drop a leaf sample here for instant analysis", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    input_tensor = PREDICT_TRANSFORM(image).unsqueeze(0)

    if selected_model_name != "Dual Benchmark Mode":
        # SINGLE ENGINE INTERACTIVE VIEW
        col_img, col_results = st.columns([1, 1.5], gap="large")

        with col_img:
            # The CSS above automatically handles the styling of this image safely
            st.image(image, use_container_width=True)

        with col_results:
            active_model = model_1 if "CNN" in selected_model_name else model_2
            with torch.no_grad():
                probabilities = F.softmax(active_model(input_tensor)[0], dim=0).numpy()

            pred_class = CLASS_NAMES[np.argmax(probabilities)]
            confidence = float(np.max(probabilities) * 100)

            if confidence < CONFIDENCE_THRESHOLD:
                st.markdown(f"""
                    <div class="status-error">
                        <b>⚠️ Unrecognized Sample Detected</b><br>
                        Engine confidence is extremely low ({confidence:.1f}%). This does not match our leaf database.
                    </div>
                """, unsafe_allow_html=True)
            else:
                # Interactive Tabs for Results
                tab_diag, tab_metrics = st.tabs(["🩺 Diagnosis & Action Plan", "📊 Confidence Metrics"])
                
                with tab_diag:
                    if pred_class == "Tomato Healthy":
                        st.markdown(f'<div class="status-healthy">✅ {pred_class} <span style="float:right; opacity:0.8;">{confidence:.1f}% Match</span></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="status-warning">⚠️ {pred_class} <span style="float:right; opacity:0.8;">{confidence:.1f}% Match</span></div>', unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.info(f"**Description:** {DIAGNOSTIC_GUIDE[pred_class]['description']}")
                    st.success(f"**Action Plan:** {DIAGNOSTIC_GUIDE[pred_class]['action']}")

                with tab_metrics:
                    st.markdown("### Probability Distribution")
                    for i, name in enumerate(CLASS_NAMES):
                        score = float(probabilities[i]) * 100
                        st.markdown(render_progress_bar(name, score, name=="Tomato Healthy"), unsafe_allow_html=True)

    else:
        # DUAL BENCHMARK INTERACTIVE VIEW
        st.markdown("<h3 style='text-align: center; color: #1e293b; margin-bottom: 2rem;'>⚔️ Live Dual-Engine Cross-Validation</h3>", unsafe_allow_html=True)
        
        col_img1, col_img2, col_img3 = st.columns([1, 2, 1])
        with col_img2: st.image(image, use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)

        with torch.no_grad():
            prob_1 = F.softmax(model_1(input_tensor)[0], dim=0).numpy()
            prob_2 = F.softmax(model_2(input_tensor)[0], dim=0).numpy()

        col_a, col_b = st.columns(2, gap="large")

        # Engine 1 interactive container
        with col_a:
            st.markdown("#### 🔹 Custom CNN Engine")
            conf_1 = float(np.max(prob_1) * 100)
            res_class_1 = CLASS_NAMES[np.argmax(prob_1)]
            
            if conf_1 < CONFIDENCE_THRESHOLD:
                st.markdown(f"<div class='status-error'>⚠️ Low Confidence ({conf_1:.1f}%)</div>", unsafe_allow_html=True)
            else:
                cls = "status-healthy" if res_class_1 == "Tomato Healthy" else "status-warning"
                st.markdown(f'<div class="{cls}" style="padding: 1rem;">{res_class_1}<br><small>{conf_1:.1f}%</small></div>', unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("View CNN Probabilities"):
                for i, name in enumerate(CLASS_NAMES):
                    st.markdown(render_progress_bar(name, float(prob_1[i]) * 100, name=="Tomato Healthy"), unsafe_allow_html=True)

        # Engine 2 interactive container
        with col_b:
            st.markdown("#### 🔸 ResNet18 Engine")
            conf_2 = float(np.max(prob_2) * 100)
            res_class_2 = CLASS_NAMES[np.argmax(prob_2)]
            
            if conf_2 < CONFIDENCE_THRESHOLD:
                st.markdown(f"<div class='status-error'>⚠️ Low Confidence ({conf_2:.1f}%)</div>", unsafe_allow_html=True)
            else:
                cls = "status-healthy" if res_class_2 == "Tomato Healthy" else "status-warning"
                st.markdown(f'<div class="{cls}" style="padding: 1rem;">{res_class_2}<br><small>{conf_2:.1f}%</small></div>', unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("View ResNet Probabilities"):
                for i, name in enumerate(CLASS_NAMES):
                    st.markdown(render_progress_bar(name, float(prob_2[i]) * 100, name=="Tomato Healthy"), unsafe_allow_html=True)
else:
    # Interactive Empty State
    st.info("👋 **Welcome to FloraScan AI.** Waiting for your leaf upload above to begin diagnostics.")