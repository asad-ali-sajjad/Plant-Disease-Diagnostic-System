"""
config.py — the ONLY file you normally edit when you retrain or change classes.
No Streamlit calls here, so it is safe to import from anywhere.
"""
from html import escape
from pathlib import Path

# ---------------------------------------------------------
# Paths (absolute, so the app works no matter where it is launched from)
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PATH_1 = BASE_DIR / "best_custom_cnn.pth"
PATH_2 = BASE_DIR / "best_resnet18.pth"
CSS_PATH = BASE_DIR / "styles.css"

# ---------------------------------------------------------
# Branding
# ---------------------------------------------------------
APP_NAME = "FloraScan AI"
APP_TAGLINE = "Calm, instant leaf diagnostics — drop a photo and let the engines read it."
CROP_EMOJI = "🍅"

# ---------------------------------------------------------
# Engines
# ---------------------------------------------------------
ENGINE_1_LABEL = "Primary Engine (CNN)"
ENGINE_2_LABEL = "Secondary Engine (ResNet)"
DUAL_LABEL = "Dual Benchmark Mode"
ENGINE_1_SHORT = "Custom CNN"
ENGINE_2_SHORT = "ResNet18"

# ---------------------------------------------------------
# Model settings
# ---------------------------------------------------------
CONFIDENCE_THRESHOLD = 65.0
INPUT_SIZE = 224
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# IMPORTANT: this order MUST match your training labels exactly.
# If you trained with ImageFolder, run `print(train_dataset.classes)` and use that order.
CLASS_NAMES = [
    "Tomato Bacterial Spot",
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Tomato Healthy",
]
NUM_CLASSES = len(CLASS_NAMES)

# ---------------------------------------------------------
# Diagnostic text (shown in the results and the report)
# ---------------------------------------------------------
DIAGNOSTIC_GUIDE = {
    "Tomato Bacterial Spot": {
        "description": "Triggered by *Xanthomonas* bacteria. Symptoms include small, water-soaked dark spots on foliage that quickly enlarge and yellow surrounding tissue.",
        "action": "Avoid overhead irrigation, apply copper-based bactericides early, and discard infected foliage to prevent spores spreading via water splash.",
    },
    "Tomato Early Blight": {
        "description": "Fungal infection caused by *Alternaria solani*. Characterized by distinct concentric target-like rings on mature leaf surfaces.",
        "action": "Ensure adequate plant spacing for airflow, stake foliage off soil, and treat proactively with suitable organic or chemical fungicides.",
    },
    "Tomato Late Blight": {
        "description": "Devastating water-mold infection caused by *Phytophthora infestans*. Causes rapidly spreading grayish-green dark foliage decay in cool, wet environments.",
        "action": "Immediately prune and dispose of infected leaves. Maintain low ambient humidity in greenhouses and apply preventive systemic fungicides.",
    },
    "Tomato Healthy": {
        "description": "Foliage exhibits rich chlorophyll saturation, standard vascular structure, and no signs of bacterial or fungal pathogen colonization.",
        "action": "Maintain optimal irrigation cycles, routine nitrogen balance, and continue periodic pest monitoring.",
    },
}

# ---------------------------------------------------------
# Visual metadata (optional per class; "tone" must be a 6-digit hex like #10b981)
# ---------------------------------------------------------
UI_META = {
    "Tomato Bacterial Spot": {
        "icon": "🦠", "tone": "#f59e0b", "bg": "linear-gradient(135deg,#fffbeb,#fef3c7)",
        "kind": "Bacterial", "severity": "Moderate",
        "signs": ["Water-soaked spots", "Yellow halos", "Rough leaf texture"],
        "desc_html": "Triggered by <em>Xanthomonas</em> bacteria. Small, water-soaked dark spots that enlarge and yellow the surrounding tissue.",
    },
    "Tomato Early Blight": {
        "icon": "🎯", "tone": "#f97316", "bg": "linear-gradient(135deg,#fff7ed,#ffedd5)",
        "kind": "Fungal", "severity": "Moderate",
        "signs": ["Concentric target rings", "Older leaves first", "Yellowing"],
        "desc_html": "Fungal infection caused by <em>Alternaria solani</em>. Distinct concentric target-like rings on mature leaves.",
    },
    "Tomato Late Blight": {
        "icon": "🌧️", "tone": "#ef4444", "bg": "linear-gradient(135deg,#fef2f2,#fee2e2)",
        "kind": "Water mold", "severity": "High",
        "signs": ["Gray-green patches", "Spreads rapidly", "Cool & wet weather"],
        "desc_html": "Devastating water-mold caused by <em>Phytophthora infestans</em>. Rapid grayish-green decay in cool, wet conditions.",
    },
    "Tomato Healthy": {
        "icon": "🌱", "tone": "#10b981", "bg": "linear-gradient(135deg,#ecfdf5,#d1fae5)",
        "kind": "Healthy", "severity": "None",
        "signs": ["Rich green color", "Even leaf surface", "No lesions"],
        "desc_html": "Rich chlorophyll saturation, standard vascular structure, and no signs of bacterial or fungal colonization.",
    },
}

# Used for any class that has no entry in UI_META (so a new class never crashes the app)
DEFAULT_META = {
    "icon": "🍃", "tone": "#0d9488",
    "bg": "linear-gradient(135deg,#f0fdfa,#ccfbf1)",
    "kind": "Condition", "severity": "Unknown",
    "signs": [], "desc_html": "",
}
UNKNOWN_TONE = "#f59e0b"  # color used for "Unrecognized" results


def get_guide(name):
    """Diagnostic text for a class, with a safe fallback for new classes."""
    return DIAGNOSTIC_GUIDE.get(name, {
        "description": "No description added yet for this class.",
        "action": "Consult an agronomist for guidance.",
    })


def get_meta(name):
    """Visual metadata for a class, with safe defaults for new classes."""
    meta = {**DEFAULT_META, **UI_META.get(name, {})}
    if not meta.get("desc_html"):
        meta["desc_html"] = escape(get_guide(name)["description"].replace("*", ""))
    return meta