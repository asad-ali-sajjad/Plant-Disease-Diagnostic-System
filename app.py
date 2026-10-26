import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING (GENERALIZED)
# ---------------------------------------------------------
st.set_page_config(
    page_title="FloraScan AI - Plant Disease Diagnostic System 🌿",
    page_icon="🌿",
    layout="wide"
)

st.title("🌿 FloraScan AI: Plant Disease Diagnostic System")
st.write("Upload a leaf image to diagnose plant health and identify potential diseases using deep learning.")
st.caption("📍 *Active Module: Solanaceae (Tomato) Classification Model*")

# Target Classes
CLASS_NAMES = [
    "Bacterial Spot",
    "Early Blight",
    "Late Blight",
    "Healthy"
]

# Image Preprocessing Transformation (Standard 224x224 & ImageNet normalization)
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------------------------------------------------
# 2. MODEL ARCHITECTURE DEFINITIONS
# ---------------------------------------------------------
class CustomCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def get_resnet18_model(num_classes=4):
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# ---------------------------------------------------------
# 3. MODEL LOADING (CACHED)
# ---------------------------------------------------------
@st.cache_resource
def load_model(model_choice):
    if model_choice == "Custom Plant CNN":
        model = CustomCNN(num_classes=4)
        model_path = "best_custom_cnn.pth"
    else:
        model = get_resnet18_model(num_classes=4)
        model_path = "best_resnet18.pth"

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# ---------------------------------------------------------
# 4. SIDEBAR CONTROLS & SYSTEM CONFIGURATION
# ---------------------------------------------------------
st.sidebar.header("⚙️ System Settings")

# Future-proof crop selection selector
selected_crop = st.sidebar.selectbox(
    "Select Crop Type:",
    ["Tomato (Active)", "Potato (Coming Soon)", "Corn (Coming Soon)"]
)

selected_model_name = st.sidebar.selectbox(
    "Choose Diagnostic Engine:",
    ["Custom Plant CNN", "Pre-trained ResNet18 Architecture"]
)

st.sidebar.markdown("""
---
**Model Performance:**
- **Custom Plant CNN**: 95.83% Test Accuracy
- **ResNet18 Backbone**: 94.00% Test Accuracy
""")

# Load active model
try:
    active_model = load_model(selected_model_name)
except Exception as e:
    st.error(f"Error loading model weights file `.pth`. Ensure weights are in the directory. Details: {e}")
    st.stop()

# ---------------------------------------------------------
# 5. DIAGNOSTIC INTERFACE & PREDICTION
# ---------------------------------------------------------
uploaded_file = st.file_uploader("Upload leaf sample (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📷 Sample View")
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("🔬 Diagnostic Analysis")
        with st.spinner("Scanning leaf features..."):
            input_tensor = TRANSFORM(image).unsqueeze(0)

            with torch.no_grad():
                outputs = active_model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0).numpy()

            predicted_class_idx = np.argmax(probabilities)
            predicted_class_name = CLASS_NAMES[predicted_class_idx]
            confidence_score = probabilities[predicted_class_idx] * 100

        # Diagnosis Output
        if predicted_class_name == "Healthy":
            st.success(f"**Condition**: {predicted_class_name} ({confidence_score:.2f}% Confidence)")
        else:
            st.error(f"**Condition Detected**: {predicted_class_name} ({confidence_score:.2f}% Confidence)")

        st.write("---")
        st.write("**Full Diagnostic Confidence Breakdown:**")
        chart_data = {CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))}
        st.bar_chart(chart_data)