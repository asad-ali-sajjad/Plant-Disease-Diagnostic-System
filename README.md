<div align="center">

# 🌿 FloraScan AI
### Intelligent Plant Disease Diagnostic System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)]()

*A real-time computer vision dashboard for precision agriculture and crop health monitoring.*

</div>

<br>

> **FloraScan AI** empowers farmers and botanists by accurately diagnosing plant leaf health and classifying agricultural crop diseases instantly using state-of-the-art Deep Learning architectures.

---

## ✨ Key Features

* 🧠 **Multi-Engine Inference:** Seamlessly toggle between two powerful backend models in real-time:
  * **Custom 3-Block CNN:** Highly optimized, lightweight architecture achieving **95.83% Accuracy**.
  * **ResNet18:** Robust transfer learning model achieving **94.00% Accuracy**.
* 📊 **Visual Confidence Metrics:** Dive deep into the AI's decision-making process with real-time probability breakdowns displayed via interactive bar charts.
* 🧩 **Modular & Scalable Design:** Purpose-built for rapid multi-crop expansion. *(Currently active module: Solanaceae / Tomato pathology)*.

---

## 📂 Repository Structure

```text
FloraScan-AI/
├── 🖥️ app.py                  # Main Streamlit web interface and prediction logic
├── 📦 requirements.txt        # Python package dependencies
├── 🧠 best_custom_cnn.pth     # Custom CNN trained model weights
├── 🧠 best_resnet18.pth       # ResNet18 trained model weights
├── 🙈 .gitignore              # Files ignored by Git control
└── 📖 README.md               # Project documentation

```

---

## 🚀 Quick Start

Get the diagnostic engine running on your local machine in seconds:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the Streamlit dashboard
streamlit run app.py

```
