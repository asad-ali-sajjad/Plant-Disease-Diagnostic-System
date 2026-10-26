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

* 🧠 **Multi-Engine Inference:** Toggle between two backend models in real-time:
  * **Custom 3-Block CNN:** Lightweight architecture achieving **95.83% Accuracy**.
  * **ResNet18:** Transfer learning model achieving **94.00% Accuracy**.
* ⚖️ **Dual Benchmark Mode:** Run both engines on the same leaf and see whether they agree, disagree, or are unsure.
* 📊 **Visual Confidence Metrics:** Circular confidence gauge and animated probability bars for every prediction.
* 🛡️ **Safety Threshold:** Predictions below 65% confidence are flagged as "Unrecognized" instead of guessed.
* 📚 **Disease Library:** Built-in reference cards with signs, severity, and recommended care for each condition.
* 🕘 **Scan History:** Session log of every analysis, with a one-click clear.
* ⬇️ **Downloadable Reports:** Export a text diagnosis report for any confident result.
* 🎨 **Calm, Modern Interface:** Large tab navigation, glass-style cards, and smooth animations.
* ⚡ **Fast & Safe:** Cached predictions, automatic reload when model files change, and friendly error messages for bad images or missing checkpoints.
* 🧩 **Modular & Scalable Design:** Built for easy retraining and multi-crop expansion. *(Currently active module: Solanaceae / Tomato pathology)*.

---

## 📂 Repository Structure

```text
FloraScan-AI/
├── 🖥️ app.py                  # Streamlit interface: sidebar, tabs, page flow
├── ⚙️ config.py               # Classes, paths, threshold, labels, disease info (edit this to retrain)
├── 🧠 model.py                # Model building, loading, cached prediction
├── 🎨 ui.py                   # HTML components: result cards, probability bars, reports
├── 💅 styles.css              # All visual styling
├── 📁 .streamlit/
│   └── config.toml            # Theme settings
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

> Requires **Streamlit 1.27 or newer**. Both `.pth` files must sit next to `app.py`.

---

## 🔄 Retraining & Customization

Most changes only touch `config.py`:

| What changed | What to edit |
|---|---|
| Same classes, new weights | Nothing. Replace the `.pth` file and the app reloads it automatically. |
| Classes added, removed, or renamed | `CLASS_NAMES` in `config.py`. Optionally add entries to `UI_META` and `DIAGNOSTIC_GUIDE`. |
| Different architecture | The matching `build_model_1` / `build_model_2` in `model.py`. |
| Different input size or normalization | `INPUT_SIZE`, `NORM_MEAN`, `NORM_STD` in `config.py`. |
| Different confidence threshold | `CONFIDENCE_THRESHOLD` in `config.py`. |
| Different crop | `CLASS_NAMES`, `CROP_EMOJI`, and text in `config.py`. |

> ⚠️ **`CLASS_NAMES` must match your training label order exactly.** If you trained with `ImageFolder`, run `print(train_dataset.classes)` and use that order. A wrong order raises no error but gives wrong diagnoses.

---

## ⚠️ Disclaimer

FloraScan AI provides AI-assisted guidance only. Confirm critical decisions with a qualified agronomist.