"""
model.py — model building, loading and prediction.
Edit this file only if you change the architecture or preprocessing.
"""
import io

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import streamlit as st

from config import (
    PATH_1, PATH_2, NUM_CLASSES, INPUT_SIZE, NORM_MEAN, NORM_STD,
)

PREDICT_TRANSFORM = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
])


def get_resnet18_model(num_classes=NUM_CLASSES):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# One builder per engine. To change an architecture, edit only the matching builder.
def build_model_1():
    return get_resnet18_model(NUM_CLASSES)


def build_model_2():
    return get_resnet18_model(NUM_CLASSES)


def model_stamp():
    """Changes whenever a checkpoint file changes (or the class count changes),
    which forces Streamlit to reload the models."""
    def mtime(path):
        return path.stat().st_mtime if path.exists() else 0
    return (NUM_CLASSES, mtime(PATH_1), mtime(PATH_2))


def _load_one(builder, path):
    """Returns (model_or_None, error_message_or_None)."""
    if not path.exists():
        return None, f"Checkpoint not found: {path.name}"
    try:
        model = builder()
        state = torch.load(path, map_location=torch.device("cpu"))
        model.load_state_dict(state)
        model.eval()
        return model, None
    except Exception as exc:
        return None, (
            f"Could not load {path.name}. The file may not match the architecture "
            f"or the number of classes ({NUM_CLASSES}). Details: {exc}"
        )


@st.cache_resource(show_spinner="Loading AI engines…")
def load_all_models(stamp):
    """`stamp` is only used as a cache key (see model_stamp)."""
    model_1, err_1 = _load_one(build_model_1, PATH_1)
    model_2, err_2 = _load_one(build_model_2, PATH_2)
    errors = [e for e in (err_1, err_2) if e]
    return model_1, model_2, errors


def predict(model, tensor):
    """Softmax probabilities (numpy array) for one image tensor."""
    with torch.no_grad():
        return F.softmax(model(tensor)[0], dim=0).numpy()


@st.cache_data(show_spinner=False, max_entries=64)
def predict_cached(image_bytes, engine, stamp):
    """Cached prediction. engine is 1 or 2.
    `stamp` makes the cache refresh when a .pth file changes."""
    model_1, model_2, _ = load_all_models(stamp)
    model = model_1 if engine == 1 else model_2
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = PREDICT_TRANSFORM(image).unsqueeze(0)
    return predict(model, tensor)