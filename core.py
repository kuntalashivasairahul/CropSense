"""Model loading and inference for CropSense AI.

Kept free of Streamlit imports so it can be run and tested headlessly
(see test_core.py).
"""

import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
from PIL import Image

MODELS_DIR = Path(__file__).parent / "models"

# ---------------------------------------------------------------------------
# Module 1 — disease detection (MobileNetV2 CNN)
# ---------------------------------------------------------------------------

# Exact output-index order of the trained model. Index 0 -> first entry.
# Do not re-sort: the model's Dense(38, softmax) head is bound to this order.
DISEASE_CLASSES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust",
    "Apple___healthy", "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot",
    "Peach___healthy", "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy", "Tomato___Bacterial_spot",
    "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus", "Tomato___healthy",
]

IMG_SIZE = (224, 224)


def pretty_class(raw_name):
    """'Tomato___Early_blight' -> 'Tomato — Early Blight'."""
    crop, _, condition = raw_name.partition("___")
    return f"{crop.replace('_', ' ').strip().title()} — {condition.replace('_', ' ').strip().title()}"


def preprocess_image(pil_image):
    """Resize to 224x224 RGB and return a raw 0-255 float batch.

    NO manual rescaling: the saved model's own graph starts with
    `x / 127.5 - 1.0` (MobileNetV2 preprocess_input) immediately after the
    input layer, so dividing here would double-scale and silently corrupt
    every prediction.
    """
    img = pil_image.convert("RGB").resize(IMG_SIZE)
    return np.expand_dims(np.asarray(img, dtype="float32"), axis=0)


def predict_disease(model, pil_image, top_k=5):
    probs = model.predict(preprocess_image(pil_image), verbose=0)[0]
    idx = int(probs.argmax())
    raw_name = DISEASE_CLASSES[idx]
    confidence = float(probs[idx])
    is_healthy = "healthy" in raw_name.lower()
    order = probs.argsort()[::-1][:top_k]
    return {
        "raw_class": raw_name,
        "display_class": pretty_class(raw_name),
        "confidence": confidence,
        "is_healthy": is_healthy,
        # High confidence in a *disease* diagnosis pulls the score toward 0.
        "disease_score": confidence if is_healthy else 1.0 - confidence,
        "top_k": [(pretty_class(DISEASE_CLASSES[i]), float(probs[i])) for i in order],
    }


# ---------------------------------------------------------------------------
# Module 2 — NDVI crop stress (Random Forest)
# ---------------------------------------------------------------------------

NDVI_FEATURES = ["ndvi_mean", "ndvi_std", "ndvi_min", "ndvi_max", "ndvi_p25", "ndvi_p75"]

NDVI_SCORE_MAP = {"healthy": 1.0, "moderate_stress": 0.5, "severe_stress": 0.1}

NDVI_DEFAULTS = {"ndvi_mean": 0.50, "ndvi_std": 0.10, "ndvi_min": 0.30,
                 "ndvi_max": 0.70, "ndvi_p25": 0.42, "ndvi_p75": 0.60}

# Found by searching the trained forest's input space and verified to produce
# their named class (see test_core.py). Note the forest keys "severe" mainly off
# high within-field variance (ndvi_std / spread), not off a low mean alone.
NDVI_EXAMPLES = {
    "healthy": {"ndvi_mean": 0.67, "ndvi_std": 0.15, "ndvi_min": 0.14,
                "ndvi_max": 0.81, "ndvi_p25": 0.39, "ndvi_p75": 0.73},
    "stressed": {"ndvi_mean": 0.25, "ndvi_std": 0.37, "ndvi_min": 0.08,
                 "ndvi_max": 0.68, "ndvi_p25": 0.21, "ndvi_p75": 0.45},
}


def predict_ndvi(model, feature_values):
    """feature_values: dict keyed by NDVI_FEATURES."""
    # DataFrame (not bare array) so the feature names match those the forest
    # was fitted with, which keeps sklearn from warning and guards the order.
    import pandas as pd
    x = pd.DataFrame([[feature_values[f] for f in NDVI_FEATURES]], columns=NDVI_FEATURES)
    probs = model.predict_proba(x)[0]
    label = str(model.classes_[int(probs.argmax())])
    return {
        "label": label,
        "confidence": float(probs.max()),
        "probs": {str(c): float(p) for c, p in zip(model.classes_, probs)},
        "ndvi_score": NDVI_SCORE_MAP[label],
    }


# ---------------------------------------------------------------------------
# Module 3 — pest risk (LSTM over 4-week weather windows)
# ---------------------------------------------------------------------------

PEST_CLASSES = ["low", "medium", "high"]
PEST_WEIGHTS = np.array([0.0, 0.5, 1.0])


def load_sequences():
    with open(MODELS_DIR / "sample_sequences.json") as f:
        return json.load(f)


def pretty_label(label):
    """'2009.0 - Week 34.0' -> '2009 · Week 34'."""
    try:
        year, _, week = label.partition(" - ")
        return f"{int(float(year))} · Week {int(float(week.replace('Week', '').strip()))}"
    except (ValueError, AttributeError):
        return label


def predict_pest(model, sequence):
    """sequence: (window_size, n_features) of raw weather values.

    Fed unscaled, exactly as exported in sample_sequences.json alongside the
    model — no scaler artifact was saved with the model.
    """
    x = np.asarray(sequence, dtype="float32")[np.newaxis, ...]
    probs = model.predict(x, verbose=0)[0]
    return {
        "probs": {c: float(p) for c, p in zip(PEST_CLASSES, probs)},
        "pred_class": PEST_CLASSES[int(probs.argmax())],
        "confidence": float(probs.max()),
        # Continuous expected risk, not just the hard class label.
        "pest_risk_score": float(np.dot(PEST_WEIGHTS, probs)),
    }


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------

def composite_score(disease_score, ndvi_score, pest_risk_score):
    return (disease_score + ndvi_score + (1.0 - pest_risk_score)) / 3.0


def bucket(score):
    """-> (label, hex colour)."""
    if score >= 0.66:
        return "Healthy", "#2E7D32"
    if score >= 0.33:
        return "Moderate Risk", "#E8A33D"
    return "Severe Risk", "#C62828"


# ---------------------------------------------------------------------------
# Loaders (wrapped in @st.cache_resource by app.py)
# ---------------------------------------------------------------------------

HF_MODELS_REPO = "rahulkuntala/cropsense-models"

LFS_POINTER_MAGIC = b"version https://git-lfs"


def model_path(filename):
    """Path to a model file, downloading from the Hub if it isn't really here.

    The artefacts are git-lfs tracked. Streamlit Community Cloud does not fetch
    lfs objects, so on that host `models/*.keras` arrive as ~130-byte pointer
    stubs that would fail to load with a confusing error. Detect the stub (and
    a plain missing file) and pull the real artefact from HF_MODELS_REPO
    instead. Locally, where lfs did run, the real file is used as-is.
    """
    local = MODELS_DIR / filename
    if local.exists():
        with open(local, "rb") as f:          # header only, not the whole 25 MB
            if f.read(len(LFS_POINTER_MAGIC)) != LFS_POINTER_MAGIC:
                return str(local)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(HF_MODELS_REPO, filename)


def load_disease_model():
    import keras
    return keras.models.load_model(model_path("module1_cnn.keras"))


def load_ndvi_model():
    import joblib
    return joblib.load(model_path("module2_rf.pkl"))


def load_pest_model():
    import keras
    return keras.models.load_model(model_path("module3_lstm.keras"))
