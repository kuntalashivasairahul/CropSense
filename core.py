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
    if "___" in str(raw_name):
        crop, _, condition = str(raw_name).partition("___")
        return f"{crop.replace('_', ' ').strip().title()} — {condition.replace('_', ' ').strip().title()}"
    return str(raw_name).replace("_", " ").strip().title()


def preprocess_image(pil_image):
    """Resize to 224x224 RGB and return a raw 0-255 float batch.

    NO manual rescaling: the saved model's own graph starts with
    `x / 127.5 - 1.0` (MobileNetV2 preprocess_input) immediately after the
    input layer, so dividing here would double-scale and silently corrupt
    every prediction.
    """
    img = pil_image.convert("RGB").resize(IMG_SIZE)
    return np.expand_dims(np.asarray(img, dtype="float32"), axis=0)


def generate_gradcam(model, pil_image, class_idx=None, alpha=0.45):
    """Compute Grad-CAM saliency map overlay on the input leaf photo.

    Extracts gradients from the MobileNetV2 backbone's final activation layer
    with respect to the target class logit, producing a spatial activation map
    highlighting the specific visual lesion features driving the diagnosis.
    """
    import tensorflow as tf

    x = preprocess_image(pil_image)
    try:
        base = model.get_layer("mobilenetv2_1.00_224")
        gap = model.get_layer("global_average_pooling2d")
        drop = model.get_layer("dropout_1")
        dense = model.get_layer("dense_2")
    except (ValueError, AttributeError):
        return None

    with tf.GradientTape() as tape:
        conv_outputs = base(x)
        tape.watch(conv_outputs)
        y = gap(conv_outputs)
        y = drop(y, training=False)
        preds = dense(y)
        if class_idx is None:
            class_idx = int(tf.argmax(preds[0]))
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        return None
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    h_np = heatmap.numpy()

    h_img = Image.fromarray(np.uint8(255 * h_np)).resize(IMG_SIZE, Image.BICUBIC)
    h_arr = np.asarray(h_img, dtype="float32") / 255.0

    # Jet colormap formula in numpy
    r = np.clip(1.5 - np.abs(4.0 * h_arr - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * h_arr - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * h_arr - 1.0), 0.0, 1.0)
    colored_cam = np.stack([r, g, b], axis=-1) * 255.0

    orig = pil_image.convert("RGB").resize(IMG_SIZE)
    orig_arr = np.asarray(orig, dtype="float32")
    # Dynamic alpha proportional to activation: 0 in background, up to 0.65 in hot spots
    alpha_map = np.clip(h_arr * 0.75, 0.0, 0.65)[..., np.newaxis]
    overlay = (1.0 - alpha_map) * orig_arr + alpha_map * colored_cam
    return Image.fromarray(np.clip(overlay, 0, 255).astype("uint8"))


# ---------------------------------------------------------------------------
# Out-of-Distribution (OOD) Leaf Sanity Checks
# ---------------------------------------------------------------------------

IMAGENET_PLANT_KEYWORDS = (
    "corn", "artichoke", "cabbage", "cauliflower", "zucchini", "squash", "cucumber",
    "mushroom", "broccoli", "bell pepper", "cardoon", "daisy", "pot", "rapeseed",
    "hay", "strawberry", "fig", "pineapple", "banana", "pomegranate", "lemon", "orange",
    "granny smith", "jackfruit", "custard apple", "acorn", "agaric", "gyromitra",
    "stinkhorn", "earthstar", "bolete", "coral fungus", "yellow lady's slipper", "greenhouse"
)

IMAGENET_NON_PLANT_EXCLUDE = {"sea_cucumber", "cornet", "Crock_Pot", "hot_pot"}


def is_imagenet_plant_class(raw_name):
    """Check if an ImageNet class name matches plant or vegetation keywords."""
    if raw_name in IMAGENET_NON_PLANT_EXCLUDE:
        return False
    clean = raw_name.lower().replace("_", " ")
    words = clean.split()
    for kw in IMAGENET_PLANT_KEYWORDS:
        if " " in kw:
            if kw in clean:
                return True
        else:
            if kw in words or any(w.startswith(kw) for w in words if len(w) <= len(kw) + 2):
                return True
    return False


def check_plant_color_coverage(pil_image, hue_low=20, hue_high=160, min_sat=40, min_val=30, threshold_pct=15.0):
    """Compute vegetation pixel percentage in HSV (greens, yellows, browns).

    Roughly hue 20-160 on OpenCV's 0-179 scale with moderate-to-high saturation and value.
    Flags as low plant-color coverage if percentage < threshold_pct (default 15%).
    """
    rgb = pil_image.convert("RGB")
    hsv = rgb.convert("HSV")
    h, s, v = np.array(hsv, dtype="float32").transpose(2, 0, 1)
    # Convert PIL hue (0..255 representing 0..360 deg) to OpenCV hue scale (0..179)
    h_cv = (h / 255.0) * 179.0
    mask = (h_cv >= hue_low) & (h_cv <= hue_high) & (s >= min_sat) & (v >= min_val)
    coverage = float(np.mean(mask) * 100.0)
    return {
        "coverage_pct": round(coverage, 2),
        "is_low_coverage": coverage < threshold_pct,
        "threshold_pct": threshold_pct,
    }


def check_imagenet_sanity(imagenet_model, pil_image, top_k=5):
    """Run image through stock ImageNet MobileNetV2 and check if top-5 has plant classes."""
    from tensorflow.keras.applications.mobilenet_v2 import decode_predictions, preprocess_input
    img = pil_image.convert("RGB").resize(IMG_SIZE)
    x = np.array(img, dtype="float32")[np.newaxis, ...]
    x = preprocess_input(x)
    preds = imagenet_model.predict(x, verbose=0)
    decoded = decode_predictions(preds, top=top_k)[0]
    results = []
    has_plant = False
    for _, name, prob in decoded:
        match = is_imagenet_plant_class(name)
        if match:
            has_plant = True
        results.append((name.replace("_", " ").title(), float(prob), match))
    return {
        "top_k": results,
        "has_plant_match": has_plant,
        "is_flagged": not has_plant,
    }


def check_leaf_sanity(pil_image, imagenet_model=None):
    """Comprehensive out-of-distribution sanity check for leaf photos."""
    color_res = check_plant_color_coverage(pil_image)
    signals = []
    if color_res["is_low_coverage"]:
        signals.append("low plant-color coverage")

    imagenet_res = None
    if imagenet_model is not None:
        imagenet_res = check_imagenet_sanity(imagenet_model, pil_image)
        if imagenet_res["is_flagged"]:
            signals.append("no plant-related classes in top-5")

    is_flagged = color_res["is_low_coverage"] or (imagenet_res["is_flagged"] if imagenet_res else False)
    return {
        "color_coverage_pct": color_res["coverage_pct"],
        "color_flagged": color_res["is_low_coverage"],
        "threshold_pct": color_res["threshold_pct"],
        "imagenet_top_5": imagenet_res["top_k"] if imagenet_res else [],
        "imagenet_flagged": imagenet_res["is_flagged"] if imagenet_res else False,
        "has_plant_match": imagenet_res["has_plant_match"] if imagenet_res else True,
        "is_flagged": is_flagged,
        "signals_fired": signals,
    }


def predict_disease(model, pil_image, top_k=5, generate_cam=True, sanity_model=None):
    probs = model.predict(preprocess_image(pil_image), verbose=0)[0]
    idx = int(probs.argmax())
    raw_name = DISEASE_CLASSES[idx]
    confidence = float(probs[idx])
    is_healthy = "healthy" in raw_name.lower()
    order = probs.argsort()[::-1][:top_k]
    cam_img = generate_gradcam(model, pil_image, idx) if generate_cam else None
    sanity = check_leaf_sanity(pil_image, sanity_model)
    return {
        "raw_class": raw_name,
        "display_class": pretty_class(raw_name),
        "confidence": confidence,
        "is_healthy": is_healthy,
        # High confidence in a *disease* diagnosis pulls the score toward 0.
        "disease_score": confidence if is_healthy else 1.0 - confidence,
        "top_k": [(pretty_class(DISEASE_CLASSES[i]), float(probs[i])) for i in order],
        "cam_image": cam_img,
        "sanity": sanity,
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
# Agronomic Advisories & Recommendations
# ---------------------------------------------------------------------------

DISEASE_INTERVENTIONS = {
    "early_blight": {
        "title": "Early Blight (Alternaria solani)",
        "action": "Apply targeted protectant fungicide (Mancozeb or Chlorothalonil). Prune lower diseased foliage to arrest spore splash.",
    },
    "late_blight": {
        "title": "Late Blight (Phytophthora infestans)",
        "action": "Emergency protocol: Apply systemic fungicide (Cymoxanil or Metalaxyl). Quarantine affected field sections to prevent epidemic spread.",
    },
    "black_rot": {
        "title": "Black Rot",
        "action": "Apply Myclobutanil or copper-based fungicide. Clear mummified fruit and prune infected canes.",
    },
    "bacterial_spot": {
        "title": "Bacterial Spot (Xanthomonas)",
        "action": "Apply fixed copper bactericide tank-mixed with Mancozeb. Avoid overhead sprinkler irrigation.",
    },
    "powdery_mildew": {
        "title": "Powdery Mildew",
        "action": "Spray potassium bicarbonate or horticultural neem oil extract. Improve canopy airflow via selective pruning.",
    },
    "rust": {
        "title": "Foliar Rust",
        "action": "Apply Triazole or Strobilurin-class fungicide. Remove alternate host weeds along field margins.",
    },
}


def get_disease_action(raw_class, is_healthy):
    if is_healthy:
        return "Foliar health nominal. Maintain prophylactic monitoring and clean sanitation practices."
    raw_lower = raw_class.lower()
    for key, data in DISEASE_INTERVENTIONS.items():
        if key in raw_lower:
            return data["action"]
    return "Foliar disease detected. Consult local extension agronomist for targeted anti-fungal treatment."


def get_ndvi_action(label):
    if label == "severe_stress":
        return "Canopy stress critical. Inspect irrigation drip pressure, verify root-zone moisture, and run an electrical conductivity (EC) soil salinity test."
    if label == "moderate_stress":
        return "Mild canopy suppression detected. Schedule supplementary irrigation and test for nitrogen or micro-nutrient deficiency."
    return "Canopy photosynthetic vigour is optimal. Maintain standard irrigation schedule."


def get_pest_action(pred_class, pest_risk_score):
    if pred_class == "high" or pest_risk_score >= 0.6:
        return "High outbreak probability. Deploy pheromone monitoring traps (10 units/ha). Prepare bio-pesticide application (Bacillus thuringiensis)."
    if pred_class == "medium" or pest_risk_score >= 0.3:
        return "Moderate pest pressure expected. Increase field scouting frequency to twice weekly."
    return "Pest pressure baseline nominal. Maintain routine surveillance and preserve natural beneficial predator insects."


def generate_report(disease, ndvi, pest, score, bucket_label):
    import datetime
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    d_act = get_disease_action(disease["raw_class"], disease["is_healthy"])
    n_act = get_ndvi_action(ndvi["label"])
    p_act = get_pest_action(pest["pred_class"], pest["pest_risk_score"])

    return f"""================================================================================
CROPSENSE AI — MULTI-MODAL CROP HEALTH ASSESSMENT REPORT
Generated: {now_str}
================================================================================

OVERALL VERDICT
--------------------------------------------------------------------------------
Composite Health Score : {score:.3f} / 1.000
Status Tier            : {bucket_label.upper()}
Evaluation Formula     : composite = 1/3*(disease) + 1/3*(ndvi) + 1/3*(1 - pest)

MODULE BREAKDOWN
--------------------------------------------------------------------------------
1. Visual Foliar Pathology (MobileNetV2 CNN)
   - Diagnosis         : {disease['display_class']}
   - Softmax Confidence: {disease['confidence']:.2%}
   - Disease Score     : {disease['disease_score']:.3f}
   - Action            : {d_act}

2. Canopy Multispectral Stress (Random Forest NDVI)
   - Classification    : {ndvi['label'].replace('_', ' ').title()}
   - Confidence        : {ndvi['confidence']:.2%}
   - NDVI Score        : {ndvi['ndvi_score']:.3f}
   - Action            : {n_act}

3. Pest Risk Forecast (Temporal Meteorological LSTM)
   - Forecast Tier     : {pest['pred_class'].title()} Risk
   - Confidence        : {pest['confidence']:.2%}
   - Pest Risk Score   : {pest['pest_risk_score']:.3f} (inverted in fusion)
   - Action            : {p_act}

AGRONOMIC ACTION PROTOCOL
--------------------------------------------------------------------------------
- Foliar Pathology   : {d_act}
- Canopy / Hydration : {n_act}
- Pest Surveillance  : {p_act}

================================================================================
CropSense AI · Final-Year B.Tech Project · Multi-Modal Agricultural Intelligence
================================================================================
"""


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


def load_imagenet_model():
    """Load stock MobileNetV2 pretrained on ImageNet for out-of-distribution sanity checking."""
    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
    return MobileNetV2(weights="imagenet", include_top=True)
