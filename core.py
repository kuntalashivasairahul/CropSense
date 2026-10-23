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


MAX_IMAGE_PIXELS = 1_048_576


def validate_image(image):
    """Check dimensions before decoding or allocating full-resolution arrays."""
    width, height = image.size
    if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
        raise ValueError(f"Image must contain 1–{MAX_IMAGE_PIXELS:,} pixels; resize it before upload.")


def preprocess_image(pil_image):
    """Resize to 224x224 RGB and return a raw 0-255 float batch.

    NO manual rescaling: the saved model's own graph starts with
    `x / 127.5 - 1.0` (MobileNetV2 preprocess_input) immediately after the
    input layer, so dividing here would double-scale and silently corrupt
    every prediction.
    """
    validate_image(pil_image)
    img = pil_image.convert("RGB").resize(IMG_SIZE)
    return np.expand_dims(np.asarray(img, dtype="float32"), axis=0)


def generate_gradcam(model, pil_image, class_idx=None, alpha=0.45):
    """Compute Grad-CAM saliency map overlay on the input leaf photo.

    Extracts gradients from the MobileNetV2 backbone's final activation layer
    with respect to the target class probability. The map shows influence,
    not biological correctness or evidence that a lesion was detected.
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
        # The outer classifier graph rescales before calling the backbone.
        conv_outputs = base(x / 127.5 - 1.0, training=False)
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
    # Blend only where the selected class has positive activation.
    alpha_map = np.clip(h_arr * alpha, 0.0, 1.0)[..., np.newaxis]
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
    validate_image(pil_image)
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
    validate_image(pil_image)
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
    """Heuristic image check; flags are not calibrated leaf-eligibility decisions."""
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


def validate_probabilities(values, count):
    probabilities = np.asarray(values, dtype=float)
    if (probabilities.shape != (count,) or not np.isfinite(probabilities).all()
            or (probabilities < 0).any() or (probabilities > 1).any()
            or not np.isclose(probabilities.sum(), 1.0, atol=1e-4)):
        raise ValueError("Model returned invalid class probabilities.")
    return probabilities / probabilities.sum()


def predict_disease(model, pil_image, top_k=5, generate_cam=True, sanity_model=None):
    probs = validate_probabilities(model.predict(preprocess_image(pil_image), verbose=0)[0], len(DISEASE_CLASSES))
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
        "disease_score": float(np.clip(sum(p for name, p in zip(DISEASE_CLASSES, probs) if "healthy" in name.lower()), 0.0, 1.0)),
        # shortcut: 0.5 is a conservative demo review rule; calibrate on a fresh validation set before field use.
        "needs_review": confidence < 0.5 or sanity["is_flagged"],
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

def predict_ndvi(model, feature_values):
    """Validate the six unscaled NDVI statistics before forest inference."""
    import pandas as pd
    try:
        mean, std, low, high, p25, p75 = values = np.asarray([feature_values[f] for f in NDVI_FEATURES], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("NDVI requires six numeric statistics.") from exc
    if (not np.isfinite(values).all() or not -1 <= low <= p25 <= p75 <= high <= 1
            or not low <= mean <= high or std < 0
            or std ** 2 > (high - mean) * (mean - low) + 1e-8):
        raise ValueError("NDVI statistics must be finite and consistent: min ≤ p25 ≤ p75 ≤ max, mean within min/max, and feasible standard deviation.")
    x = pd.DataFrame([values], columns=NDVI_FEATURES)
    probs = validate_probabilities(model.predict_proba(x)[0], len(model.classes_))
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
    with open(MODELS_DIR / "sample_sequences.json", encoding="utf-8") as f:
        return json.load(f)


def pretty_label(label):
    """'2009.0 - Week 34.0' -> '2009 · Week 34'."""
    try:
        year, _, week = label.partition(" - ")
        return f"{int(float(year))} · Week {int(float(week.replace('Week', '').strip()))}"
    except (ValueError, AttributeError):
        return label


def predict_pest(model, sequence):
    """Raw weather plus training-scaled prior pest pressure; normalization is in the graph."""
    try:
        x = np.asarray(sequence, dtype="float32")
    except (TypeError, ValueError) as exc:
        raise ValueError("Pest sequence must contain numeric values.") from exc
    if x.shape != tuple(model.input_shape[1:]) or not np.isfinite(x).all():
        raise ValueError(f"Pest sequence must be finite with shape {tuple(model.input_shape[1:])}.")
    probs = validate_probabilities(model.predict(x[np.newaxis, ...], verbose=0)[0], len(PEST_CLASSES))
    return {
        "probs": {c: float(p) for c, p in zip(PEST_CLASSES, probs)},
        "pred_class": PEST_CLASSES[int(probs.argmax())],
        "confidence": float(probs.max()),
        "pest_risk_score": float(np.clip(np.dot(PEST_WEIGHTS, probs), 0.0, 1.0)),
    }


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------

def composite_score(disease_score, ndvi_score, pest_risk_score):
    values = np.asarray([disease_score, ndvi_score, pest_risk_score], dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Composite inputs must be finite scores between 0 and 1.")
    return float((values[0] + values[1] + (1.0 - values[2])) / 3.0)


def bucket(score):
    """-> (illustrative tier, hex colour), never a validated field verdict."""
    if not np.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Score must be finite and between 0 and 1.")
    if score >= 0.66:
        return "Healthy", "#2E7D32"
    if score >= 0.33:
        return "Moderate Risk", "#E8A33D"
    return "Severe Risk", "#C62828"


def composite_assessment(disease, ndvi, pest):
    """Keep the mean index while preventing it from hiding individual warnings."""
    components = [disease["disease_score"], ndvi["ndvi_score"], 1.0 - pest["pest_risk_score"]]
    score = composite_score(components[0], components[1], pest["pest_risk_score"])
    warnings = []
    for name, value in zip(["Leaf", "Canopy", "Pest"], components):
        if value < 0.33:
            warnings.append(f"{name}: severe model signal; the mean index does not cancel it.")
        elif value < 0.66:
            warnings.append(f"{name}: model signal needs follow-up.")
    sanity = disease.get("sanity") or {}
    if sanity.get("is_flagged"):
        warnings.append("Leaf image check needs review; the forced class may be outside model scope.")
    needs_review = disease.get("needs_review", False) or sanity.get("is_flagged", False)
    if needs_review:
        warnings.append("Leaf prediction is uncertain or flagged; confirm the input before interpretation.")
        label, color = "Needs review", "#E8A33D"
    else:
        label, color = bucket(min(components))
    return {"score": score, "label": label, "color": color, "warnings": warnings}


# ---------------------------------------------------------------------------
# Agronomic Advisories & Recommendations
# ---------------------------------------------------------------------------

def get_disease_action(raw_class, is_healthy):
    if is_healthy:
        return "The model suggests healthy foliage. Confirm the photo and field observations; continue routine monitoring."
    return f"Confirm {pretty_class(raw_class)} with a qualified local agronomist before choosing any treatment. The model does not establish the cause or severity."


def get_ndvi_action(label):
    if label == "healthy":
        return "The model predicts the below-threshold anomaly class, named healthy in training. This does not establish plant health; compare field observations and verify the image acquisition conditions."
    return "Review the RGB/NIR pair and field conditions with an agronomist. This stress class does not identify irrigation, nutrient or disease causes."


def get_pest_action(pred_class, pest_risk_score):
    return f"The model predicts {pred_class} pest pressure. Confirm recent pest observations and local scouting guidance; this historical prototype does not prescribe traps or pesticides."


def generate_report(disease, ndvi, pest, score, bucket_label, *, provenance=None):
    from datetime import datetime, timezone
    assessment = composite_assessment(disease, ndvi, pest)
    if not np.isclose(score, assessment["score"], atol=1e-8) or bucket_label != assessment["label"]:
        raise ValueError("Report score and tier must match the component assessment.")
    manifest = json.loads((MODELS_DIR / "artifact_manifest.json").read_text(encoding="utf-8"))
    identities = "\n".join(f"  {name}: SHA-256 {record['sha256']}" for name, record in manifest["models"].items())
    sources = provenance or {}
    warnings = "\n".join("- " + warning for warning in assessment["warnings"]) or "- No component heuristic warning; this does not establish accuracy."
    return f"""CROPSENSE AI — SIMULATED MULTI-SIGNAL DEMONSTRATION
Generated: {datetime.now(timezone.utc).isoformat()}

This prototype combines unrelated datasets, not measurements from one farm.
No fusion accuracy or field diagnosis has been validated. Do not select treatment from this report.

SIMULATED INDEX
Mean index: {score:.3f} / 1.000
Component warning tier: {bucket_label.upper()}
Formula: (healthy-class probability sum + NDVI class score + (1 - expected pest pressure)) / 3
The tier preserves the worst component signal or requests review; it is not the mean-index bucket.
Healthy-class probability is uncalibrated; low confidence is not evidence of health.

WARNINGS
{warnings}
- Passing the image check does not establish supported leaf/crop eligibility; flowers can pass and receive confident crop-disease classes.
- Canopy classes reflect training mask-coverage thresholds; healthy means below selected anomaly thresholds, not verified plant health.
- Released canopy forest: 0/6 bundled healthy pairs correctly recognized; exact historical holdout accuracy unverified.
- Pest LSTM: 10/19 retrospective weeks correct, 0/5 medium recall; saved logistic baseline 13/19.
- Raw-band NDVI assumes aligned 8-bit RGB/NIR; sensor reflectance calibration is not established.

INPUT PROVENANCE
Leaf: {sources.get("leaf", "Not recorded")}
Canopy: {sources.get("canopy", "Not recorded")}
Pest sequence: {sources.get("pest", "Not recorded")}
Input capture timestamps: not recorded; generation time above is not acquisition time.

MODULE RESULTS
Leaf predicted class: {disease["display_class"]}
Softmax confidence: {disease["confidence"]:.2%}
Healthy-class probability sum: {disease["disease_score"]:.3f}
Follow-up: {get_disease_action(disease["raw_class"], disease["is_healthy"])}

Canopy predicted class: {ndvi["label"]}
Model confidence: {ndvi["confidence"]:.2%}
Illustrative canopy score: {ndvi["ndvi_score"]:.3f}
Follow-up: {get_ndvi_action(ndvi["label"])}

Pest predicted class: {pest["pred_class"]}
Model confidence: {pest["confidence"]:.2%}
Expected pest pressure: {pest["pest_risk_score"]:.3f}
Follow-up: {get_pest_action(pest["pred_class"], pest["pest_risk_score"])}

MODEL IDENTITIES
Release: {manifest["release"]}
{identities}
"""


# ---------------------------------------------------------------------------
# Loaders (wrapped in @st.cache_resource by app.py)
# ---------------------------------------------------------------------------

def model_path(filename):
    """Resolve only checksum-verified release artifacts, restoring from the local bundle."""
    from scripts.release_assets import ensure_local_models
    manifest = json.loads((MODELS_DIR / "artifact_manifest.json").read_text(encoding="utf-8"))
    if filename not in manifest["models"]:
        raise ValueError(f"Unknown release model: {filename}")
    ensure_local_models(MODELS_DIR.parent)
    return str(MODELS_DIR / filename)


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
