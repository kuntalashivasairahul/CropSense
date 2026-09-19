"""Runnable sanity check for CropSense AI: `python test_core.py`.

Exercises the real inference path in core.py against known inputs.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

import core

HERE = Path(__file__).parent


def check_disease():
    model = core.load_disease_model()
    assert len(core.DISEASE_CLASSES) == 38, len(core.DISEASE_CLASSES)
    assert model.output_shape[-1] == 38, model.output_shape

    # Ground truth is encoded in each fixture's filename.
    for path in sorted((HERE / "sample_images").glob("*.jpg")):
        truth = path.stem
        r = core.predict_disease(model, Image.open(path))
        total = sum(p for _, p in r["top_k"])
        print(f"  {truth:24s} -> {r['display_class']:34s} {r['confidence']:6.2%} "
              f"disease_score={r['disease_score']:.3f}")
        assert r["raw_class"] in core.DISEASE_CLASSES
        assert 0.0 <= r["confidence"] <= 1.0 and total <= 1.0 + 1e-4
        # Score formula: healthy -> confidence, diseased -> 1 - confidence.
        expected = r["confidence"] if r["is_healthy"] else 1.0 - r["confidence"]
        assert abs(r["disease_score"] - expected) < 1e-9
        # Class-index mapping must line up with the real label.
        assert r["raw_class"] == truth, f"class mapping off: {truth} -> {r['raw_class']}"

    # Full softmax sums to 1.
    probs = model.predict(core.preprocess_image(Image.open(path)), verbose=0)[0]
    assert abs(probs.sum() - 1.0) < 1e-4, probs.sum()

    # Preprocessing guard: the model rescales internally, so feeding 0-1 input
    # must give a materially different answer than the correct 0-255 input.
    wrong = model.predict(core.preprocess_image(Image.open(path)) / 255.0, verbose=0)[0]
    assert np.abs(wrong - probs).max() > 0.1, "0-255 vs 0-1 indistinguishable?"


def check_ndvi():
    model = core.load_ndvi_model()
    assert list(model.feature_names_in_) == core.NDVI_FEATURES, model.feature_names_in_

    for name, expected in [("healthy", "healthy"), ("stressed", "severe_stress")]:
        r = core.predict_ndvi(model, core.NDVI_EXAMPLES[name])
        print(f"  {name:9s} example -> {r['label']:15s} {r['confidence']:6.2%} "
              f"ndvi_score={r['ndvi_score']}")
        assert abs(sum(r["probs"].values()) - 1.0) < 1e-6
        assert r["label"] == expected, f"{name} example gave {r['label']}"
        assert r["ndvi_score"] == core.NDVI_SCORE_MAP[expected]


def check_pest():
    model = core.load_pest_model()
    data = core.load_sequences()
    n_feat = len(data["feature_names"])
    assert model.input_shape[1:] == (data["window_size"], n_feat), model.input_shape

    hits = 0
    for s in data["samples"]:
        r = core.predict_pest(model, s["sequence"])
        hits += r["pred_class"] == s["actual_class"]
        assert abs(sum(r["probs"].values()) - 1.0) < 1e-4
        assert 0.0 <= r["pest_risk_score"] <= 1.0
        # Expected-risk score must be consistent with the probabilities.
        p = r["probs"]
        assert abs(r["pest_risk_score"] - (0.5 * p["medium"] + 1.0 * p["high"])) < 1e-9
    print(f"  {len(data['samples'])} real sequences, all probs sum to 1.0, "
          f"argmax matches ground truth on {hits}/{len(data['samples'])}")


def check_fusion():
    assert abs(core.composite_score(1.0, 1.0, 0.0) - 1.0) < 1e-9
    assert abs(core.composite_score(0.0, 0.0, 1.0) - 0.0) < 1e-9
    assert abs(core.composite_score(0.5, 0.5, 0.5) - 0.5) < 1e-9
    # Monotonic in the right direction for each input.
    base = core.composite_score(0.5, 0.5, 0.5)
    assert core.composite_score(0.9, 0.5, 0.5) > base       # healthier leaf -> up
    assert core.composite_score(0.5, 1.0, 0.5) > base       # healthier NDVI -> up
    assert core.composite_score(0.5, 0.5, 0.9) < base       # more pest risk -> down
    assert core.bucket(0.9)[0] == "Healthy"
    assert core.bucket(0.5)[0] == "Moderate Risk"
    assert core.bucket(0.1)[0] == "Severe Risk"
    print(f"  composite(1,1,0)=1.0  composite(0,0,1)=0.0  "
          f"composite(.5,.5,.5)={base:.3f} -> {core.bucket(base)[0]}")


if __name__ == "__main__":
    for name, fn in [("Module 1 (disease CNN)", check_disease),
                     ("Module 2 (NDVI RF)", check_ndvi),
                     ("Module 3 (pest LSTM)", check_pest),
                     ("Fusion", check_fusion)]:
        print(f"\n{name}")
        fn()
    print("\nAll checks passed.")
