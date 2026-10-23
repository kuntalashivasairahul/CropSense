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
    for path in sorted((HERE / "sample_images").glob("*___*.jpg")):
        truth = path.stem
        r = core.predict_disease(model, Image.open(path))
        total = sum(p for _, p in r["top_k"])
        print(f"  {truth:24s} -> {r['display_class']:34s} {r['confidence']:6.2%} "
              f"disease_score={r['disease_score']:.3f}")
        assert r["raw_class"] in core.DISEASE_CLASSES
        assert 0.0 <= r["confidence"] <= 1.0 and total <= 1.0 + 1e-4
        probs = model.predict(core.preprocess_image(Image.open(path)), verbose=0)[0]
        expected = sum(float(p) for label, p in zip(core.DISEASE_CLASSES, probs) if "healthy" in label.lower()) / sum(float(p) for p in probs)
        assert abs(r["disease_score"] - expected) < 1e-9
        # Class-index mapping must line up with the real label.
        assert r["raw_class"] == truth, f"class mapping off: {truth} -> {r['raw_class']}"
        # Grad-CAM overlay must be generated as a 224x224 RGB image
        assert r["cam_image"] is not None and isinstance(r["cam_image"], Image.Image)
        assert r["cam_image"].size == (224, 224)
        # Sanity dict must be present
        assert "sanity" in r

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
    entries = json.loads((core.MODELS_DIR / "module2_samples/manifest.json").read_text())
    counts = {label: [0, 0] for label in core.NDVI_SCORE_MAP}
    for entry in entries:
        result = core.predict_ndvi(model, entry)
        assert abs(sum(result["probs"].values()) - 1.0) < 1e-6
        assert result["ndvi_score"] == core.NDVI_SCORE_MAP[result["label"]]
        counts[entry["label"]][0] += result["label"] == entry["label"]
        counts[entry["label"]][1] += 1
    assert all(total > 0 for _, total in counts.values()), "Missing a fixture class"
    for label, (correct, total) in counts.items():
        print(f"  {label}: {correct}/{total} fixture labels matched (diagnostic, not generalization accuracy)")
    print("  Known canopy quality limits remain visible; passing the runtime contract does not establish accuracy.")


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

    # Advisory & report checks
    act_d = core.get_disease_action("Potato___Early_blight", False)
    act_h = core.get_disease_action("Tomato___healthy", True)
    assert "confirm" in act_d.lower() and "treatment" in act_d.lower()
    assert "monitoring" in act_h.lower()
    assert "does not identify" in core.get_ndvi_action("severe_stress").lower()
    assert "does not prescribe" in core.get_pest_action("high", 0.8).lower()

    dummy_d = {"display_class": "Tomato — Healthy", "raw_class": "Tomato___healthy",
               "confidence": 0.99, "disease_score": 0.99, "is_healthy": True}
    dummy_n = {"label": "healthy", "confidence": 0.85, "ndvi_score": 1.0}
    dummy_p = {"pred_class": "low", "confidence": 0.90, "pest_risk_score": 0.05}
    assessment = core.composite_assessment(dummy_d, dummy_n, dummy_p)
    rep = core.generate_report(dummy_d, dummy_n, dummy_p, assessment["score"], assessment["label"])
    assert "SIMULATED MULTI-SIGNAL DEMONSTRATION" in rep
    assert "HEALTHY" in rep
    print(f"  composite(1,1,0)=1.0  composite(0,0,1)=0.0  "
          f"composite(.5,.5,.5)={base:.3f} -> {core.bucket(base)[0]}")
    print("  advisory rules and report generation validated")


def check_model_path(tmp_path=None):
    """Verify local integrity and offline restoration without touching the release."""
    import hashlib
    import shutil
    import tempfile
    from unittest.mock import patch
    original = core.MODELS_DIR
    assert core.model_path("module1_cnn.keras") == str(original / "module1_cnn.keras")
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory) / "models"
        folder.mkdir()
        for name in ["artifact_manifest.json", "release_bundle.zip", "release_bundle.sha256"]:
            shutil.copyfile(original / name, folder / name)
        (folder / "module1_cnn.keras").write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
        with patch.object(core, "MODELS_DIR", folder):
            restored = Path(core.model_path("module1_cnn.keras"))
            manifest = json.loads((folder / "artifact_manifest.json").read_text())
            assert hashlib.sha256(restored.read_bytes()).hexdigest() == manifest["models"][restored.name]["sha256"]
    print("  local release integrity and LFS pointer restoration verified without Hub fallback")


def check_ood_sanity():
    """Validates color coverage and ImageNet top-5 plant matching."""
    im_model = core.load_imagenet_model()

    # 1. Real leaf images (must NOT flag)
    leaves = ["Tomato___healthy.jpg", "Potato___Early_blight.jpg"]
    for leaf in leaves:
        p = HERE / "sample_images" / leaf
        img = Image.open(p)
        res = core.check_leaf_sanity(img, im_model)
        top5_str = ", ".join(f"{n} ({pr:.1%})" for n, pr, _ in res["imagenet_top_5"][:3])
        print(f"  {leaf:24s} -> cov={res['color_coverage_pct']:.1f}% plant_match={res['has_plant_match']} "
              f"is_flagged={res['is_flagged']} [top: {top5_str}]")
        assert not res["color_flagged"], f"{leaf} color falsely flagged"
        assert res["has_plant_match"], f"{leaf} missed ImageNet plant match"
        assert not res["is_flagged"], f"{leaf} falsely flagged as OOD"

    # 2. Adversarial non-leaf image (MUST flag)
    adv_path = HERE / "sample_images" / "adversarial_non_leaf.jpg"
    assert adv_path.exists(), "adversarial fixture missing"
    adv_img = Image.open(adv_path)
    adv_res = core.check_leaf_sanity(adv_img, im_model)
    top5_str = ", ".join(f"{n} ({pr:.1%})" for n, pr, _ in adv_res["imagenet_top_5"][:3])
    print(f"  {'adversarial_non_leaf':24s} -> cov={adv_res['color_coverage_pct']:.1f}% "
          f"plant_match={adv_res['has_plant_match']} is_flagged={adv_res['is_flagged']} [top: {top5_str}]")
    assert adv_res["color_flagged"], "adversarial image color check did not flag"
    assert not adv_res["has_plant_match"], "adversarial image unexpectedly matched plant class"
    assert adv_res["is_flagged"], "adversarial image was not flagged as OOD"
    apple = core.check_leaf_sanity(Image.open(HERE / "sample_images/Apple___Black_rot.jpg"), im_model)
    print(f"  All three labelled leaf fixtures: {int(apple['is_flagged'])}/3 heuristic false alarms; not an OOD benchmark.")
    print("  The non-leaf fixture is flagged; a real diseased leaf can also be flagged.")


if __name__ == "__main__":
    for name, fn in [("Model resolution", check_model_path),
                     ("Module 1 (disease CNN)", check_disease),
                     ("Module 1 (OOD sanity checks)", check_ood_sanity),
                     ("Module 2 (NDVI RF)", check_ndvi),
                     ("Module 3 (pest LSTM)", check_pest),
                     ("Fusion", check_fusion)]:
        print(f"\n{name}")
        fn()
    print("\nAll checks passed.")
