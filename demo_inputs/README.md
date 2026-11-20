# Demo inputs for the Windows college presentation

These are unchanged copies of the project's existing fixtures, gathered in one easy-to-find folder. They are examples, not a new evaluation dataset. `manifest.json` records every original path and SHA-256.

## 1. Leaf disease: upload one JPG

In **Disease detection**, select **Manual / Custom inputs** in the scenario dropdown before uploading. Otherwise the chosen demonstration scenario deliberately keeps its bundled input.

| File in `01_leaf/` | What to show |
|---|---|
| `Tomato___healthy.jpg` | Clear reference leaf; released model predicts Tomato Healthy |
| `Potato___Early_blight.jpg` | Diseased reference leaf; released model predicts Potato Early Blight; switch to Grad-CAM |
| `Apple___Black_rot.jpg` | Another supported crop/disease example; image screening may request review even for a real leaf |
| `plantdoc_field_tomato_late_blight.jpg` | Real-world field limitation example; source label is Tomato Late Blight, not a promise of a correct prediction |
| `adversarial_non_leaf.jpg` | Non-leaf sports-car photo: show the image-screen warning and explain why forced crop-class output should not be trusted |

Press **Classify leaf** after uploading. The original source label comes from the fixture, while the app displays its prediction. Softmax confidence is not measured accuracy. Grad-CAM shows influence, not verified lesions.

## 2. Canopy stress: upload a matching pair

In **Canopy stress**, remain in **Manual / Custom inputs**. Upload `RGB.jpg` into the RGB uploader and `NIR.jpg` into the NIR uploader from the **same subfolder**. Run **Compute NDVI and classify**. NIR files are genuine bundled grayscale near-infrared bands; do not substitute a colour photo or mix folders.

| Subfolder in `02_canopy/` | Source label | Released model behavior |
|---|---|---|
| `healthy_label/` | Healthy by training mask thresholds | Predicts Moderate Stress; this illustrates the documented healthy-class weakness |
| `moderate/` | Moderate stress | Predicts Moderate Stress |
| `severe/` | Severe stress | Predicts Severe Stress; assigned canopy score 0.100 |

Inspect RGB, low-NDVI overlay, heatmap, and six summary statistics. Equal dimensions do not establish sensor alignment/calibration. Source labels reflect mask-coverage rules, not agronomist diagnoses. Original parcel IDs remain in `manifest.json`.

## 3. Pest risk: use the built-in sequence selector

No photo is needed or accepted. Read `03_pest/README.md`, select an existing historical week in the app, and press **Forecast pest outbreak**. Do not upload screenshots as weather data.

## 4. Composite: run the three modules, then download

No separate photo is needed. After all three modules have results, open **Composite** and download the text report. Changing an input clears its old result and invalidates the completed composite. Inputs are from unrelated datasets; the mean is a simulated illustration, not one farm's validated health score.

For a quick full walk-through, choose **Scenario 3 · Severe Combined Risk** instead of manual uploads: mean **0.177**, **Severe Risk**. Its High-labelled pest week predicts Medium, which is a known error. Use manual mode to demonstrate the files in this folder.

Follow [the Windows setup and offline checklist](../docs/WINDOWS_DEMO.md) before college. The app needs online setup once on the laptop; the prepared models and examples then run locally without Wi-Fi.
