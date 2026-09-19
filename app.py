"""CropSense AI — composite crop-health dashboard over three trained models."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

import core

st.set_page_config(page_title="CropSense AI", page_icon="🌿", layout="wide")

GREEN, OLIVE, EARTH, AMBER, RED = "#2E7D32", "#6B8E23", "#6D4C41", "#E8A33D", "#C62828"

st.markdown(
    """
    <style>
      .stApp { background: linear-gradient(180deg, #FAF8F3 0%, #F2F0E6 100%); }
      h1, h2, h3 { color: #1B4D22; }
      .cs-hero { background: linear-gradient(100deg, #2E7D32 0%, #6B8E23 100%);
                 padding: 1.4rem 1.8rem; border-radius: 12px; color: #fff;
                 margin-bottom: 1.2rem; }
      .cs-hero h1 { color: #fff; margin: 0; font-size: 2.1rem; }
      .cs-hero p  { margin: .35rem 0 0; opacity: .92; }
      .cs-card { background: #fff; border: 1px solid #DCD7C7; border-left: 5px solid #6B8E23;
                 border-radius: 10px; padding: 1rem 1.1rem; height: 100%; }
      .cs-card .lbl { font-size: .78rem; text-transform: uppercase; letter-spacing: .06em;
                      color: #6D4C41; }
      .cs-card .val { font-size: 2rem; font-weight: 700; color: #1B4D22; line-height: 1.2; }
      .cs-card .sub { font-size: .85rem; color: #6D4C41; }
      .stTabs [data-baseweb="tab-list"] { gap: .4rem; }
      .stTabs [data-baseweb="tab"] { background: #EDE8DC; border-radius: 8px 8px 0 0;
                                     padding: 0 1.1rem; }
      .stTabs [aria-selected="true"] { background: #2E7D32 !important; color: #fff !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """<div class="cs-hero"><h1>🌿 CropSense AI</h1>
    <p>Fusing leaf-disease vision, NDVI field stress and pest-risk forecasting
    into one composite crop-health score.</p></div>""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Cached loaders — models stay in memory across tab switches and reruns.
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading disease model…")
def disease_model():
    return core.load_disease_model()


@st.cache_resource(show_spinner="Loading NDVI model…")
def ndvi_model():
    return core.load_ndvi_model()


@st.cache_resource(show_spinner="Loading pest model…")
def pest_model():
    return core.load_pest_model()


@st.cache_resource
def sequences():
    return core.load_sequences()


def card(label, value, sub, accent=OLIVE):
    st.markdown(
        f"""<div class="cs-card" style="border-left-color:{accent}">
        <div class="lbl">{label}</div><div class="val">{value}</div>
        <div class="sub">{sub}</div></div>""",
        unsafe_allow_html=True,
    )


def prob_bars(labels, values, colors, title):
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h",
                           marker_color=colors,
                           text=[f"{v:.1%}" for v in values], textposition="auto"))
    fig.update_layout(title=title, height=max(180, 46 * len(labels)),
                      margin=dict(l=8, r=8, t=44, b=8),
                      xaxis=dict(range=[0, 1], tickformat=".0%", title=None),
                      yaxis=dict(autorange="reversed"),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch")


tab1, tab2, tab3, tab4 = st.tabs(
    ["🍃 Disease Detection", "🛰️ NDVI Crop Stress", "🦗 Pest Risk", "📊 Fusion Dashboard"]
)

# ==========================================================================
# Module 1 — Disease Detection
# ==========================================================================
with tab1:
    st.subheader("Module 1 · Leaf Disease Detection")
    st.caption("MobileNetV2 transfer-learning CNN · 38 crop/disease classes · 224×224 RGB input")
    st.warning(
        "Trained and validated on lab-condition leaf images (PlantVillage dataset); "
        "accuracy on real-world field photos is lower (~30%) due to domain shift — "
        "see project report for details.",
        icon="⚠️",
    )

    left, right = st.columns([1, 1.25], gap="large")
    with left:
        upload = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])
        samples = sorted((core.MODELS_DIR.parent / "sample_images").glob("*.jpg"))
        pick = st.selectbox(
            "…or try a labelled PlantVillage sample",
            ["—"] + [core.pretty_class(p.stem) for p in samples],
        )
        image = None
        if upload is not None:
            image = Image.open(upload)
            source = "your upload"
        elif pick != "—":
            image = Image.open(samples[[core.pretty_class(p.stem) for p in samples].index(pick)])
            source = f"sample · true label {pick}"
        if image is not None:
            st.image(image, caption=f"Input ({source}) — resized to 224×224", width=320)

    with right:
        if image is None:
            st.info("Upload a leaf image or pick a sample to run the classifier.", icon="🍃")
        elif st.button("Analyse leaf", type="primary", key="run_disease"):
            st.session_state.disease = core.predict_disease(disease_model(), image)

        result = st.session_state.get("disease")
        if result and image is not None:
            c1, c2, c3 = st.columns(3)
            with c1:
                card("Predicted class", result["display_class"],
                     "argmax over 38-way softmax", GREEN if result["is_healthy"] else RED)
            with c2:
                card("Confidence", f"{result['confidence']:.1%}", "model probability", EARTH)
            with c3:
                card("disease_score", f"{result['disease_score']:.3f}",
                     "confidence if healthy, else 1 − confidence",
                     GREEN if result["disease_score"] >= 0.5 else RED)
            st.write("")
            labels = [n for n, _ in result["top_k"]]
            values = [p for _, p in result["top_k"]]
            prob_bars(labels, values, [GREEN] + [OLIVE] * (len(labels) - 1),
                      "Top-5 class probabilities")
            st.caption(
                "Preprocessing: resize to 224×224 RGB, pixels passed through as raw 0–255. "
                "The saved model rescales internally (`x / 127.5 − 1`), so no manual "
                "normalisation is applied here."
            )

# ==========================================================================
# Module 2 — NDVI Crop Stress
# ==========================================================================
with tab2:
    st.subheader("Module 2 · NDVI Crop Stress")
    st.caption("Random Forest (200 trees) over six NDVI summary statistics from field imagery")

    for key, val in core.NDVI_DEFAULTS.items():
        st.session_state.setdefault(f"ndvi_{key}", val)

    def load_example(which):
        for k, v in core.NDVI_EXAMPLES[which].items():
            st.session_state[f"ndvi_{k}"] = v
        st.session_state.ndvi_pending = True

    b1, b2, _ = st.columns([1, 1, 2])
    b1.button("🌱 Healthy field example", key="ex_healthy", width="stretch",
              on_click=load_example, args=("healthy",))
    b2.button("🥀 Stressed field example", key="ex_stressed", width="stretch",
              on_click=load_example, args=("stressed",))

    left, right = st.columns([1, 1.25], gap="large")
    with left:
        st.markdown("**NDVI summary statistics**")
        for feat in core.NDVI_FEATURES:
            st.slider(feat, 0.0, 1.0, step=0.01, key=f"ndvi_{feat}")
        run = st.button("Analyse field", type="primary", key="run_ndvi")

    values = {f: st.session_state[f"ndvi_{f}"] for f in core.NDVI_FEATURES}
    pending = st.session_state.pop("ndvi_pending", False)
    if run or pending:
        st.session_state.ndvi = core.predict_ndvi(ndvi_model(), values)

    with right:
        result = st.session_state.get("ndvi")
        if not result:
            st.info("Set the six NDVI statistics (or load an example) and press "
                    "**Analyse field**.", icon="🛰️")
        else:
            tone = {"healthy": GREEN, "moderate_stress": AMBER, "severe_stress": RED}[result["label"]]
            c1, c2, c3 = st.columns(3)
            with c1:
                card("Stress class", result["label"].replace("_", " ").title(),
                     "Random Forest vote", tone)
            with c2:
                card("Confidence", f"{result['confidence']:.1%}", "share of trees", EARTH)
            with c3:
                card("ndvi_score", f"{result['ndvi_score']:.2f}",
                     "healthy 1.0 · moderate 0.5 · severe 0.1", tone)
            st.write("")
            order = ["healthy", "moderate_stress", "severe_stress"]
            prob_bars([o.replace("_", " ").title() for o in order],
                      [result["probs"][o] for o in order],
                      [GREEN, AMBER, RED], "Class probabilities")
            st.caption(
                "NDVI ranges 0–1 here; higher mean greenness indicates more vigorous canopy. "
                "This forest keys strongly off **spread** (`ndvi_std`, `ndvi_p25`, `ndvi_p75`) — "
                "a patchy field scores worse than a uniformly moderate one."
            )

# ==========================================================================
# Module 3 — Pest Risk
# ==========================================================================
with tab3:
    st.subheader("Module 3 · Pest Risk Forecast")
    data = sequences()
    st.caption(
        f"LSTM (16 units) over a {data['window_size']}-week rolling window of "
        f"{len(data['feature_names'])} weather/pest variables → low / medium / high risk"
    )
    st.warning(
        "Trained on a limited historical dataset (~5 seasons, one region); documented as a "
        "data-volume-limited baseline in the project report, included per the module's "
        "required architecture.",
        icon="⚠️",
    )

    labels = [core.pretty_label(s["label"]) for s in data["samples"]]
    choice = st.selectbox("Historical scenario (real sequences from the test set)", labels)
    sample = data["samples"][labels.index(choice)]

    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.markdown(f"**Input window — {choice}**")
        st.dataframe(
            pd.DataFrame(sample["sequence"], columns=data["feature_names"],
                         index=[f"Week −{data['window_size'] - i}" for i in range(data["window_size"])]),
            width="stretch",
        )
        if st.button("Forecast pest risk", type="primary", key="run_pest"):
            st.session_state.pest = core.predict_pest(pest_model(), sample["sequence"])
            st.session_state.pest_truth = sample["actual_class"]
            st.session_state.pest_label = choice

    with right:
        result = st.session_state.get("pest")
        if not result:
            st.info("Pick a historical week and press **Forecast pest risk**.", icon="🦗")
        else:
            if st.session_state.get("pest_label") != choice:
                st.caption(f"Showing result for **{st.session_state.get('pest_label')}** — "
                           "press the button to re-run on the selected week.")
            tone = {"low": GREEN, "medium": AMBER, "high": RED}[result["pred_class"]]
            truth = st.session_state.get("pest_truth")
            c1, c2, c3 = st.columns(3)
            with c1:
                card("Predicted risk", result["pred_class"].title(),
                     f"{result['confidence']:.1%} confidence", tone)
            with c2:
                card("Ground truth", truth.title(), "actual recorded class",
                     GREEN if truth == result["pred_class"] else EARTH)
            with c3:
                card("pest_risk_score", f"{result['pest_risk_score']:.3f}",
                     "0·P(low) + 0.5·P(med) + 1·P(high)", tone)
            st.write("")
            prob_bars([c.title() for c in core.PEST_CLASSES],
                      [result["probs"][c] for c in core.PEST_CLASSES],
                      [GREEN, AMBER, RED], "Risk-class probabilities")
            st.caption(
                "Ground truth is shown for demo transparency. The expected-risk score is "
                "continuous, so a near-tie between medium and high still moves the fusion "
                "score — unlike a hard class label."
            )

# ==========================================================================
# Fusion
# ==========================================================================
with tab4:
    st.subheader("Composite Crop-Health Score")
    st.caption("Equal-weighted fusion of all three module outputs")

    disease, ndvi, pest = (st.session_state.get(k) for k in ("disease", "ndvi", "pest"))
    missing = [name for name, r in
               [("Module 1 · Disease Detection", disease),
                ("Module 2 · NDVI Crop Stress", ndvi),
                ("Module 3 · Pest Risk", pest)] if not r]

    if missing:
        st.info("Waiting on: " + "  ·  ".join(missing), icon="⏳")
        st.markdown("Run each module on its own tab — the fusion score needs all three "
                    "signals, and is deliberately **not** computed from defaults.")

    left, right = st.columns([1.15, 1], gap="large")
    with left:
        if missing:
            st.markdown(
                f"""<div class="cs-card" style="border-left-color:{EARTH};text-align:center;
                padding:3rem 1rem"><div class="val" style="color:#9A9384">—</div>
                <div class="sub">{3 - len(missing)} of 3 modules ready</div></div>""",
                unsafe_allow_html=True,
            )
        else:
            score = core.composite_score(disease["disease_score"], ndvi["ndvi_score"],
                                         pest["pest_risk_score"])
            label, colour = core.bucket(score)
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                number={"valueformat": ".3f", "font": {"size": 46, "color": colour}},
                title={"text": f"<b>{label}</b>", "font": {"size": 24, "color": colour}},
                gauge={
                    "axis": {"range": [0, 1], "tickwidth": 1, "tickcolor": EARTH},
                    "bar": {"color": colour, "thickness": 0.7},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 1,
                    "bordercolor": "#DCD7C7",
                    "steps": [
                        {"range": [0, 0.33], "color": "#F6DADA"},
                        {"range": [0.33, 0.66], "color": "#FBEFD8"},
                        {"range": [0.66, 1.0], "color": "#DDEBDC"},
                    ],
                },
            ))
            fig.update_layout(height=360, margin=dict(l=20, r=20, t=70, b=10),
                              paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width="stretch")
            st.latex(r"\text{composite} = \tfrac{1}{3}\,\text{disease} + "
                     r"\tfrac{1}{3}\,\text{ndvi} + \tfrac{1}{3}\,(1 - \text{pest risk})")

    with right:
        st.markdown("**Component scores**")
        rows = [
            ("Module 1 · Disease Detection", disease, "disease_score",
             lambda r: r["display_class"]),
            ("Module 2 · NDVI Crop Stress", ndvi, "ndvi_score",
             lambda r: r["label"].replace("_", " ").title()),
            ("Module 3 · Pest Risk", pest, "pest_risk_score",
             lambda r: f"{r['pred_class'].title()} risk (inverted in fusion)"),
        ]
        for title, result, key, describe in rows:
            if result:
                val = result[key]
                good = val >= 0.5 if key != "pest_risk_score" else val < 0.5
                card(title, f"{val:.3f}", f"{key} — {describe(result)}",
                     GREEN if good else AMBER)
            else:
                card(title, "—", f"{key} — run this module first", EARTH)
            st.write("")

    with st.expander("ℹ️ Why equal weights?"):
        st.markdown(
            "With three independently-trained models on three unrelated datasets, there's "
            "no statistically grounded way to derive optimal weights without labeled ground "
            "truth linking all three signals to one real outcome. Equal weighting is used as "
            "the honest, defensible default."
        )

    st.divider()
    st.caption("CropSense AI · final-year B.Tech project · models are served as trained, "
               "no retraining or fine-tuning is performed by this dashboard.")
