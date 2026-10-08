"""CropSense AI — composite crop-health dashboard over three trained models.

Presentation layer only. All model loading, inference and fusion arithmetic
lives in core.py and is called from here unchanged.
"""

import base64
import textwrap
from html import escape
import io
import json
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import core

# --------------------------------------------------------------------------
# Design tokens & Light Agriculture Palette
# --------------------------------------------------------------------------
INK = "#0E1A12"
SECONDARY = "#5B6760"
WHITE = "#FFFFFF"
LIME = "#C6E94A"
LIME_DARK = "#3F6B12"

HEALTHY = "#2E7D32"
HEALTHY_BG = "#E8F5E9"
MODERATE = "#E0A526"
MODERATE_DARK = "#946300"
MODERATE_BG = "#FFF7E6"
SEVERE = "#E4572E"
SEVERE_DARK = "#C62828"
SEVERE_BG = "#FBE9E7"

MONO = "'IBM Plex Mono', ui-monospace, SFMono-Regular, monospace"
SANS = "'IBM Plex Sans', system-ui, sans-serif"

# Chlorophyll gradient ramp
RAMP = f"linear-gradient(90deg, {SEVERE} 0%, {MODERATE} 38%, #8BC34A 70%, {HEALTHY} 100%)"


def _favicon():
    """A 4-band chip of the canopy ramp."""
    bands = [(228, 87, 46), (224, 165, 38), (198, 233, 74), (46, 125, 50)]
    img = Image.new("RGB", (4, 1))
    img.putdata(bands)
    return img.resize((64, 64), Image.NEAREST)


st.set_page_config(
    page_title="CropSense AI",
    page_icon=_favicon(),
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------------------------------
# Icons — Lucide line paths, 24x24 grid, single stroke weight, currentColor.
# --------------------------------------------------------------------------
ICONS = {
    "microscope": '<path d="M6 18h8"/><path d="M3 22h18"/><path d="M14 22a7 7 0 1 0 0-14h-1"/>'
                  '<path d="M9 14h2"/><path d="M9 12a2 2 0 0 1-2-2V6h6v4a2 2 0 0 1-2 2Z"/>'
                  '<path d="M12 6V3a1 1 0 0 0-1-1H9a1 1 0 0 0-1 1v3"/>',
    "satellite": '<path d="M13 7 9 3 5 7l4 4"/><path d="m17 11 4 4-4 4-4-4"/>'
                 '<path d="m8 12 4 4 6-6-4-4Z"/><path d="m16 8 3-3"/>'
                 '<path d="M9 21a6 6 0 0 0-6-6"/>',
    "bug": '<path d="m8 2 1.88 1.88"/><path d="M14.12 3.88 16 2"/>'
           '<path d="M9 7.13v-1a3.003 3.003 0 1 1 6 0v1"/>'
           '<path d="M12 20c-3.3 0-6-2.7-6-6v-3a4 4 0 0 1 4-4h4a4 4 0 0 1 4 4v3c0 3.3-2.7 6-6 6"/>'
           '<path d="M12 20v-9"/><path d="M6.53 9C4.6 8.8 3 7.1 3 5"/><path d="M6 13H2"/>'
           '<path d="M3 21c0-2.1 1.7-3.9 3.8-4"/><path d="M20.97 5c0 2.1-1.6 3.8-3.5 4"/>'
           '<path d="M22 13h-4"/><path d="M17.2 17c2.1.1 3.8 1.9 3.8 4"/>',
    "gauge": '<path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/>',
    "alert": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/>'
             '<path d="M12 9v4"/><path d="M12 17h.01"/>',
    "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
    "check": '<path d="M20 6 9 17l-5-5"/>',
    "leaf": '<path d="M11 20A7 7 0 0 1 4 13a7 7 0 0 1 7-7 7 7 0 0 1 7 7v7h-7Z"/><path d="M11 20V10"/>',
    "download": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" x2="12" y1="15" y2="3"/>',
}


def icon(name, size=16, stroke=1.5):
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="currentColor" stroke-width="{stroke}" stroke-linecap="round" '
            f'stroke-linejoin="round" style="vertical-align:middle;display:inline-block;" aria-hidden="true">{ICONS.get(name, "")}</svg>')


def status_svg(tone, size=14):
    """Clean inline SVG indicator for healthy, moderate, severe, or idle."""
    if tone == HEALTHY:
        return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
                f'stroke="{HEALTHY}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" '
                f'style="vertical-align:middle;display:inline-block;" aria-hidden="true">'
                f'<polyline points="20 6 9 17 4 12"/></svg>')
    elif tone == MODERATE:
        return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
                f'stroke="{MODERATE}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" '
                f'style="vertical-align:middle;display:inline-block;" aria-hidden="true">'
                f'<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/>'
                f'<line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>')
    elif tone == SEVERE:
        return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
                f'stroke="{SEVERE}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" '
                f'style="vertical-align:middle;display:inline-block;" aria-hidden="true">'
                f'<circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>')
    else:
        return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
                f'stroke="{SECONDARY}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
                f'style="vertical-align:middle;display:inline-block;" aria-hidden="true">'
                f'<circle cx="12" cy="12" r="9"/></svg>')


CSS = """
<style>
/* Self-hosted fonts (strictly local in static/fonts) */
@font-face {
  font-family: 'IBM Plex Sans';
  src: url('/app/static/fonts/plex-sans-var.woff2') format('woff2');
  font-weight: 100 700;
  font-style: normal;
  font-display: swap;
}
@font-face {
  font-family: 'IBM Plex Mono';
  src: url('/app/static/fonts/plex-mono-400.woff2') format('woff2');
  font-weight: 400;
  font-style: normal;
  font-display: swap;
}
@font-face {
  font-family: 'IBM Plex Mono';
  src: url('/app/static/fonts/plex-mono-500.woff2') format('woff2');
  font-weight: 500;
  font-style: normal;
  font-display: swap;
}

/* CSS Tokens & Light Airy Agriculture Design */
:root {
  --canvas-grad: linear-gradient(135deg, #DDEBF3 0%, #F0F4C3 100%);
  --ink: #0E1A12;
  --secondary: #5B6760;
  --white: #FFFFFF;
  --lime: #C6E94A;
  --lime-dark: #3F6B12;
  --healthy: #2E7D32;
  --healthy-bg: #E8F5E9;
  --moderate: #E0A526;
  --moderate-dark: #946300;
  --moderate-bg: #FFF7E6;
  --severe: #E4572E;
  --severe-dark: #C62828;
  --severe-bg: #FBE9E7;
  --card-radius: 20px;
  --card-shadow: 0 4px 20px -2px rgba(14, 26, 18, 0.05), 0 2px 6px -1px rgba(14, 26, 18, 0.03);
  --card-border: 1px solid rgba(14, 26, 18, 0.07);
  --sans: 'IBM Plex Sans', system-ui, sans-serif;
  --mono: 'IBM Plex Mono', ui-monospace, SFMono-Regular, monospace;
  --ease-spring: cubic-bezier(.22, 1, .36, 1);
}

@supports (animation-timing-function: linear(0, 1)) {
  :root {
    --ease-spring: linear(0, 0.006, 0.025 2.8%, 0.101 6.1%, 0.539 18.9%, 0.721 25.3%, 0.849 31.5%, 0.937 38.1%, 0.968 41.8%, 0.991 45.7%, 1.006 50.1%, 1.015 55%, 1.017 60.7%, 1.01 68.4%, 1.004 78.4%, 1);
  }
}

/* Base Body and Fixed Diagonal Gradient Canvas */
html, body, [data-testid="stAppViewContainer"], .stApp {
  background: var(--canvas-grad) !important;
  background-attachment: fixed !important;
  font-family: var(--sans) !important;
  color: var(--ink) !important;
}

[data-testid="stMainBlockContainer"] {
  padding: 0.8rem 2.5rem 4rem !important;
  max-width: 1440px !important;
}

/* Chrome cleanups */
#MainMenu, footer, [data-testid="stToolbarActions"], [data-testid="stActionButton"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"],
[data-testid="stHeaderActionElements"] { display: none !important; }

header[data-testid="stHeader"] { background: transparent !important; pointer-events: none; }
header[data-testid="stHeader"] * { pointer-events: auto; }

/* Top App Header & Pill Bar */
.top-nav-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1.5rem;
  background: rgba(255, 255, 255, 0.75);
  backdrop-filter: blur(18px);
  -webkit-backdrop-filter: blur(18px);
  border: var(--card-border);
  border-radius: 9999px;
  padding: 0.45rem 1rem 0.45rem 1.4rem;
  box-shadow: 0 2px 12px rgba(14, 26, 18, 0.04);
  margin-bottom: 1.5rem;
}

.brand-wrap {
  display: flex;
  align-items: center;
  gap: 0.65rem;
}

.brand-mark {
  width: 28px;
  height: 28px;
  border-radius: 8px;
  background: #0E1A12;
  color: #C6E94A;
  display: flex;
  align-items: center;
  justify-content: center;
}

.brand-text {
  font-size: 1.08rem;
  font-weight: 600;
  letter-spacing: -0.015em;
  color: #0E1A12;
}

.brand-sub {
  font-size: 13px;
  color: #5B6760;
  margin-left: 0.4rem;
}

.hdr-right-wrap {
  display: flex;
  align-items: center;
  gap: 0.85rem;
}

.hdr-comp-pill {
  display: inline-flex;
  align-items: center;
  gap: 0.55rem;
  background: #FFFFFF;
  border: 1px solid rgba(14, 26, 18, 0.09);
  padding: 0.32rem 0.85rem;
  border-radius: 9999px;
  font-size: 13px;
  font-family: var(--mono);
  box-shadow: 0 1px 4px rgba(14, 26, 18, 0.03);
}

.hdr-score-num {
  font-weight: 500;
  color: #0E1A12;
}

.hdr-status-badge {
  font-size: 13px;
  padding: 0.18rem 0.55rem;
  border-radius: 9999px;
  font-weight: 500;
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
}

/* Secondary Section Header */
.sub-header-bar {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 1.5rem;
  margin-bottom: 1.4rem;
  padding: 0 0.2rem;
}

.mod-eyebrow {
  font-size: 13px;
  font-weight: 500;
  color: #5B6760;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-bottom: 0.25rem;
}

.mod-title {
  font-size: 2.1rem;
  font-weight: 400;
  letter-spacing: -0.03em;
  color: #0E1A12;
  margin: 0;
  line-height: 1.15;
}

.mod-title-anim {
  animation: titleReveal 420ms var(--ease-spring) forwards;
}

@keyframes titleReveal {
  from { opacity: 0; transform: translateY(12px); }
  to { opacity: 1; transform: translateY(0); }
}

/* Native Streamlit Bordered Container as White 20px Card */
[data-testid="stVerticalBlockBorderWrapper"] {
  background: #FFFFFF !important;
  border-radius: 20px !important;
  border: 1px solid rgba(14, 26, 18, 0.07) !important;
  box-shadow: 0 4px 20px -2px rgba(14, 26, 18, 0.05), 0 2px 6px -1px rgba(14, 26, 18, 0.03) !important;
  padding: 1.5rem 1.6rem !important;
  margin-bottom: 1.4rem !important;
  position: relative !important;
  transition: transform 240ms ease, box-shadow 240ms ease !important;
}

[data-testid="stVerticalBlockBorderWrapper"]:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 26px -4px rgba(14, 26, 18, 0.08) !important;
}

.card-anim-reveal [data-testid="stVerticalBlockBorderWrapper"] {
  animation: cardReveal 460ms var(--ease-spring) forwards !important;
}

@keyframes cardReveal {
  from {
    clip-path: inset(0 0 100% 0 round 20px);
    opacity: 0;
    transform: translateY(12px);
  }
  to {
    clip-path: inset(0 0 0% 0 round 20px);
    opacity: 1;
    transform: translateY(0);
  }
}

.card-eyebrow {
  font-size: 13px;
  font-weight: 500;
  color: #5B6760;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-bottom: 0.8rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

/* Hero Numerals & Labels */
.hero-num-wrap {
  display: flex;
  align-items: baseline;
  gap: 0.6rem;
  margin-bottom: 0.5rem;
}

.hero-num {
  font-size: 3.2rem;
  font-weight: 300;
  line-height: 1.05;
  letter-spacing: -0.04em;
  color: #0E1A12;
  font-family: var(--sans);
  font-variant-numeric: tabular-nums;
}

.hero-unit {
  font-size: 14px;
  color: #5B6760;
  font-weight: 400;
}

.verdict-title {
  font-size: 1.45rem;
  font-weight: 500;
  letter-spacing: -0.02em;
  color: #0E1A12;
  margin-bottom: 1.1rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

/* Tone Classes & Badges */
.tone-healthy { color: #2E7D32 !important; }
.tone-moderate { color: #946300 !important; }
.tone-severe { color: #C62828 !important; }
.tone-idle { color: #5B6760 !important; }

.badge-healthy { background: var(--healthy-bg); color: var(--healthy); }
.badge-moderate { background: var(--moderate-bg); color: var(--moderate-dark); }
.badge-severe { background: var(--severe-bg); color: var(--severe-dark); }
.badge-idle { background: #F4F8FA; color: #5B6760; }

.status-mark {
  display: inline-flex;
  align-items: center;
  margin-right: 0.35rem;
}

/* CSS @property and Counter for Result Number Count-Up */
@property --num-int { syntax: '<integer>'; initial-value: 0; inherits: false; }
@property --num-dec { syntax: '<integer>'; initial-value: 0; inherits: false; }

@keyframes countInt { from { --num-int: 0; } to { --num-int: var(--target-int, 0); } }
@keyframes countDec { from { --num-dec: 0; } to { --num-dec: var(--target-dec, 0); } }

.count-num {
  font-variant-numeric: tabular-nums;
  display: inline-flex;
  align-items: baseline;
}
.count-num .count-val { display: none; }
@supports (animation-timing-function: linear(0, 1)) {
  .count-num .count-fallback { display: none; }
  .count-num .count-val {
    display: inline;
    animation: countInt 500ms var(--ease-spring) forwards, countDec 500ms var(--ease-spring) forwards;
    counter-reset: int var(--num-int) dec var(--num-dec);
  }
  .count-num.is-pct .count-val::before {
    content: counter(int) "." counter(dec) "%";
  }
  .count-num.is-dec .count-val::before {
    content: counter(int) "." counter(dec, decimal-leading-zero);
  }
}

/* Buttons — Pill Shapes */
.stButton button, [data-testid="stBaseButton-primary"], [data-testid="stBaseButton-secondary"] {
  border-radius: 9999px !important;
  font-family: var(--sans) !important;
  font-size: 0.88rem !important;
  padding: 0.45rem 1.25rem !important;
  transition: transform 180ms ease, box-shadow 180ms ease, background-color 180ms ease !important;
}

.stButton button[kind="primary"], [data-testid="stBaseButton-primary"] {
  background: #0E1A12 !important;
  color: #FFFFFF !important;
  border: 1px solid #0E1A12 !important;
  font-weight: 500 !important;
}
.stButton button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {
  background: #233429 !important;
  border-color: #233429 !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 12px rgba(14, 26, 18, 0.12) !important;
}

.stButton button[kind="secondary"], [data-testid="stBaseButton-secondary"] {
  background: transparent !important;
  color: #0E1A12 !important;
  border: 1px solid rgba(14, 26, 18, 0.22) !important;
  font-weight: 400 !important;
}
.stButton button[kind="secondary"]:hover, [data-testid="stBaseButton-secondary"]:hover {
  background: rgba(14, 26, 18, 0.04) !important;
  border-color: #0E1A12 !important;
  transform: translateY(-1px) !important;
}

.stDownloadButton button {
  border-radius: 9999px !important;
  background: #0E1A12 !important;
  color: #FFFFFF !important;
  border: 1px solid #0E1A12 !important;
}

/* Nav Pills Styling */
div[data-testid="stPills"] > div {
  display: flex !important;
  flex-wrap: nowrap !important;
  overflow-x: auto !important;
  scrollbar-width: none !important;
  gap: 0.35rem !important;
}
div[data-testid="stPills"] > div::-webkit-scrollbar {
  display: none !important;
}
div[data-testid="stButtonGroup"] button, div[data-testid="stPills"] button {
  border-radius: 9999px !important;
  border: 1px solid transparent !important;
  font-family: var(--sans) !important;
  font-size: 0.88rem !important;
  color: #5B6760 !important;
  background: transparent !important;
  padding: 0.35rem 1.1rem !important;
  transition: all 180ms ease !important;
}
div[data-testid="stButtonGroup"] button[aria-checked="true"],
div[data-testid="stButtonGroup"] button[aria-pressed="true"],
div[data-testid="stPills"] button[aria-checked="true"],
div[data-testid="stPills"] button[aria-pressed="true"] {
  background: #0E1A12 !important;
  color: #FFFFFF !important;
  border-color: #0E1A12 !important;
  font-weight: 500 !important;
}
div[data-testid="stButtonGroup"] button:hover, div[data-testid="stPills"] button:hover {
  color: #0E1A12 !important;
  background: rgba(14, 26, 18, 0.05) !important;
}

/* Form inputs & Uploaders */
[data-testid="stFileUploaderDropzone"] {
  background: #F8FBFC !important;
  border: 1.5px dashed rgba(14, 26, 18, 0.16) !important;
  border-radius: 16px !important;
  padding: 1.2rem !important;
}
[data-baseweb="select"] > div {
  background: #FFFFFF !important;
  border: 1px solid rgba(14, 26, 18, 0.14) !important;
  border-radius: 12px !important;
}
[data-testid="stExpander"] details {
  background: #FFFFFF !important;
  border: var(--card-border) !important;
  border-radius: 16px !important;
}

/* Gauge SVG Styling */
.gauge-card-wrap {
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 1rem 0 0.5rem;
}
.gauge-svg {
  width: 100%;
  max-width: 320px;
  overflow: visible;
}
.gauge-fill {
  animation: gaugeSweep 800ms var(--ease-spring) forwards;
}
@keyframes gaugeSweep {
  from { stroke-dashoffset: 398; }
  to { stroke-dashoffset: var(--gauge-offset); }
}
.gauge-center-val {
  position: absolute;
  top: 48%;
  left: 50%;
  transform: translate(-50%, -40%);
  text-align: center;
}
.gauge-num {
  font-size: 3.2rem;
  font-weight: 300;
  line-height: 1;
  letter-spacing: -0.04em;
  color: #0E1A12;
  font-family: var(--sans);
  font-variant-numeric: tabular-nums;
}
.gauge-status-chip {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  font-size: 0.95rem;
  font-weight: 500;
  margin-top: 0.35rem;
}
.dial-chips-row {
  display: flex;
  justify-content: center;
  gap: 0.6rem;
  margin-top: 0.9rem;
  flex-wrap: wrap;
}
.dial-chip {
  font-size: 13px;
  padding: 0.32rem 0.8rem;
  border-radius: 9999px;
  background: #F4F8FA;
  color: var(--secondary);
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  animation: chipSlideUp 400ms var(--ease-spring) backwards;
}
.dial-chip:nth-child(1) { animation-delay: 50ms; }
.dial-chip:nth-child(2) { animation-delay: 100ms; }
.dial-chip:nth-child(3) { animation-delay: 150ms; }
@keyframes chipSlideUp {
  from { transform: translateY(8px); opacity: 0; }
  to { transform: translateY(0); opacity: 1; }
}

/* Horizontal Probability Bars */
.bars-container {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  margin-top: 0.4rem;
}
.bar-row {
  display: flex;
  flex-direction: column;
  gap: 0.28rem;
}
.bar-header {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  font-size: 0.84rem;
}
.bar-label { color: #0E1A12; font-weight: 400; }
.bar-val { font-family: var(--mono); color: #5B6760; font-size: 13px; }
.bar-track {
  width: 100%;
  height: 8px;
  background: #EDF2F4;
  border-radius: 9999px;
  overflow: hidden;
}
.bar-fill {
  height: 100%;
  border-radius: 9999px;
  animation: barWipe 450ms var(--ease-spring) backwards;
}
.bar-row:nth-child(1) .bar-fill { animation-delay: 40ms; }
.bar-row:nth-child(2) .bar-fill { animation-delay: 90ms; }
.bar-row:nth-child(3) .bar-fill { animation-delay: 140ms; }
.bar-row:nth-child(4) .bar-fill { animation-delay: 190ms; }
.bar-row:nth-child(5) .bar-fill { animation-delay: 240ms; }
@keyframes barWipe {
  from { clip-path: inset(0 100% 0 0 round 9999px); }
  to { clip-path: inset(0 0 0 0 round 9999px); }
}

/* Pest Area Chart SVG */
.pest-chart-wrap {
  width: 100%;
  position: relative;
  margin: 0.8rem 0 0.4rem;
}
.pest-chart-svg {
  width: 100%;
  height: 160px;
  overflow: visible;
}
.chart-line-anim {
  stroke-dasharray: 600;
  stroke-dashoffset: 600;
  animation: chartDraw 650ms var(--ease-spring) forwards;
}
@keyframes chartDraw {
  from { stroke-dashoffset: 600; }
  to { stroke-dashoffset: 0; }
}
.chart-pin {
  transform-origin: center;
  animation: pinPop 400ms var(--ease-spring) backwards;
}
@keyframes pinPop {
  0% { transform: scale(0); opacity: 0; }
  75% { transform: scale(1.05); opacity: 1; }
  100% { transform: scale(1); opacity: 1; }
}

/* Module 2 Image Tile with Floating Blur Ribbon */
.field-tile-wrap {
  position: relative;
  border-radius: 16px;
  overflow: hidden;
  background: #0E1A12;
  width: 100%;
  height: 280px;
  margin-top: 0.8rem;
  box-shadow: 0 4px 14px rgba(14, 26, 18, 0.08);
}
.field-tile-img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
  animation: tileSettle 600ms var(--ease-spring) forwards;
}
@keyframes tileSettle {
  from { transform: scale(1.04); }
  to { transform: scale(1); }
}
.field-overlay-img {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
  animation: fadeIn 500ms ease 100ms forwards;
}
@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}
.stats-ribbon {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  background: rgba(255, 255, 255, 0.88);
  backdrop-filter: blur(14px);
  -webkit-backdrop-filter: blur(14px);
  border-top: 1px solid rgba(14, 26, 18, 0.08);
  padding: 0.65rem 0.85rem;
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 0.35rem;
  animation: ribbonSlide 550ms var(--ease-spring) 180ms forwards;
}
@keyframes ribbonSlide {
  from { transform: translateY(100%); opacity: 0; }
  to { transform: translateY(0); opacity: 1; }
}
.ribbon-cell { text-align: center; }
.ribbon-val {
  font-family: var(--mono);
  font-size: 14px;
  font-weight: 500;
  color: #0E1A12;
}
.ribbon-lbl {
  font-size: 13px;
  color: #5B6760;
  text-transform: uppercase;
  margin-top: 1px;
}

/* Warnings and Notes */
.note {
  display: flex;
  gap: 0.75rem;
  align-items: flex-start;
  padding: 0.85rem 1rem;
  border-radius: 12px;
  font-size: 0.85rem;
  line-height: 1.5;
  margin-bottom: 1.2rem;
}
.note svg { flex: 0 0 16px; margin-top: 0.2rem; }
.note.caution {
  color: #946300;
  background: #FFF7E6;
  border: 1px solid #FFE58F;
}
.note.info {
  color: #2E7D32;
  background: #E8F5E9;
  border: 1px solid #C8E6C9;
}
.note.fault {
  color: #C62828;
  background: #FBE9E7;
  border: 1px solid #FFCDD2;
}

/* Composite Ledger */
.ledger-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.95rem 0.2rem;
  border-bottom: 1px solid rgba(14, 26, 18, 0.07);
}
.ledger-row:last-child { border-bottom: none; }
.ledger-title { font-weight: 500; font-size: 0.95rem; color: #0E1A12; }
.ledger-sub { font-size: 13px; color: #5B6760; margin-top: 2px; }
.ledger-score {
  font-family: var(--mono);
  font-size: 1.15rem;
  font-weight: 500;
  display: flex;
  align-items: center;
  gap: 0.4rem;
}

.plan-header {
  display: flex;
  align-items: center;
  gap: 0.45rem;
  font-weight: 600;
  font-size: 0.95rem;
  color: #0E1A12;
  margin-bottom: 0.35rem;
}

[data-testid="stCaptionContainer"], .stCaption {
  font-size: 13px !important;
  color: #5B6760 !important;
}

/* NO REPLAY ON SLIDER DRAG / STATIC STATE */
.card-static [data-testid="stVerticalBlockBorderWrapper"],
.card-static * {
  animation: none !important;
  transition: none !important;
}
.card-static .gauge-fill {
  stroke-dashoffset: var(--gauge-offset) !important;
}
.card-static .chart-line-anim {
  stroke-dashoffset: 0 !important;
}
.card-static .chart-pin {
  transform: scale(1) !important;
  opacity: 1 !important;
}
.card-static .bar-fill {
  clip-path: inset(0 0 0 0 round 9999px) !important;
}
.card-static .field-tile-img {
  transform: scale(1) !important;
}
.card-static .field-overlay-img {
  opacity: 1 !important;
}
.card-static .stats-ribbon {
  transform: translateY(0) !important;
  opacity: 1 !important;
}
.card-static .count-val {
  display: none !important;
}
.card-static .count-fallback {
  display: inline !important;
}

/* Reduced Motion Override */
@media (prefers-reduced-motion: reduce) {
  *, ::before, ::after {
    animation-duration: 0.001ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.001ms !important;
    transform: none !important;
    clip-path: none !important;
    stroke-dashoffset: 0 !important;
  }
  .gauge-fill {
    stroke-dashoffset: var(--gauge-offset) !important;
  }
  .count-val {
    display: none !important;
  }
  .count-fallback {
    display: inline !important;
  }
}

/* Responsive Media Queries */
@media (max-width: 900px) {
  [data-testid="stMainBlockContainer"] { padding: 0.6rem 1.2rem 3rem !important; }
  .top-nav-bar { flex-direction: column; align-items: stretch; border-radius: 20px; }
  .sub-header-bar { flex-direction: column; align-items: flex-start; }
  .stats-ribbon { grid-template-columns: repeat(3, 1fr); gap: 0.5rem; }
}

@media (max-width: 420px) {
  .mod-title { font-size: 1.7rem; }
  .hero-num { font-size: 2.6rem; }
  .stats-ribbon { grid-template-columns: repeat(2, 1fr); }
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Cached Loaders
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


@st.cache_resource(show_spinner="Loading ImageNet sanity model…")
def imagenet_model():
    return core.load_imagenet_model()


@st.cache_resource
def sequences():
    return core.load_sequences()


@st.cache_resource
def module2_manifest():
    manifest_path = core.MODELS_DIR / "module2_samples" / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    return []


# --------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------
TONE_CLASS = {HEALTHY: "tone-healthy", MODERATE: "tone-moderate", SEVERE: "tone-severe"}
BADGE_CLASS = {HEALTHY: "badge-healthy", MODERATE: "badge-moderate", SEVERE: "badge-severe"}


def html(markup):
    st.markdown(textwrap.dedent(markup).strip(), unsafe_allow_html=True)


def note(text, tone="caution"):
    html(f'<div class="note {tone}">{icon("info" if tone == "info" else "alert")}<div>{text}</div></div>')


def disease_tone(res):
    return HEALTHY if res["is_healthy"] else (SEVERE if res["disease_score"] < 0.33 else MODERATE)


def ndvi_tone(res):
    return HEALTHY if res["label"] == "healthy" else (SEVERE if res["label"] == "severe_stress" else MODERATE)


def pest_tone(res):
    score = res["pest_risk_score"]
    return HEALTHY if score < 0.33 else (SEVERE if score >= 0.66 else MODERATE)


def pil_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def compute_ndvi_and_stats(rgb_img, nir_img):
    """Compute per-pixel NDVI, heatmap image, low-NDVI stress overlay, and 6 stats."""
    if nir_img.size != rgb_img.size:
        nir_img = nir_img.resize(rgb_img.size, Image.Resampling.BILINEAR)

    rgb_arr = np.array(rgb_img.convert("RGB")).astype(float)
    red = rgb_arr[:, :, 0]
    nir_arr = np.array(nir_img).astype(float)
    if nir_arr.ndim == 3:
        nir_arr = nir_arr[:, :, 0]

    ndvi = (nir_arr - red) / (nir_arr + red + 1e-8)
    stats = {
        "ndvi_mean": float(np.mean(ndvi)),
        "ndvi_std": float(np.std(ndvi)),
        "ndvi_min": float(np.min(ndvi)),
        "ndvi_max": float(np.max(ndvi)),
        "ndvi_p25": float(np.percentile(ndvi, 25)),
        "ndvi_p75": float(np.percentile(ndvi, 75)),
    }

    # RdYlGn continuous colormap
    norm = np.clip((ndvi + 0.2) / 1.0, 0.0, 1.0)
    cmap = np.zeros((*ndvi.shape, 3), dtype=np.uint8)
    mask_low = norm < 0.5
    t_low = norm[mask_low] / 0.5
    cmap[mask_low, 0] = (228 + (224 - 228) * t_low).astype(np.uint8)
    cmap[mask_low, 1] = (87 + (165 - 87) * t_low).astype(np.uint8)
    cmap[mask_low, 2] = (46 + (38 - 46) * t_low).astype(np.uint8)
    mask_high = ~mask_low
    t_high = (norm[mask_high] - 0.5) / 0.5
    cmap[mask_high, 0] = (224 + (46 - 224) * t_high).astype(np.uint8)
    cmap[mask_high, 1] = (165 + (125 - 165) * t_high).astype(np.uint8)
    cmap[mask_high, 2] = (38 + (50 - 38) * t_high).astype(np.uint8)
    heatmap_img = Image.fromarray(cmap)

    # Low-NDVI stress overlay: highlight pixels where NDVI < 0.20
    overlay_arr = rgb_arr.copy()
    stress_mask = ndvi < 0.20
    alpha = 0.55
    overlay_arr[stress_mask, 0] = (1.0 - alpha) * overlay_arr[stress_mask, 0] + alpha * 228
    overlay_arr[stress_mask, 1] = (1.0 - alpha) * overlay_arr[stress_mask, 1] + alpha * 87
    overlay_arr[stress_mask, 2] = (1.0 - alpha) * overlay_arr[stress_mask, 2] + alpha * 46
    overlay_img = Image.fromarray(np.clip(overlay_arr, 0, 255).astype(np.uint8))

    return stats, heatmap_img, overlay_img


def render_prob_bars(labels, values, colors, title="Probability distribution"):
    rows = []
    for lbl, val, col in zip(labels, values, colors):
        pct = max(0.0, min(val * 100.0, 100.0))
        rows.append(
            f'<div class="bar-row">'
            f'<div class="bar-header">'
            f'<span class="bar-label">{lbl}</span>'
            f'<span class="bar-val">{val:.1%}</span>'
            f'</div>'
            f'<div class="bar-track">'
            f'<div class="bar-fill" style="width:{pct:.1f}%;background:{col};"></div>'
            f'</div>'
            f'</div>'
        )
    bars_html = "".join(rows)
    html(f'<div class="card-eyebrow">{title}</div><div class="bars-container">{bars_html}</div>')


def composite_gauge_svg(score, tone=HEALTHY, label="Healthy"):
    total_len = 398.0
    if score is not None:
        target_offset = total_len * (1.0 - min(max(score, 0.0), 1.0))
        num_str = f"{score:.3f}"
        int_part = int(score)
        dec_two = int(round((score - int_part) * 100))
        stroke_col = LIME if tone == HEALTHY else tone
    else:
        target_offset = total_len
        num_str = "—"
        int_part, dec_two = 0, 0
        stroke_col = "transparent"

    return f"""
    <div class="gauge-card-wrap">
      <svg class="gauge-svg" viewBox="0 0 280 185" fill="none">
        <path d="M 57.73 175 A 95 95 0 1 1 222.27 175" stroke="#E9ECEF" stroke-width="12" stroke-linecap="round"/>
        <path class="gauge-fill" d="M 57.73 175 A 95 95 0 1 1 222.27 175"
              stroke="{stroke_col}" stroke-width="12" stroke-linecap="round"
              stroke-dasharray="398" stroke-dashoffset="398"
              style="--gauge-offset:{target_offset:.1f}px;"/>
      </svg>
      <div class="gauge-center-val">
        <div class="gauge-num">
          <span class="count-num is-dec" style="--target-int:{int_part};--target-dec:{dec_two};" aria-label="{num_str}">
            <span class="count-val"></span><span class="count-fallback">{num_str}</span>
          </span>
        </div>
        <div class="gauge-status-chip {TONE_CLASS.get(tone, 'tone-idle')}">
          <span class="status-mark">{status_svg(tone, 14)}</span> {label if score is not None else 'Awaiting modules'}
        </div>
      </div>
    </div>
    <div class="dial-chips-row">
      <span class="dial-chip"><span class="status-mark">{status_svg(SEVERE, 13)}</span>0.00 – 0.33 Severe</span>
      <span class="dial-chip"><span class="status-mark">{status_svg(MODERATE, 13)}</span>0.33 – 0.66 Moderate</span>
      <span class="dial-chip"><span class="status-mark">{status_svg(HEALTHY, 13)}</span>0.66 – 1.00 Healthy</span>
    </div>
    """


def pest_svg_chart(seq, pred_class, pest_score):
    temps = [float(row[0]) for row in seq]
    min_t, max_t = min(temps), max(temps) + 1e-5
    norm_vals = [(t - min_t) / (max_t - min_t) for t in temps]

    xs = [40, 150, 260, 370]
    ys = [int(125 - v * 70) for v in norm_vals]

    path_d = f"M {xs[0]},{ys[0]} C 95,{ys[0]} 95,{ys[1]} {xs[1]},{ys[1]} C 205,{ys[1]} 205,{ys[2]} {xs[2]},{ys[2]} C 315,{ys[2]} 315,{ys[3]} {xs[3]},{ys[3]}"
    area_d = f"{path_d} L {xs[3]},145 L {xs[0]},145 Z"

    pins = []
    for i, (x, y, v) in enumerate(zip(xs, ys, temps)):
        delay = int((x / 370.0) * 550)
        is_target = i == 3
        r_size = 6 if is_target else 4.5
        fill_col = INK if is_target else LIME_DARK
        pins.append(
            f'<g class="chart-pin" style="animation-delay:{delay}ms;">'
            f'<circle cx="{x}" cy="{y}" r="{r_size}" fill="{fill_col}" stroke="#FFFFFF" stroke-width="2"/>'
            f'<text x="{x}" y="{y - 10}" font-family="var(--mono)" font-size="13" fill="#5B6760" text-anchor="middle">{v:.1f}°C</text>'
            f'</g>'
        )
    pins_markup = "".join(pins)

    svg = f"""
    <div class="pest-chart-wrap">
      <svg class="pest-chart-svg" viewBox="0 0 410 160" fill="none">
        <defs>
          <linearGradient id="pestLimeGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="#C6E94A" stop-opacity="0.45"/>
            <stop offset="100%" stop-color="#C6E94A" stop-opacity="0.0"/>
          </linearGradient>
        </defs>
        <!-- Area fill -->
        <path d="{area_d}" fill="url(#pestLimeGrad)"/>
        <!-- Line curve -->
        <path class="chart-line-anim" d="{path_d}" stroke="#3F6B12" stroke-width="2.6" stroke-linecap="round"/>
        <!-- Pins -->
        {pins_markup}
        <!-- X axis labels -->
        <text x="40" y="155" font-family="var(--mono)" font-size="13" fill="#5B6760" text-anchor="middle">W1</text>
        <text x="150" y="155" font-family="var(--mono)" font-size="13" fill="#5B6760" text-anchor="middle">W2</text>
        <text x="260" y="155" font-family="var(--mono)" font-size="13" fill="#5B6760" text-anchor="middle">W3</text>
        <text x="370" y="155" font-family="var(--mono)" font-size="13" fill="#0E1A12" font-weight="600" text-anchor="middle">W4 (Pin)</text>
      </svg>
    </div>
    """
    return svg


# --------------------------------------------------------------------------
# State Management & Care for Animations (No replay on slider drag)
# --------------------------------------------------------------------------
st.session_state.setdefault("nav", "Disease detection")


def check_module_animation(mod_key):
    """Determine whether animation should play for this module.
    Only plays on module open or when a new inference result is computed.
    Never plays on slider drags, dropdown changes, or minor reruns.
    """
    just_opened = st.session_state.get(f"{mod_key}_just_opened", False)
    just_computed = st.session_state.get(f"{mod_key}_just_computed", False)
    should_anim = just_opened or just_computed
    st.session_state[f"{mod_key}_just_opened"] = False
    st.session_state[f"{mod_key}_just_computed"] = False
    return should_anim


def run_model(slot, fn):
    try:
        st.session_state[slot] = fn()
        st.session_state.pop(f"{slot}_error", None)
        mod_key = {"disease": "m1", "ndvi": "m2", "pest": "m3"}.get(slot, "m1")
        st.session_state[f"{mod_key}_version"] = st.session_state.get(f"{mod_key}_version", 0) + 1
        st.session_state[f"{mod_key}_just_computed"] = True
    except Exception as e:
        st.session_state[slot] = None
        st.session_state[f"{slot}_error"] = str(e)


def apply_scenario(name):
    dm, nm, pm = disease_model(), ndvi_model(), pest_model()
    im = imagenet_model()
    seq_data = sequences()
    seq_map = {s["actual_class"]: s for s in seq_data["samples"]}
    manifest = module2_manifest()

    if "1 · Pristine" in name:
        sample_path = core.MODELS_DIR.parent / "sample_images" / "Tomato___healthy.jpg"
        img = Image.open(sample_path)
        pick_label = "Tomato — Healthy"
        st.session_state.disease = core.predict_disease(dm, img, sanity_model=im)
        st.session_state.disease_pick = pick_label
        st.session_state.disease_src = f"sample:{pick_label}"
        st.session_state.disease_img = (img, f"Sample image, true label {pick_label}", f"sample:{pick_label}")

        # Module 2 healthy pair from manifest
        entry = next((e for e in manifest if e["label"] == "healthy"), None)
        if entry:
            src = entry["source_filename"].replace(".jpg", "")
            rgb_p = core.MODELS_DIR / "module2_samples" / f"healthy_{src}_rgb.jpg"
            nir_p = core.MODELS_DIR / "module2_samples" / f"healthy_{src}_nir.jpg"
            rgb, nir = Image.open(rgb_p), Image.open(nir_p)
            stats, hm, ov = compute_ndvi_and_stats(rgb, nir)
            st.session_state.ndvi = core.predict_ndvi(nm, stats)
            st.session_state.ndvi_imgs = (rgb, ov, hm, stats, "healthy sample pair")
            for f, val in stats.items():
                st.session_state[f"ndvi_{f}"] = val

        p_samp = seq_map["low"]
        st.session_state.pest = core.predict_pest(pm, p_samp["sequence"])
        st.session_state.pest_seq = p_samp["sequence"]
        st.session_state.pest_week = f"{p_samp['label']} (recorded: Low)"

    elif "2 · Early Warning" in name:
        sample_path = core.MODELS_DIR.parent / "sample_images" / "Tomato___healthy.jpg"
        img = Image.open(sample_path)
        pick_label = "Tomato — Healthy"
        st.session_state.disease = core.predict_disease(dm, img, sanity_model=im)
        st.session_state.disease_pick = pick_label
        st.session_state.disease_src = f"sample:{pick_label}"
        st.session_state.disease_img = (img, f"Sample image, true label {pick_label}", f"sample:{pick_label}")

        # Module 2 stressed pair from manifest
        entry = next((e for e in manifest if e["label"] == "severe_stress"), None)
        if entry:
            src = entry["source_filename"].replace(".jpg", "")
            rgb_p = core.MODELS_DIR / "module2_samples" / f"severe_stress_{src}_rgb.jpg"
            nir_p = core.MODELS_DIR / "module2_samples" / f"severe_stress_{src}_nir.jpg"
            rgb, nir = Image.open(rgb_p), Image.open(nir_p)
            stats, hm, ov = compute_ndvi_and_stats(rgb, nir)
            st.session_state.ndvi = core.predict_ndvi(nm, stats)
            st.session_state.ndvi_imgs = (rgb, ov, hm, stats, "severe stress sample pair")
            for f, val in stats.items():
                st.session_state[f"ndvi_{f}"] = val

        p_samp = seq_map["medium"]
        st.session_state.pest = core.predict_pest(pm, p_samp["sequence"])
        st.session_state.pest_seq = p_samp["sequence"]
        st.session_state.pest_week = f"{p_samp['label']} (recorded: Medium)"

    elif "3 · Severe" in name:
        sample_path = core.MODELS_DIR.parent / "sample_images" / "Potato___Early_blight.jpg"
        img = Image.open(sample_path)
        pick_label = "Potato — Early Blight"
        st.session_state.disease = core.predict_disease(dm, img, sanity_model=im)
        st.session_state.disease_pick = pick_label
        st.session_state.disease_src = f"sample:{pick_label}"
        st.session_state.disease_img = (img, f"Sample image, true label {pick_label}", f"sample:{pick_label}")

        entry = next((e for e in manifest if e["label"] == "severe_stress"), None)
        if entry:
            src = entry["source_filename"].replace(".jpg", "")
            rgb_p = core.MODELS_DIR / "module2_samples" / f"severe_stress_{src}_rgb.jpg"
            nir_p = core.MODELS_DIR / "module2_samples" / f"severe_stress_{src}_nir.jpg"
            rgb, nir = Image.open(rgb_p), Image.open(nir_p)
            stats, hm, ov = compute_ndvi_and_stats(rgb, nir)
            st.session_state.ndvi = core.predict_ndvi(nm, stats)
            st.session_state.ndvi_imgs = (rgb, ov, hm, stats, "severe stress sample pair")
            for f, val in stats.items():
                st.session_state[f"ndvi_{f}"] = val

        p_samp = seq_map["high"]
        st.session_state.pest = core.predict_pest(pm, p_samp["sequence"])
        st.session_state.pest_seq = p_samp["sequence"]
        st.session_state.pest_week = f"{p_samp['label']} (recorded: High)"

    elif "Adversarial test" in name:
        sample_path = core.MODELS_DIR.parent / "sample_images" / "adversarial_non_leaf.jpg"
        img = Image.open(sample_path)
        st.session_state.disease = core.predict_disease(dm, img, sanity_model=im)
        st.session_state.disease_pick = "None"
        st.session_state.disease_src = "sample:adversarial_non_leaf"
        st.session_state.disease_img = (img, "Adversarial test: Non-leaf stock photo (sports car)", "sample:adversarial_non_leaf")
        st.session_state["_target_nav"] = "Disease detection"

    # Bump version numbers for all modules on scenario apply
    for m in ["m1", "m2", "m3", "m4"]:
        st.session_state[f"{m}_version"] = st.session_state.get(f"{m}_version", 0) + 1
        st.session_state[f"{m}_just_computed"] = True


# Apply targeted nav redirection if set
if "_target_nav" in st.session_state:
    st.session_state["nav"] = st.session_state.pop("_target_nav")


# --------------------------------------------------------------------------
# Top Navigation & Live Header Bar
# --------------------------------------------------------------------------
_d, _n, _p = (st.session_state.get(k) for k in ("disease", "ndvi", "pest"))
if _d and _n and _p:
    _score = core.composite_score(_d["disease_score"], _n["ndvi_score"], _p["pest_risk_score"])
    _label, _ = core.bucket(_score)
    _tone = {"Healthy": HEALTHY, "Moderate Risk": MODERATE, "Severe Risk": SEVERE}[_label]
    _comp_readout = f"""
    <div class="hdr-comp-pill">
      <span class="hdr-score-num">{_score:.3f}</span>
      <span class="hdr-status-badge {BADGE_CLASS[_tone]}">{status_svg(_tone, 13)} {_label}</span>
    </div>
    """
else:
    _ready = sum(1 for r in (_d, _n, _p) if r)
    _comp_readout = f"""
    <div class="hdr-comp-pill">
      <span class="hdr-score-num">—</span>
      <span class="hdr-status-badge badge-idle">{_ready} of 3 modules ready</span>
    </div>
    """

# Track navigation switch
if st.session_state.get("_prev_nav") != st.session_state.get("nav"):
    st.session_state["_prev_nav"] = st.session_state.get("nav", "Disease detection")
    curr_mod = {"Disease detection": "m1", "Canopy stress": "m2", "Pest risk": "m3", "Composite": "m4"}.get(st.session_state["_prev_nav"], "m1")
    st.session_state[f"{curr_mod}_just_opened"] = True

# Top Pill Nav Layout
col_brand, col_pills, col_stat = st.columns([0.85, 3.2, 0.95], vertical_alignment="center")

with col_brand:
    html(f"""
    <div class="brand-wrap">
      <div class="brand-mark">{icon("leaf", 16, 2)}</div>
      <div>
        <span class="brand-text">CropSense</span><span class="brand-sub">Analytics</span>
      </div>
    </div>
    """)

with col_pills:
    nav = st.pills(
        "Navigation",
        ["Disease detection", "Canopy stress", "Pest risk", "Composite"],
        default="Disease detection",
        key="nav",
        label_visibility="collapsed",
    )
    if not nav:
        nav = "Disease detection"

with col_stat:
    html(f'<div class="hdr-right-wrap" style="justify-content:flex-end;">{_comp_readout}</div>')

# Secondary Action Bar (Module Title + Demo Scenarios Picker)
MODULE_META = {
    "Disease detection": ("Module 1 · Foliar Vision", "Data Meets Foliar Health"),
    "Canopy stress": ("Module 2 · Remote Sensing", "Canopy Multispectral Vigour"),
    "Pest risk": ("Module 3 · Agro-meteorology", "Microclimate Outbreak Forecast"),
    "Composite": ("Module 4 · Holistic Fusion", "Holistic Crop Health Synthesis"),
}
eyebrow, title_text = MODULE_META.get(nav, ("Module 1 · Foliar Vision", "Data Meets Foliar Health"))

sec_left, sec_right = st.columns([2.2, 1.2], vertical_alignment="center")
with sec_left:
    mod_id = {"Disease detection": "m1", "Canopy stress": "m2", "Pest risk": "m3", "Composite": "m4"}.get(nav, "m1")
    should_anim_title = check_module_animation(mod_id)
    title_anim_class = "mod-title-anim" if should_anim_title else ""
    html(f'<div class="mod-eyebrow">{eyebrow}</div><h1 class="mod-title {title_anim_class}">{title_text}</h1>')

with sec_right:
    sc_choice = st.selectbox(
        "Load demonstration scenario",
        [
            "Manual / Custom inputs",
            "Scenario 1 · Pristine Field (Healthy)",
            "Scenario 2 · Early Warning (Hidden Stress)",
            "Scenario 3 · Severe Outbreak (High Risk)",
            "Adversarial test — non-leaf image",
        ],
        key="scenario_select",
        label_visibility="collapsed",
    )
    if sc_choice != st.session_state.get("_last_scenario", "Manual / Custom inputs"):
        st.session_state["_last_scenario"] = sc_choice
        if sc_choice != "Manual / Custom inputs":
            apply_scenario(sc_choice)
            st.rerun()


# ==========================================================================
# MODULE 1: DISEASE DETECTION
# ==========================================================================
if nav == "Disease detection":
    note("Trained and validated on lab-condition leaf images (PlantVillage dataset); accuracy "
         "on real-world field photos is lower (~30%) due to domain shift — see project report "
         "for details.")

    anim_m1 = should_anim_title
    wrapper_cls = "card-anim-reveal" if anim_m1 else "card-static"
    res_id_m1 = st.session_state.get("m1_version", 0)

    left, right = st.columns([1, 1.25], gap="large")
    with left:
        with st.container(border=True):
            html('<div class="card-eyebrow">Leaf Photo Input</div>')
            upload = st.file_uploader("Upload leaf photograph", type=["jpg", "jpeg", "png"], key="disease_upload")
            samples = sorted([p for p in (core.MODELS_DIR.parent / "sample_images").glob("*.jpg") if p.stem != "adversarial_non_leaf"])
            sample_names = [core.pretty_class(p.stem) for p in samples]
            pick = st.selectbox("Or load labelled PlantVillage sample", ["None"] + sample_names, key="disease_pick")

            image, source, src_key = None, "", None
            if upload is not None:
                try:
                    decoded = Image.open(upload)
                    decoded.load()
                except Exception:
                    note(f"<strong>{escape(upload.name)}</strong> could not be decoded as an image.", "fault")
                else:
                    image = decoded
                    source = "Uploaded leaf photograph (224×224)"
                    src_key = f"upload:{upload.name}:{upload.size}"
            elif pick != "None":
                image = Image.open(samples[sample_names.index(pick)])
                source = f"PlantVillage reference ({pick})"
                src_key = f"sample:{pick}"

            if image is None and st.session_state.get("disease_img"):
                image, source, src_key = st.session_state["disease_img"]

            if image is not None:
                st.write("")
                if st.button("Classify leaf", type="primary", key="run_disease"):
                    im = imagenet_model()
                    run_model("disease", lambda: core.predict_disease(disease_model(), image, sanity_model=im))
                    st.session_state.disease_src = src_key
                    st.session_state.disease_img = (image, source, src_key)
                    st.rerun()

                res_live = st.session_state.get("disease")
                has_cam = bool(res_live and st.session_state.get("disease_src") == src_key and res_live.get("cam_image"))
                if has_cam:
                    view_mode = st.segmented_control("View Mode", ["Original photo", "Grad-CAM saliency"], default="Original photo", key="disease_cam_mode", label_visibility="collapsed")
                    if view_mode == "Grad-CAM saliency":
                        st.image(res_live["cam_image"], width=320)
                        st.caption("Grad-CAM: warm gradients highlight visual lesion features driving the diagnosis.")
                    else:
                        st.image(image, width=320)
                        st.caption(source)
                else:
                    st.image(image, width=320)
                    st.caption(source)

    with right:
        html(f'<div class="{wrapper_cls}">')
        with st.container(border=True, key=f"m1_container_{res_id_m1}"):
            result = st.session_state.get("disease")
            st.session_state.disease_stale = bool(result) and st.session_state.get("disease_src") != src_key

            html('<div class="card-eyebrow">Neural Diagnosis</div>')
            if image is None:
                note("Upload a leaf photo or load a reference sample to begin analysis.", "info")
            elif not result:
                note("Ready. Press Classify leaf to evaluate with MobileNetV2.", "info")
            else:
                sanity = result.get("sanity")
                if sanity and sanity.get("is_flagged"):
                    signals_html = ""
                    if sanity.get("color_flagged"):
                        signals_html += (f"<div>&bull; <strong>Low plant-color coverage:</strong> only "
                                         f"<code>{sanity['color_coverage_pct']:.1f}%</code> of pixels fall in "
                                         f"vegetation HSV ranges (expected &ge; {sanity['threshold_pct']:.0f}%).</div>")
                    if sanity.get("imagenet_flagged"):
                        guesses = ", ".join(f"{name} ({prob:.1%})" for name, prob, _ in sanity.get("imagenet_top_5", []))
                        signals_html += (f"<div>&bull; <strong>No plant-related classes in ImageNet top-5:</strong> "
                                         f"stock backbone identified {guesses}.</div>")

                    warning_html = (
                        f"<strong>Out-of-distribution input detected:</strong> This image does not "
                        f"resemble plant or leaf material based on automated sanity checks.<br/>"
                        f"{signals_html}"
                        f"<div style='margin-top:0.35rem'>Because this classifier operates on a closed-set 38-class softmax "
                        f"with no built-in &ldquo;unknown&rdquo; category, it always assigns one of its "
                        f"trained crop-disease labels regardless of input &mdash; so the prediction below "
                        f"should be treated as unreliable, not dismissed outright.</div>"
                    )
                    note(warning_html, "caution")

                tone = disease_tone(result)
                conf_val = result["confidence"]
                conf_int = int(conf_val * 100)
                conf_dec = int(round((conf_val * 100 - conf_int) * 10))

                html(f"""
                <div class="hero-num-wrap">
                  <div class="hero-num">
                    <span class="count-num is-pct" style="--target-int:{conf_int};--target-dec:{conf_dec};" aria-label="{conf_val:.1%}">
                      <span class="count-val"></span><span class="count-fallback">{conf_val:.1%}</span>
                    </span>
                  </div>
                  <span class="hero-unit">Softmax Confidence</span>
                </div>
                <div class="verdict-title {TONE_CLASS[tone]}">
                  <span class="status-mark">{status_svg(tone, 18)}</span> {result['display_class']}
                </div>
                """)

                colors = [tone] + ["#CBD5E1"] * (len(result["top_k"]) - 1)
                render_prob_bars([n for n, _ in result["top_k"]], [p for _, p in result["top_k"]], colors, "Top five probability classes")

                with st.expander("Explainable AI · Grad-CAM activation details"):
                    st.markdown(
                        "**Gradient-weighted Class Activation Mapping (Grad-CAM)** computes gradient saliency "
                        "backpropagated directly to the final convolutional layer (`out_relu`), proving the model "
                        "anchors its diagnosis in active necrosis rather than background noise."
                    )
        html('</div>')


# ==========================================================================
# MODULE 2: CANOPY STRESS (NDVI REAL IMAGERY PIPELINE)
# ==========================================================================
elif nav == "Canopy stress":
    manifest = module2_manifest()
    anim_m2 = should_anim_title
    wrapper_cls = "card-anim-reveal" if anim_m2 else "card-static"
    res_id_m2 = st.session_state.get("m2_version", 0)

    left, right = st.columns([1.1, 1.2], gap="large")
    with left:
        with st.container(border=True):
            html('<div class="card-eyebrow">Multispectral Imagery Inputs</div>')
            st.caption("Upload paired visible (RGB) and near-infrared (NIR) multispectral field tiles:")
            m2_rgb_file = st.file_uploader("RGB image", type=["jpg", "jpeg", "png"], key="m2_rgb_file")
            m2_nir_file = st.file_uploader("NIR image (near-infrared band)", type=["jpg", "jpeg", "png"], key="m2_nir_file")

            if m2_rgb_file and m2_nir_file:
                if st.button("Compute NDVI and classify", type="primary", key="btn_run_m2_upload"):
                    rgb_img = Image.open(m2_rgb_file)
                    nir_img = Image.open(m2_nir_file)
                    stats, hm, ov = compute_ndvi_and_stats(rgb_img, nir_img)
                    st.session_state.ndvi = core.predict_ndvi(ndvi_model(), stats)
                    st.session_state.ndvi_imgs = (rgb_img, ov, hm, stats, "Uploaded multispectral pair")
                    for f, val in stats.items():
                        st.session_state[f"ndvi_{f}"] = val
                    st.session_state["m2_version"] = st.session_state.get("m2_version", 0) + 1
                    st.session_state["m2_just_computed"] = True
                    st.rerun()

            st.write("")
            st.caption("Or load ground-truth multispectral satellite samples:")
            b1, b2, b3 = st.columns(3)

            def load_sample_by_label(lbl_target):
                entry = next((e for e in manifest if e["label"] == lbl_target), None)
                if entry:
                    src = entry["source_filename"].replace(".jpg", "")
                    rgb_p = core.MODELS_DIR / "module2_samples" / f"{entry['label']}_{src}_rgb.jpg"
                    nir_p = core.MODELS_DIR / "module2_samples" / f"{entry['label']}_{src}_nir.jpg"
                    rgb_img = Image.open(rgb_p)
                    nir_img = Image.open(nir_p)
                    stats, hm, ov = compute_ndvi_and_stats(rgb_img, nir_img)
                    st.session_state.ndvi = core.predict_ndvi(ndvi_model(), stats)
                    st.session_state.ndvi_imgs = (rgb_img, ov, hm, stats, f"Sample parcel ({entry['label'].replace('_', ' ').title()})")
                    for f, val in stats.items():
                        st.session_state[f"ndvi_{f}"] = val
                    st.session_state["m2_version"] = st.session_state.get("m2_version", 0) + 1
                    st.session_state["m2_just_computed"] = True
                    st.rerun()

            if b1.button("Healthy", key="btn_m2_h", width="stretch"):
                load_sample_by_label("healthy")
            if b2.button("Moderate", key="btn_m2_m", width="stretch"):
                load_sample_by_label("moderate_stress")
            if b3.button("Severe", key="btn_m2_s", width="stretch"):
                load_sample_by_label("severe_stress")

            # Manual stats expander (demoted fallback)
            with st.expander("Enter NDVI statistics manually"):
                for feat in core.NDVI_FEATURES:
                    st.slider(feat, -1.0 if "min" in feat else 0.0, 1.0, step=0.01, value=float(st.session_state.get(f"ndvi_{feat}", 0.35)), key=f"manual_slider_{feat}")
                if st.button("Classify from manual stats", key="btn_manual_ndvi"):
                    man_stats = {feat: float(st.session_state[f"manual_slider_{feat}"]) for feat in core.NDVI_FEATURES}
                    st.session_state.ndvi = core.predict_ndvi(ndvi_model(), man_stats)
                    st.session_state.ndvi_imgs = None
                    for f, val in man_stats.items():
                        st.session_state[f"ndvi_{f}"] = val
                    st.session_state["m2_version"] = st.session_state.get("m2_version", 0) + 1
                    st.session_state["m2_just_computed"] = True
                    st.rerun()

    with right:
        html(f'<div class="{wrapper_cls}">')
        with st.container(border=True, key=f"m2_container_{res_id_m2}"):
            res_ndvi = st.session_state.get("ndvi")
            img_state = st.session_state.get("ndvi_imgs")

            html('<div class="card-eyebrow">Canopy Health Diagnosis</div>')
            if not res_ndvi:
                note("Load a multispectral satellite pair or run NDVI calculation.", "info")
            else:
                tone = ndvi_tone(res_ndvi)
                score_val = res_ndvi["ndvi_score"]
                s_int = int(score_val)
                s_dec = int(round((score_val - s_int) * 100))

                html(f"""
                <div class="hero-num-wrap">
                  <div class="hero-num">
                    <span class="count-num is-dec" style="--target-int:{s_int};--target-dec:{s_dec};" aria-label="{score_val:.3f}">
                      <span class="count-val"></span><span class="count-fallback">{score_val:.3f}</span>
                    </span>
                  </div>
                  <span class="hero-unit">NDVI Score (Composite Signal)</span>
                </div>
                <div class="verdict-title {TONE_CLASS[tone]}">
                  <span class="status-mark">{status_svg(tone, 18)}</span> {res_ndvi['label'].replace('_', ' ').title()}
                </div>
                """)

                # Class Probability Bars
                lbls = ["Healthy", "Moderate stress", "Severe stress"]
                raw_keys = ["healthy", "moderate_stress", "severe_stress"]
                vals = [res_ndvi["probs"][k] for k in raw_keys]
                cols = [HEALTHY if k == "healthy" else (MODERATE if k == "moderate_stress" else SEVERE) for k in raw_keys]
                render_prob_bars(lbls, vals, cols, "Random forest class confidence")

                # Field Tile Visual Display
                if img_state is not None:
                    rgb_img, ov_img, hm_img, stats, src_desc = img_state
                    view_choice = st.segmented_control(
                        "Field Visualization",
                        ["RGB Field Tile", "Low-NDVI Stress Overlay", "Per-Pixel NDVI Heatmap"],
                        default="Low-NDVI Stress Overlay",
                        key="ndvi_view_toggle",
                        label_visibility="collapsed",
                    )

                    if view_choice == "Per-Pixel NDVI Heatmap":
                        shown_img = hm_img
                        overlay_html = ""
                    elif view_choice == "Low-NDVI Stress Overlay":
                        shown_img = rgb_img
                        overlay_html = f'<img class="field-overlay-img" src="{pil_to_b64(ov_img)}" alt="Low-NDVI overlay"/>'
                    else:
                        shown_img = rgb_img
                        overlay_html = ""

                    b64_tile = pil_to_b64(shown_img)
                    html(f"""
                    <div class="field-tile-wrap">
                      <img class="field-tile-img" src="{b64_tile}" alt="Field tile"/>
                      {overlay_html}
                      <div class="stats-ribbon">
                        <div class="ribbon-cell"><div class="ribbon-val">{stats['ndvi_mean']:.2f}</div><div class="ribbon-lbl">Mean</div></div>
                        <div class="ribbon-cell"><div class="ribbon-val">{stats['ndvi_std']:.2f}</div><div class="ribbon-lbl">Std</div></div>
                        <div class="ribbon-cell"><div class="ribbon-val">{stats['ndvi_min']:.2f}</div><div class="ribbon-lbl">Min</div></div>
                        <div class="ribbon-cell"><div class="ribbon-val">{stats['ndvi_max']:.2f}</div><div class="ribbon-lbl">Max</div></div>
                        <div class="ribbon-cell"><div class="ribbon-val">{stats['ndvi_p25']:.2f}</div><div class="ribbon-lbl">P25</div></div>
                        <div class="ribbon-cell"><div class="ribbon-val">{stats['ndvi_p75']:.2f}</div><div class="ribbon-lbl">P75</div></div>
                      </div>
                    </div>
                    """)
                    st.caption(f"{src_desc} · Real per-pixel multispectral computation")
                else:
                    st.info("Manual NDVI statistics loaded (no spatial multispectral imagery provided).")
        html('</div>')


# ==========================================================================
# MODULE 3: PEST RISK FORECASTING (TEMPORAL LSTM)
# ==========================================================================
elif nav == "Pest risk":
    seq_data = sequences()
    samples = seq_data["samples"]
    labels = [f"{s['label']} (recorded: {s['actual_class'].title()})" for s in samples]

    anim_m3 = should_anim_title
    wrapper_cls = "card-anim-reveal" if anim_m3 else "card-static"
    res_id_m3 = st.session_state.get("m3_version", 0)

    left, right = st.columns([1.1, 1.2], gap="large")
    with left:
        with st.container(border=True):
            html('<div class="card-eyebrow">Meteorological Window</div>')
            pick_seq = st.selectbox("Select 4-week test sequence", labels, key="pest_week_select")
            chosen_sample = samples[labels.index(pick_seq)]
            seq_matrix = chosen_sample["sequence"]

            if st.button("Forecast pest outbreak", type="primary", key="btn_run_pest"):
                run_model("pest", lambda: core.predict_pest(pest_model(), seq_matrix))
                st.session_state.pest_seq = seq_matrix
                st.session_state.pest_truth = chosen_sample["actual_class"]
                st.session_state.pest_label = chosen_sample["label"]
                st.rerun()

            st.write("")
            st.caption("Microclimate incubation curve (temperature, humidity, precipitation):")
            res_p = st.session_state.get("pest")
            p_class = res_p["pred_class"] if res_p else chosen_sample["actual_class"]
            p_score = res_p["pest_risk_score"] if res_p else 0.5
            html(pest_svg_chart(seq_matrix, p_class, p_score))

    with right:
        html(f'<div class="{wrapper_cls}">')
        with st.container(border=True, key=f"m3_container_{res_id_m3}"):
            res_pest = st.session_state.get("pest")
            html('<div class="card-eyebrow">Outbreak Risk Assessment</div>')
            if not res_pest:
                note("Select a weather sequence and run the LSTM forecast.", "info")
            else:
                tone = pest_tone(res_pest)
                risk_score = res_pest["pest_risk_score"]
                r_int = int(risk_score)
                r_dec = int(round((risk_score - r_int) * 100))

                html(f"""
                <div class="hero-num-wrap">
                  <div class="hero-num">
                    <span class="count-num is-dec" style="--target-int:{r_int};--target-dec:{r_dec};" aria-label="{risk_score:.3f}">
                      <span class="count-val"></span><span class="count-fallback">{risk_score:.3f}</span>
                    </span>
                  </div>
                  <span class="hero-unit">Expected Pest Pressure [0–1]</span>
                </div>
                <div class="verdict-title {TONE_CLASS[tone]}">
                  <span class="status-mark">{status_svg(tone, 18)}</span> {res_pest['pred_class'].title()} Risk Outbreak
                </div>
                """)

                # 3-tier probability distribution bars
                tier_lbls = ["Low outbreak risk", "Medium outbreak risk", "High outbreak risk"]
                tier_vals = [res_pest["probs"]["low"], res_pest["probs"]["medium"], res_pest["probs"]["high"]]
                tier_cols = [HEALTHY, MODERATE, SEVERE]
                render_prob_bars(tier_lbls, tier_vals, tier_cols, "LSTM sequence probability")

                st.caption("Pest pressure is inverted in the composite formula: (1.0 − pest_risk_score).")
        html('</div>')


# ==========================================================================
# MODULE 4: COMPOSITE MULTI-SIGNAL FUSION
# ==========================================================================
elif nav == "Composite":
    missing = []
    if not _d: missing.append("Disease detection")
    if not _n: missing.append("Canopy stress")
    if not _p: missing.append("Pest risk")

    anim_m4 = should_anim_title
    wrapper_cls = "card-anim-reveal" if anim_m4 else "card-static"
    res_id_m4 = st.session_state.get("m4_version", 0)

    c_left, c_right = st.columns([1.1, 1.3], gap="large")

    with c_left:
        html(f'<div class="{wrapper_cls}">')
        with st.container(border=True, key=f"m4_gauge_container_{res_id_m4}"):
            html('<div class="card-eyebrow">Holistic Health Synthesis</div>')
            if missing:
                html(composite_gauge_svg(None, HEALTHY, ""))
                note(f"Awaiting signals from: {', '.join(missing)}.", "info")
            else:
                score = core.composite_score(_d["disease_score"], _n["ndvi_score"], _p["pest_risk_score"])
                label, _ = core.bucket(score)
                tone = {"Healthy": HEALTHY, "Moderate Risk": MODERATE, "Severe Risk": SEVERE}[label]
                html(composite_gauge_svg(score, tone, label))
                st.write("")
                st.caption("Evaluation Formula: composite = (disease_score + ndvi_score + (1.0 − pest_risk_score)) / 3")
        html('</div>')

    with c_right:
        html(f'<div class="{wrapper_cls}">')
        with st.container(border=True, key=f"m4_ledger_container_{res_id_m4}"):
            html('<div class="card-eyebrow">Orthogonal Signal Breakdown</div>')
            rows_data = [
                ("Foliar Pathology", "MobileNetV2 CNN", _d, "disease_score", lambda r: r["display_class"], disease_tone),
                ("Canopy Hydration", "Random Forest NDVI", _n, "ndvi_score", lambda r: r["label"].replace("_", " ").title(), ndvi_tone),
                ("Pest Pressure (Inverted)", "Temporal LSTM", _p, "pest_risk_score", lambda r: f"{r['pred_class'].title()} risk ({1.0 - r['pest_risk_score']:.3f})", pest_tone),
            ]

            ledger_html = ""
            for name, arch, res, key, desc_fn, tone_fn in rows_data:
                if res:
                    val = 1.0 - res[key] if key == "pest_risk_score" else res[key]
                    t = tone_fn(res)
                    score_str = f'<span class="status-mark">{status_svg(t, 14)}</span> {val:.3f}'
                    sub = desc_fn(res)
                else:
                    score_str = f'<span class="status-mark">{status_svg(None, 14)}</span> —'
                    sub = "Awaiting module execution"

                ledger_html += f"""
                <div class="ledger-row">
                  <div>
                    <div class="ledger-title">{name} <span style="font-size:13px;color:#5B6760;font-weight:400;">({arch})</span></div>
                    <div class="ledger-sub">{sub}</div>
                  </div>
                  <div class="ledger-score">{score_str}</div>
                </div>
                """

            html(ledger_html)

            with st.expander("Why equal weighting (1/3 each)?"):
                st.markdown(
                    "With three independently trained models on distinct datasets, there is no statistically "
                    "grounded empirical joint outcome dataset to train learned meta-weights without severe "
                    "overfitting. Equal weighting serves as the honest, transparent baseline."
                )
        html('</div>')

    # Agronomic Action Protocol Cards
    if not missing:
        html(f'<div class="{wrapper_cls}">')
        with st.container(border=True):
            html('<div class="card-eyebrow">Agronomic Action Plan & Interventions</div>')
            d_act = core.get_disease_action(_d["raw_class"], _d["is_healthy"])
            n_act = core.get_ndvi_action(_n["label"])
            p_act = core.get_pest_action(_p["pred_class"], _p["pest_risk_score"])

            col1, col2, col3 = st.columns(3, gap="medium")
            with col1:
                html(f'<div class="plan-header">{icon("leaf", 18)} <span>Foliar Pathology</span></div>')
                st.caption(d_act)
            with col2:
                html(f'<div class="plan-header">{icon("satellite", 18)} <span>Canopy & Hydration</span></div>')
                st.caption(n_act)
            with col3:
                html(f'<div class="plan-header">{icon("bug", 18)} <span>Pest Surveillance</span></div>')
                st.caption(p_act)

            st.write("")
            report_txt = core.generate_report(_d, _n, _p, score, label)
            st.download_button(
                "Download Agronomic Assessment Report (TXT)",
                data=report_txt,
                file_name=f"CropSense_Assessment_{label.replace(' ', '_')}.txt",
                mime="text/plain",
                key="download_report",
            )
        html('</div>')
