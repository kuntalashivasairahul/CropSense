#!/bin/sh
set -eu
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=-1
.venv/bin/python scripts/verify_environment.py --offline
exec .venv/bin/python -m streamlit run app.py
