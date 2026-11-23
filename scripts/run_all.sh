#!/usr/bin/env bash
# Rerun the whole workflow. Run from anywhere:  bash scripts/run_all.sh
# Optional: PYTHON=/path/to/python  DRONE_DATA_DIR=/path/for/large/raw/files
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"

"$PY" 01_download_data.py          # about 1.2 GB on the first run, then skipped
"$PY" 02_data_qa.py
"$PY" 03_vegetation_indices.py
"$PY" 04_vegetation_height.py
"$PY" 05_reference_labels.py
"$PY" 06_classify.py
"$PY" 07_accuracy_assessment.py
"$PY" 08_fractional_cover.py
"$PY" 09_scale_comparison.py
echo "All steps finished. See outputs/figures, outputs/tables and outputs/maps."
