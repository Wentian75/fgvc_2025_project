#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${1:-"../repo_fgvc_2025"}
OUT_DIR=${2:-"data"}

if [ ! -d "$REPO_DIR" ]; then
  echo "Repo dir '$REPO_DIR' not found. Clone https://github.com/fathomnet/fgvc-comp-2025 first." >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

python "$REPO_DIR/download.py" "$REPO_DIR/datasets/dataset_train.json" "$OUT_DIR/train" --num-downloads 16 -v
python "$REPO_DIR/download.py" "$REPO_DIR/datasets/dataset_test.json"  "$OUT_DIR/test"  --num-downloads 16 -v

echo "Done. CSVs at $OUT_DIR/train/annotations.csv and $OUT_DIR/test/annotations.csv"

