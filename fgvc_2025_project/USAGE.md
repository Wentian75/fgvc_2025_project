FGVC 2025 — How to Train, Validate, and Submit

Prerequisites
- Python 3.10+
- Git, wget/curl
- Optional: CUDA GPU (A100 recommended)

1) Environment Setup
- Create a virtualenv and install requirements:
  - `cd fgvc_2025_project`
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -U pip && pip install -r requirements.txt`
- Optional: Login to Weights & Biases (for online logging):
  - `wandb login` (or set `WANDB_API_KEY`)

2) Download and Prepare Data
- Clone the official competition repo once (to get the COCO JSONs and downloader):
  - `git clone https://github.com/fathomnet/fgvc-comp-2025 ../repo_fgvc_2025`
- Prepare train/test data (downloads images and crops ROIs):
  - `bash scripts/prepare_data.sh ../repo_fgvc_2025 data`
- Outputs:
  - Train CSV: `data/train/annotations.csv` (columns: `path,label`)
  - Test CSV: `data/test/annotations.csv` (columns: `path,label=None`)
  - Cropped ROIs under `data/*/rois/`

3) Configure Hyperparameters (YAML)
- Edit `configs/default.yaml` to set:
  - Data: `data_root`, `val_split`, `image_size`, `batch_size`, `num_workers`
  - Model: `backbone`, `use_lora`, `lora_r`, `lora_alpha`, `lora_dropout`
  - Optimization: `epochs`, `lr`, `weight_decay`, `lr_schedule` (cosine), `min_lr`, `amp_dtype`, `compile`
  - Loss weights: `ce_loss_weight`, `hd_loss_weight`, `enable_taxonomy`
  - Inference: `hierarchical`, `rank_thresholds`

4) Train the Model
- Start training with YAML config:
  - `python -m src.train --config configs/default.yaml`
- A100/CUDA tips:
  - The config defaults enable AMP (bfloat16), TF32, cuDNN autotune, and `--compile`.
- Checkpoints and labels mapping are saved to `outputs/checkpoints/`:
  - `best.pt` (weights) and `model_config.json` (backbone/LoRA) and `labels.json`.
- WandB metrics of interest:
  - `val/mean_hierarchical_distance` (observed MHD — lower is better)
  - `train/hd_expected`, `val/hd_expected` (expected MHD under predictions)
  - `train/entropy`, `val/entropy` (uncertainty)
  - `train/loss` (CE + HD convex combination)
  - `val/confusion_matrix_phylum` or class-level fallback

5) Validate on the Training Split
- Evaluate the saved checkpoint on your validation subset:
  - `python -m src.eval --config configs/default.yaml --model-dir outputs/checkpoints --data-root data/train`
- This logs the same metrics/visuals and prints summary: acc and MHD.

6) Generate Submission on the Test Set
- Ensure test data is prepared: `data/test/annotations.csv` exists (step 2).
- Create a Kaggle submission with hierarchical backoff:
  - `python -m src.infer --config configs/default.yaml --model-dir outputs/checkpoints --data-root data/test --output submission.csv --hierarchical --enable-taxonomy`
- Output: `submission.csv` (headers: `annotation_id,concept_name`).
  - This file is ready for Kaggle upload.

7) Troubleshooting
- Taxonomy resolution issues (e.g., ‘Vertebrata’ replacing ‘Chordata’):
  - Remove old cache and rebuild: `rm -f cache/taxonomy_reference.json` and re-run.
  - Ensure network access so WoRMS can resolve ranks correctly.
- Missing keys when loading a checkpoint at inference:
  - We automatically rebuild the exact model using `outputs/checkpoints/model_config.json`.
  - If you moved checkpoints, keep `best.pt`, `model_config.json`, and `labels.json` together.
- Slow training on CPU/MPS:
  - Prefer CUDA. Reduce `image_size`/`batch_size` for debugging, then scale up on GPU.
- WandB offline:
  - Set `WANDB_MODE=offline` to log locally; later run `wandb sync` to push.

8) Customization Tips
- Adjust hierarchical loss vs CE:
  - Increase `hd_loss_weight` to focus more on taxonomic closeness.
  - Monitor `val/mean_hierarchical_distance` — model selection prefers lower MHD.
- LoRA adapters:
  - Enable `use_lora: true` and tune `lora_r`, `lora_alpha`, `lora_dropout` in YAML.
- Inference thresholds:
  - Set `rank_thresholds` (e.g., `species:0.55,genus:0.60,...`) to control when to back off to higher ranks.

Reference Commands
- Train (YAML): `python -m src.train --config configs/default.yaml`
- Eval (YAML): `python -m src.eval --config configs/default.yaml --model-dir outputs/checkpoints --data-root data/train`
- Infer (YAML): `python -m src.infer --config configs/default.yaml --model-dir outputs/checkpoints --data-root data/test --output submission.csv --hierarchical --enable-taxonomy`

