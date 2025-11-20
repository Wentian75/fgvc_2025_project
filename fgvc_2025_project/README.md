FGVC 2025 — Training Scaffold with WandB + Hierarchical Metrics

Overview
- End-to-end scaffold for the FathomNet FGVC 2025 competition.
- Includes dataset prep from the official COCO JSONs, a PyTorch+timm baseline, optional LoRA adapters, and WandB visualizations: MHD, grouped confusion, error table, and LoRA gradient norms.

Quick Start
1) Create and activate a Python 3.10+ environment.
2) Install dependencies:
   - `pip install -r requirements.txt`
3) Prepare data (downloads and crops ROIs):
   - Clone the official repo if you haven’t: `git clone https://github.com/fathomnet/fgvc-comp-2025`
   - Run the downloader on train/test:
     - `python ../repo_fgvc_2025/download.py ../repo_fgvc_2025/datasets/dataset_train.json data/train --num-downloads 16 -v`
     - `python ../repo_fgvc_2025/download.py ../repo_fgvc_2025/datasets/dataset_test.json  data/test  --num-downloads 16 -v`
   - This writes `data/train/annotations.csv` with `path,label` and crops into `data/train/rois/`.

4) Train a baseline (ViT-B/16 + LoRA off by default):
   - `python -m src.train --data-root data/train --val-split 0.1 --epochs 10 --batch-size 64 --project fgvc2025`

5) Enable LoRA (applied to Linear layers; see flags):
   - `python -m src.train --data-root data/train --use-lora --lora-r 8 --lora-alpha 16 --lora-dropout 0.05`

6) Evaluate and log visuals on a validation split:
   - `python -m src.eval --data-root data/train`

7) Export Kaggle submission (test set):
   - Prepare test data using the downloader (step 3 produces `data/test/annotations.csv`).
   - Run inference with your best checkpoint directory (created during training):
     - `python -m src.infer --data-root data/test --model-dir outputs/checkpoints --output submission.csv`
   - The output CSV has headers `annotation_id,concept_name` and can be submitted to Kaggle.

See also: USAGE.md for a step-by-step training/validation/submission guide.

WandB Logging
- Primary metric: `val/mean_hierarchical_distance` (MHD). Decreasing MHD indicates better taxonomic awareness.
- Grouped confusion (by Phylum) if taxonomy is resolved; otherwise logs class-level.
- Visual error table with top-k worst hierarchical mistakes per epoch.
- LoRA gradient norms when adapters are enabled.

Taxonomy Resolution
- Uses concept names from `annotations.csv` and attempts to resolve their full 7-level taxonomy via `fathomnet`’s WoRMS API. Results are cached in `cache/taxonomy_reference.json`.
- If the `fathomnet` package or network is unavailable, training proceeds and MHD/grouped confusion will be skipped with a warning. You can still train and log accuracy.

Files
- `src/datasets.py`: CSV ROI dataset with deterministic splits and label indexing.
- `src/model.py`: timm backbone and optional LoRA adapters.
- `src/taxonomy_wandb_utils.py`: taxonomy resolver, MHD computation, grouped confusion, error table, LoRA grad logging, rank groups and distance alignment.
- `src/train.py`: training loop with WandB integration.
- `src/eval.py`: standalone evaluation/visualization over a split.
- `src/infer.py`: Kaggle submission generation, supports hierarchical backoff predictions by rank thresholds.

Notes
- The official hierarchical scoring uses the FathomNet WoRMS module. This scaffold mirrors that logic by resolving taxonomy and computing path distance via Lowest Common Ancestor. Exact parity depends on the external service and naming normalization.
- Keep runs reproducible: WandB logs include a taxonomy hash of the cache. Rebuild the cache if you change concept spellings.
 - If you see `UnsupportedFieldAttributeWarning: The 'repr' attribute...` from Pydantic dataclasses during imports, it is benign; the project suppresses it automatically. You can also set `PYTHONWARNINGS="ignore::pydantic.dataclasses.UnsupportedFieldAttributeWarning"`.

Devices
- The code auto-selects `cuda`, then `mps` (macOS), then `cpu`.
- You may see harmless MPS pin_memory warnings on macOS.

CUDA/A100 Tips
- Defaults are tuned for A100: AMP uses `bfloat16` and TF32 is enabled when CUDA is detected.
- Recommended command on A100:
  - `python -m src.train --data-root data/train --epochs 50 --batch-size 128 --enable-taxonomy --amp-dtype bfloat16 --compile`
- Internals configured for CUDA:
  - `torch.backends.cudnn.benchmark = True`
  - `torch.backends.cudnn.allow_tf32 = True`
  - `torch.backends.cuda.matmul.allow_tf32 = True`
  - `torch.set_float32_matmul_precision('high')`
  - Mixed precision: `torch.autocast(device_type='cuda', dtype=torch.bfloat16)` during forward on CUDA

Hierarchical Training + Inference
- Loss targets MHD explicitly via expected hierarchical distance: total loss = `ce_loss_weight * CE + hd_loss_weight * E[D(y, \hat{y})]`.
- Enable taxonomy for training to activate HD loss: `--enable-taxonomy --hd-loss-weight 1.0` (falls back to CE if taxonomy unavailable).
- Save-best uses lowest validation MHD when available, otherwise accuracy.
- Hierarchical backoff predictions for submissions (when uncertain):
  - Enable with `src.infer --hierarchical --enable-taxonomy`.
  - Thresholds per rank via `--rank-thresholds`, default `species:0.55,genus:0.60,family:0.65,order:0.70,class:0.75,phylum:0.80,kingdom:0.85`.
  - If species top-1 prob < threshold, it backs off progressively to higher ranks and outputs the ancestor name.

Config via YAML
- Edit `configs/default.yaml` and run with:
  - Train: `python -m src.train --config configs/default.yaml`
  - Eval: `python -m src.eval --config configs/default.yaml --model-dir outputs/checkpoints --data-root data/train`
  - Infer: `python -m src.infer --config configs/default.yaml --model-dir outputs/checkpoints --data-root data/test --output submission.csv`
- YAML keys cover: epochs, lr, lr_schedule (cosine), loss weights (CE vs HD), LoRA toggles/params, AMP dtype/compile, and hierarchical thresholds for inference.
