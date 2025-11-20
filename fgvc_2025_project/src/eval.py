import argparse
from .warnings_filters import install_warning_filters

install_warning_filters()
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb
from PIL import Image

from .datasets import make_loader_for_infer
from .model import build_model
from .taxonomy_wandb_utils import (
    define_wandb_metrics,
    init_taxonomy,
    compute_mhd,
    log_grouped_confusion,
    build_error_table,
)
from .utils import load_labels, label2idx_from_idx2label, load_checkpoint, get_device


def run_eval(args):
    data_root = Path(args.data_root)
    csv_path = data_root / "annotations.csv"
    assert csv_path.exists(), f"{csv_path} not found"
    labels_path = Path(args.model_dir) / "labels.json"
    ckpt_path = Path(args.model_dir) / "best.pt"
    assert labels_path.exists(), f"labels.json not found at {labels_path}"
    assert ckpt_path.exists(), f"checkpoint not found at {ckpt_path}"

    idx2label = load_labels(labels_path)
    label2idx = label2idx_from_idx2label(idx2label)
    loader, idx2label_check = make_loader_for_infer(csv_path, label2idx, batch_size=args.batch_size, num_workers=args.num_workers, image_size=args.image_size)
    assert idx2label_check == idx2label

    if args.project:
        run = wandb.init(project=args.project, name=args.run_name)
        define_wandb_metrics()
    else:
        run = None

    tax = init_taxonomy(idx2label, cache_dir=Path(args.cache_dir)) if args.enable_taxonomy else None

    device = get_device()
    model = build_model(num_classes=len(idx2label), backbone=args.backbone, pretrained=False)
    load_checkpoint(model, ckpt_path, map_location=device)
    model.to(device).eval()

    all_preds: List[int] = []
    all_labels: List[int] = []
    images_for_table = []
    probs_for_table = []

    use_amp = get_device().type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else None

    with torch.no_grad():
        for xb, yb, paths, names in tqdm(loader, desc="eval", ncols=100):
            xb = xb.to(device, non_blocking=True)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = model(xb)
            else:
                logits = model(xb)
            probs = F.softmax(logits, dim=-1)
            preds = probs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend([label for label in yb.cpu().tolist() if label >= 0])
            # capture example images
            if len(images_for_table) < args.error_table_max:
                for p in paths:
                    if len(images_for_table) >= args.error_table_max:
                        break
                    images_for_table.append(Image.open(p).convert("RGB"))
            probs_for_table.append(probs.cpu())

    val_acc = float((np.array(all_preds[:len(all_labels)]) == np.array(all_labels)).mean()) if all_labels else 0.0
    mhd = None
    if tax is not None and all_labels:
        mhd = compute_mhd(all_labels, all_preds[:len(all_labels)], idx2label, tax)

    log_data = {"val/acc": val_acc}
    if mhd is not None and mhd == mhd:
        log_data["val/mean_hierarchical_distance"] = float(mhd)
    if run is not None:
        wandb.log(log_data)

        # confusion
        try:
            log_grouped_confusion(all_labels, all_preds[:len(all_labels)], idx2label, tax, plot_key="val/confusion_matrix_phylum")
        except Exception as e:
            print(f"[eval] confusion skipped: {e}")

        # error table
        try:
            probs_np = torch.cat(probs_for_table, dim=0)
            table = build_error_table(images_for_table[:len(all_labels)], probs_np[:len(all_labels)], all_labels, all_preds[:len(all_labels)], idx2label, tax, topk=5, min_distance=args.error_table_min_hd, limit=args.error_table_max)
            wandb.log({"val/error_table": table})
        except Exception as e:
            print(f"[eval] error table skipped: {e}")

    print(f"eval done. acc={val_acc:.4f} mhd={mhd if mhd is not None else 'NA'}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--model-dir", type=str, required=True, help="Directory containing best.pt and labels.json")
    p.add_argument("--project", type=str, default=None)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--enable-taxonomy", action="store_true")
    p.add_argument("--cache-dir", type=str, default="cache")
    p.add_argument("--backbone", type=str, default="vit_base_patch16_224")
    p.add_argument("--error-table-min-hd", type=int, default=6)
    p.add_argument("--error-table-max", type=int, default=200)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eval(args)
