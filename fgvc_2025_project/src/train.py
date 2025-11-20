import argparse
from contextlib import nullcontext
from .warnings_filters import install_warning_filters

install_warning_filters()
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from PIL import Image

from .datasets import make_dataloaders, read_annotations_csv
from .model import build_model
from .taxonomy_wandb_utils import (
    define_wandb_metrics,
    init_taxonomy,
    compute_mhd,
    log_grouped_confusion,
    build_error_table,
    log_lora_grads,
)
from .utils import save_checkpoint, save_labels, get_device


def accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    return float((preds == labels).mean()) if len(labels) else 0.0


def run_train(args):
    data_root = Path(args.data_root)
    csv_path = data_root / "annotations.csv"
    assert csv_path.exists(), f"{csv_path} not found. Run the official downloader first."

    project = args.project or "fgvc2025"
    run = wandb.init(project=project, name=args.run_name, config={
        "backbone": args.backbone,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "val_split": args.val_split,
        "use_lora": args.use_lora,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
    })
    define_wandb_metrics()

    train_loader, val_loader, label2idx, idx2label = make_dataloaders(
        csv_path=csv_path, val_split=args.val_split, batch_size=args.batch_size,
        num_workers=args.num_workers, image_size=args.image_size, seed=args.seed
    )

    wandb.config.update({
        "num_classes": len(idx2label),
    }, allow_val_change=True)

    # Taxonomy init (optional, uses WoRMS REST; cached)
    tax = init_taxonomy(idx2label, cache_dir=Path(args.cache_dir)) if args.enable_taxonomy else None

    device = get_device()
    # CUDA/A100 performance knobs
    if device.type == "cuda":
        try:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    model = build_model(
        num_classes=len(idx2label), backbone=args.backbone, pretrained=True,
        use_lora=args.use_lora, lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout
    ).to(device)
    if args.compile and device.type == "cuda":
        try:
            model = torch.compile(model, mode="max-autotune")
        except Exception:
            pass

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # AMP setup (A100-friendly)
    use_amp = device.type == "cuda" and args.amp_dtype.lower() != "none"
    amp_dtype = None
    scaler = None
    if use_amp:
        if args.amp_dtype.lower() == "float16":
            amp_dtype = torch.float16
            scaler = torch.cuda.amp.GradScaler(enabled=True)
        elif args.amp_dtype.lower() == "bfloat16":
            amp_dtype = torch.bfloat16
            scaler = torch.cuda.amp.GradScaler(enabled=False)

    ckpt_dir = Path(args.out_dir) / "checkpoints"
    labels_path = ckpt_dir / "labels.json"
    best_ckpt_path = ckpt_dir / "best.pt"

    # Persist labels mapping for inference
    save_labels(idx2label, labels_path)

    global_step = 0
    best_val = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: List[float] = []
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]", ncols=100)
        for xb, yb, paths, names in pbar:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = model(xb)
                    loss = criterion(logits, yb)
                if scaler and scaler.is_enabled():
                    # Unscale for consistent grad norm logging
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if args.use_lora:
                        log_lora_grads(model, step=global_step, prefix="lora/")
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if args.use_lora:
                        log_lora_grads(model, step=global_step, prefix="lora/")
                    optimizer.step()
            else:
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                if args.use_lora:
                    log_lora_grads(model, step=global_step, prefix="lora/")
                optimizer.step()

            train_losses.append(loss.item())
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
            acc = correct / max(total, 1)
            pbar.set_postfix({"loss": f"{loss.item():.3f}", "acc": f"{acc:.3f}"})
            if global_step % 50 == 0:
                wandb.log({"train/loss": loss.item(), "train/acc": acc, "epoch": epoch})
            global_step += 1

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        train_acc = correct / max(total, 1)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        all_preds: List[int] = []
        all_labels: List[int] = []
        val_images_log = []
        val_probs_log = []
        with torch.no_grad():
            for xb, yb, paths, names in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]", ncols=100):
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        logits = model(xb)
                else:
                    logits = model(xb)
                probs = F.softmax(logits, dim=-1)
                preds = probs.argmax(dim=1)
                val_correct += (preds == yb).sum().item()
                val_total += yb.numel()
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(yb.cpu().tolist())
                # Sample a few images for error table logging
                if len(val_images_log) < args.error_table_max:
                    for p in paths:
                        if len(val_images_log) >= args.error_table_max:
                            break
                        val_images_log.append(Image.open(p).convert("RGB"))
                val_probs_log.append(probs.cpu())

        val_acc = val_correct / max(val_total, 1)
        # Concatenate probs for error table
        val_probs_np = torch.cat(val_probs_log, dim=0) if len(val_probs_log) else None

        mhd = None
        if tax is not None:
            try:
                mhd = compute_mhd(all_labels, all_preds, idx2label, tax)
            except Exception:
                mhd = None

        log_data = {
            "epoch": epoch,
            "train/loss": train_loss,
            "train/acc": train_acc,
            "val/acc": val_acc,
        }
        if mhd is not None and mhd == mhd:  # check for NaN
            log_data["val/mean_hierarchical_distance"] = float(mhd)
        wandb.log(log_data)

        # Grouped confusion
        try:
            log_grouped_confusion(all_labels, all_preds, idx2label, tax, plot_key="val/confusion_matrix_phylum")
        except Exception as e:
            print(f"[val] grouped confusion skipped: {e}")

        # Error table (worst by hierarchical distance when available, else skip filter)
        try:
            if val_probs_np is not None and len(val_images_log) >= len(all_labels):
                table = build_error_table(
                    images=val_images_log[:len(all_labels)],
                    probs_or_logits=val_probs_np[:len(all_labels)],
                    labels=all_labels,
                    preds=all_preds,
                    idx2label=idx2label,
                    tax=tax,
                    topk=5,
                    min_distance=args.error_table_min_hd if tax is not None else 0,
                    limit=args.error_table_max,
                )
                wandb.log({"val/error_table": table})
        except Exception as e:
            print(f"[val] error table skipped: {e}")

        # Save best by val accuracy (simple criterion)
        if val_acc > best_val:
            best_val = val_acc
            save_checkpoint(model, best_ckpt_path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, required=True, help="Directory containing annotations.csv and rois/")
    p.add_argument("--project", type=str, default="fgvc2025")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--backbone", type=str, default="vit_base_patch16_224")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--use-lora", action="store_true")
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--enable-taxonomy", action="store_true", help="Resolve taxonomy via WoRMS and log MHD/Grouped confusion")
    p.add_argument("--cache-dir", type=str, default="cache")
    p.add_argument("--error-table-min-hd", type=int, default=6)
    p.add_argument("--error-table-max", type=int, default=200)
    p.add_argument("--out-dir", type=str, default="outputs", help="Directory to store checkpoints and logs")
    p.add_argument("--amp-dtype", type=str, default="bfloat16", choices=["none", "float16", "bfloat16"], help="AMP dtype for CUDA; A100 supports bfloat16 well")
    p.add_argument("--compile", action="store_true", help="Use torch.compile on CUDA for potential speedups")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_train(args)
