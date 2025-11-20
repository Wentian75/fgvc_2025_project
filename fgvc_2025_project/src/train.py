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
    align_distance_matrix_for_labels,
)
from .utils import save_checkpoint, save_labels, get_device, save_model_config, update_args_from_yaml


def accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    return float((preds == labels).mean()) if len(labels) else 0.0


def run_train(args):
    data_root = Path(args.data_root)
    csv_path = data_root / "annotations.csv"
    assert csv_path.exists(), f"{csv_path} not found. Run the official downloader first."

    project = args.project or "fgvc2025"
    # Ensure a fresh run each time; do not resume or compare with past run IDs
    run = wandb.init(project=project, name=args.run_name, resume="never", reinit=True, config={
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
        "ce_loss_weight": args.ce_loss_weight,
        "hd_loss_weight": args.hd_loss_weight,
        "lr_schedule": args.lr_schedule,
        "min_lr": args.min_lr,
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
    # Precompute distance matrix aligned to our class order (for hierarchical loss)
    dist_mat_torch = None
    if tax is not None and args.hd_loss_weight > 0:
        D_np = align_distance_matrix_for_labels(tax, idx2label)
        dist_mat_torch = torch.tensor(D_np, dtype=torch.float32)

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
    if dist_mat_torch is not None:
        dist_mat_torch = dist_mat_torch.to(device, non_blocking=True)

    # LR scheduler
    scheduler = None
    if args.lr_schedule == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.min_lr)

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
    model_cfg_path = ckpt_dir / "model_config.json"

    # Persist labels mapping for inference
    save_labels(idx2label, labels_path)

    # Persist model configuration for consistent inference
    save_model_config({
        "backbone": args.backbone,
        "use_lora": args.use_lora,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "num_classes": len(idx2label),
    }, model_cfg_path)

    global_step = 0
    best_val = -1.0
    no_improve = 0
    use_mhd_for_selection = False
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: List[float] = []
        train_ce_vals: List[float] = []
        train_hd_vals: List[float] = []
        train_entropy_vals: List[float] = []
        val_hd_vals: List[float] = []
        val_entropy_vals: List[float] = []
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
                    ce = criterion(logits, yb)
                    if dist_mat_torch is not None and args.hd_loss_weight > 0:
                        # Expected hierarchical distance under predicted distribution
                        D_dev = dist_mat_torch.to(device, non_blocking=True)
                        probs = F.softmax(logits, dim=-1).float()
                        row = D_dev[yb]  # [B, C]
                        hd = (probs * row).sum(dim=-1).mean()
                        # entropy
                        entropy = (-probs * (probs.clamp_min(1e-9).log())).sum(dim=-1).mean()
                        loss = args.ce_loss_weight * ce + args.hd_loss_weight * hd
                    else:
                        probs = F.softmax(logits, dim=-1)
                        entropy = (-probs * (probs.clamp_min(1e-9).log())).sum(dim=-1).mean()
                        loss = ce
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
                ce = criterion(logits, yb)
                if dist_mat_torch is not None and args.hd_loss_weight > 0:
                    D_dev = dist_mat_torch.to(device, non_blocking=True)
                    probs = F.softmax(logits, dim=-1).float()
                    row = D_dev[yb]
                    hd = (probs * row).sum(dim=-1).mean()
                    entropy = (-probs * (probs.clamp_min(1e-9).log())).sum(dim=-1).mean()
                    loss = args.ce_loss_weight * ce + args.hd_loss_weight * hd
                else:
                    probs = F.softmax(logits, dim=-1)
                    entropy = (-probs * (probs.clamp_min(1e-9).log())).sum(dim=-1).mean()
                    loss = ce
                loss.backward()
                if args.use_lora:
                    log_lora_grads(model, step=global_step, prefix="lora/")
                optimizer.step()

            train_losses.append(loss.item())
            train_ce_vals.append(float(ce.item()))
            train_hd_vals.append(float(hd.item() if dist_mat_torch is not None and args.hd_loss_weight > 0 else 0.0))
            train_entropy_vals.append(float(entropy.item()))
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
            acc = correct / max(total, 1)
            pbar.set_postfix({"loss": f"{loss.item():.3f}", "acc": f"{acc:.3f}"})
            if global_step % 50 == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/acc": acc,
                    "train/ce": float(ce.item()),
                    "train/hd_expected": float(hd.item() if dist_mat_torch is not None and args.hd_loss_weight > 0 else 0.0),
                    "train/entropy": float(entropy.item()),
                    "epoch": epoch
                })
            global_step += 1

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        train_ce = float(np.mean(train_ce_vals)) if train_ce_vals else 0.0
        train_hd = float(np.mean(train_hd_vals)) if train_hd_vals else 0.0
        train_entropy = float(np.mean(train_entropy_vals)) if train_entropy_vals else 0.0
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
                # Expected HD and Entropy on val
                if dist_mat_torch is not None and args.hd_loss_weight > 0:
                    row = dist_mat_torch[yb]
                    hd_val = (probs * row).sum(dim=-1).mean().item()
                    val_hd_vals.append(hd_val)
                entropy_val = (-probs * (probs.clamp_min(1e-9).log())).sum(dim=-1).mean().item()
                val_entropy_vals.append(entropy_val)
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
            "train/ce": train_ce,
            "train/hd_expected": train_hd,
            "train/entropy": train_entropy,
        }
        if mhd is not None and mhd == mhd:  # check for NaN
            log_data["val/mean_hierarchical_distance"] = float(mhd)
        # Log val expected HD/entropy (averaged in loop above via appends)
        if dist_mat_torch is not None and args.hd_loss_weight > 0:
            log_data["val/hd_expected"] = float(np.mean(val_hd_vals)) if val_hd_vals else 0.0
        log_data["val/entropy"] = float(np.mean(val_entropy_vals)) if val_entropy_vals else 0.0
        # Optional: log separate heads
        # Note: train CE/HD components were combined; expose combined only to keep UI tidy
        wandb.log(log_data)

        # Step LR scheduler per epoch
        if scheduler is not None:
            scheduler.step()
        # Log LR
        try:
            wandb.log({"lr": optimizer.param_groups[0]["lr"]}, commit=False)
        except Exception:
            pass

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

        # Save best and early stop: prefer lowest MHD if available, else highest accuracy
        save_best = False
        improved = False
        metric_name = "val/acc"
        current_metric = val_acc
        if mhd is not None and mhd == mhd:
            metric_name = "val/mean_hierarchical_distance"
            current_metric = mhd
            use_mhd_for_selection = True
            if not hasattr(run_train, "_best_mhd"):
                run_train._best_mhd = mhd
                save_best = True
                improved = True
            elif (run_train._best_mhd - mhd) > args.early_delta:
                run_train._best_mhd = mhd
                save_best = True
                improved = True
        else:
            if (val_acc - best_val) > args.early_delta:
                best_val = val_acc
                save_best = True
                improved = True
        if save_best:
            # Save in a forward-compatible wrapper
            best_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict()}, best_ckpt_path)

        # Early stopping check
        if args.early_stop:
            no_improve = 0 if improved else (no_improve + 1)
            try:
                wandb.log({"early_stop/no_improve_epochs": no_improve, "early_stop/metric": current_metric, "early_stop/metric_name": metric_name}, commit=False)
            except Exception:
                pass
            if no_improve >= args.early_patience:
                print(f"Early stopping: no improvement in {metric_name} for {no_improve} epochs.")
                break


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
    p.add_argument("--ce-loss-weight", type=float, default=1.0, help="Weight for cross-entropy loss")
    p.add_argument("--hd-loss-weight", type=float, default=1.0, help="Weight for expected hierarchical distance loss")
    p.add_argument("--lr-schedule", type=str, default="cosine", choices=["none", "cosine"], help="Learning rate schedule")
    p.add_argument("--min-lr", type=float, default=1e-6, help="Minimum LR for cosine schedule")
    p.add_argument("--config", type=str, default=None, help="Path to YAML config to override args")
    # Early stopping
    p.add_argument("--early-stop", action="store_true", help="Enable early stopping on val metric (MHD if available else accuracy)")
    p.add_argument("--early-patience", type=int, default=5, help="Epochs with no improvement before stopping")
    p.add_argument("--early-delta", type=float, default=0.0, help="Minimum change to qualify as improvement")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.config:
        args = update_args_from_yaml(args, Path(args.config))
    run_train(args)
