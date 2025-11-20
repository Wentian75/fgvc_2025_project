import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .datasets import make_loader_for_infer, read_annotations_csv
from .model import build_model
from .utils import load_labels, label2idx_from_idx2label, load_checkpoint, get_device
from .taxonomy_wandb_utils import init_taxonomy, build_rank_groups_for_labels, RANKS


def parse_annotation_id_from_roi_path(path: str) -> str:
    # ROI filename pattern: {image_id}_{annotation.id}.png
    # We return the annotation.id part
    fname = Path(path).name
    m = re.match(r"^(\d+)_([\d]+)\.[a-zA-Z0-9]+$", fname)
    if not m:
        # try more permissive: split last underscore
        stem = Path(path).stem
        if "_" in stem:
            return stem.split("_")[-1]
        return stem
    return m.group(2)


def run_infer(args):
    data_root = Path(args.data_root)
    csv_path = data_root / "annotations.csv"
    assert csv_path.exists(), f"{csv_path} not found"
    model_dir = Path(args.model_dir)
    labels_path = model_dir / "labels.json"
    ckpt_path = model_dir / "best.pt"
    assert labels_path.exists(), f"labels.json not found at {labels_path}"
    assert ckpt_path.exists(), f"checkpoint not found at {ckpt_path}"

    idx2label = load_labels(labels_path)
    label2idx = label2idx_from_idx2label(idx2label)
    loader, idx2label_check = make_loader_for_infer(csv_path, label2idx, batch_size=args.batch_size, num_workers=args.num_workers, image_size=args.image_size)

    device = get_device()
    model = build_model(num_classes=len(idx2label), backbone=args.backbone, pretrained=False)
    load_checkpoint(model, ckpt_path, map_location=device)
    model.to(device).eval()

    # Read original rows to preserve order between ROI path and annotation ids
    rows = read_annotations_csv(csv_path)
    roi_paths = [r.path for r in rows]

    # Optional taxonomy init for hierarchical backoff
    tax = init_taxonomy(idx2label, cache_dir=Path(args.cache_dir)) if args.enable_taxonomy or args.hierarchical else None
    rank_groups: Dict[str, Dict[str, List[int]]] = {}
    if tax is not None:
        for r in RANKS[::-1]:  # species ... kingdom
            rank_groups[r] = build_rank_groups_for_labels(tax, idx2label, rank=r)

    thresholds = parse_rank_thresholds(args.rank_thresholds)

    preds_names: List[str] = []
    with torch.no_grad():
        idx = 0
        for xb, yb, paths, names in tqdm(loader, desc="infer", ncols=100):
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            probs = F.softmax(logits, dim=-1)
            if not args.hierarchical or tax is None:
                pred = probs.argmax(dim=1).cpu().tolist()
                for p in pred:
                    preds_names.append(idx2label[p])
            else:
                for i in range(probs.size(0)):
                    pvec = probs[i].cpu()
                    # Species (leaf)
                    s_idx = int(torch.argmax(pvec))
                    s_prob = float(pvec[s_idx])
                    if s_prob >= thresholds.get("species", 0.55):
                        preds_names.append(idx2label[s_idx])
                        continue
                    chosen: Optional[str] = None
                    # Backoff sequence
                    for r in ["genus", "family", "order", "class", "phylum", "kingdom"]:
                        groups = rank_groups.get(r, {})
                        if not groups:
                            continue
                        thr = thresholds.get(r, 0.6)
                        best_name = None
                        best_prob = -1.0
                        for gname, idxs in groups.items():
                            if not idxs:
                                continue
                            prob = float(pvec[idxs].sum().item())
                            if prob > best_prob:
                                best_prob = prob
                                best_name = gname
                        if best_name is not None and best_prob >= thr:
                            chosen = best_name
                            break
                    preds_names.append(chosen if chosen is not None else idx2label[s_idx])
            idx += logits.size(0)

    # write submission
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["annotation_id", "concept_name"])
        for rp, cname in zip(roi_paths, preds_names):
            ann_id = parse_annotation_id_from_roi_path(rp)
            writer.writerow([ann_id, cname])
    print(f"Saved submission: {out_path}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--model-dir", type=str, required=True)
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--backbone", type=str, default="vit_base_patch16_224")
    ap.add_argument("--enable-taxonomy", action="store_true")
    ap.add_argument("--cache-dir", type=str, default="cache")
    ap.add_argument("--hierarchical", action="store_true", help="Enable hierarchical backoff prediction by rank thresholds")
    ap.add_argument("--rank-thresholds", type=str, default="species:0.55,genus:0.60,family:0.65,order:0.70,class:0.75,phylum:0.80,kingdom:0.85",
                    help="Comma-separated rank:threshold list")
    return ap.parse_args()


def parse_rank_thresholds(spec: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not spec:
        return out
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    for p in parts:
        if ":" in p:
            k, v = p.split(":", 1)
            try:
                out[k.strip().lower()] = float(v.strip())
            except ValueError:
                continue
    return out


if __name__ == "__main__":
    args = parse_args()
    run_infer(args)
