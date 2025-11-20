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
from .utils import load_labels, label2idx_from_idx2label, load_checkpoint, get_device, update_args_from_yaml
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


def _load_submission_ids(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"submission list not found: {p}")
    ext = p.suffix.lower()
    ids: List[str] = []
    if ext == ".json":
        # Expect COCO-like dataset_test.json with `annotations` list and `id` field
        import json
        data = json.loads(p.read_text())
        anns = data.get("annotations", [])
        ids = [str(a.get("id")) for a in anns if "id" in a]
    elif ext == ".csv":
        import csv as _csv
        with open(p, "r") as f:
            reader = _csv.DictReader(f)
            # Prefer column named 'annotation_id'; else try first column
            field = "annotation_id" if reader.fieldnames and "annotation_id" in reader.fieldnames else (reader.fieldnames[0] if reader.fieldnames else None)
            if field is None:
                raise ValueError("CSV must have a header with annotation_id column")
            for r in reader:
                if r.get(field):
                    ids.append(str(r[field]).strip())
    else:
        # Assume txt: one id per line
        ids = [line.strip() for line in p.read_text().splitlines() if line.strip()]
    if not ids:
        raise ValueError(f"No annotation ids found in {p}")
    return ids


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

    # Optional taxonomy init for hierarchical backoff (safe: does not require wandb)
    tax = init_taxonomy(idx2label, cache_dir=Path(args.cache_dir)) if args.enable_taxonomy or args.hierarchical else None
    rank_groups: Dict[str, Dict[str, List[int]]] = {}
    if tax is not None:
        for r in RANKS[::-1]:  # species ... kingdom
            rank_groups[r] = build_rank_groups_for_labels(tax, idx2label, rank=r)

    thresholds = parse_rank_thresholds(args.rank_thresholds)

    # For de-duplication and exact 1-row-per-annotation: track best prediction per annotation_id
    preds_names: List[str] = []  # legacy list (will be superseded by map)
    best_for_ann: Dict[str, Dict[str, object]] = {}  # ann_id -> {name:str, conf:float}
    with torch.no_grad():
        idx = 0
        for xb, yb, paths, names in tqdm(loader, desc="infer", ncols=100):
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            probs = F.softmax(logits, dim=-1)
            for i in range(probs.size(0)):
                pvec = probs[i].cpu()
                ann = parse_annotation_id_from_roi_path(paths[i])
                if not args.hierarchical or tax is None:
                    s_idx = int(torch.argmax(pvec))
                    s_prob = float(pvec[s_idx])
                    cname = idx2label[s_idx]
                    conf = s_prob
                else:
                    # Species (leaf)
                    s_idx = int(torch.argmax(pvec))
                    s_prob = float(pvec[s_idx])
                    if s_prob >= thresholds.get("species", 0.55):
                        cname = idx2label[s_idx]
                        conf = s_prob
                    else:
                        chosen: Optional[str] = None
                        conf = s_prob
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
                                conf = best_prob
                                break
                        cname = chosen if chosen is not None else idx2label[s_idx]
                prev = best_for_ann.get(ann)
                if (prev is None) or (conf > float(prev.get("conf", -1.0))):
                    best_for_ann[ann] = {"name": cname, "conf": conf}
            idx += logits.size(0)

    # Optionally restrict to a provided submission ID list (e.g., dataset_test.json or sample_submission.csv)
    if args.submission_list:
        sub_ids = _load_submission_ids(args.submission_list)
        unique_rows = []
        missing = 0
        for ann in sub_ids:
            rec = best_for_ann.get(str(ann))
            if rec is None:
                missing += 1
                # Fallback to a valid high-rank taxon; competition allows any rank
                cname = "Animalia"
            else:
                cname = rec["name"]
            unique_rows.append((str(ann), cname))
        if missing:
            print(f"[infer] Warning: {missing} annotation_ids not found in predictions; filled with 'Animalia'.")
    else:
        # Build from whatever was predicted: one row per unique annotation_id
        unique_rows = sorted([(ann, v["name"]) for ann, v in best_for_ann.items()], key=lambda x: x[0])

    # write submission
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["annotation_id", "concept_name"])
        for ann, cname in unique_rows:
            writer.writerow([ann, cname])
    print(f"Saved submission: {out_path} (rows: {len(unique_rows)})")


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
    ap.add_argument("--submission-list", type=str, default=None,
                    help="Optional path to a list of annotation_ids (CSV with header annotation_id, JSON dataset_test.json, or TXT one id per line). If provided, output rows will match this list and count exactly.")
    ap.add_argument("--config", type=str, default=None, help="Path to YAML to override args")
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
    if args.config:
        args = update_args_from_yaml(args, Path(args.config))
    # Allow loading backbone/lora settings from model_config.json if present
    model_dir = Path(args.model_dir)
    cfg_path = model_dir / "model_config.json"
    if cfg_path.exists():
        try:
            from .utils import load_model_config
            mcfg = load_model_config(cfg_path)
            args.backbone = mcfg.get("backbone", args.backbone)
            # Rebuild model with correct LoRA configuration if used in training
            args_lora = {
                "use_lora": mcfg.get("use_lora", False),
                "lora_r": mcfg.get("lora_r", 8),
                "lora_alpha": mcfg.get("lora_alpha", 16.0),
                "lora_dropout": mcfg.get("lora_dropout", 0.0),
            }
        except Exception:
            args_lora = {"use_lora": False, "lora_r": 8, "lora_alpha": 16.0, "lora_dropout": 0.0}
    else:
        args_lora = {"use_lora": False, "lora_r": 8, "lora_alpha": 16.0, "lora_dropout": 0.0}
    # Monkey patch build_model kwargs via global variable
    _orig_build_model = build_model
    def _build_model(num_classes, backbone, pretrained=False):
        return _orig_build_model(num_classes=num_classes, backbone=backbone, pretrained=pretrained,
                                 use_lora=args_lora.get("use_lora", False),
                                 lora_r=args_lora.get("lora_r", 8),
                                 lora_alpha=args_lora.get("lora_alpha", 16.0),
                                 lora_dropout=args_lora.get("lora_dropout", 0.0))
    globals()["build_model"] = _build_model
    run_infer(args)
