import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import wandb

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    import requests
except Exception:  # pragma: no cover
    requests = None


RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]


def define_wandb_metrics():
    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")
    wandb.define_metric("val/mean_hierarchical_distance", summary="min")


@dataclass
class Taxonomy:
    concepts: List[str]
    # concept -> full path of 7 canonical ranks (strings or None)
    concept_to_path: Dict[str, List[Optional[str]]]
    # index mapping for quick distance lookups
    index_of: Dict[str, int]
    distance_matrix: np.ndarray  # shape [C, C]
    hash: str


def _lcp_len(a: List[Optional[str]], b: List[Optional[str]]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i] and a[i] is not None:
        i += 1
    return i


def _build_distance_matrix(concept_to_path: Dict[str, List[Optional[str]]], concepts_sorted: List[str]) -> Tuple[np.ndarray, Dict[str, int]]:
    idx_of = {c: i for i, c in enumerate(concepts_sorted)}
    C = len(concepts_sorted)
    D = np.zeros((C, C), dtype=np.int32)
    for i, ci in enumerate(concepts_sorted):
        pi = concept_to_path[ci]
        li = sum(1 for x in pi if x is not None)
        for j, cj in enumerate(concepts_sorted):
            pj = concept_to_path[cj]
            lj = sum(1 for x in pj if x is not None)
            l = _lcp_len(pi, pj)
            D[i, j] = (li - l) + (lj - l)
    return D, idx_of


def _taxonomy_hash(concepts_sorted: List[str], concept_to_path: Dict[str, List[Optional[str]]]) -> str:
    h = hashlib.sha256()
    for c in concepts_sorted:
        h.update(c.encode())
        for r in concept_to_path[c]:
            h.update((r or "").encode())
    return h.hexdigest()[:12]


def _worms_records_by_name(name: str):
    # REST: https://www.marinespecies.org/rest/AphiaRecordsByName/{name}?like=false&marine_only=true
    url = f"https://www.marinespecies.org/rest/AphiaRecordsByName/{requests.utils.quote(name)}?like=false&marine_only=true"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()


def _worms_classification_by_id(aphia_id: int):
    # REST: https://www.marinespecies.org/rest/AphiaClassificationByAphiaID/{id}
    url = f"https://www.marinespecies.org/rest/AphiaClassificationByAphiaID/{aphia_id}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()


def _normalize_rank_key(rank: str) -> Optional[str]:
    if not rank:
        return None
    r = rank.strip().lower()
    # normalize synonyms/variants
    mapping = {
        "subphylum": "phylum",
        "phylum": "phylum",
        "class": "class",
        "order": "order",
        "family": "family",
        "genus": "genus",
        "species": "species",
        "kingdom": "kingdom",
    }
    return mapping.get(r, None)


def _extract_path_from_worms_classification(tree: dict) -> List[Optional[str]]:
    # The classification response is a nested linked list from Kingdom downwards, e.g. {"rank":"Kingdom","name":"Animalia","child":{...}}
    # We'll traverse and collect the canonical 7 ranks.
    values: Dict[str, Optional[str]] = {r: None for r in RANKS}
    node = tree
    visited = 0
    while isinstance(node, dict) and visited < 128:
        rank = _normalize_rank_key(node.get("rank") or node.get("Rank"))
        name = node.get("scientificname") or node.get("ScientificName") or node.get("name")
        if rank in values and name:
            values[rank] = name
        node = node.get("child") or node.get("Child") or None
        visited += 1
    return [values[r] for r in RANKS]


def resolve_concept_paths(concepts: List[str], cache_path: Path) -> Dict[str, List[Optional[str]]]:
    concept_to_path: Dict[str, List[Optional[str]]] = {}
    cache: Dict[str, List[Optional[str]]] = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except Exception:
            cache = {}
    # Fill from cache
    for c in concepts:
        if c in cache:
            concept_to_path[c] = cache[c]

    unresolved = [c for c in concepts if c not in concept_to_path]
    if not unresolved:
        return concept_to_path

    if requests is None:
        # Can't resolve without requests; leave unresolved as None-paths
        for c in unresolved:
            concept_to_path[c] = [None] * len(RANKS)
        return concept_to_path

    for c in unresolved:
        try:
            records = _worms_records_by_name(c)
            record = None
            # Try exact scientific name match first
            if isinstance(records, list) and records:
                for r in records:
                    if str(r.get("scientificname", "")).lower() == c.lower():
                        record = r
                        break
                if record is None:
                    record = records[0]
            if not record:
                concept_to_path[c] = [None] * len(RANKS)
                continue
            aphia_id = record.get("AphiaID") or record.get("aphiaID") or record.get("AphiaId")
            if not aphia_id:
                concept_to_path[c] = [None] * len(RANKS)
                continue
            cls = _worms_classification_by_id(int(aphia_id))
            path = _extract_path_from_worms_classification(cls)
            concept_to_path[c] = path
        except Exception:
            concept_to_path[c] = [None] * len(RANKS)

    # Save/merge cache
    merged = {**cache, **concept_to_path}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(merged, indent=2))
    return concept_to_path


def init_taxonomy(concepts: List[str], cache_dir: Path) -> Optional[Taxonomy]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "taxonomy_reference.json"
    try:
        concept_to_path = resolve_concept_paths(concepts, cache_path=cache_path)
        concepts_sorted = sorted(concepts)
        D, idx_of = _build_distance_matrix(concept_to_path, concepts_sorted)
        tax_hash = _taxonomy_hash(concepts_sorted, concept_to_path)
        wandb.config.update({"taxonomy_hash": tax_hash, "taxonomy_ranks": RANKS}, allow_val_change=True)
        return Taxonomy(
            concepts=concepts_sorted,
            concept_to_path=concept_to_path,
            index_of=idx_of,
            distance_matrix=D,
            hash=tax_hash,
        )
    except Exception as e:  # safe fallback
        print(f"[taxonomy] Warning: could not initialize taxonomy: {e}")
        return None


def compute_mhd(labels: List[int], preds: List[int], idx2label: List[str], tax: Taxonomy) -> float:
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    # Map indices -> concept names -> taxonomy indices
    try:
        lnames = [idx2label[i] for i in labels]
        pnames = [idx2label[i] for i in preds]
        li = np.vectorize(lambda n: tax.index_of.get(n, -1))(lnames)
        pi = np.vectorize(lambda n: tax.index_of.get(n, -1))(pnames)
        mask = (li >= 0) & (pi >= 0)
        if not mask.any():
            return float('nan')
        vals = tax.distance_matrix[li[mask], pi[mask]]
        return float(vals.mean())
    except Exception:
        return float('nan')


def log_grouped_confusion(labels: List[int], preds: List[int], idx2label: List[str], tax: Optional[Taxonomy], plot_key: str):
    try:
        if tax is not None:
            # Map to Phylum
            phylum_map: Dict[str, Optional[str]] = {}
            for c in tax.concepts:
                path = tax.concept_to_path.get(c, [None] * 7)
                phylum_map[c] = path[1] if len(path) >= 2 else None
            y_true_all = [phylum_map.get(idx2label[i], None) for i in labels]
            y_pred_all = [phylum_map.get(idx2label[i], None) for i in preds]
            pairs = [(t, p) for t, p in zip(y_true_all, y_pred_all) if (t is not None and p is not None)]
            if pairs:
                y_true = [t for t, _ in pairs]
                y_pred = [p for _, p in pairs]
                names = sorted(set(y_true) | set(y_pred))
                plot = wandb.plot.confusion_matrix(y_true=y_true, preds=y_pred, class_names=names)
                wandb.log({plot_key: plot}, commit=False)
                return
        # Fallback to class-level confusion
        class_names = idx2label
        plot = wandb.plot.confusion_matrix(y_true=[idx2label[i] for i in labels],
                                           preds=[idx2label[i] for i in preds],
                                           class_names=class_names)
        wandb.log({plot_key.replace("_phylum", "_class"): plot}, commit=False)
    except Exception as e:
        print(f"[taxonomy] grouped confusion failed: {e}")


def build_error_table(images, probs_or_logits, labels: List[int], preds: List[int], idx2label: List[str], tax: Optional[Taxonomy], topk: int = 5, min_distance: int = 0, limit: int = 200):
    if torch is None:
        raise RuntimeError("torch is required for building error table")
    if hasattr(probs_or_logits, "detach"):
        probs = torch.softmax(probs_or_logits, dim=-1).detach().cpu().numpy()
    else:
        probs = probs_or_logits
    table = wandb.Table(columns=[
        "image","true_id","true_name","pred_id","pred_name",
        "pred_prob","top5_names","top5_probs","hierarchical_distance","true_phylum","pred_phylum","correct"
    ])
    rows = []
    for i in range(len(labels)):
        y = int(labels[i]); yhat = int(preds[i])
        yname = idx2label[y]; pyname = idx2label[yhat]
        hdist = None
        true_phy = None
        pred_phy = None
        if tax is not None:
            try:
                yi = tax.index_of.get(yname, -1)
                pi = tax.index_of.get(pyname, -1)
                if yi >= 0 and pi >= 0:
                    hdist = int(tax.distance_matrix[yi, pi])
                path_y = tax.concept_to_path.get(yname, [None] * 7)
                path_p = tax.concept_to_path.get(pyname, [None] * 7)
                true_phy = path_y[1] if len(path_y) > 1 else None
                pred_phy = path_p[1] if len(path_p) > 1 else None
            except Exception:
                pass
        if hdist is None:
            hdist = -1
        if hdist < min_distance:
            continue
        p_row = probs[i]
        top_idx = np.argsort(p_row)[-topk:][::-1]
        top = [(int(j), float(p_row[j])) for j in top_idx]
        top_names = [idx2label[j] for j, _ in top]
        top_probs = [float(p_row[j]) for j, _ in top]
        rows.append([
            images[i],
            y, yname,
            yhat, pyname,
            float(p_row[yhat]),
            top_names,
            top_probs,
            hdist,
            true_phy,
            pred_phy,
            bool(y == yhat),
        ])
    rows.sort(key=lambda r: r[7], reverse=True)
    rows = rows[:limit]
    for r in rows:
        table.add_data(
            wandb.Image(r[0]),  # image
            r[1],               # true_id
            r[2],               # true_name
            r[3],               # pred_id
            r[4],               # pred_name
            r[5],               # pred_prob
            r[6],               # top5_names
            r[7],               # top5_probs
            r[8],               # hierarchical_distance
            r[9],               # true_phylum
            r[10],              # pred_phylum
            r[11],              # correct
        )
    return table


def log_lora_grads(model, step=None, prefix="lora/"):
    grads = []
    for name, p in model.named_parameters():
        if not p.requires_grad or p.grad is None:
            continue
        lname = name.lower()
        if ("lora" in lname) or (lname.endswith(".lora_a")) or (lname.endswith(".lora_b")):
            try:
                g = p.grad.detach()
                n = float(g.norm(2).item())
                grads.append(n)
            except Exception:
                continue
    if grads:
        total = float(np.linalg.norm(grads))
        mx = float(np.max(grads))
        wandb.log({
            f"{prefix}grad_norm_l2": total,
            f"{prefix}grad_norm_max": mx,
            f"{prefix}grad_norm_hist": wandb.Histogram(grads),
        }, step=step)
