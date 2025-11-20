import json
from pathlib import Path
from typing import List, Dict, Any

import torch


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_labels(idx2label: List[str], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"idx2label": idx2label}, f, indent=2)


def load_labels(path: Path) -> List[str]:
    with open(path, "r") as f:
        data = json.load(f)
    return data["idx2label"]


def label2idx_from_idx2label(idx2label: List[str]) -> Dict[str, int]:
    return {name: i for i, name in enumerate(idx2label)}


def save_checkpoint(model: torch.nn.Module, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model: torch.nn.Module, path: Path, map_location=None):
    state = torch.load(path, map_location=map_location)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    return model


def save_model_config(cfg: Dict[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)


def load_model_config(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def update_args_from_yaml(args, config_path: Path):
    try:
        import yaml
    except Exception as e:
        raise RuntimeError("pyyaml is required to use --config") from e
    data = yaml.safe_load(open(config_path, "r")) or {}
    for k, v in data.items():
        # nested dicts (e.g., train: {epochs: 50}) will be flattened as top-level keys if match exists
        if isinstance(v, dict):
            for kk, vv in v.items():
                if hasattr(args, kk):
                    setattr(args, kk, vv)
        else:
            if hasattr(args, k):
                setattr(args, k, v)
    return args
