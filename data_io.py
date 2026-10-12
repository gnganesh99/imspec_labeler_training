"""Find, check and load one dataset built by ImSpec Labeler ("Build dataset").

A dataset is the group of files that share a prefix ``<mode>_<kind>``, for example ``binary_image``:

    binary_image_info.json       shapes, layout of a sample, what the class numbers mean
    binary_image_normalize.json  how the arrays were normalized
    binary_image_input.npy       the samples
    binary_image_output.npy      class numbers, or scores
    binary_image_aux.npy         scan size / bias range, mean, std of every sample   (optional)

The training code assumes the data was built with **normalize = zscore**, **NaN to zero** and
**Add NaN mask** ticked. ``load_dataset`` checks this and stops with a clear message otherwise.
"""
import csv
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REQUIRED = ("info.json", "normalize.json", "input.npy", "output.npy")


@dataclass
class LabelData:
    folder: Path
    prefix: str            # e.g. "binary_image"
    mode: str              # binary | multiclass | score
    kind: str              # image | spectrum | rgb
    info: dict
    norm: dict
    x: np.ndarray          # float32, channels first: (n, C, H, W) or (n, C, P)
    y: np.ndarray          # as stored: class numbers or scores
    aux: np.ndarray        # float32 (n, k), or None
    aux_columns: list = field(default_factory=list)

    @property
    def n(self):
        return len(self.x)


# ----------------------------------------------------------------------------- finding the files

def find_prefix(folder, name=None):
    """The dataset in ``folder``. With several of them, ``name`` (e.g. "binary_image") picks one."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")
    found = sorted(p.name[:-len("_info.json")] for p in folder.glob("*_info.json"))
    if name:
        if name not in found:
            raise FileNotFoundError(f"No dataset '{name}' in {folder}. Found: {', '.join(found) or 'none'}.")
        return name
    if not found:
        raise FileNotFoundError(
            f"No '*_info.json' in {folder}. Point to the 'labeled/training_data' folder and press "
            "'Build dataset' in the app first.")
    if len(found) > 1:
        raise ValueError(
            f"Several datasets in {folder}: {', '.join(found)}. Choose one with --set, e.g. --set {found[0]}.")
    return found[0]


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"{path.name} is not valid JSON: {e}") from None


# ----------------------------------------------------------------------------- loading

def load_dataset(folder, name=None, use_aux=True):
    """Load and check one dataset. Raises FileNotFoundError / ValueError with what is wrong."""
    folder = Path(folder)
    prefix = find_prefix(folder, name)
    missing = [f"{prefix}_{part}" for part in REQUIRED if not (folder / f"{prefix}_{part}").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing in {folder}: {', '.join(missing)}. Build the dataset again in the app.")

    info = _read_json(folder / f"{prefix}_info.json")
    norm = _read_json(folder / f"{prefix}_normalize.json")
    for key in ("mode", "kind", "input", "output"):
        if key not in info:
            raise ValueError(f"{prefix}_info.json has no '{key}' entry: it does not look like an ImSpec Labeler file.")
    mode, kind = info["mode"], info["kind"]
    if mode not in ("binary", "multiclass", "score"):
        raise ValueError(f"Unknown mode '{mode}' in {prefix}_info.json.")
    if kind not in ("image", "spectrum", "rgb"):
        raise ValueError(f"Unknown kind of data '{kind}' in {prefix}_info.json.")

    # the three things the model relies on
    problems = []
    if norm.get("normalize") != "zscore":
        problems.append('"normalize" is not "zscore"')
    if not norm.get("nan_to_zero"):
        problems.append('"NaN to zero" was not ticked')
    if not norm.get("nan_mask"):
        problems.append('"Add NaN mask" was not ticked')
    if problems:
        raise ValueError(
            f"{prefix} was not built the way this training code expects: {'; '.join(problems)}. "
            "In the app: Data > Training dataset, set \"normalize\": \"zscore\", tick both NaN boxes, Build dataset.")

    x = np.load(folder / f"{prefix}_input.npy", allow_pickle=False)
    y = np.load(folder / f"{prefix}_output.npy", allow_pickle=False)
    expected = info["input"].get("shape")
    if expected and list(x.shape) != list(expected):
        raise ValueError(f"{prefix}_input.npy has shape {x.shape}, but {prefix}_info.json says {tuple(expected)}. "
                         "The files are from different builds: build the dataset again.")
    if len(x) != len(y):
        raise ValueError(f"{prefix}: {len(x)} samples in the input but {len(y)} in the output.")
    if not np.isfinite(x).all():
        raise ValueError(f"{prefix}_input.npy contains NaN or infinite values although 'NaN to zero' is recorded.")

    x = _channels_first(np.asarray(x, dtype=np.float32), kind, prefix)

    aux, columns = None, []
    aux_path = folder / f"{prefix}_aux.npy"
    if use_aux:
        if not aux_path.is_file():
            raise FileNotFoundError(f"Missing in {folder}: {aux_path.name}. Build again, or train with --no-aux.")
        aux = np.asarray(np.load(aux_path, allow_pickle=False), dtype=np.float32)
        columns = list(info.get("aux", {}).get("columns", []))
        if aux.ndim != 2 or len(aux) != len(x):
            raise ValueError(f"{aux_path.name} has shape {aux.shape}; expected ({len(x)}, k).")
        if len(columns) != aux.shape[1]:
            columns = [f"aux_{i}" for i in range(aux.shape[1])]
        if not np.isfinite(aux).all():
            raise ValueError(f"{aux_path.name} contains NaN or infinite values.")
    return LabelData(folder, prefix, mode, kind, info, norm, x, y, aux, columns)


def _channels_first(x, kind, prefix):
    """(n, C, H, W) for images and photos, (n, C, P) for spectra."""
    if kind == "image":
        if x.ndim != 4 or x.shape[1] != 2:      # with the mask: [:, 0] data, [:, 1] mask
            raise ValueError(f"{prefix}_input.npy has shape {x.shape}; expected (n, 2, H, W) = data + NaN mask.")
        return x
    if kind == "rgb":
        if x.ndim != 4 or x.shape[-1] != 4:     # (n, H, W, 4): red, green, blue, mask
            raise ValueError(f"{prefix}_input.npy has shape {x.shape}; expected (n, H, W, 4) = RGB + NaN mask.")
        return np.ascontiguousarray(np.moveaxis(x, -1, 1))
    if x.ndim != 3 or x.shape[1] not in (3, 5):  # bias, signal(s), mask(s)
        raise ValueError(f"{prefix}_input.npy has shape {x.shape}; expected (n, 3, P) or (n, 5, P) = "
                         "bias, signal row(s), mask row(s).")
    return x


def channel_names(data):
    """What each input channel of the model is."""
    if data.kind == "image":
        return ["data", "mask"]
    if data.kind == "rgb":
        return ["red", "green", "blue", "mask"]
    return ["bias", "forward", "mask_forward"] if data.x.shape[1] == 3 else \
        ["bias", "forward", "backward", "mask_forward", "mask_backward"]


# ----------------------------------------------------------------------------- how the samples were made

def source_summary(folder, prefix):
    """Which channel, scan direction, flatten and slice the samples were labeled with.

    These are only recorded per sample, in ``<prefix>_meta.csv``. Inference on a new file needs them, so the
    most common combination is returned, with ``consistent`` False when the samples do not all agree.
    Returns None when there is no meta file.
    """
    path = Path(folder) / f"{prefix}_meta.csv"
    if not path.is_file():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    keys = ("channel", "direction", "flatten", "slice_index")
    counts = Counter(tuple(r.get(k, "") or "" for k in keys) for r in rows)
    best = counts.most_common(1)[0][0]
    summary = dict(zip(keys, best))
    summary["slice_index"] = int(float(summary["slice_index"])) if summary["slice_index"] != "" else None
    summary["consistent"] = len(counts) == 1
    summary["combinations"] = [dict(zip(keys, combo), samples=n) for combo, n in counts.most_common()]
    return summary


def sample_files(folder, prefix):
    """The file name of every sample (same order as the arrays), from ``<prefix>_meta.csv``. None without it."""
    path = Path(folder) / f"{prefix}_meta.csv"
    if not path.is_file():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        return [row.get("file", "") for row in csv.DictReader(f)]


# ----------------------------------------------------------------------------- targets

def make_targets(data):
    """Targets for training, and a description of them for the config.

    Classification (binary / multiclass): the class numbers of the file become indices 0..C-1.
    Regression (score): the scores as stored (z-scored); mean / std convert predictions back.
    """
    if data.mode == "score":
        out = data.norm.get("output", {})
        if out.get("scope") != "dataset":
            raise ValueError(f"{data.prefix}_normalize.json does not record the score mean / std.")
        y = np.asarray(data.y, dtype=np.float32)
        if not np.isfinite(y).all():
            raise ValueError(f"{data.prefix}_output.npy contains NaN or infinite values.")
        return y, {"type": "regression", "n_outputs": 1,
                   "score_mean": float(out["mean"]), "score_std": float(out["std"]) or 1.0}

    names = data.info["output"].get("classes") or {}          # label -> number stored in the output
    by_value = {float(v): k for k, v in names.items()}
    values = sorted(set(by_value) | {float(v) for v in np.unique(data.y)})
    if len(values) < 2:
        raise ValueError(f"{data.prefix} has only one class ({by_value.get(values[0], values[0])}): nothing to learn.")
    index = {v: i for i, v in enumerate(values)}
    y = np.array([index[float(v)] for v in data.y], dtype=np.int64)
    classes = [{"index": i, "class_id": int(v) if float(v).is_integer() else v, "label": by_value.get(v, str(v))}
               for v, i in index.items()]
    return y, {"type": "classification", "n_outputs": len(values), "classes": classes}


# ----------------------------------------------------------------------------- split

def split_indices(n, y=None, val_fraction=0.2, seed=42):
    """(train, val) indices. With class indices ``y`` the split is stratified: every class keeps
    about ``val_fraction`` of its samples for validation (a class with one sample stays in training)."""
    rng = np.random.default_rng(seed)
    if y is None:
        order = rng.permutation(n)
        n_val = min(max(1, int(round(n * val_fraction))), n - 1)
        val, train = order[:n_val], order[n_val:]
    else:
        train, val = [], []
        for c in np.unique(y):
            members = rng.permutation(np.flatnonzero(y == c))
            n_val = int(round(len(members) * val_fraction))
            if len(members) >= 2:
                n_val = min(max(1, n_val), len(members) - 1)
            val.extend(members[:n_val])
            train.extend(members[n_val:])
        train, val = np.array(train, dtype=int), np.array(val, dtype=int)
    if len(train) == 0 or len(val) == 0:
        raise ValueError(f"Only {n} samples: too few for a train / validation split. Label more files.")
    return np.sort(train), np.sort(val)


def scale_bias_channel(x, train_idx):
    """Spectra: the bias row is in volts. Bring it to mean 0 / std 1 with the training samples' statistics."""
    mean = float(x[train_idx, 0].mean())
    std = float(x[train_idx, 0].std()) or 1.0
    x = x.copy()
    x[:, 0] = (x[:, 0] - mean) / std
    return x, {"mean": mean, "std": std}
