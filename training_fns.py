"""Training helpers: device, data loaders, the training loop with early stopping, evaluation, saving.

The loop watches the validation loss. When it has not improved for ``patience`` epochs in a row the
training stops, and the weights of the best epoch are the ones that are kept and saved.
"""
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import build_model


def get_device(name="auto"):
    """'auto' picks the GPU when there is one (cuda, then Apple's mps), else the CPU."""
    if name in (None, "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was asked for, but this PyTorch does not see a CUDA GPU.")
    return device


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------- data

def make_loaders(x, aux, y, train_idx, val_idx, batch_size, device, seed=42):
    """Training (shuffled) and validation loaders. ``aux`` may be None."""
    def dataset(idx):
        tensors = [torch.from_numpy(x[idx])]
        tensors.append(torch.from_numpy(aux[idx]) if aux is not None else torch.zeros(len(idx), 0))
        tensors.append(torch.from_numpy(y[idx]))
        return TensorDataset(*tensors)

    pin = device.type == "cuda"
    generator = torch.Generator().manual_seed(seed)
    train = DataLoader(dataset(train_idx), batch_size=batch_size, shuffle=True, generator=generator, pin_memory=pin)
    val = DataLoader(dataset(val_idx), batch_size=batch_size, shuffle=False, pin_memory=pin)
    return train, val


def class_weights(y_train, n_classes):
    """Inverse-frequency weights, so that a rare class counts as much as a common one."""
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)
    weights = np.where(counts > 0, len(y_train) / (n_classes * np.maximum(counts, 1)), 0.0)
    return torch.tensor(weights, dtype=torch.float32)


# ----------------------------------------------------------------------------- one pass over the data

def _forward(model, batch, device):
    x, aux, y = (t.to(device, non_blocking=True) for t in batch)
    out = model(x, aux if model.n_aux else None)
    return out, y


def run_epoch(model, loader, criterion, device, task, optimizer=None):
    """One pass. With an optimizer the model is trained, without it only evaluated.
    Returns (mean loss, predictions, targets) with predictions / targets as numpy arrays."""
    training = optimizer is not None
    model.train(training)
    total, count, preds, targets = 0.0, 0, [], []
    with torch.set_grad_enabled(training):
        for batch in loader:
            out, y = _forward(model, batch, device)
            if task == "regression":
                out = out.squeeze(1)
            loss = criterion(out, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total += loss.item() * len(y)
            count += len(y)
            preds.append((out.argmax(1) if task == "classification" else out).detach().cpu())
            targets.append(y.detach().cpu())
    return total / max(count, 1), torch.cat(preds).numpy(), torch.cat(targets).numpy()


# ----------------------------------------------------------------------------- training

def train(model, train_loader, val_loader, task, device, epochs=100, patience=10, lr=1e-3, weight_decay=1e-4,
          weights=None, min_delta=1e-4, log=print):
    """Train with early stopping on the validation loss. The model ends up with its best weights.

    An epoch counts as an improvement only if the validation loss drops by more than ``min_delta``;
    after ``patience`` epochs without one, training stops.

    Returns a dict: history (one entry per epoch), best_epoch, best_val_loss, epochs_run, stopped_early.
    """
    model.to(device)
    if task == "classification":
        criterion = nn.CrossEntropyLoss(weight=weights.to(device) if weights is not None else None)
    else:
        criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history, best_loss, best_state, best_epoch, waited = [], float("inf"), None, 0, 0
    for epoch in range(1, epochs + 1):
        train_loss, tp, tt = run_epoch(model, train_loader, criterion, device, task, optimizer)
        val_loss, vp, vt = run_epoch(model, val_loader, criterion, device, task)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        if task == "classification":
            row.update(train_accuracy=float((tp == tt).mean()), val_accuracy=float((vp == vt).mean()))
        history.append(row)

        improved = val_loss < best_loss - min_delta
        if improved:
            best_loss, best_epoch, waited = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            waited += 1
        extra = f"  acc {row['train_accuracy']:.3f} / {row['val_accuracy']:.3f}" if task == "classification" else ""
        log(f"epoch {epoch:3d}/{epochs}  train loss {train_loss:.4f}  val loss {val_loss:.4f}{extra}"
            f"{'  *' if improved else ''}")
        if waited >= patience:
            log(f"Stopped early: the validation loss has not improved for {patience} epochs "
                f"(best: epoch {best_epoch}, val loss {best_loss:.4f}).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"history": history, "best_epoch": best_epoch, "best_val_loss": best_loss,
            "epochs_run": len(history), "stopped_early": len(history) < epochs}


# ----------------------------------------------------------------------------- evaluation

def evaluate(model, loader, task, device, task_info):
    """Metrics of the (best) model on one loader."""
    criterion = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()
    loss, pred, target = run_epoch(model, loader, criterion, device, task)
    if task == "classification":
        n = task_info["n_outputs"]
        confusion = np.zeros((n, n), dtype=int)          # rows: true class, columns: predicted class
        for t, p in zip(target, pred):
            confusion[int(t), int(p)] += 1
        return {"loss": loss, "accuracy": float((pred == target).mean()), "confusion_matrix": confusion.tolist()}
    std, mean = task_info["score_std"], task_info["score_mean"]
    error = (pred - target) * std                         # back in the units of the scores
    return {"loss": loss, "mae_score": float(np.abs(error).mean()), "rmse_score": float(np.sqrt((error ** 2).mean())),
            "predicted_scores": [round(float(v), 4) for v in pred * std + mean],
            "true_scores": [round(float(v), 4) for v in target * std + mean]}


def predict_arrays(model, x, aux, task, device, batch_size=64):
    """Model outputs for arrays that are already prepared like the training data.

    Classification: the class probabilities, shape (n, classes). Regression: the z-scored score, shape (n,).
    """
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start:start + batch_size]).to(device)
            ab = torch.from_numpy(aux[start:start + batch_size]).to(device) if model.n_aux else None
            out = model(xb, ab)
            outputs.append((torch.softmax(out, dim=1) if task == "classification" else out.squeeze(1)).cpu())
    return torch.cat(outputs).numpy()


# ----------------------------------------------------------------------------- saving / loading

def save_model(out_dir, model, config, history):
    """model.pt (weights + config), model_config.txt (the same config, readable), history.csv."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": {k: v.cpu() for k, v in model.state_dict().items()}, "config": config},
               out_dir / "model.pt")
    (out_dir / "model_config.txt").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    if history:
        keys = list(history[0])
        lines = [",".join(keys)] + [",".join(f"{row[k]:.6g}" if isinstance(row[k], float) else str(row[k]) for k in keys)
                                    for row in history]
        (out_dir / "history.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_dir


def read_config(path):
    """A model_config.txt (it is JSON text)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"{path} is not a model_config.txt written by this program: {e}") from None


def load_model(path, device="cpu"):
    """(model, config) from a model.pt saved by ``save_model``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")
    saved = torch.load(path, map_location=device, weights_only=True)
    model = build_model(saved["config"]["model"])
    model.load_state_dict(saved["model_state"])
    return model.to(device), saved["config"]
