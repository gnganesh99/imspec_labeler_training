"""Train a small CNN on a dataset built with ImSpec Labeler.

    python main.py PATH/TO/labeled/training_data

The folder must hold one built dataset (for example the ``binary_image_*`` files). The program reads
``<set>_info.json`` and ``<set>_normalize.json`` to find out what the data is, builds a matching network
(2D for images, 1D for spectra; classification for binary / multiclass, regression for scores), splits the
samples 80 : 20 (seed 42), trains with early stopping, and saves into ``<folder>/model/``:

    model.pt           the best weights, with the configuration
    model_config.txt   the configuration: data, network, training settings, split, results, and what
                       infer.py needs for new files (channel / flatten used, the whole normalize.json)
    history.csv        the loss of every epoch

Options:
    --set NAME         which dataset, when the folder holds several (e.g. --set binary_image)
    --epochs 100       maximum number of epochs
    --patience 10      stop when the validation loss has not improved for this many epochs
    --min-delta 0.0001 how much the validation loss must drop to count as an improvement
    --batch-size 16    --lr 0.001    --val-fraction 0.2    --seed 42
    --device auto      auto | cpu | cuda | cuda:0 ...   (auto uses the GPU when there is one)
    --out DIR          where to save (default: <folder>/model)
    --no-aux           do not use the scan-size / bias-range values
    --no-class-weights do not weight the classes by how rare they are
    --config FILE      reuse the network settings of an earlier model_config.txt
    --weights FILE     start from the weights of an earlier model.pt (retraining)

The data must be built with "normalize": "zscore" and both NaN boxes ticked (NaN to zero, Add NaN mask).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

import data_io
import training_fns as tf
from model import build_model, count_parameters, default_model_config


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train a small CNN on a dataset built with ImSpec Labeler.")
    p.add_argument("data_dir", help="the labeled/training_data folder")
    p.add_argument("--set", dest="name", default=None, help="dataset prefix, e.g. binary_image")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=None)
    p.add_argument("--no-aux", action="store_true")
    p.add_argument("--no-class-weights", action="store_true")
    p.add_argument("--config", default=None, help="an earlier model_config.txt whose network settings are reused")
    p.add_argument("--weights", default=None, help="an earlier model.pt to start from")
    return p.parse_args(argv)


def describe(config):
    """A few plain lines for the top of model_config.txt."""
    d, m, t, tr = config["data"], config["model"], config["task"], config["training"]
    lines = [
        f"Dataset {d['prefix']}: {d['samples']} samples of kind '{d['kind']}', mode '{d['mode']}'.",
        f"Model input: {d['model_input_shape']} = channels {d['channels']}"
        + (f"; aux values: {', '.join(d['aux_columns'])}." if d["aux_columns"] else "; no aux values."),
        f"Network: {m['dims']}D CNN, conv channels {m['channels']}, kernel {m['kernel_size']}, "
        f"hidden {m['hidden']}, dropout {m['dropout']}, {config['parameters']} parameters.",
    ]
    if t["type"] == "classification":
        lines.append("Output: " + ", ".join(f"index {c['index']} = {c['label']} (class_id {c['class_id']})" for c in t["classes"]) + ".")
    else:
        lines.append(f"Output: one number, the z-scored score. score = output * {t['score_std']:.6g} + {t['score_mean']:.6g}.")
    lines.append(f"Training: {tr['split']} split {tr['n_train']} / {tr['n_val']} (seed {tr['seed']}), "
                 f"{tr['epochs_run']} of {tr['epochs']} epochs, best epoch {tr['best_epoch']}, patience {tr['patience']}.")
    return lines


def build_config(data, x, task, model, model_config, preprocessing, source, train_idx, val_idx, result, metrics,
                 weights, settings, device, data_dir, seconds=None, started_from=None):
    """Everything that goes into model_config.txt (and into model.pt).

    ``settings``: epochs, patience, min_delta, batch_size, lr, weight_decay, seed, val_fraction.
    ``result``: what training_fns.train returned. ``metrics``: {"train": ..., "validation": ...}.
    """
    classification = task["type"] == "classification"
    config = {
        "description": [],
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data": {
            "folder": str(Path(data_dir).resolve()), "prefix": data.prefix, "mode": data.mode, "kind": data.kind,
            "samples": data.n, "built": data.info.get("built"), "stored_input_shape": data.info["input"].get("shape"),
            "sample_layout": data.info["input"].get("sample_layout"), "model_input_shape": list(x.shape[1:]),
            "channels": data_io.channel_names(data), "aux_columns": data.aux_columns,
            "normalize": data.norm.get("normalize"), "nan_to_zero": data.norm.get("nan_to_zero"),
            "nan_mask": data.norm.get("nan_mask"), "transform": data.info.get("transform"),
        },
        "preprocessing": preprocessing,
        "source": source,            # channel / direction / flatten / slice the samples were labeled with
        "normalize": data.norm,      # the dataset's normalize.json, needed again for new files
        "task": task,
        "model": model_config,
        "parameters": count_parameters(model),
        "training": {
            "epochs": settings["epochs"], "epochs_run": result["epochs_run"], "best_epoch": result["best_epoch"],
            "stopped_early": result["stopped_early"], "patience": settings["patience"],
            "min_delta": settings["min_delta"], "batch_size": settings["batch_size"],
            "optimizer": "Adam", "lr": settings["lr"], "weight_decay": settings["weight_decay"],
            "loss": "CrossEntropyLoss" if classification else "MSELoss (on the z-scored score)",
            "class_weights": weights.tolist() if weights is not None else None,
            "seed": settings["seed"], "val_fraction": settings["val_fraction"],
            "split": "stratified random" if classification else "random",
            "n_train": int(len(train_idx)), "n_val": int(len(val_idx)),
            "train_indices": [int(i) for i in train_idx], "val_indices": [int(i) for i in val_idx],
            "device": str(device), "torch": str(torch.__version__),
            "seconds": round(seconds, 1) if seconds is not None else None, "started_from": started_from,
        },
        "results": {"best_val_loss": result["best_val_loss"], **metrics},
    }
    config["description"] = describe(config)
    return config


def main(argv=None):
    args = parse_args(argv)
    tf.set_seed(args.seed)
    device = tf.get_device(args.device)

    # ---- data
    data = data_io.load_dataset(args.data_dir, args.name, use_aux=not args.no_aux)
    y, task = data_io.make_targets(data)
    classification = task["type"] == "classification"
    source = data_io.source_summary(args.data_dir, data.prefix)
    if source is None:
        print(f"Note: {data.prefix}_meta.csv not found. Training works, but infer.py will need --channel (and --flatten).")
    elif not source["consistent"]:
        print("Note: the samples were not all labeled the same way: "
              + "; ".join(f"{c['samples']} x channel {c['channel']} {c['direction']} flatten {c['flatten']}".strip()
                          for c in source["combinations"])
              + f". infer.py will use the most common one ({source['channel']}).")
    train_idx, val_idx = data_io.split_indices(data.n, y if classification else None, args.val_fraction, args.seed)
    x, preprocessing = data.x, {}
    if data.kind == "spectrum":
        x, preprocessing["bias_channel"] = data_io.scale_bias_channel(x, train_idx)
    print(f"Dataset {data.prefix}: {data.n} samples, input {tuple(x.shape[1:])}, "
          f"{'classes ' + str([c['label'] for c in task['classes']]) if classification else 'scores'}, "
          f"aux {data.aux.shape[1] if data.aux is not None else 0}. Train {len(train_idx)}, validation {len(val_idx)}.")

    # ---- model
    n_aux = data.aux.shape[1] if data.aux is not None else 0
    model_config = default_model_config(data.kind, x.shape[1:], n_aux, task["n_outputs"])
    if args.config:
        earlier = tf.read_config(args.config).get("model")
        if not earlier:
            raise ValueError(f"{args.config} has no 'model' section.")
        for key in ("dims", "in_channels", "n_aux", "n_outputs"):
            if earlier.get(key) != model_config[key]:
                raise ValueError(f"{args.config} does not fit this data: {key} is {earlier.get(key)} there, "
                                 f"{model_config[key]} here.")
        model_config = earlier
    model = build_model(model_config)
    if args.weights:
        saved = torch.load(args.weights, map_location="cpu", weights_only=True)
        try:
            model.load_state_dict(saved["model_state"])
        except (RuntimeError, KeyError) as e:
            raise ValueError(f"{args.weights} does not fit this network: {e}") from None
        print(f"Starting from the weights in {args.weights}.")
    print(f"Network: {model_config['dims']}D CNN, {count_parameters(model)} parameters, on {device}.")

    # ---- training
    weights = None
    if classification and not args.no_class_weights:
        weights = tf.class_weights(y[train_idx], task["n_outputs"])
    train_loader, val_loader = tf.make_loaders(x, data.aux, y, train_idx, val_idx, args.batch_size, device, args.seed)
    started = time.time()
    result = tf.train(model, train_loader, val_loader, task["type"], device, args.epochs, args.patience,
                      args.lr, args.weight_decay, weights, args.min_delta)
    metrics = {"train": tf.evaluate(model, train_loader, task["type"], device, task),
               "validation": tf.evaluate(model, val_loader, task["type"], device, task)}

    # ---- save
    settings = {"epochs": args.epochs, "patience": args.patience, "min_delta": args.min_delta,
                "batch_size": args.batch_size, "lr": args.lr, "weight_decay": args.weight_decay,
                "seed": args.seed, "val_fraction": args.val_fraction}
    config = build_config(data, x, task, model, model_config, preprocessing, source, train_idx, val_idx, result,
                          metrics, weights, settings, device, args.data_dir, time.time() - started, args.weights)
    out_dir = tf.save_model(args.out or Path(args.data_dir) / "model", model, config, result["history"])

    v = metrics["validation"]
    summary = f"accuracy {v['accuracy']:.3f}" if classification else f"mean error {v['mae_score']:.3f} (in score units)"
    print(f"Best epoch {result['best_epoch']}: validation loss {result['best_val_loss']:.4f}, {summary}.")
    print(f"Saved model.pt, model_config.txt and history.csv in {out_dir}")
    return config


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        sys.exit(f"Error: {e}")
