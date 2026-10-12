"""Run a trained model on new files.

    python infer.py MODEL_DIR FILE_OR_FOLDER [more files or folders ...] [--csv predictions.csv]

``MODEL_DIR`` is the folder written by main.py (model.pt and model_config.txt). The inputs are raw files
(.sxm, .dat, .3ds, jpg / png / tif ...); a folder means every supported file directly inside it.

Each file goes through the same steps as the training data, taken from the model's configuration:

    1. read the file and take the channel, scan direction, flatten (and slice) the samples were labeled with
    2. measure the sample: its own mean and std, scan size or bias range        -> the aux values
    3. resize / resample as the dataset was built
    4. normalize the input with the sample's own mean and std; NaN to zero; add the NaN mask
    5. normalize the aux values with the dataset's mean and std from normalize.json (log where it was used)
    6. the model; a score is converted back with the mean and std of normalize.json

Reading and transforming use the app's own code (``py/reader.py`` and ``py/dataset.py`` of the ImSpec Labeler
repository), so a new file is treated exactly like a labeled one. That needs the SciFiReaders *fork* the app
uses: run this with the Python that has it (the repository's ``.venv``).

Options:
    --channel NAME   --direction forward|backward   --flatten none|offset|line|plane   --slice N
                     use these instead of what the samples were labeled with
    --csv FILE       also write the predictions to a csv file
    --device auto    auto | cpu | cuda ...
    --app-py DIR     where the app's py/ folder is (default: ../../py next to this script)
"""
import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch

import training_fns as tf

FORK_HINT = ("Reading this file needs the SciFiReaders fork that ImSpec Labeler uses "
             "(https://github.com/gnganesh99/SciFiReaders). Run infer.py with the Python that has it, "
             "for example the repository's .venv:  .venv\\Scripts\\python to_ignore\\label_training\\infer.py ...")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run a model trained with main.py on new files.")
    p.add_argument("model", help="the model folder (or its model.pt)")
    p.add_argument("inputs", nargs="+", help="files, or folders of files")
    p.add_argument("--channel", default=None)
    p.add_argument("--direction", default=None, choices=["forward", "backward"])
    p.add_argument("--flatten", default=None, choices=["none", "offset", "line", "plane"])
    p.add_argument("--slice", dest="slice_index", type=int, default=None)
    p.add_argument("--csv", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--app-py", default=None)
    return p.parse_args(argv)


def import_app(app_py=None):
    """The app's reader and dataset modules."""
    folder = Path(app_py) if app_py else Path(__file__).resolve().parents[2] / "py"
    if not (folder / "reader.py").is_file() or not (folder / "dataset.py").is_file():
        raise FileNotFoundError(f"The app's py/ folder was not found at {folder}. Give its location with --app-py.")
    sys.path.insert(0, str(folder))
    import dataset
    import reader
    return reader, dataset


# ----------------------------------------------------------------------------- the recipe

def load_recipe(config):
    """Everything about the data that inference needs, from model_config.txt (or the original files)."""
    data = config["data"]
    norm = config.get("normalize")
    if not norm:   # a model saved before the normalize.json was stored in the config
        path = Path(data["folder"]) / f"{data['prefix']}_normalize.json"
        if not path.is_file():
            raise FileNotFoundError(f"The model has no normalization record and {path} is gone. Train again with main.py.")
        import json
        norm = json.loads(path.read_text(encoding="utf-8"))
    if not data.get("transform"):
        raise ValueError("model_config.txt has no 'transform' (how the samples were resized). Train again with main.py.")
    return {"kind": data["kind"], "transform": data["transform"], "shape": tuple(data["model_input_shape"]),
            "aux_columns": data.get("aux_columns") or [], "norm": norm, "source": config.get("source") or {},
            "bias": (config.get("preprocessing") or {}).get("bias_channel")}


def choose_source(recipe, args):
    """Channel / direction / flatten / slice: from the command line, else as the samples were labeled."""
    src = recipe["source"]
    channel = args.channel or src.get("channel")
    if not channel:
        raise ValueError("The model does not record which channel was labeled (no meta.csv at training time). "
                         "Give it with --channel, e.g. --channel Z.")
    return {"channel": channel,
            "direction": args.direction or src.get("direction") or "forward",
            "flatten": args.flatten or src.get("flatten") or "none",
            "index": args.slice_index if args.slice_index is not None else src.get("slice_index")}


# ----------------------------------------------------------------------------- one file -> model input

def aux_value(name, meta, stats, nan_to_zero):
    """One aux column for a new sample, normalized like the dataset's column. Returns (value, note)."""
    if stats is None:
        raise ValueError(f"normalize.json has no statistics for the aux column '{name}'.")
    if stats.get("constant"):
        return 0.0, None                       # the model never saw this column vary
    x = meta.get(name)
    if x is None or not math.isfinite(x):
        return (0.0 if nan_to_zero else float("nan")), f"{name} missing"
    v = math.log(max(x, stats["floor"])) if stats.get("log") else x
    z = (v - stats["mean"]) / stats["std"]
    note = f"{name} = {x:.4g} is far from the training data" if abs(z) > 4 else None
    return z, note


def prepare(path, recipe, source, reader, dataset):
    """Read one file and return (x, aux, details): the arrays the model takes, built like a training sample."""
    path = str(path)
    try:
        reader.load(path)
    except AttributeError as e:               # the original SciFiReaders returns lists, not the fork's dicts
        raise RuntimeError(FORK_HINT) from e
    try:
        out = reader.export(path, source["channel"], source["direction"], source["flatten"], source["index"])
    except KeyError:
        have = ", ".join(reader.load(path)["channels"])
        raise ValueError(f"no channel '{source['channel']}' (the file has: {have})") from None
    finally:
        reader.drop(path)
    meta, kind = out["meta"], out["kind"]
    if kind != recipe["kind"]:
        raise ValueError(f"the model was trained on '{recipe['kind']}' data, but this channel gives '{kind}'")

    norm = recipe["norm"]
    params = dict(recipe["transform"], normalize=norm.get("normalize", "none"), mean=meta.get("sample_mean"),
                  std=meta.get("sample_std"), nan_to_zero=bool(norm.get("nan_to_zero")), nan_mask=bool(norm.get("nan_mask")))
    t = dataset.transform(out["npy"], kind, params)
    x = np.frombuffer(t["data"], dtype="<f4").reshape(t["shape"]).copy()
    if kind == "rgb":
        x = np.moveaxis(x, -1, 0)             # (H, W, 4) -> (4, H, W), as in training
    if x.shape != recipe["shape"]:
        raise ValueError(f"prepared shape {x.shape} does not match the model's input {recipe['shape']}")
    if recipe["bias"]:                        # spectra: the bias row, scaled as in training
        x[0] = (x[0] - recipe["bias"]["mean"]) / recipe["bias"]["std"]

    notes, aux = [], []
    stats = norm.get("aux", {}).get("columns", {})
    for name in recipe["aux_columns"]:
        z, note = aux_value(name, meta, stats.get(name), bool(norm.get("nan_to_zero")))
        aux.append(z)
        if note:
            notes.append(note)
    if not np.isfinite(x).all():
        notes.append("NaN in the input")
    return x.astype(np.float32), np.asarray(aux, dtype=np.float32), {"meta": meta, "notes": notes}


# ----------------------------------------------------------------------------- prediction

def predict(model, config, x, aux, device):
    """A dict with the prediction for one prepared sample."""
    task = config["task"]
    xb = torch.from_numpy(x).unsqueeze(0).to(device)
    ab = torch.from_numpy(aux).unsqueeze(0).to(device) if model.n_aux else None
    with torch.no_grad():
        out = model(xb, ab)[0].cpu()
    if task["type"] == "regression":
        return {"score": float(out[0]) * task["score_std"] + task["score_mean"]}
    probs = torch.softmax(out, dim=0).numpy()
    best = task["classes"][int(probs.argmax())]
    return {"prediction": best["label"], "class_id": best["class_id"], "probability": float(probs.max()),
            "probabilities": {c["label"]: float(probs[c["index"]]) for c in task["classes"]}}


def collect_files(inputs, reader):
    files = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files += [p / name for name in reader.list_files(str(p))]
        elif p.is_file():
            files.append(p)
        else:
            raise FileNotFoundError(f"Not found: {p}")
    if not files:
        raise FileNotFoundError("No supported files in the given inputs.")
    return files


def main(argv=None):
    args = parse_args(argv)
    device = tf.get_device(args.device)
    model_path = Path(args.model)
    if model_path.is_dir():
        model_path = model_path / "model.pt"
    model, config = tf.load_model(model_path, device)
    model.eval()
    recipe = load_recipe(config)
    source = choose_source(recipe, args)
    reader, dataset = import_app(args.app_py)
    files = collect_files(args.inputs, reader)

    data = config["data"]
    print(f"Model: {data['prefix']} ({data['mode']}, {data['kind']}), trained on {data['samples']} samples.")
    print(f"Files are read as: channel {source['channel']}"
          + (f", {source['direction']}, flatten {source['flatten']}" if recipe["kind"] != "spectrum" else "")
          + (f", slice {source['index']}" if source["index"] is not None else "") + ".")
    if recipe["source"] and not recipe["source"].get("consistent", True) and not args.channel:
        print("Note: the training samples were not all labeled the same way; the most common setting is used.")

    rows = []
    for path in files:
        row = {"file": path.name, "channel": source["channel"]}
        try:
            x, aux, details = prepare(path, recipe, source, reader, dataset)
            row.update(predict(model, config, x, aux, device))
            row["notes"] = "; ".join(details["notes"])
        except (ValueError, OSError) as e:
            row["notes"] = f"skipped: {e}"
        rows.append(row)
        if "score" in row:
            shown = f"score {row['score']:.3f}"
        elif "prediction" in row:
            shown = f"{row['prediction']} (class_id {row['class_id']}, p = {row['probability']:.2f})"
        else:
            shown = ""
        print(f"  {path.name}: {shown}{'  [' + row['notes'] + ']' if row['notes'] else ''}")

    if args.csv:
        labels = [c["label"] for c in config["task"].get("classes", [])]
        head = ["file", "channel"] + (["score"] if config["task"]["type"] == "regression"
                                      else ["prediction", "class_id", "probability"] + [f"p_{name}" for name in labels]) + ["notes"]
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(head)
            for r in rows:
                probs = r.get("probabilities", {})
                flat = dict(r, **{f"p_{k}": v for k, v in probs.items()})
                w.writerow([flat.get(h, "") for h in head])
        print(f"Written: {args.csv}")
    return rows


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        sys.exit(f"Error: {e}")
