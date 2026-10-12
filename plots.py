"""Plots for looking at a dataset and at a training run (used by the notebook)."""
import matplotlib.pyplot as plt
import numpy as np


def target_names(y, task):
    """A short text for every target: the label, or the score in its original units."""
    if task["type"] == "classification":
        labels = {c["index"]: c["label"] for c in task["classes"]}
        return [labels[int(v)] for v in y]
    return [f"{float(v) * task['score_std'] + task['score_mean']:.2f}" for v in y]


def show_samples(kind, x, titles=None, origin="lower", ncols=4, bias_label="bias (scaled)"):
    """Samples as the model sees them. ``x``: (m, C, H, W) for images / photos, (m, C, P) for spectra.

    Images: the data on top, the NaN mask below (white = data, black = was NaN).
    Spectra: the signal against the bias row; points the file did not measure are left out.
    ``bias_label``: the bias row is in volts as built, and scaled once the training split is made.
    ``origin``: "lower" shows row 0 at the bottom (Nanonis scans), "upper" at the top (ordinary images).
    """
    x = np.asarray(x)
    m = len(x)
    titles = titles or [str(i) for i in range(m)]
    if kind == "spectrum":
        ncols = min(ncols, m)
        nrows = int(np.ceil(m / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.6 * nrows), squeeze=False)
        n_signal = (x.shape[1] - 1) // 2
        for k, ax in enumerate(axes.flat):
            if k >= m:
                ax.axis("off")
                continue
            bias = x[k, 0]
            for r, name in zip(range(n_signal), ["forward", "backward"]):
                shown = np.where(x[k, 1 + n_signal + r] > 0.5, x[k, 1 + r], np.nan)
                ax.plot(bias, shown, lw=1.2, label=name)
            ax.set_title(titles[k], fontsize=9)
            ax.set_xlabel(bias_label)
            if k == 0 and n_signal > 1:
                ax.legend(fontsize=8)
        fig.tight_layout()
        return fig
    fig, axes = plt.subplots(2, m, figsize=(2.3 * m, 4.8), squeeze=False)
    for k in range(m):
        if kind == "rgb":
            rgb = np.moveaxis(x[k, :3], 0, -1)
            span = rgb.max() - rgb.min()
            axes[0, k].imshow((rgb - rgb.min()) / (span if span > 0 else 1), origin=origin)
        else:
            lo, hi = np.percentile(x[k, 0], [1, 99])
            axes[0, k].imshow(x[k, 0], origin=origin, cmap="afmhot", vmin=lo, vmax=hi if hi > lo else lo + 1)
        axes[1, k].imshow(x[k, -1], origin=origin, cmap="gray", vmin=0, vmax=1)
        axes[0, k].set_title(titles[k], fontsize=9)
        for ax in axes[:, k]:
            ax.set_xticks([])
            ax.set_yticks([])
    axes[0, 0].set_ylabel("data")
    axes[1, 0].set_ylabel("mask")
    fig.tight_layout()
    return fig


def plot_targets(y, task, train_idx=None, val_idx=None):
    """How many samples per class (or the distribution of the scores), for all / training / validation."""
    groups = [("all", np.arange(len(y)))] if train_idx is None else [("training", train_idx), ("validation", val_idx)]
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    if task["type"] == "classification":
        labels = [c["label"] for c in task["classes"]]
        width = 0.8 / len(groups)
        for g, (name, idx) in enumerate(groups):
            counts = np.bincount(y[idx], minlength=len(labels))
            bars = ax.bar(np.arange(len(labels)) + g * width, counts, width, label=name)
            ax.bar_label(bars, fontsize=8)
        ax.set_xticks(np.arange(len(labels)) + width * (len(groups) - 1) / 2, labels)
        ax.set_ylabel("samples")
    else:
        scores = y * task["score_std"] + task["score_mean"]
        ax.hist([scores[idx] for _, idx in groups], bins=15, label=[name for name, _ in groups])
        ax.set_xlabel("score")
        ax.set_ylabel("samples")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_aux(aux, columns, y, task):
    """Every aux column (z-scores), one dot per sample, grouped by class or against the score."""
    fig, axes = plt.subplots(1, len(columns), figsize=(2.4 * len(columns), 3), squeeze=False)
    rng = np.random.default_rng(0)
    for j, (ax, name) in enumerate(zip(axes[0], columns)):
        if task["type"] == "classification":
            for c in task["classes"]:
                v = aux[y == c["index"], j]
                ax.scatter(c["index"] + rng.uniform(-0.15, 0.15, len(v)), v, s=10, alpha=0.7)
            ax.set_xticks([c["index"] for c in task["classes"]], [c["label"] for c in task["classes"]], fontsize=8)
        else:
            ax.scatter(y * task["score_std"] + task["score_mean"], aux[:, j], s=10, alpha=0.7)
            ax.set_xlabel("score", fontsize=8)
        ax.set_title(name, fontsize=9)
    axes[0, 0].set_ylabel("z-score")
    fig.tight_layout()
    return fig


def plot_history(history, best_epoch=None):
    """Loss (and accuracy) per epoch; the dashed line marks the epoch whose weights are kept."""
    epochs = [h["epoch"] for h in history]
    has_acc = "val_accuracy" in history[0]
    fig, axes = plt.subplots(1, 2 if has_acc else 1, figsize=(10 if has_acc else 5.5, 3.4), squeeze=False)
    ax = axes[0, 0]
    ax.plot(epochs, [h["train_loss"] for h in history], label="training")
    ax.plot(epochs, [h["val_loss"] for h in history], label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    if has_acc:
        axes[0, 1].plot(epochs, [h["train_accuracy"] for h in history], label="training")
        axes[0, 1].plot(epochs, [h["val_accuracy"] for h in history], label="validation")
        axes[0, 1].set_xlabel("epoch")
        axes[0, 1].set_ylabel("accuracy")
        axes[0, 1].set_ylim(0, 1.05)
    for a in axes[0]:
        if best_epoch:
            a.axvline(best_epoch, color="gray", ls="--", lw=1)
        a.legend()
    fig.tight_layout()
    return fig


def plot_confusion(matrix, labels, title="validation"):
    """Rows: the true class. Columns: what the model predicted."""
    matrix = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(1.3 * len(labels) + 2, 1.3 * len(labels) + 1.5))
    ax.imshow(matrix, cmap="Blues")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_scores(true, predicted, title="validation"):
    """Predicted against true score; the line is a perfect prediction."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(true, predicted, s=18)
    lo, hi = min(min(true), min(predicted)), max(max(true), max(predicted))
    ax.plot([lo, hi], [lo, hi], color="gray", lw=1)
    ax.set_xlabel("true score")
    ax.set_ylabel("predicted score")
    ax.set_title(title)
    fig.tight_layout()
    return fig
