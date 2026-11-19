"""Grafik çizimi (Matplotlib)."""

import os

import matplotlib
matplotlib.use("Agg")  # ekransız ortamlarda da çalışsın
import matplotlib.pyplot as plt


def _series(run, key):
    return [r["epoch"] for r in run["history"]], [r[key] for r in run["history"]]


def plot_accuracy_curves(runs, title, path, label_fn):
    """Verilen koşular için train/validation başarısını tek şekilde çizer.

    Her koşu ayrı bir alt grafiktir (üst satır: başarı, alt satır: kayıp).
    Overfitting başlangıcı (min val loss epoch'u) dikey çizgiyle gösterilir.
    label_fn(run) alt grafik başlığını üretir.
    """
    n = len(runs)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 6.4), squeeze=False)
    for j, run in enumerate(runs):
        for i, (key_tr, key_va, ylabel) in enumerate(
                [("train_acc", "val_acc", "Başarı (accuracy)"),
                 ("train_loss", "val_loss", "Kayıp (cross-entropy)")]):
            ax = axes[i][j]
            x, ytr = _series(run, key_tr)
            _, yva = _series(run, key_va)
            ax.plot(x, ytr, "o-", ms=3, label="Train")
            ax.plot(x, yva, "s-", ms=3, label="Validation")
            if run["overfit_observed"]:
                ax.axvline(run["min_val_loss_epoch"], color="gray", ls="--",
                           lw=1, label="Overfitting başlangıcı")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(label_fn(run), fontsize=10)
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_overlay(runs, title, path, label_fn):
    """Birden çok koşunun train (düz) ve validation (kesikli) başarısını aynı
    eksende karşılaştırır (örn. nöron sayısı 32/64/128)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k, run in enumerate(runs):
        c = f"C{k}"
        x, ytr = _series(run, "train_acc")
        _, yva = _series(run, "val_acc")
        ax.plot(x, ytr, "-", color=c, label=f"{label_fn(run)} - Train")
        ax.plot(x, yva, "--", color=c, label=f"{label_fn(run)} - Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Başarı (accuracy)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
