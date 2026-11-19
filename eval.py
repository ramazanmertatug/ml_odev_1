"""(g) ve (h) maddeleri: tek ve iki gizli katmanlı ağların karşılaştırılması ve
en başarılı ayarlar için confusion matrix.

Önce `python train.py` çalıştırılmış olmalıdır (results/ altındaki koşular ve
kaydedilmiş modeller kullanılır).

Kullanım:  python eval.py
"""

import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix)

from dataset import CLASS_NAMES, get_splits
from experiment import load_runs, write_summary
from mlp import MLP
from plots import plot_overlay

RESULTS = "results"


def best_run(runs):
    """En yüksek validation başarısına sahip koşuyu döndürür."""
    return max(runs, key=lambda r: r["best_val_acc"])


def train_acc_at(run, epoch):
    """Koşunun belirtilen epoch'taki train başarısı."""
    return next(r["train_acc"] for r in run["history"] if r["epoch"] == epoch)


def seconds_per_epoch(run, data, n_epochs=2, repeats=3):
    """Aynı mimariyi sıfırdan kurup epoch süresini ölçer.

    Isınma için 1 epoch atılır, ardından `repeats` kez n_epochs epoch ölçülür
    ve ortanca alınır. Ölçüm sırasında bilgisayarda başka ağır bir işlem
    çalışmamalıdır; aksi halde süreler güvenilir olmaz.
    """
    net = MLP(data["X_train"].shape[0], run["hidden_sizes"], 10,
              activation=run["activation"], seed=0)
    kw = dict(lr=run["lr"], batch_size=run["batch_size"])
    net.fit(data["X_train"], data["y_train"], 1, **kw)  # ısınma
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        net.fit(data["X_train"], data["y_train"], n_epochs, **kw)
        times.append((time.perf_counter() - t) / n_epochs)
    return sorted(times)[len(times) // 2]


def describe(run, net, data):
    """Karşılaştırma tablosu için bir koşunun özet satırını üretir."""
    ep = run["best_val_epoch"]
    return {
        "mimari": "-".join(map(str, [784] + run["hidden_sizes"] + [10])),
        "aktivasyon": run["activation"], "lr": run["lr"],
        "parametre": run["n_params"], "en_iyi_epoch": ep,
        "train": train_acc_at(run, ep), "val": run["best_val_acc"],
        "test": net.accuracy(data["X_test"], data["y_test"]),
        "genelleme_farki": train_acc_at(run, ep) - run["best_val_acc"],
        "epoch_sayisi": run["epochs_run"], "sn_epoch": seconds_per_epoch(run, data),
    }


def comparison_text(rows):
    """(g) için karşılaştırma tablosunu düz metne çevirir."""
    head = (f"{'model':>14s} {'mimari':>16s} {'aktiv.':>8s} {'lr':>5s} "
            f"{'param.':>8s} {'en iyi ep.':>10s} {'train':>7s} {'val':>7s} "
            f"{'test':>7s} {'train-val':>10s} {'sn/epoch':>9s}")
    lines = [head, "-" * len(head)]
    for name, r in rows:
        lines.append(
            f"{name:>14s} {r['mimari']:>16s} {r['aktivasyon']:>8s} {r['lr']:>5} "
            f"{r['parametre']:>8d} {r['en_iyi_epoch']:>10d} {r['train']:>7.4f} "
            f"{r['val']:>7.4f} {r['test']:>7.4f} {r['genelleme_farki']:>10.4f} "
            f"{r['sn_epoch']:>9.2f}")
    return "\n".join(lines)


def plot_confusions(models, X, y, title, path):
    """Her model için normalize edilmemiş confusion matrix'i yan yana çizer."""
    fig, axes = plt.subplots(1, len(models), figsize=(7.5 * len(models), 7))
    for ax, (name, net) in zip(axes, models):
        cm = confusion_matrix(y, net.predict(X), labels=range(10))
        ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
            ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
        ax.set_title(f"{name} (doğruluk = {net.accuracy(X, y):.4f})")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    de_runs = load_runs(os.path.join(RESULTS, "de_runs.json"))
    f_runs = load_runs(os.path.join(RESULTS, "f_runs.json"))
    # özet tabloları kayıtlı geçmişten yeniden üret (çöküş/overfitting etiketleri)
    write_summary(de_runs, os.path.join(RESULTS, "de_summary.txt"))
    write_summary(f_runs, os.path.join(RESULTS, "f_summary.txt"))
    two, one = best_run(de_runs), best_run(f_runs)
    data = get_splits()
    models = [("Tek gizli katman", MLP.load(one["model_path"])),
              ("İki gizli katman", MLP.load(two["model_path"]))]

    # (g) karşılaştırma tablosu
    rows = [(name, describe(run, net, data))
            for (name, net), run in zip(models, [one, two])]
    text = comparison_text(rows)
    print(text)
    out = [text, "", "Not: 'en iyi ep.' validation başarısının en yüksek olduğu "
           "epoch'tur; train/val/test bu modele aittir. sn/epoch, "
           "tek başına çalışırken ölçülmüştür."]

    # (g) eğitim eğrilerinin karşılaştırması
    plot_overlay([one, two], "Tek ve iki gizli katmanlı ağ: en iyi ayarlar",
                 os.path.join(RESULTS, "fig_compare_one_two.png"),
                 lambda r: f"{len(r['hidden_sizes'])} gizli katman "
                           f"{r['hidden_sizes']} ({r['activation']}, lr={r['lr']})")

    # (h) confusion matrix: validation ve test, ayrıca sınıf bazında rapor
    plot_confusions(models, data["X_val"], data["y_val"],
                    "Confusion matrix - validation kümesi",
                    os.path.join(RESULTS, "fig_confusion_val.png"))
    plot_confusions(models, data["X_test"], data["y_test"],
                    "Confusion matrix - test kümesi",
                    os.path.join(RESULTS, "fig_confusion_test.png"))
    for name, net in models:
        for split in ("val", "test"):
            X, y = data[f"X_{split}"], data[f"y_{split}"]
            rep = classification_report(y, net.predict(X), target_names=CLASS_NAMES,
                                        digits=4)
            out.append(f"\n[{name} | {split}]\n{rep}")
            print(f"\n[{name} | {split}]\n{rep}")
    with open(os.path.join(RESULTS, "eval_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
