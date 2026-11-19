"""Deney yardımcıları: 10'ar epoch'luk adımlarla eğitim, sonuçların özeti ve
kaydı.
"""

import json
import os
import time

import numpy as np

from mlp import MLP


def run_until_perfect(data, hidden_sizes, activation, lr, max_epochs=300,
                      step=10, batch_size=32, seed=0, target_acc=1.0,
                      model_path=None, verbose=True):
    """Ağı `step` epoch'luk adımlarla, train başarısı target_acc'e ulaşana
    (ödevde %100) veya max_epochs'a gelene kadar eğitir.

    Her adım sonunda train/validation kaybı ve başarısı kaydedilir. En yüksek
    validation başarısına ulaşılan andaki model model_path'e kaydedilir.

    Dönüş: koşunun yapılandırmasını, geçmişini ve özetini içeren sözlük.
    Overfitting başlangıcı, validation kaybının en küçük olduğu epoch olarak
    alınır (bu epoch'tan sonra val kaybı artıyorsa overfitting başlamıştır).
    """
    net = MLP(data["X_train"].shape[0], hidden_sizes, 10,
              activation=activation, seed=seed)
    history, best_val, best_epoch = [], -1.0, 0
    diverged = False
    t_start = time.perf_counter()

    while net.epochs_done < max_epochs:
        rec = net.fit(data["X_train"], data["y_train"], step, lr,
                      batch_size=batch_size, X_val=data["X_val"],
                      y_val=data["y_val"], log_every=step)[-1]
        history.append(rec)
        if verbose:
            print(f"  {activation:7s} lr={lr} h={hidden_sizes} "
                  f"epoch {rec['epoch']:4d} | train acc {rec['train_acc']:.4f} "
                  f"loss {rec['train_loss']:.4f} | val acc {rec['val_acc']:.4f} "
                  f"loss {rec['val_loss']:.4f}", flush=True)
        if not np.isfinite(rec["train_loss"]):
            diverged = True
            break
        if rec["val_acc"] > best_val:
            best_val, best_epoch = rec["val_acc"], rec["epoch"]
            if model_path:
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                net.save(model_path)
        if rec["train_acc"] >= target_acc:
            break

    valid = [r for r in history if np.isfinite(r["val_loss"])]
    min_loss_epoch = min(valid, key=lambda r: r["val_loss"])["epoch"] if valid else None
    last = history[-1]
    return {
        "hidden_sizes": list(hidden_sizes), "activation": activation,
        "lr": lr, "batch_size": batch_size, "seed": seed,
        "n_params": net.num_parameters(), "model_path": model_path,
        "history": history,
        "reached_target": bool(last["train_acc"] >= target_acc),
        "diverged": diverged,
        "epochs_run": last["epoch"],
        "best_val_acc": best_val, "best_val_epoch": best_epoch,
        "min_val_loss_epoch": min_loss_epoch,
        "not_learning": bool(last["train_acc"] < 0.2),
        "overfit_observed": bool(min_loss_epoch is not None
                                 and min_loss_epoch < last["epoch"]
                                 and last["train_acc"] >= 0.2),
        "final_train_acc": last["train_acc"], "final_val_acc": last["val_acc"],
        "seconds": time.perf_counter() - t_start,
        "seconds_per_epoch": (time.perf_counter() - t_start) / last["epoch"],
    }


def analyze(run):
    """Bir koşunun durumunu ve overfitting özetini kayıtlı geçmişten türetir.

    - "çöktü": ağ bir süre öğrenmiş (train >= %50) ama sonra şansa (< %20)
      düşmüştür (ör. yüksek öğrenme oranında ölü ReLU).
    - "öğrenmedi": train başarısı hiç %50'ye ulaşmamış ve sonda < %20.
    Overfitting (validation kaybının en küçük olduğu epoch) çöküşten önceki
    noktalarla hesaplanır.
    """
    hist = run["history"]
    collapse_epoch, valid = None, hist
    seen_good = False
    for i, r in enumerate(hist):
        if r["train_acc"] >= 0.5:
            seen_good = True
        elif seen_good and r["train_acc"] < 0.2:
            collapse_epoch, valid = r["epoch"], hist[:i]
            break
    final = hist[-1]["train_acc"]
    if run.get("diverged"):
        status = "diverge"
    elif collapse_epoch is not None:
        status = f"çöktü({collapse_epoch})"
    elif final < 0.2:
        status = "öğrenmedi"
    elif run["reached_target"]:
        status = "evet"
    else:
        status = "hayır"
    finite = [r for r in valid if np.isfinite(r["val_loss"])]
    min_ep = min(finite, key=lambda r: r["val_loss"])["epoch"] if finite else None
    overfit = bool(min_ep is not None and min_ep < valid[-1]["epoch"]
                   and valid[-1]["train_acc"] >= 0.2)
    return {"status": status, "min_val_loss_epoch": min_ep, "overfit": overfit}


def summary_table(runs):
    """Koşuların özetini düz metin tablo olarak döndürür."""
    head = (f"{'katmanlar':>12s} {'aktiv.':>8s} {'lr':>5s} {'epoch':>6s} "
            f"{'%100?':>11s} {'son tr':>7s} {'son val':>8s} {'en iyi val':>11s} "
            f"{'@epoch':>7s} {'min val loss @':>15s} {'overfit':>8s}")
    lines = [head, "-" * len(head)]
    for r in runs:
        a = analyze(r)
        lines.append(
            f"{str(r['hidden_sizes']):>12s} {r['activation']:>8s} {r['lr']:>5} "
            f"{r['epochs_run']:>6d} {a['status']:>11s} {r['final_train_acc']:>7.4f} "
            f"{r['final_val_acc']:>8.4f} {r['best_val_acc']:>11.4f} "
            f"{r['best_val_epoch']:>7d} {str(a['min_val_loss_epoch']):>15s} "
            f"{'evet' if a['overfit'] else 'hayır':>8s}")
    return "\n".join(lines)


def save_runs(runs, json_path, txt_path):
    """Koşuları JSON olarak ve özet tabloyu .txt olarak kaydeder."""
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(runs, f, indent=1)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary_table(runs) + "\n")


def write_summary(runs, txt_path):
    """Özet tabloyu yalnızca .txt olarak (yeniden) yazar."""
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary_table(runs) + "\n")


def load_runs(json_path):
    """save_runs ile kaydedilmiş koşuları okur."""
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)
