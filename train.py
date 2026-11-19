"""Ödev deneylerini çalıştırır: (d), (e) ve (f) maddeleri.

Kullanım:
    python train.py                 # tüm deneyler (de + f)
    python train.py --part de       # yalnızca iki gizli katmanlı ağ (d, e)
    python train.py --part f        # yalnızca tek gizli katmanlı ağ (f)
    python train.py --max-epochs 30 # hızlı deneme
"""

import argparse
import os

from dataset import get_splits
from experiment import load_runs, run_until_perfect, save_runs, summary_table
from plots import plot_accuracy_curves, plot_overlay

RESULTS = "results"
ACTIVATIONS = ["sigmoid", "relu"]
LEARNING_RATES = [0.1, 0.3, 0.5, 0.7]
SINGLE_HIDDEN_SIZES = [32, 64, 128]
DE_JSON = os.path.join(RESULTS, "de_runs.json")
F_JSON = os.path.join(RESULTS, "f_runs.json")


def run_part_de(data, args):
    """(d) ve (e): 784-256-64-10 ağı, sigmoid/ReLU x lr 0.1/0.3/0.5/0.7."""
    runs = []
    for act in ACTIVATIONS:
        for lr in LEARNING_RATES:
            print(f"\n=== İki gizli katman [256, 64] | {act} | lr={lr} ===")
            model_path = os.path.join(RESULTS, "models", f"two_hidden_{act}_lr{lr}.npz")
            runs.append(run_until_perfect(
                data, [256, 64], act, lr, max_epochs=args.max_epochs, step=args.step,
                batch_size=args.batch_size, seed=args.seed, model_path=model_path))
    save_runs(runs, DE_JSON, os.path.join(RESULTS, "de_summary.txt"))
    for act in ACTIVATIONS:
        subset = [r for r in runs if r["activation"] == act]
        plot_accuracy_curves(
            subset, f"İki gizli katmanlı MLP (256-64), {act} aktivasyonu",
            os.path.join(RESULTS, f"fig_two_hidden_{act}.png"),
            lambda r: f"lr = {r['lr']}")
    print("\n" + summary_table(runs))
    return runs


def best_config(runs):
    """En yüksek validation başarısına ulaşan (aktivasyon, lr) çiftini seçer."""
    best = max(runs, key=lambda r: r["best_val_acc"])
    return best["activation"], best["lr"]


def run_part_f(data, args, activation, lr):
    """(f): tek gizli katman, 32/64/128 nöron."""
    runs = []
    for h in SINGLE_HIDDEN_SIZES:
        print(f"\n=== Tek gizli katman [{h}] | {activation} | lr={lr} ===")
        model_path = os.path.join(RESULTS, "models", f"one_hidden_{h}_{activation}_lr{lr}.npz")
        runs.append(run_until_perfect(
            data, [h], activation, lr, max_epochs=args.max_epochs, step=args.step,
            batch_size=args.batch_size, seed=args.seed, model_path=model_path))
    save_runs(runs, F_JSON, os.path.join(RESULTS, "f_summary.txt"))
    plot_accuracy_curves(
        runs, f"Tek gizli katmanlı MLP, {activation}, lr = {lr}",
        os.path.join(RESULTS, "fig_one_hidden_panels.png"),
        lambda r: f"{r['hidden_sizes'][0]} nöron")
    plot_overlay(
        runs, f"Tek gizli katman: nöron sayısının etkisi ({activation}, lr = {lr})",
        os.path.join(RESULTS, "fig_one_hidden_overlay.png"),
        lambda r: f"{r['hidden_sizes'][0]} nöron")
    print("\n" + summary_table(runs))
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--part", choices=["all", "de", "f"], default="all")
    parser.add_argument("--max-epochs", type=int, default=300,
                        help="train başarısı %%100 olmazsa durma sınırı")
    parser.add_argument("--step", type=int, default=5,
                        help="kaç epoch'ta bir ölçüm/kayıt yapılacağı (10'un "
                             "katları ödevdeki 10'ar epoch ızgarasını içerir)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = get_splits()
    if args.part in ("all", "de"):
        de_runs = run_part_de(data, args)
    else:
        de_runs = load_runs(DE_JSON)
    if args.part in ("all", "f"):
        act, lr = best_config(de_runs)
        print(f"\n(f) için (d)-(e)'deki en iyi ayar kullanılıyor: {act}, lr={lr}")
        run_part_f(data, args, act, lr)


if __name__ == "__main__":
    main()
