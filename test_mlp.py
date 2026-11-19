"""MLP doğrulama testleri: gradient check ve küçük veride öğrenme kontrolü.

Çalıştırma: python test_mlp.py
"""

import numpy as np

from mlp import MLP, one_hot


def gradient_check(activation, hidden_sizes=(7, 5), eps=1e-6, seed=1):
    """Geri yayılım gradyanını merkezi sayısal türevle karşılaştırır.

    Dönüş: en büyük göreli hata (küçük olmalı, ~1e-7 veya daha az).
    """
    rng = np.random.default_rng(seed)
    n_in, n_out, m = 6, 4, 9
    net = MLP(n_in, hidden_sizes, n_out, activation=activation, seed=seed)
    X = rng.normal(size=(n_in, m))
    y = rng.integers(0, n_out, size=m)
    Y = one_hot(y, n_out)

    _, cache = net.forward(X)
    dW, db = net.backward(cache, Y)

    max_rel_err = 0.0
    for params, grads in ((net.W, dW), (net.b, db)):
        for P, G in zip(params, grads):
            it = np.nditer(P, flags=["multi_index"], op_flags=["readwrite"])
            for _ in it:
                i = it.multi_index
                old = P[i]
                P[i] = old + eps
                loss_plus = net.loss(X, y)
                P[i] = old - eps
                loss_minus = net.loss(X, y)
                P[i] = old
                num = (loss_plus - loss_minus) / (2 * eps)
                ana = G[i]
                denom = max(abs(num) + abs(ana), 1e-12)
                max_rel_err = max(max_rel_err, abs(num - ana) / denom)
    return max_rel_err


def overfit_small_batch(activation, lr):
    """Çok küçük bir veri kümesine ezberleme testi (doğruluk -> 1.0 olmalı)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 40))
    y = rng.integers(0, 5, size=40)
    net = MLP(20, [32, 16], 5, activation=activation, seed=0)
    net.fit(X, y, epochs=400, lr=lr, batch_size=40)
    return net.accuracy(X, y)


if __name__ == "__main__":
    ok = True
    for act in ("sigmoid", "relu", "tanh"):
        err = gradient_check(act)
        status = "OK" if err < 1e-5 else "HATA"
        ok &= err < 1e-5
        print(f"[gradient check] {act:8s} max göreli hata = {err:.2e}  {status}")

    for act, lr in (("sigmoid", 0.5), ("relu", 0.1)):
        acc = overfit_small_batch(act, lr)
        status = "OK" if acc > 0.99 else "HATA"
        ok &= acc > 0.99
        print(f"[ezberleme testi] {act:8s} train acc = {acc:.3f}  {status}")

    print("\nTÜM TESTLER GEÇTİ" if ok else "\nBAZI TESTLER BAŞARISIZ")
