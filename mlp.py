"""Çok Katmanlı Algılayıcı (MLP) - yalnızca NumPy ile.

Veri düzeni: X matrisi (özellik_sayısı, örnek_sayısı) şeklindedir; yani her
örnek bir sütundur (Fashion-MNIST için 784x1). Etiketler tam sayı vektörü
(örnek_sayısı,) olarak verilir, one-hot dönüşümü sınıf içinde yapılır.

Mimari: giriş -> [gizli katmanlar, parametrik] -> softmax çıkış
Kayıp : cross-entropy
Eğitim: mini-batch stokastik gradyan inişi (SGD)
"""

import time

import numpy as np


# ---------------------------------------------------------------------------
# Aktivasyon fonksiyonları ve türevleri
# ---------------------------------------------------------------------------
def sigmoid(z):
    """Sayısal olarak kararlı sigmoid."""
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def sigmoid_grad(z, a):
    """Sigmoid türevi; a = sigmoid(z) önceden hesaplanmış çıktıdır."""
    return a * (1.0 - a)


def relu(z):
    """ReLU: max(0, z)."""
    return np.maximum(0.0, z)


def relu_grad(z, a):
    """ReLU türevi: z > 0 ise 1, değilse 0."""
    return (z > 0).astype(z.dtype)


def tanh(z):
    """Hiperbolik tanjant."""
    return np.tanh(z)


def tanh_grad(z, a):
    """tanh türevi; a = tanh(z)."""
    return 1.0 - a * a


ACTIVATIONS = {
    "sigmoid": (sigmoid, sigmoid_grad),
    "relu": (relu, relu_grad),
    "tanh": (tanh, tanh_grad),
}


def softmax(z):
    """Sütun bazında softmax (max çıkarılarak kararlı hale getirilmiştir)."""
    z = z - np.max(z, axis=0, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=0, keepdims=True)


def one_hot(y, num_classes):
    """(m,) tam sayı etiketlerini (num_classes, m) one-hot matrisine çevirir."""
    Y = np.zeros((num_classes, y.shape[0]))
    Y[y, np.arange(y.shape[0])] = 1.0
    return Y


# ---------------------------------------------------------------------------
# MLP sınıfı
# ---------------------------------------------------------------------------
class MLP:
    """Parametrik çok katmanlı algılayıcı.

    Parametreler
    ------------
    input_size : int
        Giriş katmanı nöron sayısı (Fashion-MNIST için 784).
    hidden_sizes : list[int]
        Gizli katmanların nöron sayıları. Uzunluğu gizli katman sayısını
        belirler. Örn. [256, 64] -> iki gizli katman.
    output_size : int
        Çıkış katmanı nöron sayısı (sınıf sayısı, 10).
    activation : str | list[str]
        Gizli katman aktivasyonu: "sigmoid", "relu" veya "tanh". Tek bir
        string tüm gizli katmanlara uygulanır; liste verilirse her katman
        için ayrı aktivasyon kullanılır.
    seed : int
        Ağırlık başlatma ve karıştırma için rastgelelik tohumu.
    dtype : numpy dtype
        Hesaplama tipi (varsayılan float64).
    """

    def __init__(self, input_size, hidden_sizes, output_size,
                 activation="relu", seed=0, dtype=np.float64):
        self.input_size = input_size
        self.hidden_sizes = list(hidden_sizes)
        self.output_size = output_size
        self.dtype = dtype
        self.rng = np.random.default_rng(seed)

        n_hidden = len(self.hidden_sizes)
        if isinstance(activation, str):
            activation = [activation] * n_hidden
        if len(activation) != n_hidden:
            raise ValueError("activation listesinin uzunluğu gizli katman "
                             "sayısına eşit olmalı.")
        for name in activation:
            if name not in ACTIVATIONS:
                raise ValueError(f"Bilinmeyen aktivasyon: {name}")
        self.activation_names = list(activation)

        self.layer_sizes = [input_size] + self.hidden_sizes + [output_size]
        self.W, self.b = [], []
        self.epochs_done = 0
        self._init_parameters()

    # -- kaydet / yükle ----------------------------------------------------
    def save(self, path):
        """Mimariyi ve ağırlıkları .npz dosyasına kaydeder."""
        arrays = {f"W{l}": w for l, w in enumerate(self.W)}
        arrays.update({f"b{l}": b for l, b in enumerate(self.b)})
        np.savez(path, input_size=self.input_size,
                 hidden_sizes=np.array(self.hidden_sizes),
                 output_size=self.output_size,
                 activation=np.array(self.activation_names),
                 epochs_done=self.epochs_done, **arrays)

    @classmethod
    def load(cls, path):
        """save() ile kaydedilmiş bir modeli yükler."""
        d = np.load(path)
        net = cls(int(d["input_size"]), d["hidden_sizes"].tolist(),
                  int(d["output_size"]), activation=d["activation"].tolist())
        for l in range(len(net.W)):
            net.W[l] = d[f"W{l}"]
            net.b[l] = d[f"b{l}"]
        net.epochs_done = int(d["epochs_done"])
        return net

    # -- başlatma ----------------------------------------------------------
    def _init_parameters(self):
        """Xavier (sigmoid/tanh/softmax) veya He (ReLU) ile ağırlık başlatır."""
        n_layers = len(self.layer_sizes) - 1
        for l in range(n_layers):
            fan_in, fan_out = self.layer_sizes[l], self.layer_sizes[l + 1]
            act = self.activation_names[l] if l < n_layers - 1 else "softmax"
            if act == "relu":
                std = np.sqrt(2.0 / fan_in)                 # He
            else:
                std = np.sqrt(2.0 / (fan_in + fan_out))     # Xavier
            self.W.append(self.rng.normal(0.0, std, (fan_out, fan_in))
                          .astype(self.dtype))
            self.b.append(np.zeros((fan_out, 1), dtype=self.dtype))

    def num_parameters(self):
        """Toplam eğitilebilir parametre sayısı."""
        return int(sum(w.size + b.size for w, b in zip(self.W, self.b)))

    # -- ileri yayılım ------------------------------------------------------
    def forward(self, X):
        """İleri yayılım.

        X: (input_size, m). Dönüş: (olasılıklar (output_size, m), cache).
        cache, geri yayılım için her katmanın (Z, A) değerlerini tutar.
        """
        A = X.astype(self.dtype, copy=False)
        cache = {"A": [A], "Z": []}
        n_layers = len(self.W)
        for l in range(n_layers):
            Z = self.W[l] @ A + self.b[l]
            if l < n_layers - 1:
                A = ACTIVATIONS[self.activation_names[l]][0](Z)
            else:
                A = softmax(Z)
            cache["Z"].append(Z)
            cache["A"].append(A)
        return A, cache

    # -- kayıp --------------------------------------------------------------
    @staticmethod
    def cross_entropy(probs, Y):
        """Ortalama cross-entropy kaybı. probs, Y: (sınıf, m)."""
        eps = 1e-12
        return float(-np.sum(Y * np.log(probs + eps)) / probs.shape[1])

    def loss(self, X, y):
        """Verilen (X, y) üzerinde ortalama cross-entropy."""
        probs, _ = self.forward(X)
        return self.cross_entropy(probs, one_hot(y, self.output_size))

    # -- geri yayılım -------------------------------------------------------
    def backward(self, cache, Y):
        """Geri yayılım. Dönüş: (dW listesi, db listesi).

        Softmax + cross-entropy birleşiminde çıkış katmanı hatası
        dZ_L = (A_L - Y) / m şeklindedir.
        """
        m = Y.shape[1]
        n_layers = len(self.W)
        dW = [None] * n_layers
        db = [None] * n_layers

        dZ = (cache["A"][-1] - Y) / m
        for l in range(n_layers - 1, -1, -1):
            dW[l] = dZ @ cache["A"][l].T
            db[l] = np.sum(dZ, axis=1, keepdims=True)
            if l > 0:
                dA = self.W[l].T @ dZ
                grad_fn = ACTIVATIONS[self.activation_names[l - 1]][1]
                dZ = dA * grad_fn(cache["Z"][l - 1], cache["A"][l])
        return dW, db

    # -- eğitim -------------------------------------------------------------
    def fit(self, X, y, epochs, lr, batch_size=32,
            X_val=None, y_val=None, log_every=1, verbose=False):
        """Mini-batch SGD ile eğitir.

        Fonksiyon tekrar çağrılırsa eğitim kaldığı yerden devam eder
        (ağırlıklar ve rastgelelik durumu korunur). Böylece "10 epoch, sonra
        10 epoch daha..." deneyi tek bir koşu olarak yürütülebilir.

        Dönüş: epoch başına sözlük listesi
        [{"epoch", "train_loss", "train_acc", "val_loss", "val_acc"}, ...]
        (val_* yalnızca X_val verilmişse bulunur). "epoch", bu çağrı
        içindeki sıra değil, ağın toplam gördüğü epoch sayısıdır.
        """
        m = X.shape[1]
        Y = one_hot(y, self.output_size)
        history = []

        for _ in range(epochs):
            t0 = time.time()
            perm = self.rng.permutation(m)
            for start in range(0, m, batch_size):
                idx = perm[start:start + batch_size]
                _, cache = self.forward(X[:, idx])
                dW, db = self.backward(cache, Y[:, idx])
                for l in range(len(self.W)):
                    self.W[l] -= lr * dW[l]
                    self.b[l] -= lr * db[l]
            self.epochs_done += 1

            if self.epochs_done % log_every == 0:
                rec = {"epoch": self.epochs_done}
                rec["train_loss"], rec["train_acc"] = self.evaluate(X, y)
                if X_val is not None:
                    rec["val_loss"], rec["val_acc"] = self.evaluate(X_val, y_val)
                rec["seconds"] = time.time() - t0
                history.append(rec)
                if verbose:
                    msg = (f"epoch {rec['epoch']:4d} | "
                           f"train loss {rec['train_loss']:.4f} acc {rec['train_acc']:.4f}")
                    if X_val is not None:
                        msg += (f" | val loss {rec['val_loss']:.4f} "
                                f"acc {rec['val_acc']:.4f}")
                    print(msg)
        return history

    # -- tahmin / değerlendirme --------------------------------------------
    def predict_proba(self, X):
        """Sınıf olasılıkları (output_size, m)."""
        probs, _ = self.forward(X)
        return probs

    def predict(self, X):
        """Tahmin edilen sınıf indisleri (m,)."""
        return np.argmax(self.predict_proba(X), axis=0)

    def evaluate(self, X, y):
        """(ortalama cross-entropy, doğruluk) çifti döndürür."""
        probs = self.predict_proba(X)
        loss = self.cross_entropy(probs, one_hot(y, self.output_size))
        acc = float(np.mean(np.argmax(probs, axis=0) == y))
        return loss, acc

    def accuracy(self, X, y):
        """Doğruluk oranı [0, 1]."""
        return float(np.mean(self.predict(X) == y))
