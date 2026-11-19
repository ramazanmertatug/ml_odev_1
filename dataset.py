"""Fashion-MNIST yükleme ve ödevde istenen bölmelerin hazırlanması.

Veri yüklemek için Keras kullanılır (ödev bu işlem için izin veriyor).
İlk çalıştırmada veri indirilir ve Keras önbelleğine (~/.keras) kaydedilir.
"""

import os

import numpy as np

CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


CACHE_PATH = os.path.join("dataset", "fashion_mnist_raw.npz")


def load_raw():
    """Ham Fashion-MNIST'i yükler (ilk indirmeden sonra dataset/ altında
    önbelleğe alınır).

    Dönüş: (x_train, y_train), (x_test, y_test); görüntüler (n, 28, 28) uint8.
    """
    if os.path.exists(CACHE_PATH):
        d = np.load(CACHE_PATH)
        return (d["x_tr"], d["y_tr"]), (d["x_te"], d["y_te"])
    (x_tr, y_tr), (x_te, y_te) = _download_raw()
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    np.savez_compressed(CACHE_PATH, x_tr=x_tr, y_tr=y_tr, x_te=x_te, y_te=y_te)
    return (x_tr, y_tr), (x_te, y_te)


def _download_raw():
    """Keras (varsa) veya Scikit-learn ile veriyi indirir."""
    try:
        from tensorflow.keras.datasets import fashion_mnist
    except ImportError:
        return _load_raw_sklearn()
    return fashion_mnist.load_data()


def _load_raw_sklearn():
    """TensorFlow kurulu değilse Scikit-learn ile OpenML'den yükler.

    OpenML'deki Fashion-MNIST (data_id=40996) 70.000 örnektir; ilk 60.000'i
    eğitim, son 10.000'i resmi test kümesidir (Keras ile aynı sıra).
    """
    from sklearn.datasets import fetch_openml
    print("TensorFlow bulunamadı; Fashion-MNIST OpenML'den indiriliyor...")
    ds = fetch_openml(data_id=40996, as_frame=False, parser="liac-arff")
    x = ds.data.astype(np.uint8).reshape(-1, 28, 28)
    y = ds.target.astype(np.int64)
    x_tr, y_tr, x_te, y_te = x[:60000], y[:60000], x[60000:], y[60000:]
    # Doğrulama: resmi test kümesinde her sınıftan tam 1000 örnek vardır.
    if not np.all(np.bincount(y_te, minlength=10) == 1000):
        raise RuntimeError("Veri sırası beklenenden farklı: son 10.000 örnek "
                           "resmi test kümesi değil.")
    return (x_tr, y_tr), (x_te, y_te)


def preprocess(images):
    """(n, 28, 28) görüntüleri [0,1] aralığına normalize edip (784, n)
    matrisine çevirir; her sütun bir örnektir (784x1 giriş vektörü)."""
    x = images.astype(np.float64) / 255.0
    return x.reshape(x.shape[0], -1).T


def class_slice(y, start, stop, num_classes=10):
    """Her sınıfın, veri kümesindeki sırasıyla start..stop-1. örneklerinin
    indislerini birleştirip döndürür (sıra korunur)."""
    idx = []
    for c in range(num_classes):
        class_idx = np.flatnonzero(y == c)
        idx.append(class_idx[start:stop])
    return np.concatenate(idx)


def get_splits(n_train_per_class=1000, n_val_per_class=500):
    """Ödevdeki bölmeleri hazırlar.

    - Eğitim: eğitim kümesinde her sınıfın ilk 1000 görüntüsü (10.000).
    - Validation: her sınıftan 500 örnek (5.000). Ödev kaynağı belirtmediği
      için, eğitim ile örtüşmesin diye eğitim kümesinden her sınıfın
      1000-1499. görüntüleri seçilir.
    - Test: Fashion-MNIST'in resmi 10.000'lik test kümesi (ek raporlama için).

    Dönüş: sözlük; X_* (784, n), y_* (n,).
    """
    (x_tr, y_tr), (x_te, y_te) = load_raw()
    n, v = n_train_per_class, n_val_per_class
    tr_idx = class_slice(y_tr, 0, n)
    val_idx = class_slice(y_tr, n, n + v)
    return {
        "X_train": preprocess(x_tr[tr_idx]), "y_train": y_tr[tr_idx],
        "X_val": preprocess(x_tr[val_idx]), "y_val": y_tr[val_idx],
        "X_test": preprocess(x_te), "y_test": y_te,
    }


if __name__ == "__main__":
    d = get_splits()
    for k, v in d.items():
        extra = f", min={v.min():.2f}, max={v.max():.2f}" if k.startswith("X") else \
                f", sınıf başına={np.bincount(v).tolist()}"
        print(f"{k:8s} shape={v.shape}{extra}")
