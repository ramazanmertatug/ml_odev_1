## Açıklama

Makine Öğrenmesi dersi 1. ödevi: Çok Katmanlı Algılayıcı (MLP).
MLP yalnızca NumPy ile sıfırdan yazılmıştır (mlp.py). Fashion-MNIST
üzerinde performans analizi yapılır. Veri yükleme için TensorFlow/Keras
(kuruluysa) veya Scikit-learn (OpenML), grafikler için Matplotlib, confusion
matrix için Scikit-learn kullanılır.

## Gereksinimler

pip install -r requirements.txt

(İlk çalıştırmada Fashion-MNIST internetten indirilir ve dataset/ klasörüne
önbelleğe alınır; sonraki çalıştırmalarda internet gerekmez.)

## Çalıştırma

1) MLP doğrulama testi (gradient check):

    python test_mlp.py

2) Deneyler: (d) ve (e) iki gizli katmanlı ağ (256-64; sigmoid ve ReLU;
   öğrenme oranı 0.1, 0.3, 0.5, 0.7) ve (f) tek gizli katmanlı ağ (32, 64, 128):

    python train.py

   Seçenekler:
    --part de|f|all     hangi maddelerin çalışacağı (varsayılan: all)
    --max-epochs N      train başarısı %100 olmazsa durma sınırı (varsayılan 300)
    --step N            kaç epoch'ta bir ölçüm alınacağı (varsayılan 5)
    --batch-size N      mini-batch boyutu (varsayılan 32)
    --seed N            rastgelelik tohumu (varsayılan 0)

3) (g) karşılaştırma ve (h) confusion matrix (train.py'den sonra çalıştırılır):

    python eval.py

## Çıktılar (results/ klasörü)

- de_summary.txt, f_summary.txt : (d), (e), (f) özet tabloları
- de_runs.json, f_runs.json     : her ölçümün train/val başarısı ve kaybı
- eval_summary.txt              : (g) karşılaştırma tablosu, (h) sınıf bazında rapor
- fig_two_hidden_*.png          : (d), (e) train/validation grafikleri
- fig_one_hidden_*.png          : (f) grafikleri
- fig_compare_one_two.png       : (g) karşılaştırma grafiği
- fig_confusion_val.png, fig_confusion_test.png : (h) confusion matrix
- models/                       : en iyi validation başarısındaki ağırlıklar

## Dosya Düzeni

proje_klasoru/
 |- mlp.py            MLP sınıfı (yalnızca NumPy)
 |- dataset.py        veri yükleme, normalizasyon, eğitim/validation bölmesi
 |- experiment.py     eğitim döngüsü ve sonuç özetleri
 |- plots.py          grafik fonksiyonları
 |- train.py          (d), (e), (f) deneyleri
 |- eval.py           (g), (h) karşılaştırma ve confusion matrix
 |- test_mlp.py       gradient check ve ezberleme testi
 |- requirements.txt
 |- ReadMe.txt
 |- dataset/          indirilen veri (önbellek)
 |- results/          sayısal sonuçlar, grafikler, modeller

## Varsayımlar

- Eğitim kümesi: her sınıfın ilk 1000 görüntüsü (10.000).
- Validation kümesi: her sınıfın 1000-1499. görüntüleri (5.000); eğitimle
  örtüşmez.
- Batch size 32; ağırlık başlatma: sigmoid için Xavier, ReLU için He.
- (f) için aktivasyon ve öğrenme oranı, (d)-(e)'de en yüksek validation
  başarısına ulaşan ayardır.
