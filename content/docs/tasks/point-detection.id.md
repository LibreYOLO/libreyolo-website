---
title: Deteksi titik
seo_title: Deteksi titik dan penghitungan di LibreYOLO
description: >-
  Lokalisasi objek sebagai satu titik, bukan box, di LibreYOLO. Prediksi
  centroid, hitung objek, latih FOMO, dan baca metrik point.
lead: >-
  Deteksi titik mengembalikan satu lokasi x, y per objek sebagai pengganti
  bounding box. LibreYOLO menyediakannya sebagai task point, dan prediksi memuat
  satu baris x, y, class, serta confidence per objek.
keywords:
  - deteksi titik Python
  - menghitung objek Python
  - deteksi centroid
  - lokalisasi point FOMO
  - object counting gambar
  - point localization
last_verified: 1.5.0
snippets:
  predict:
    - label: Prediksi titik dan hitung objek
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Bobot LibreFOMO tidak diunduh otomatis. Ambil checkpoint dari

        # https://huggingface.co/LibreYOLO dahulu dan muat berdasarkan path
        lokal.

        model = LibreYOLO("./LibreFOMOs-point.pt")

        result = model(SAMPLE_IMAGE, save=True)


        points = result.points

        print(len(points))     # jumlah objek

        print(points.xy)       # (N, 2) pusat dalam piksel gambar asli

        print(points.cls, points.conf)
    - label: Koordinat ternormalisasi dan jumlah per class
      language: python
      code: >
        from collections import Counter


        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("./LibreFOMOs-point.pt")

        result = model(SAMPLE_IMAGE)


        points = result.points.numpy()

        print(points.xyn)                          # pusat yang sama dalam [0,
        1]

        print(Counter(points.cls.astype(int).tolist()))
  train:
    - label: Latih FOMO pada dataset YOLO
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        model.train(data="my-dataset.yaml", epochs=40, batch=32, lr0=3e-4)
    - label: Prediksi dengan checkpoint hasil pelatihan
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("./LibreFOMOs-point.pt")

        results = model.train(data="my-dataset.yaml", epochs=40)


        # train() memuat ulang checkpoint terbaik ke objek yang sama, sehingga

        # model memprediksi dengan bobot hasil pelatihan saat pemanggilan
        selesai.

        print(results["best_checkpoint"])

        print(model(SAMPLE_IMAGE).points.xy)
  val:
    - label: Validasi dan baca key metrik
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/precision"], metrics["metrics/recall"])
        print(metrics["metrics/f1"])
        print(metrics["metrics/mAP@[0.01:0.10]"])   # fitness
        print(metrics["metrics/MLE"])               # mean localization error
        print(metrics["metrics/MAE"], metrics["metrics/RMSE"])   # error jumlah
    - label: Ubah ambang batas jarak
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("./LibreFOMOs-point.pt")


        # Batas sweep menjadi bagian teks key, sehingga sweep kustom

        # mengubah nama key mAP yang dihasilkannya.

        metrics = model.val(data="my-dataset.yaml", dist_thresholds=[0.02,
        0.05])


        print(metrics["metrics/mAP@0.02"])

        print(metrics["metrics/mAP@[0.02:0.05]"])
  export:
    - label: Ekspor
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        model.export(format="onnx")
    - label: Jalankan file hasil ekspor
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Factory merutekan berdasarkan akhiran file, sehingga artefak hasil

        # ekspor dimuat seperti checkpoint dan mengembalikan objek Results yang
        sama.

        model = LibreYOLO("./LibreFOMOs-point.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.points.xy)
source_hash: 932153c8870d1c7c
---

## Definisi

Task `point` melokalisasi setiap objek dengan satu koordinat x, y dan class,
tanpa lebar, tinggi, atau mask. Karena prediksi merupakan daftar datar objek,
jumlah baris adalah jumlah objek, yang menjadikannya task penghitungan.

Prediksi mengisi `result.points`, payload `Points` yang membungkus array `(N, 4)`
berisi baris `x, y, class, confidence` dalam piksel gambar asli. `.xy`
mengembalikan koordinat, `.xyn` mengembalikan koordinat yang sama dibagi ukuran
gambar, `.cls` mengembalikan indeks class, dan `.conf` mengembalikan skor;
`len()` mengembalikan jumlah titik. `result.boxes` tetap kosong, sehingga `iou`
dan `max_det` tidak memiliki apa pun untuk diproses.

## Model

Tiga family melayani `point`, dan ketiganya tidak dapat saling dipertukarkan.

[FOMO](/docs/models/fomo) adalah pilihan vocabulary tetap: grid classifier yang
memberi label setiap sel grid beresolusi rendah sebagai background atau pusat
objek. Ini satu-satunya family point yang dapat dilatih LibreYOLO dan satu-satunya
yang dapat diekspor.

[LocateAnything](/docs/models/locate-anything) menerima teks, bukan indeks class,
sehingga vocabulary-nya adalah frasa apa pun yang Anda tulis. Model ini memerlukan
extra `vlm`, dibuat sebagai `LibreLocateAnything` dan bukan melalui factory
`LibreYOLO()`, serta bobotnya dibatasi untuk penggunaan nonkomersial. Ketentuan
persis dan dua lisensi tambahan yang digabungkan checkpoint tersedia pada halamannya.

[SenseNova-Vision](/docs/models/sensenova-vision) menjalankan `point` melalui
checkpoint prompted-generation yang sama untuk enam task lainnya, dimuat dengan
`LibreVLM("sensenova-vision", task="point")`. Model ini memerlukan extra
`sensenova`, dan setiap prediksi merupakan generation pass pada model 7B, sehingga
latensi per gambar jauh lebih tinggi daripada detector khusus. Bobotnya
nonkomersial; lisensinya tersedia pada halaman model.

## Prediksi

Bobot LibreFOMO adalah satu-satunya pengecualian untuk unduhan otomatis di situs
ini. `LibreYOLO("LibreFOMOs-point.pt")` mencari file tersebut di disk dan
memunculkan `ValueError` yang menyebut namanya, bukan mengunduhnya. Unduh checkpoint
dari [organisasi LibreYOLO](https://huggingface.co/LibreYOLO) di Hugging Face
terlebih dahulu dan muat berdasarkan path lokal, atau latih model sendiri.

<code-tabs name="predict" />

Nama file harus memuat akhiran task `-point` agar loader mengenalinya.
`predict(..., nms_radius=1)` mengontrol jarak minimum dalam sel grid agar dua
deteksi FOMO sama-sama bertahan. Lihat [prediksi](/docs/predict) untuk sumber,
streaming, dan penanganan hasil.

## Format dataset

`point` tidak memiliki format label sendiri. Family point membaca tata letak
deteksi YOLO standar dan menurunkan satu pusat dari setiap baris box, sehingga
`cx cy` adalah titiknya dan `w h` hanya menentukan apakah baris valid.

```text
dataset/
  data.yaml
  images/
    train/scene.jpg
    val/scene.jpg
  labels/
    train/scene.txt
    val/scene.txt
```

Setiap file label memuat satu baris per objek dengan koordinat ternormalisasi:

```text
<class_id> <cx> <cy> <w> <h>
```

```yaml
path: dataset
train: images/train
val: images/val
nc: 1
names: {0: seedling}
```

File label yang tidak ada atau kosong berarti tidak ada objek. Lihat
[format dataset](/docs/reference/dataset-formats) untuk kontrak lengkap.

## Pelatihan

FOMO adalah satu-satunya family point dengan implementasi pelatihan. `train()`
pada LocateAnything dan SenseNova-Vision memunculkan `NotImplementedError`;
lakukan fine-tuning di upstream dan muat hasilnya.

<code-tabs name="train" />

`imgsz` bukan pilihan bebas untuk FOMO: default-nya adalah resolusi native dari
checkpoint yang dimuat, dan memberikan nilai berbeda memunculkan `ValueError`
yang menyebutkan ukuran yang diharapkan. Lihat [pelatihan](/docs/train) untuk
dataset, logger, dan multi-GPU, serta [halaman FOMO](/docs/models/fomo) untuk
default family ini.

## Validasi

`val()` mencocokkan titik prediksi dengan titik ground truth satu lawan satu
menggunakan algoritma Hungarian pada sweep ambang batas jarak. Ambang batas
adalah jarak Euclidean dalam koordinat gambar ternormalisasi, dan sweep default
terdiri dari sepuluh nilai dari 0,01 hingga 0,10.

<code-tabs name="val" />

`metrics/precision`, `metrics/recall`, dan `metrics/f1` dirata-ratakan secara
makro pada semua class di ambang batas paling ketat dalam sweep, yaitu 0,01 secara
default. `metrics/mAP@0.01` adalah average precision pada ambang batas yang sama,
dan `metrics/mAP@[0.01:0.10]` adalah mean untuk seluruh sweep. Nilai sweep tersebut
juga merupakan `fitness`, angka yang dibaca saat memilih checkpoint terbaik.
Kedua key mAP dibangun dari ambang batas yang digunakan, sehingga memberikan
`dist_thresholds=` akan mengubah namanya.

`metrics/MLE` adalah mean jarak antara pasangan yang cocok pada ambang batas
paling ketat, dalam satuan ternormalisasi yang sama. `metrics/MAE` dan
`metrics/RMSE` merupakan metrik penghitungan, bukan lokalisasi: keduanya mengukur
perbedaan per gambar antara jumlah titik prediksi dan ground truth.

FOMO menambahkan kelompok kedua pada tingkat grid di atas metrik tersebut. Model
melakukan sweep confidence dan `nms_radius`, lalu menerbitkan kombinasi F1 terbaik
sebagai `metrics/grid_F1`, `metrics/grid_precision`, `metrics/grid_recall`,
`metrics/grid_mean_distance`, `metrics/grid_TP`, `metrics/grid_FP`, dan
`metrics/grid_FN`, dengan pengaturan yang menghasilkannya dalam `decode/threshold`
serta `decode/nms_radius`.

## Ekspor

FOMO diekspor melalui jalur ekspor bersama, dan artefak hasil ekspor dimuat
kembali melalui `LibreYOLO()` berdasarkan akhiran filenya, sehingga file `.onnx`
atau `.engine` berperilaku seperti checkpoint dan mengembalikan `Results` yang sama.

<code-tabs name="export" />

Cakupan per format tersedia di [halaman FOMO](/docs/models/fomo) dan
[matriks ekspor lengkap](/docs/reference/export-matrix). LocateAnything dan
SenseNova-Vision tidak dapat diekspor: `export()` memunculkan error pada keduanya
karena model generatif tidak memiliki graph deteksi yang dapat di-trace.
