---
title: Estimasi kedalaman
seo_title: Estimasi kedalaman monokular dalam LibreYOLO
description: >-
  Memprediksi peta kedalaman relatif yang padat dari satu gambar di LibreYOLO.
  Bandingkan keluarga kedalaman, baca metrik kedalaman, dan ekspor model
  kedalaman.
lead: >-
  Estimasi kedalaman memprediksi seberapa jauh setiap piksel dari kamera
  menggunakan satu gambar. LibreYOLO mengeksposnya sebagai task kedalaman, yang
  mengembalikan peta kedalaman invers relatif yang padat di kanvas gambar asli.
keywords:
  - Estimasi kedalaman monokular python
  - peta kedalaman dari satu gambar
  - model kedalaman relatif
  - depth anything libreyolo
  - prediksi kedalaman padat
last_verified: 1.5.0
snippets:
  predict:
    - label: Memprediksi peta kedalaman
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.data.shape)              # (H, W) di kanvas asli
        print(depth.min, depth.max, depth.mean)
    - label: Bekerja dengan nilai-nilai tersebut
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")

        result = model(SAMPLE_IMAGE)


        depth = result.depth_map

        raw = depth.data          # semakin tinggi semakin dekat; tidak ada
        satuan metrik, tidak ada skala

        gray = depth.normalized() # diubah skala ke [0, 1] untuk visualisasi

        print(raw.shape, float(gray.max()))
    - label: Alternatif yang ringkas
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Kontrak task yang sama, jaringan yang jauh lebih kecil dibangun untuk
        runtime edge.

        model = LibreYOLO("LibreZipDepthb-depth.pt")

        result = model(SAMPLE_IMAGE)


        print(result.depth_map.data.shape)
  val:
    - label: Validasi dan baca kunci metrik
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])   # kebugaran
        print(metrics["metrics/delta2"], metrics["metrics/delta3"])
  export:
    - label: Export
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        model.export(format="onnx")
    - label: Jalankan file yang diekspor
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Pabrik mengarahkan berdasarkan sufiks file, sehingga artefak yang
        diekspor dimuat

        # seperti checkpoint apa pun dan mengembalikan objek Results yang sama.

        model = LibreYOLO("LibreDepthAnythingV2s-depth.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.depth_map.data.shape)
source_hash: e0612c59f9c999b4
---

## Definisi

`depth` task memprediksi satu nilai per piksel dari satu gambar RGB. LibreYOLO
mendefinisikan nilai itu sebagai kedalaman terbalik relatif: lebih tinggi berarti lebih dekat ke kamera,
dan angka tersebut tidak memiliki satuan metrik dan tidak memiliki skala yang berlaku di antara dua gambar.
Membandingkan kedalaman antara dua piksel dari prediksi yang sama adalah bermakna;
membandingkan suatu nilai dengan nilai dari gambar lain tidak demikian.

Sebuah prediksi mengisi `result.depth_map`, sebuah muatan `DepthMap` yang memegang
array `(H, W)` pada kanvas gambar asli. `.min`, `.max` dan `.mean` membaca
nilai terbatas, dan `.normalized()` mengubah skala peta ke `[0, 1]` untuk ditampilkan.
`result.boxes` tetap kosong, sehingga `conf`, `iou` dan `max_det` tidak berpengaruh,
dan `save=True` menulis gambar berpeta-warna dari peta tersebut alih-alih foto yang diberi anotasi.
foto.

## Model

Enam keluarga melayani `depth`.

[Depth Anything V2](/docs/models/depth-anything-v2) memadukan encoder DINOv2 dengan
adalah decoder DPT dan merupakan default serba guna di sini. Lisensi menentukan
ukuran sama seperti akurasi: checkpoint Kecil adalah Apache-2.0 sedangkan Base dan
Besar bersifat non-komersial, jadi periksa tabel checkpoint di halamannya sebelum
memilih salah satu.

[Depth Anything 3](/docs/models/depth-anything-3) mem-port DA3MONO-LARGE
checkpoint, sebuah transformer biasa tanpa spesialisasi arsitektur untuk kedalaman.

[ZipDepth](/docs/models/zipdepth) adalah tingkatan kompak: CNN yang dapat direparameterisasi
yang didistilasi dari Depth Anything V2 Large, dengan checkpoint kedua yang decodernya
menghindari operasi gather dan unfold untuk compiler NPU yang tidak memilikinya.

[MiDaS](/docs/models/midas) adalah lini kerja yang menetapkan zero-shot
yang digunakan untuk mengukur keluarga lain. Ini adalah satu
kedalaman family LibreYOLO tidak memublikasikan ulang: permintaan checkpoint mengunduh
aset resmi dari rilis GitHub penulisnya dan memeriksa SHA-256 yang dipin.

[LibreMODUS](/docs/models/libremodus) mencapai kedalaman sebagai satu target dari
model apa pun-ke-apa pun daripada sebagai head khusus. Ini membutuhkan `modus` tambahan dan
akun Hugging Face Anda yang terautentikasi, dan tidak menawarkan `val()` maupun
`export()`.

[SenseNova-Vision](/docs/models/sensenova-vision) menghasilkan peta kedalaman sebagai
gambar melalui dekode difusi, dari checkpoint 7B yang sama yang melayani
enam tugas lainnya. Ini membutuhkan `sensenova` tambahan, dan bobotnya dibatasi
untuk penggunaan non-komersial; lisensinya ada di halamannya.

## Prediksi

Bobot diunduh dari Hugging Face pada penggunaan pertama dan disimpan secara lokal dalam cache, kecuali
untuk dua keluarga yang disebutkan di atas.

<code-tabs name="predict" />

Resolusi input dibatasi sesuai family. Depth Anything V2 dan Depth Anything
3 dibangun di atas grid patch DINOv2, jadi `imgsz` harus dapat dibagi habis dengan 14, yang
LibreYOLO periksa sebelum dijalankan. `Results.plot()` tidak mencakup task ini; ini
didefinisikan hanya untuk normal permukaan dan tepi. Lihat [prediction](/docs/predict)
untuk sumber, streaming, dan penanganan hasil.

## Format Dataset

Validasi kedalaman memasangkan setiap gambar dengan peta kedalaman saluran tunggal yang padat yang memiliki
resolusi yang sama, ditemukan dengan menggantikan direktori kedalaman ke dalam gambar
jalur.

```text
dataset/
  data.yaml
  images/
    val/room.jpg
  depths/
    val/room.png
```

```yaml
path: dataset
val: images/val
depths_dir: depths
nc: 1
names: {0: depth}
```

Peta adalah PNG atau TIF saluran tunggal, atau `.npy`. Nilai adalah kedalaman biasa dalam suatu satuan
dataset tetap konsisten, dan `0`, negatif, NaN dan piksel tak hingga ditandai
sampel tidak valid yang dikecualikan dari metrik. Peta bilangan bulat dibagi oleh
`depth_scale`, yang default ke `256.0`, konvensi PNG 16-bit; float
Peta `.npy` digunakan apa adanya. `depth_stem_suffix` dan `depth_mask_suffix`
menutupi dataset yang menamai file kedalaman atau mask validitas mereka dengan cara yang berbeda. Lihat
[dataset memformat ](/docs/reference/dataset-formats) untuk kontrak penuh.

## Latih

Tidak ada kedalaman family di LibreYOLO yang memiliki implementasi pelatihan: `train()` meningkatkan
`NotImplementedError` pada keenamnya. Setiap halaman model menamai skrip konversi
yang mengubah checkpoint yang dilatih di hulu menjadi satu LibreYOLO dapat memuat.

## Validasi

`val()` menjalankan validator kedalaman bersama. Kedalaman relatif tidak memiliki skala absolut,
jadi setiap prediksi pertama-tama disesuaikan dengan invers dari ground truth-nya dengan
skala dan pergeseran kuadrat terkecil per gambar, kemudian dibalik kembali ke kedalaman. Setiap
metrik di bawah dihitung per gambar pada peta yang selaras itu dan dirata-rata di seluruh
dataset, menghitung hanya piksel yang ditandai dataset sebagai valid.

<code-tabs name="val" />

`metrics/abs_rel` adalah kesalahan relatif absolut rata-rata, residu dibagi dengan
kedalaman ground-truth, dan nilai yang lebih rendah lebih baik. `metrics/rmse` adalah akar rata-rata
kuadrat kesalahan dalam satuan kedalaman dataset itu sendiri, juga lebih rendah lebih baik.
`metrics/delta1`, `metrics/delta2` dan `metrics/delta3` adalah akurasi ambang:
fraksi piksel valid yang rasio terhadap ground truth, diambil dalam
arah manapun yang lebih besar, berada di bawah 1,25, 1,25 kuadrat dan 1,25 pangkat tiga, sehingga
nilai yang lebih tinggi lebih baik. `metrics/delta1` juga `fitness`, jumlah
bacaan pemilihan best-checkpoint.

## Ekspor

Model kedalaman yang diekspor dimuat kembali melalui `LibreYOLO()` berdasarkan akhiran file-nya, sehingga
file `.onnx` atau `.engine` berperilaku seperti checkpoint dan mengembalikan `.onnx` yang sama,
dengan `depth_map` menggantikan kotak.

<code-tabs name="export" />

Cakupan berbeda per family, dan Depth Anything 3 menolak format apa pun di luar
set yang divalidasi alih-alih mencoba konversi yang tidak divalidasi. Periksa halaman model
dan [matriks ekspor penuh ](/docs/reference/export-matrix) sebelum
berkomitmen ke target. LibreMODUS dan SenseNova-Vision tidak mengekspor sama sekali.
[Ekspor ](/docs/export) mencantumkan argumen yang diterima setiap format.

