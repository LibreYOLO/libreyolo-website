---
title: FOMO
families:
  - fomo
seo_title: 'FOMO: lokalisasi titik, pelatihan, dan ekspor di LibreYOLO'
description: >-
  Jalankan FOMO (Faster Objects, More Objects) di LibreYOLO: detector lokalisasi
  titik berukuran kecil untuk menghitung banyak objek kecil. Instal, prediksi,
  latih, dan ekspor.
lead: >-
  FOMO adalah pelokal titik berbasis grid: setiap sel pada grid beresolusi
  rendah diklasifikasikan sebagai background atau pusat objek, tanpa regresi
  bounding box. LibreYOLO mendukungnya untuk task titik.
keywords:
  - FOMO
  - Faster Objects More Objects
  - lokalisasi titik
  - deteksi centroid
  - tiny object detection
  - edge AI
  - MCU detection
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Bobot LibreFOMO tidak diunduh otomatis (lihat Checkpoint di bawah).
        # Arahkan ini ke checkpoint yang telah diunduh secara lokal.
        model = LibreYOLO("./LibreFOMOs-point.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for point in result.points:
            print(point.cls, point.conf, point.xy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=./LibreFOMOs-point.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=40, batch=32, lr0=3e-4,
        )
    - label: CLI
      language: bash
      code: >
        # imgsz harus diteruskan: CLI memakai nilai default 640, sedangkan
        checkpoint s

        # hanya menerima resolusi native 96.

        libreyolo train model=./LibreFOMOs-point.pt data=my-dataset.yaml
        imgsz=96 epochs=40 batch=32 lr0=3e-4
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/grid_F1"])
        print(metrics["metrics/grid_precision"], metrics["metrics/grid_recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=./LibreFOMOs-point.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("./LibreFOMOs-point.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=./LibreFOMOs-point.pt format=onnx
    - label: Gunakan berkas hasil ekspor
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Factory merutekan berdasarkan akhiran berkas, sehingga artefak hasil
        ekspor dimuat

        # seperti checkpoint lain dan mengembalikan objek Results yang sama.

        model = LibreYOLO("./LibreFOMOs-point.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.points.xy)
source_hash: 03015f2bcd9fe99d
---

## Instalasi

FOMO tidak memerlukan extra selain paket dasar.

```bash
pip install libreyolo
```

## Prediksi

Tidak seperti semua family lain di situs ini, bobot LibreFOMO tidak diunduh
otomatis. `LibreYOLO("LibreFOMOs-point.pt")` mencari berkas tersebut di disk
dan memunculkan `ValueError` yang menyebutkan namanya alih-alih mengambilnya
dari Hugging Face. Unduh checkpoint dari [organisasi LibreYOLO](https://huggingface.co/LibreYOLO)
lebih dulu dan muat berdasarkan path lokal, atau latih sendiri (lihat Pelatihan
di bawah).

<code-tabs name="predict" />

Hasil memuat payload `points`, bukan `boxes`. Setiap baris adalah
`x, y, class, confidence`, yang tersedia sebagai `result.points.data` atau
melalui accessor `.xy`, `.xyn`, `.cls`, dan `.conf`. Tidak ada ambang batas
`iou` untuk ditetapkan karena tidak ada box yang perlu di-suppress.
`predict(..., nms_radius=1)` mengontrol jarak minimum dalam jumlah sel grid agar
dua deteksi sama-sama bertahan, dan nama berkas harus memuat akhiran task FOMO
`-point` agar dikenali loader. Lihat [prediksi](/docs/predict) untuk sumber,
streaming, dan penanganan hasil.

## Varian

Tiga ukuran, `s`, `m`, dan `l`, menggunakan backbone bergaya MobileNetV2 yang
makin lebar pada resolusi input tetap yang juga makin besar, masing-masing di
belakang satu head klasifikasi 1x1. Family ini tidak memiliki tabel benchmark
di sini. Ukuran berkas checkpoint dalam tabel di bawah adalah indikator
per ukuran paling jelas yang saat ini dipublikasikan.

## Pelatihan

<code-tabs name="train" />

`imgsz` bukan pilihan bebas. Nilainya default ke resolusi native checkpoint
yang dimuat, dan meneruskan nilai lain memunculkan `ValueError` yang menyebutkan
ukuran yang diharapkan. Ukurannya adalah 96 untuk `s`, 192 untuk `m`, dan 224
untuk `l`. CLI menetapkan `imgsz` default ke 640, sehingga perintah
`libreyolo train` harus mengaturnya secara eksplisit agar cocok dengan
checkpoint.

Jika pengaturan lain dibiarkan apa adanya, trainer berjalan selama 40 epoch
dengan batch 32 memakai Adam pada `lr0=3e-4`, tanpa weight decay, dan kelas
foreground diberi bobot 100 kali kelas background dalam loss cross-entropy per
sel karena hampir setiap sel grid adalah background dalam adegan biasa. EMA dan
mixed precision sama-sama nonaktif secara default. Augmentasi geometris maupun
warna yang digunakan di bagian lain LibreYOLO juga tidak diterapkan: mosaic,
mixup, HSV jitter, flip, rotasi, translasi, dan shear semuanya nol.

Ini adalah jalur yang digunakan untuk melatih checkpoint LibreFOMO yang
dipublikasikan, dari nol pada COCO.

Lihat [pelatihan](/docs/train) untuk dataset dan logger.

## Validasi

`val()` mengarahkan eksekusi ke validator tingkat grid yang dibuat untuk family
ini. Di samping key point matching `metrics/precision`, `metrics/recall`, dan
`metrics/mAP@` yang sama dengan task titik lain, validator menyapu ambang batas
confidence dan nilai `nms_radius`, lalu memublikasikan kombinasi F1 terbaik pada
`metrics/grid_F1`, `metrics/grid_precision`, `metrics/grid_recall`, dan
`metrics/grid_mean_distance`. Ambang batas dan radius yang menghasilkannya
tersedia pada `decode/threshold` dan `decode/nms_radius`.

<code-tabs name="val" />

## Ekspor

<export-matrix />

Artefak hasil ekspor dapat dimuat kembali melalui `LibreYOLO()` berdasarkan
akhiran berkasnya, sehingga berkas `.onnx` atau `.engine` berperilaku seperti
checkpoint dan mengembalikan `Results` yang sama. Menjalankan graph pada runtime
polos tanpa LibreYOLO juga didukung, tetapi preprocessing dan postprocessing
harus ditulis sendiri.

<code-tabs name="export" />

## Checkpoint

Semua berkas bobot yang dipublikasikan untuk family ini. Tidak satu pun diunduh
secara otomatis. Ambil berkas yang diinginkan dari halaman Hugging Face yang
ditautkan dan teruskan path lokalnya ke `LibreYOLO()`.

<checkpoint-table />

## Lisensi

<provenance-box>

Tidak ada repository kode upstream untuk FOMO yang dapat ditautkan. Edge Impulse
menjelaskan teknik ini melalui posting blog dan dokumentasi produknya, tetapi
belum merilis kode pelatihan atau inferensi FOMO. Arsitektur dan pelatihan di
sini adalah implementasi LibreYOLO sendiri atas deskripsi yang dipublikasikan,
dan checkpoint LibreFOMO yang dipublikasikan dilatih dari nol pada COCO. Karena
itu, kode dan bobot ini sama-sama berlisensi MIT milik LibreYOLO. Nama FOMO dan
teknik yang dijelaskannya tetap milik Edge Impulse.

</provenance-box>
