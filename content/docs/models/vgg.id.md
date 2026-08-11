---
title: VGG
families:
  - vgg
seo_title: 'VGG: jalankan pengklasifikasi gambar VGG-16/19 di LibreYOLO'
description: >-
  Prediksi, validasi, dan ekspor pengklasifikasi VGG dengan LibreYOLO. Bobot
  torchvision BSD-3-Clause; fine-tuning belum didukung.
lead: >-
  VGG adalah pengklasifikasi gambar konvolusional yang dibuat dari tumpukan
  seragam konvolusi kecil 3x3, bukan filter yang lebih besar. LibreYOLO
  menyediakan ukuran 16 dan 19 lapisan, versi biasa dan dengan batch
  normalization, untuk klasifikasi gambar.
keywords:
  - VGG
  - VGG-16
  - VGG-19
  - convolutional neural network
  - klasifikasi gambar
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreVGG16-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        probs = result.probs
        print(probs.top1, probs.top1conf)
        print(probs.top5, probs.top5conf)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreVGG16-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreVGG16-cls.pt")

        # data adalah root direktori dengan split folder kelas train/ dan val/
        # (tata letak ImageFolder), bukan YAML dataset.
        metrics = model.val(data="imagenet-1k/")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreVGG16-cls.pt data=imagenet-1k/
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreVGG16-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreVGG16-cls.pt format=onnx
        libreyolo export model=LibreVGG16-cls.pt format=tensorrt half=True
    - label: Gunakan berkas hasil ekspor
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Factory merutekan berdasarkan sufiks berkas, sehingga artefak hasil
        ekspor

        # dimuat seperti checkpoint lain dan mengembalikan objek Results yang
        sama.

        model = LibreYOLO("LibreVGG16-cls.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.probs.top1)
source_hash: 26eb6ff5811533fd
---

## Instalasi

VGG tidak memerlukan komponen tambahan opsional. Semua impornya tersedia dalam instalasi dasar.

```bash
pip install libreyolo
```

## Prediksi

Bobot diunduh dari Hugging Face saat pertama kali digunakan dan disimpan dalam cache lokal.

<code-tabs name="predict" />

Pengklasifikasi mengembalikan `result.probs`, bukan `result.boxes`: `top1` dan `top5`
memberikan indeks kelas, sedangkan `top1conf` dan `top5conf` memberikan confidence-nya.
Prediksi berjalan pada input tetap 224px dan memunculkan error jika `imgsz` berbeda
diteruskan. Lihat [prediksi](/docs/predict) untuk sumber, streaming, dan penanganan hasil.

## Varian

Empat ukuran: 16 dan 19 lapisan konvolusional, masing-masing dengan varian biasa dan varian
batch-normalized. Bobot yang disediakan berasal dari pelatihan ImageNet dari nol versi lebih
baru milik torchvision, bukan hasil konversi rilis Caffe asli kelompok Oxford tahun 2014.
LibreYOLO menyediakan family ini khusus untuk inferensi: prediksi, validasi top-1/top-5
bergaya ImageNet, dan ekspor didukung, sedangkan fine-tuning tidak diimplementasikan.

## Validasi

`val()` berjalan terhadap split bergaya ImageFolder (direktori dengan subfolder `train/`
dan `val/`, satu folder per kelas) dan mengembalikan akurasi top-1 dan top-5.

<code-tabs name="val" />

## Ekspor

<export-matrix />

Artefak hasil ekspor dimuat kembali melalui `LibreYOLO()` berdasarkan sufiks berkasnya,
sehingga berkas `.onnx` atau `.engine` berperilaku seperti checkpoint dan mengembalikan
`Results` yang sama. [Ekspor](/docs/export) mencantumkan argumen yang diterima setiap format
beserta komponen tambahan yang disediakan beberapa format.

<code-tabs name="export" />

## Checkpoint

Setiap berkas bobot yang dipublikasikan untuk family ini.

<checkpoint-table />

## Lisensi

<provenance-box></provenance-box>
