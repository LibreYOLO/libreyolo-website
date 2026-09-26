---
title: AlexNet
families:
  - alexnet
seo_title: 'AlexNet: jalankan classifier ImageNet klasik di LibreYOLO'
description: >-
  Memprediksi, memvalidasi dan mengekspor AlexNet dengan LibreYOLO. Bobot
  torchvision berlisensi BSD-3-Clause; fine-tuning belum didukung.
lead: >-
  AlexNet adalah jaringan konvolusional yang memenangkan ILSVRC 2012 dan ikut
  memulai era deep learning di computer vision. LibreYOLO menyediakan revisi
  arsitektur single-tower yang lebih baru untuk klasifikasi gambar.
keywords:
  - AlexNet
  - ImageNet
  - klasifikasi gambar python
  - image classification
  - convolutional neural network
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreAlexNetb-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        probs = result.probs
        print(probs.top1, probs.top1conf)
        print(probs.top5, probs.top5conf)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreAlexNetb-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreAlexNetb-cls.pt")

        # data adalah direktori root berisi split train/ dan val/ berupa folder
        # per kelas (tata letak ImageFolder), bukan YAML dataset.
        metrics = model.val(data="imagenet-1k/")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreAlexNetb-cls.pt data=imagenet-1k/
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreAlexNetb-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreAlexNetb-cls.pt format=onnx
        libreyolo export model=LibreAlexNetb-cls.pt format=tensorrt half=True
    - label: Memakai berkas hasil ekspor
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory memilih rute berdasarkan sufiks berkas, jadi artefak hasil
        # ekspor dimuat seperti checkpoint biasa dan mengembalikan objek
        # Results yang sama.
        model = LibreYOLO("LibreAlexNetb-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
source_hash: 68c09f080c74bb87
---

## Instalasi

AlexNet tidak memerlukan extra opsional. Semua yang diimpornya sudah ada di
instalasi dasar.

```bash
pip install libreyolo
```

## Prediksi

Bobot diunduh dari Hugging Face saat pertama kali dipakai dan disimpan di cache
lokal.

<code-tabs name="predict" />

Classifier mengembalikan `result.probs`, bukan `result.boxes`: `top1`
dan `top5` memberi indeks kelas, `top1conf` dan `top5conf` memberi skor
keyakinannya. Lihat [prediksi](/docs/predict) untuk sumber, streaming dan
penanganan hasil.

## Varian

Satu ukuran. Graph yang disertakan adalah revisi single-tower yang lebih baru
dari torchvision, dengan 64 filter pada lapisan pertama dan tanpa local
response normalization, bukan arsitektur dua-GPU asli dari 2012. LibreYOLO
menyediakan family ini hanya untuk inferensi: prediksi, validasi top-1/top-5
ala ImageNet dan ekspor didukung, sedangkan fine-tuning belum
diimplementasikan.

## Validasi

`val()` berjalan atas split bergaya ImageFolder (direktori dengan subfolder
`train/` dan `val/`, satu folder per kelas) dan mengembalikan akurasi top-1 dan
top-5.

<code-tabs name="val" />

## Ekspor

<export-matrix />

Artefak hasil ekspor dimuat kembali lewat `LibreYOLO()` berdasarkan sufiks
berkasnya, jadi berkas `.onnx` atau `.engine` berperilaku seperti checkpoint dan
mengembalikan `Results` yang sama. [Ekspor](/docs/export) memuat daftar argumen
yang diterima setiap format dan tambahan yang dipakai beberapa di antaranya.

<code-tabs name="export" />

## Checkpoint

Semua berkas bobot yang dipublikasikan untuk family ini.

<checkpoint-table />

## Lisensi

<provenance-box></provenance-box>
