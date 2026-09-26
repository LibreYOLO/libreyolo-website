---
title: DeiT
families:
  - deit
seo_title: 'Classifier gambar DeiT: prediksi, validasi, ekspor'
description: >-
  Menjalankan classifier gambar DeiT di LibreYOLO: family museum yang dibekukan
  dan hanya untuk inferensi, dalam ukuran tiny, small dan base, berlisensi
  Apache-2.0.
lead: >-
  DeiT (Data-efficient image Transformer) adalah classifier Vision Transformer
  polos yang dilatih hanya pada ImageNet-1k, tanpa data pretraining tambahan.
  LibreYOLO menyediakan ukuran tiny, small dan base patch-16 sebagai koleksi
  museum yang dibekukan dan hanya untuk inferensi.
keywords:
  - DeiT
  - Vision Transformer
  - ViT
  - klasifikasi gambar python
  - image classification
  - ImageNet
  - data-efficient training
  - museum family
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDeiTb-cls.pt")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1, result.probs.top1conf)
        print(result.probs.top5)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDeiTb-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeiTb-cls.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDeiTb-cls.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeiTb-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDeiTb-cls.pt format=onnx
        libreyolo export model=LibreDeiTb-cls.pt format=tensorrt half=True
    - label: Memakai berkas hasil ekspor
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory memilih rute berdasarkan sufiks berkas, jadi artefak hasil
        # ekspor dimuat seperti checkpoint biasa dan mengembalikan objek
        # Results yang sama.
        model = LibreYOLO("LibreDeiTb-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
source_hash: 9c67c8554b2af5c6
---

## Instalasi

DeiT tidak memerlukan extra apa pun di luar paket dasar.

```bash
pip install libreyolo
```

## Prediksi

Family ini hanya untuk inferensi: `train()` memunculkan `NotImplementedError`,
jadi halaman ini tidak punya bagian Pelatihan. Prediksi, validasi dan ekspor
semuanya didukung. Bobot diunduh dari Hugging Face saat pertama kali dipakai dan
disimpan di cache lokal. Sufiks `-cls` pada nama berkas wajib ada dan memilih
task klasifikasi.

<code-tabs name="predict" />

Objek `Results` yang dikembalikan membawa tensor `probs`, bukan `boxes`; `top1`
dan `top5` memberi indeks ke 1.000 kelas ImageNet-1k, sedangkan `top1conf`
adalah skor softmax untuk prediksi teratas. Setiap ukuran punya resolusi input
tetap yang berasal dari positional embedding-nya: praproses mengubah ukuran dan
melakukan center-crop ke resolusi tersebut, dan memberikan `imgsz` yang berbeda
akan memunculkan error alih-alih diam-diam melakukan resampling. Lihat
[prediksi](/docs/predict) untuk sumber, streaming dan penanganan hasil.

## Validasi

`val()` mengembalikan dictionary berisi akurasi top-1 dan top-5, diukur terhadap
dataset yang ditata dalam struktur folder konvensional `train/<class>/` dan
`val/<class>/`.

<code-tabs name="val" />

## Ekspor

<export-matrix />

Artefak hasil ekspor dimuat kembali lewat `LibreYOLO()` berdasarkan sufiks
berkasnya, jadi berkas `.onnx` atau `.engine` berperilaku seperti checkpoint dan
mengembalikan `Results` yang sama. Menjalankan graph di runtime polos, tanpa
LibreYOLO terpasang, juga didukung, tetapi praproses dan pascaproses menjadi
tanggung jawab Anda.

<code-tabs name="export" />

## Checkpoint

Semua berkas bobot yang dipublikasikan untuk family ini.

<checkpoint-table />

## Lisensi

<provenance-box></provenance-box>

## Sitasi

<citation-block />
