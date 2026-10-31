---
title: PIDNet
families:
  - pidnet
seo_title: 'PIDNet: prediksi dan ekspor segmentasi real-time di bawah MIT'
description: >-
  Gunakan PIDNet di LibreYOLO untuk segmentasi semantik real-time. Instal,
  prediksi, validasi, dan ekspor checkpoint Cityscapes s/m/l di bawah MIT.
lead: >-
  Jaringan segmentasi semantik tiga cabang yang menambahkan cabang batas khusus
  pada desain yang terinspirasi proportional-integral-derivative, dengan sasaran
  inferensi real-time. LibreYOLO menyediakannya hanya untuk segmentasi semantik.
keywords:
  - PIDNet
  - segmentasi semantik real-time
  - boundary-aware segmentation
  - Cityscapes
  - dense prediction
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePIDNets-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape)   # (H, W) id kelas
        print(mask.classes)      # id kelas terurut yang ada dalam gambar
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibrePIDNets-sem.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePIDNets-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibrePIDNets-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePIDNets-sem.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibrePIDNets-sem.pt format=onnx
        libreyolo export model=LibrePIDNets-sem.pt format=tensorrt half=True
    - label: Gunakan berkas hasil ekspor
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Factory merutekan berdasarkan sufiks berkas, sehingga artefak hasil
        ekspor

        # dimuat seperti checkpoint lain dan mengembalikan objek Results yang
        sama.

        model = LibreYOLO("LibrePIDNets-sem.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.semantic_mask.data.shape)
source_hash: 489db64a39e3a61a
---

## Instalasi

PIDNet tidak memerlukan komponen tambahan opsional. Semua impornya tersedia dalam instalasi dasar.

```bash
pip install libreyolo
```

## Prediksi

Bobot diunduh dari Hugging Face saat pertama kali digunakan dan disimpan dalam cache lokal.
Sufiks nama berkas `-sem` wajib digunakan untuk family ini.

<code-tabs name="predict" />

Segmentasi semantik mengembalikan satu id kelas per piksel, bukan kotak, sehingga
`result.semantic_mask` menyimpan array `(H, W)` pada `.data` dan daftar id kelas
yang ada dalam gambar pada `.classes`. `conf`, `iou`, dan `max_det` diterima
demi kesetaraan API tetapi tidak berpengaruh: model menetapkan kelas untuk setiap
piksel dengan argmax, tanpa ambang confidence atau tahap NMS. Lihat
[prediksi](/docs/predict) untuk sumber, streaming, dan penanganan hasil.

## Varian

Tiga ukuran, semuanya dengan input tetap 1024 px. Checkpoint yang dipublikasikan
merupakan hasil konversi bobot Cityscapes PIDNet resmi, dengan 19 kelas.

LibreYOLO tidak melatih PIDNet: `train()` memunculkan `NotImplementedError` untuk
family ini, yang ditandai sebagai khusus inferensi pada [tier dukungan](/docs/models) di atas.

## Validasi

`val()` mengembalikan `metrics/mIoU` dan `metrics/pixel_accuracy`, yang diukur terhadap
dataset apa pun dalam format yang digunakan untuk pelatihan.

<code-tabs name="val" />

## Ekspor

<export-matrix />

Artefak hasil ekspor dimuat kembali melalui `LibreYOLO()` berdasarkan sufiks berkasnya,
sehingga berkas `.onnx` atau `.engine` berperilaku seperti checkpoint dan mengembalikan
`Results` yang sama. [Ekspor](/docs/export) mencantumkan argumen yang diterima setiap format.

<code-tabs name="export" />

## Checkpoint

Setiap berkas bobot yang dipublikasikan untuk family ini.

<checkpoint-table />

## Lisensi

<provenance-box></provenance-box>

## Sitasi

<citation-block />
