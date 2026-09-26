---
title: FCN
families:
  - fcn
seo_title: 'FCN: prediksi dan ekspor FCN ResNet berlisensi BSD-3-Clause'
description: >-
  Gunakan FCN di LibreYOLO untuk segmentasi semantik. Instal, prediksi,
  validasi, dan ekspor checkpoint FCN ResNet berdilasi dari torchvision.
lead: >-
  Classifier padat per piksel yang mengganti lapisan fully connected milik
  detector dengan konvolusi, sehingga menghasilkan peta kelas beresolusi penuh,
  bukan box. LibreYOLO menyediakannya hanya untuk segmentasi semantik.
keywords:
  - FCN
  - fully convolutional network
  - segmentasi semantik Python
  - dense prediction
  - ResNet
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFCNr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape)   # ID kelas (H, W)
        print(mask.classes)      # ID kelas terurut yang ada dalam gambar
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreFCNr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFCNr50.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreFCNr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFCNr50.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreFCNr50.pt format=onnx
        libreyolo export model=LibreFCNr50.pt format=tensorrt half=True
    - label: Gunakan berkas hasil ekspor
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Factory merutekan berdasarkan akhiran berkas, sehingga artefak hasil
        ekspor dimuat

        # seperti checkpoint lain dan mengembalikan objek Results yang sama.

        model = LibreYOLO("LibreFCNr50.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.semantic_mask.data.shape)
source_hash: 7776b0fc85a208fb
---

## Instalasi

FCN tidak memerlukan extra opsional. Semua yang diimpornya tersedia dalam
instalasi dasar.

```bash
pip install libreyolo
```

## Prediksi

Bobot diunduh dari Hugging Face saat pertama kali digunakan dan disimpan dalam
cache lokal.

<code-tabs name="predict" />

Segmentasi semantik mengembalikan satu ID kelas per piksel, bukan box, sehingga
`result.semantic_mask` menyimpan array `(H, W)` pada `.data` dan daftar ID
kelas yang ada dalam gambar pada `.classes`. `conf`, `iou`, dan `max_det`
diterima demi paritas API, tetapi tidak berpengaruh. Model menetapkan kelas
untuk setiap piksel dengan argmax, tanpa ambang batas confidence atau langkah
NMS. Lihat [prediksi](/docs/predict) untuk sumber, streaming, dan penanganan
hasil.

## Varian

Ada dua kedalaman ResNet, keduanya dengan input tetap 520 px. Graph inferensi
library ini adalah FCN ResNet berdilasi dari torchvision, bukan jaringan FCN-8s
berbasis VGG dengan skip connection dari makalah asli.

LibreYOLO tidak melatih FCN: `train()` memunculkan `NotImplementedError` untuk
family ini, yang ditandai sebagai hanya inferensi pada [tingkat dukungan](/docs/models)
di atas. Kedua checkpoint yang dipublikasikan merupakan bobot milik torchvision
yang dilatih pada COCO dan dikonversi untuk loader LibreYOLO.

## Validasi

`val()` mengembalikan `metrics/mIoU` dan `metrics/pixel_accuracy`, yang diukur
terhadap dataset apa pun dalam format yang digunakan saat pelatihan.

<code-tabs name="val" />

## Ekspor

<export-matrix />

Artefak hasil ekspor dapat dimuat kembali melalui `LibreYOLO()` berdasarkan
akhiran berkasnya, sehingga berkas `.onnx` atau `.engine` berperilaku seperti
checkpoint dan mengembalikan `Results` yang sama. [Ekspor](/docs/export)
mencantumkan argumen yang diterima oleh setiap format.

<code-tabs name="export" />

## Checkpoint

Semua berkas bobot yang dipublikasikan untuk family ini.

<checkpoint-table />

## Lisensi

<provenance-box></provenance-box>
