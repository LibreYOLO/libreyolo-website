---
title: DeepLabv3
families:
  - deeplabv3
seo_title: 'DeepLabv3: prediksi dan ekspor segmentasi semantik ASPP'
description: >-
  Pakai DeepLabv3 di LibreYOLO untuk segmentasi semantik. Pasang, prediksi,
  validasi dan ekspor checkpoint ResNet dan MobileNetV3 dari torchvision.
lead: >-
  Jaringan segmentasi semantik yang melakukan pooling fitur pada beberapa
  dilation rate secara paralel (atrous spatial pyramid pooling) sebelum
  mengklasifikasikan setiap piksel. LibreYOLO menyediakannya hanya untuk
  segmentasi semantik.
keywords:
  - DeepLabv3
  - ASPP
  - segmentasi semantik python
  - semantic segmentation
  - atrous spatial pyramid pooling
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDeepLabv3r50-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape)   # (H, W) id kelas
        print(mask.classes)      # id kelas yang ada di gambar, terurut
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDeepLabv3r50-sem.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeepLabv3r50-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDeepLabv3r50-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeepLabv3r50-sem.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDeepLabv3r50-sem.pt format=onnx

        libreyolo export model=LibreDeepLabv3r50-sem.pt format=tensorrt
        half=True
    - label: Memakai berkas hasil ekspor
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory memilih rute berdasarkan sufiks berkas, jadi artefak hasil
        # ekspor dimuat seperti checkpoint biasa dan mengembalikan objek
        # Results yang sama.
        model = LibreYOLO("LibreDeepLabv3r50-sem.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
source_hash: 7abf11ebb6cece18
---

## Instalasi

DeepLabv3 tidak memerlukan extra opsional. Semua yang diimpornya sudah ada di
instalasi dasar.

```bash
pip install libreyolo
```

## Prediksi

Bobot diunduh dari Hugging Face saat pertama kali dipakai dan disimpan di cache
lokal. Sufiks nama berkas `-sem` wajib untuk family ini.

<code-tabs name="predict" />

Segmentasi semantik mengembalikan satu id kelas per piksel, bukan bounding box,
jadi `result.semantic_mask` membawa array `(H, W)` di `.data` dan daftar id
kelas yang ada di gambar di `.classes`. `conf`, `iou` dan `max_det` tetap
diterima demi kesetaraan API, tetapi tidak berpengaruh: model menetapkan satu
kelas untuk setiap piksel lewat argmax, tanpa ambang batas skor keyakinan
maupun langkah NMS. Lihat [prediksi](/docs/predict) untuk sumber, streaming dan
penanganan hasil.

## Varian

Tiga backbone: ResNet-50 dilated, ResNet-101 dilated, dan MobileNetV3-Large
dilated. Ini DeepLabv3, bukan DeepLabv3+, jadi tidak ada tahap decoder maupun
penghalusan CRF, sesuai implementasi torchvision alih-alih kode referensi dari
paper aslinya.

LibreYOLO tidak melatih DeepLabv3: `train()` memunculkan `NotImplementedError`
untuk family ini, dan [tingkat dukungan](/docs/models) di atas menandainya
sebagai inferensi saja. Ketiga checkpoint yang dipublikasikan adalah bobot
COCO-dengan-label-VOC milik torchvision sendiri, yang dikonversi untuk loader
LibreYOLO.

## Validasi

`val()` mengembalikan `metrics/mIoU` dan `metrics/pixel_accuracy`, yang diukur
terhadap dataset apa pun dengan format yang dipakai saat pelatihan.

<code-tabs name="val" />

## Ekspor

<export-matrix />

Artefak hasil ekspor dimuat kembali lewat `LibreYOLO()` berdasarkan sufiks
berkasnya, jadi berkas `.onnx` atau `.engine` berperilaku seperti checkpoint dan
mengembalikan `Results` yang sama. [Ekspor](/docs/export) memuat daftar argumen
yang diterima setiap format.

<code-tabs name="export" />

## Checkpoint

Semua berkas bobot yang dipublikasikan untuk family ini.

<checkpoint-table />

## Lisensi

<provenance-box></provenance-box>
