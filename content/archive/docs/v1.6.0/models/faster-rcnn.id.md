---
title: Faster R-CNN
families:
  - faster_rcnn
seo_title: 'Faster R-CNN di LibreYOLO: prediksi, validasi, dan ekspor'
description: >-
  Jalankan Faster R-CNN di LibreYOLO untuk deteksi objek dengan empat backbone.
  Instal, prediksi, validasi, dan ekspor port torchvision berlisensi
  BSD-3-Clause.
lead: >-
  Faster R-CNN mendeteksi objek dengan jaringan proposal region yang memasok
  classifier dua tahap, arsitektur yang menjadikan proposal region bagian dari
  jaringan terlatih yang sama, bukan langkah terpisah. LibreYOLO menyediakan
  port implementasi torchvision untuk deteksi.
keywords:
  - Faster R-CNN
  - deteksi objek Python
  - region proposal network
  - two-stage detector
  - torchvision
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFasterRCNNl.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreFasterRCNNl.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFasterRCNNl.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreFasterRCNNl.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFasterRCNNl.pt")
        model.export(format="onnx", imgsz=800)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreFasterRCNNl.pt format=onnx imgsz=800
    - label: Gunakan berkas hasil ekspor
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Factory merutekan berdasarkan akhiran berkas, sehingga artefak hasil
        ekspor dimuat

        # seperti checkpoint lain dan mengembalikan objek Results yang sama.

        model = LibreYOLO("LibreFasterRCNNl.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 3fd82eb835399560
---

## Instalasi

Faster R-CNN tidak memerlukan extra opsional. Semua yang diimpornya tersedia
dalam instalasi dasar.

```bash
pip install libreyolo
```

## Prediksi

Bobot diunduh dari Hugging Face saat pertama kali digunakan dan disimpan dalam
cache lokal.

<code-tabs name="predict" />

Objek `Results` yang dikembalikan sama seperti yang dikembalikan setiap family,
sehingga mengganti detector hanya memerlukan perubahan satu baris. `conf` dan
`iou` menetapkan ambang batas confidence dan NMS. Faster R-CNN mempertahankan
langkah NMS upstream, tidak seperti detector berbasis query. Lihat
[prediksi](/docs/predict) untuk sumber, streaming, dan penanganan hasil.

## Varian

Ada empat ukuran, masing-masing merupakan konfigurasi torchvision yang berbeda,
bukan versi berskala dari konfigurasi yang sama: `n` adalah MobileNetV3-Large
dengan input 320 px, `s` memakai backbone yang sama pada 800 px, `m` adalah
ResNet-50 dengan feature pyramid, dan `l` adalah revisi v2 dengan head proposal
region yang lebih dalam serta head box empat konvolusi sebagai pengganti milik
`m`. `n` dan `s` menukar akurasi dengan backbone yang lebih ringan.

## Validasi

`val()` mengembalikan dictionary berisi key `metrics/` yang mencakup presisi,
recall, mAP 50, dan mAP 50-95, yang diukur terhadap dataset apa pun dalam format
yang digunakan saat pelatihan.

<code-tabs name="val" />

## Ekspor

<export-matrix />

Faster R-CNN hanya dapat diekspor ke ONNX dengan ukuran batch 1. Graph hasil
ekspor mempertahankan langkah resize upstream di dalamnya, sehingga LibreYOLO
memaksa `dynamic=True` apa pun nilai yang diteruskan agar graph tetap valid
untuk sumber yang tidak berbentuk persegi. Berkas `.onnx` hasil ekspor dapat
dimuat kembali melalui `LibreYOLO()` berdasarkan akhiran berkasnya dan
mengembalikan `Results` yang sama.

<code-tabs name="export" />

## Checkpoint

Semua berkas bobot yang dipublikasikan untuk family ini.

<checkpoint-table />

## Lisensi

<provenance-box></provenance-box>

## Sitasi

<citation-block />
