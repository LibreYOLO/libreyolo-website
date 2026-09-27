---
title: YOLOv4
families:
  - yolo4
seo_title: 'YOLOv4: jalankan, validasi, dan ekspor di LibreYOLO'
description: >-
  Jalankan YOLOv4 di LibreYOLO: family museum yang dibekukan dan hanya untuk
  inferensi dengan backbone CSPDarknet-53. Prediksi, validasi, dan ekspor dengan
  lisensi domain publik.
lead: >-
  YOLOv4 menggabungkan backbone CSPDarknet-53, blok SPP, dan neck PANet dengan
  aktivasi Mish. LibreYOLO menyediakannya sebagai pameran yang dibekukan dan
  hanya untuk inferensi dalam ukuran tiny dan base.
keywords:
  - YOLOv4
  - Darknet
  - CSPDarknet-53
  - PANet
  - deteksi objek
  - Mish activation
  - museum family
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO4b.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreYOLO4b.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO4b.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO4b.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO4b.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO4b.pt format=onnx
        libreyolo export model=LibreYOLO4b.pt format=tensorrt half=True
    - label: Gunakan berkas hasil ekspor
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Factory merutekan berdasarkan akhiran berkas, sehingga artefak hasil
        ekspor dimuat

        # seperti checkpoint lain dan mengembalikan objek Results yang sama.

        model = LibreYOLO("LibreYOLO4b.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 6070bb4a09d75416
---

## Instalasi

YOLOv4 tidak memerlukan extra selain paket dasar.

```bash
pip install libreyolo
```

## Prediksi

Family ini hanya untuk inferensi. `train()` memunculkan `NotImplementedError`,
sehingga halaman ini tidak memiliki bagian Pelatihan. Prediksi, validasi, dan
ekspor semuanya didukung. Bobot diunduh dari Hugging Face saat pertama kali
digunakan dan disimpan dalam cache lokal.

<code-tabs name="predict" />

Objek `Results` yang dikembalikan sama seperti yang dikembalikan setiap family,
sehingga mengganti detector hanya memerlukan perubahan satu baris. `conf`
memfilter ambang batas confidence dan `iou` memfilter ambang batas NMS yang
diterapkan setelah penskalaan pusat `scale_x_y` milik setiap head. Lihat
[prediksi](/docs/predict) untuk sumber, streaming, dan penanganan hasil.

## Validasi

`val()` mengembalikan dictionary berisi key `metrics/` yang mencakup presisi,
recall, mAP 50, dan mAP 50-95, yang diukur terhadap dataset apa pun dalam format
validasi.

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

Semua berkas bobot yang dipublikasikan untuk family ini.

<checkpoint-table />

## Lisensi

<provenance-box></provenance-box>

## Sitasi

<citation-block />
