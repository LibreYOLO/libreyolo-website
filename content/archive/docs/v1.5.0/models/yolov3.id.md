---
title: YOLOv3
families:
  - yolo3
seo_title: 'YOLOv3 di LibreYOLO: prediksi, validasi, ekspor'
description: >-
  Jalankan YOLOv3 di LibreYOLO: family museum yang dibekukan dan hanya untuk
  inferensi dengan ukuran tiny, base, dan SPP. Prediksi, validasi, dan ekspor
  dengan lisensi domain publik.
lead: >-
  YOLOv3 adalah detector Darknet-53 yang menambahkan prediksi multi-scale dan
  classifier logistic independen ke rangkaian YOLO. LibreYOLO menyediakannya
  sebagai pameran yang dibekukan dan hanya untuk inferensi dalam ukuran tiny,
  base, dan SPP.
keywords:
  - YOLOv3
  - Darknet
  - Darknet-53
  - deteksi objek
  - multi-scale detection
  - museum family
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO3b.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreYOLO3b.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Ukuran SPP
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Varian SPP menambahkan blok spatial pyramid pooling sebelum head
        deteksi

        # dan berjalan pada ukuran input native-nya sendiri.

        model = LibreYOLO("LibreYOLO3spp.pt")

        result = model(SAMPLE_IMAGE)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO3b.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO3b.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO3b.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO3b.pt format=onnx
        libreyolo export model=LibreYOLO3b.pt format=tensorrt half=True
    - label: Gunakan berkas hasil ekspor
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Factory merutekan berdasarkan akhiran berkas, sehingga artefak hasil
        ekspor dimuat

        # seperti checkpoint lain dan mengembalikan objek Results yang sama.

        model = LibreYOLO("LibreYOLO3b.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: a4c652bb2707fc8f
---

## Instalasi

YOLOv3 tidak memerlukan extra selain paket dasar.

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
memfilter ambang batas confidence dan `iou` memfilter ambang batas NMS, yang
diterapkan per skala sebelum box dari ketiga head digabungkan. Lihat
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
