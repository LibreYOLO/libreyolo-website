---
title: YOLOv1
families:
  - yolo1
seo_title: 'YOLOv1 di LibreYOLO: prediksi, validasi, ekspor'
description: >-
  Jalankan detektor YOLOv1 asli di LibreYOLO: family museum yang dibekukan dan
  khusus inferensi. Prediksi, validasi, dan ekspor di bawah lisensi domain
  publik.
lead: >-
  YOLOv1 adalah detektor asli tahun 2016 yang memberi nama pada family YOLO:
  satu jaringan konvolusional dengan head fully connected memprediksi setiap
  kotak dan skor kelas dalam satu tahap, tanpa anchor box. LibreYOLO
  menyediakannya sebagai pameran yang dibekukan dan khusus inferensi.
keywords:
  - YOLOv1
  - YOLO v1
  - Darknet
  - deteksi objek
  - Pascal VOC
  - model YOLO lama
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO1b.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreYOLO1b.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO1b.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO1b.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO1b.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO1b.pt format=onnx
        libreyolo export model=LibreYOLO1b.pt format=tensorrt half=True
    - label: Gunakan berkas hasil ekspor
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Factory merutekan berdasarkan sufiks berkas, sehingga artefak hasil
        ekspor

        # dimuat seperti checkpoint lain dan mengembalikan objek Results yang
        sama.

        model = LibreYOLO("LibreYOLO1b.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: a786372dba86f2f8
---

## Instalasi

YOLOv1 tidak memerlukan komponen tambahan selain paket dasar.

```bash
pip install libreyolo
```

## Prediksi

Family ini khusus inferensi: `train()` memunculkan `NotImplementedError`, sehingga halaman
ini tidak memiliki bagian Pelatihan. Prediksi, validasi, dan ekspor semuanya didukung.
Bobot diunduh dari Hugging Face saat pertama kali digunakan dan disimpan dalam cache lokal.

<code-tabs name="predict" />

Objek `Results` yang dikembalikan sama dengan yang dikembalikan setiap family, jadi mengganti
detektor hanya memerlukan perubahan satu baris. Dua hal khusus untuk family ini. Checkpoint
yang diterbitkan dilatih pada Pascal VOC (2007+2012), bukan COCO, sehingga `box.cls`
mengindeks 20 kategori VOC (aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair,
cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor),
bukan 80 kategori COCO. Selain itu, head deteksi fully connected menerima satu gambar pada
satu waktu, sehingga daftar sumber dijalankan dalam loop, bukan sebagai batch sejati.
Lihat [prediksi](/docs/predict) untuk sumber, streaming, dan penanganan hasil.

## Validasi

`val()` mengembalikan dictionary dengan key `metrics/` yang mencakup presisi, recall,
mAP 50, dan mAP 50-95, yang diukur terhadap dataset dalam ruang label bergaya VOC yang
sama seperti yang digunakan untuk melatih checkpoint.

<code-tabs name="val" />

## Ekspor

<export-matrix />

Artefak hasil ekspor dimuat kembali melalui `LibreYOLO()` berdasarkan sufiks berkasnya,
sehingga berkas `.onnx` atau `.engine` berperilaku seperti checkpoint dan mengembalikan
`Results` yang sama. Menjalankan graph pada runtime mandiri tanpa memasang LibreYOLO juga
didukung, tetapi prapemrosesan dan pascapemrosesannya harus ditulis sendiri.

<code-tabs name="export" />

## Checkpoint

Setiap berkas bobot yang dipublikasikan untuk family ini.

<checkpoint-table />

## Lisensi

<provenance-box></provenance-box>
