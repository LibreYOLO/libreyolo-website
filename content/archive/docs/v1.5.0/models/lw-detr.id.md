---
title: LW-DETR
families:
  - lwdetr
seo_title: 'LW-DETR: prediksi dan ekspor berlisensi Apache-2.0'
description: >-
  Jalankan LW-DETR di LibreYOLO untuk deteksi objek real-time. Instal, prediksi,
  validasi, dan ekspor lima ukuran berbasis ViT, semuanya berlisensi Apache-2.0.
lead: >-
  Detection transformer berbasis ViT polos yang diposisikan Baidu sebagai
  alternatif real-time untuk detector YOLO. LibreYOLO menyediakan lima ukuran
  untuk deteksi, hanya inferensi.
keywords:
  - LW-DETR
  - detection transformer
  - deteksi objek real-time
  - plain ViT
  - DETR
  - Baidu
  - Atten4Vis
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreLWDETRt.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreLWDETRt.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLWDETRt.pt")

        # val() mengembalikan dict biasa, bukan objek
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreLWDETRt.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLWDETRt.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreLWDETRt.pt format=onnx imgsz=640

        libreyolo export model=LibreLWDETRt.pt format=tensorrt imgsz=640
        half=True
    - label: Gunakan berkas hasil ekspor
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Factory merutekan berdasarkan akhiran berkas, sehingga artefak hasil
        ekspor dimuat

        # seperti checkpoint lain dan mengembalikan objek Results yang sama.

        model = LibreYOLO("LibreLWDETRt.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: badd1d8255df5bbd
---

## Instalasi

LW-DETR tidak memerlukan extra opsional. Semua yang diimpornya tersedia dalam
instalasi dasar.

```bash
pip install libreyolo
```

## Prediksi

Bobot diunduh dari Hugging Face saat pertama kali digunakan dan disimpan dalam
cache lokal.

<code-tabs name="predict" />

Objek `Results` yang dikembalikan sama seperti yang dikembalikan setiap family,
sehingga mengganti detector hanya memerlukan perubahan satu baris. `conf` dan
`max_det` memfilter pemilihan query. `iou` diterima demi paritas API, tetapi
tidak berpengaruh karena decoder adalah set predictor tanpa langkah NMS. Lihat
[prediksi](/docs/predict) untuk sumber, streaming, dan penanganan hasil.

LW-DETR hanya untuk inferensi di LibreYOLO. Upstream melatih dengan supervisi
one-to-many Group-DETR pada beberapa grup query dan loss klasifikasi IoU-aware.
Resep tersebut tidak dihubungkan di sini, sehingga `train()` memunculkan
`NotImplementedError`.

## Varian

Ada lima ukuran. Semuanya memakai encoder ViT polos, projector multi-scale, dan
decoder deformable DETR yang sama serta berjalan pada resolusi input yang sama.
Dua ukuran terkecil memakai lebar encoder yang sama dan dibedakan oleh kedalaman
blok. Dua berikutnya memakai encoder lebih lebar dan dibedakan oleh jumlah level
projector yang memasok decoder. Ukuran terbesar beralih ke encoder paling lebar.

## Validasi

`val()` mengembalikan dictionary berisi key `metrics/` yang mencakup presisi,
recall, mAP 50, dan mAP 50-95, yang diukur terhadap dataset apa pun dalam format
yang digunakan saat pelatihan.

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

## Sitasi

<citation-block />
