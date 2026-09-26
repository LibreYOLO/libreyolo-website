---
title: DETR
families:
  - detr
seo_title: 'DETR: prediksi dan ekspor di bawah Apache-2.0'
description: >-
  Jalankan DETR, detection transformer yang pertama, di LibreYOLO. Pasang,
  prediksi, validasi dan ekspor empat ukuran berbasis ResNet, semuanya
  berlisensi Apache-2.0.
lead: >-
  DETR adalah detection transformer yang pertama, memprediksi sekumpulan objek
  berjumlah tetap lewat decoder transformer dengan pencocokan Hungarian, bukan
  lewat anchor atau grid padat. LibreYOLO menyediakan empat ukuran untuk
  deteksi, khusus inferensi.
keywords:
  - DETR
  - detection transformer
  - deteksi objek python
  - object detection
  - Hungarian matching
  - transformer decoder
  - Meta AI
  - Facebook AI Research
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDETRr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDETRr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDETRr50.pt")

        # val() mengembalikan dict biasa, bukan sebuah objek
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDETRr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDETRr50.pt")
        model.export(format="onnx", imgsz=800)
        model.export(format="tensorrt", imgsz=800, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDETRr50.pt format=onnx imgsz=800

        libreyolo export model=LibreDETRr50.pt format=tensorrt imgsz=800
        half=True
    - label: Memakai berkas hasil ekspor
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Factory-nya memilih berdasarkan sufiks berkas, jadi artefak hasil
        ekspor

        # dimuat seperti checkpoint biasa dan mengembalikan objek Results yang
        sama.

        model = LibreYOLO("LibreDETRr50.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: c5549a596742d2a5
---

## Instalasi

DETR tidak memerlukan extra opsional. Semua yang diimpornya sudah tersedia di
instalasi dasar.

```bash
pip install libreyolo
```

## Prediksi

Bobot diunduh dari Hugging Face saat pertama kali dipakai lalu disimpan di
cache lokal.

<code-tabs name="predict" />

Objek `Results` yang dikembalikan sama dengan yang dikembalikan setiap family,
jadi mengganti detektor dengan yang lain hanya butuh perubahan satu baris.
`conf` dan `max_det` menyaring pemilihan query; `iou` tetap diterima demi
keseragaman API tetapi tidak berpengaruh, karena decoder-nya adalah set
predictor tanpa langkah NMS. Lihat [prediksi](/docs/predict) untuk source,
streaming dan penanganan hasil.

Di LibreYOLO, DETR hanya untuk inferensi. Proyek upstream melatihnya selama 500
epoch dengan pencocokan Hungarian; resep itu tidak diimplementasikan di sini,
jadi `train()` memunculkan `NotImplementedError`.

## Varian

Empat checkpoint memadukan dua kedalaman backbone, ResNet-50 atau ResNet-101,
dengan tahap C5 berdilasi yang opsional: varian DC5 mempertahankan tahap
backbone terakhir pada resolusi penuh alih-alih menurunkannya lebih jauh,
sehingga decoder membaca feature map yang lebih halus dari ukuran input yang
sama. Keempatnya sama-sama memakai 100 object query terlatih dan encoder-decoder
transformer enam lapisan, dan semuanya berjalan pada resolusi input yang sama.

## Validasi

`val()` mengembalikan dictionary berisi key `metrics/` yang mencakup presisi,
recall, mAP 50 dan mAP 50-95, diukur terhadap dataset apa pun dalam format yang
dipakai saat pelatihan.

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
