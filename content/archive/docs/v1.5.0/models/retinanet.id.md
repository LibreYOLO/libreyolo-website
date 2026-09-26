---
title: RetinaNet
families:
  - retinanet
seo_title: 'RetinaNet di LibreYOLO: prediksi, validasi, dan ekspor'
description: >-
  Jalankan RetinaNet di LibreYOLO untuk deteksi objek satu tahap dengan focal
  loss. Instal, prediksi, validasi, dan ekspor port torchvision BSD-3-Clause.
lead: >-
  RetinaNet adalah detektor satu tahap yang dilatih dengan focal loss, yang
  menurunkan bobot negatif mudah sehingga grid anchor padat tidak lagi
  memerlukan tahap proposal terpisah untuk mempertahankan akurasi. LibreYOLO
  mem-porting implementasi torchvision untuk deteksi.
keywords:
  - RetinaNet
  - focal loss
  - deteksi objek
  - one-stage detector
  - torchvision
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRetinaNetr50v2.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreRetinaNetr50v2.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRetinaNetr50v2.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRetinaNetr50v2.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRetinaNetr50v2.pt")
        model.export(format="onnx", imgsz=800)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreRetinaNetr50v2.pt format=onnx imgsz=800
    - label: Gunakan berkas hasil ekspor
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Factory merutekan berdasarkan sufiks berkas, sehingga artefak hasil
        ekspor

        # dimuat seperti checkpoint lain dan mengembalikan objek Results yang
        sama.

        model = LibreYOLO("LibreRetinaNetr50v2.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 1cc7ceb6de290bdb
---

## Instalasi

RetinaNet tidak memerlukan komponen tambahan opsional. Semua impornya tersedia dalam
instalasi dasar.

```bash
pip install libreyolo
```

## Prediksi

Bobot diunduh dari Hugging Face saat pertama kali digunakan dan disimpan dalam cache lokal.

<code-tabs name="predict" />

Objek `Results` yang dikembalikan sama dengan yang dikembalikan setiap family, jadi mengganti
detektor hanya memerlukan perubahan satu baris. `conf` dan `iou` menetapkan ambang batas
confidence dan NMS; RetinaNet mempertahankan tahap NMS upstream pada grid anchor
padat. Lihat [prediksi](/docs/predict) untuk sumber, streaming, dan penanganan hasil.

## Varian

Dua ukuran, keduanya memakai ResNet-50 dengan feature pyramid: `r50` memakai head
asli, sedangkan `r50v2` menggantinya dengan head GroupNorm dan block P6 lebih lebar
yang menerima input dari tahap terakhir backbone, bukan output FPN.

## Validasi

`val()` mengembalikan dictionary dengan key `metrics/` yang mencakup presisi, recall,
mAP 50, dan mAP 50-95, yang diukur terhadap dataset apa pun dalam format yang digunakan
untuk pelatihan.

<code-tabs name="val" />

## Ekspor

<export-matrix />

RetinaNet hanya dapat diekspor ke ONNX dengan ukuran batch 1. RetinaNet mengubah ukuran
ke input variabel dengan rasio aspek dipertahankan, sehingga LibreYOLO memaksa
`dynamic=True` terlepas dari nilai yang diteruskan agar graph tetap valid untuk sumber
dengan bentuk berbeda. Berkas `.onnx` hasil ekspor dimuat kembali melalui `LibreYOLO()`
berdasarkan sufiks berkas dan mengembalikan `Results` yang sama.

<code-tabs name="export" />

## Checkpoint

Setiap berkas bobot yang dipublikasikan untuk family ini.

<checkpoint-table />

## Lisensi

<provenance-box></provenance-box>
