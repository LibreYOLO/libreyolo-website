---
title: Mask R-CNN
families:
  - mask_rcnn
seo_title: 'Mask R-CNN di LibreYOLO: prediksi, validasi, dan ekspor'
description: >-
  Jalankan Mask R-CNN di LibreYOLO untuk deteksi objek dan segmentasi instance.
  Instal, prediksi, validasi, dan ekspor port torchvision berlisensi
  BSD-3-Clause.
lead: >-
  Mask R-CNN menambahkan cabang mask per region ke Faster R-CNN, dengan
  memprediksi mask segmentasi bersama setiap box yang dideteksi. LibreYOLO
  menyediakan port implementasi torchvision untuk deteksi dan segmentasi
  instance.
keywords:
  - Mask R-CNN
  - segmentasi instance
  - deteksi objek Python
  - Faster R-CNN
  - torchvision
  - two-stage detector
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMaskRCNNr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreMaskRCNNr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Hanya box
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # task="detect" melewati head mask dan mengembalikan box dari checkpoint
        yang sama,

        # tanpa mask dalam hasil.

        model = LibreYOLO("LibreMaskRCNNr50.pt", task="detect")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMaskRCNNr50.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])      # mask
        print(metrics["metrics/mAP50-95(B)"])   # box
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreMaskRCNNr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMaskRCNNr50.pt")
        model.export(format="onnx", imgsz=800)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreMaskRCNNr50.pt format=onnx imgsz=800
    - label: Gunakan berkas hasil ekspor
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Factory merutekan berdasarkan akhiran berkas, sehingga artefak hasil
        ekspor dimuat

        # seperti checkpoint lain dan mengembalikan objek Results yang sama.

        model = LibreYOLO("LibreMaskRCNNr50.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.masks.data.shape)
source_hash: 9608459b801aa6d5
---

## Instalasi

Mask R-CNN tidak memerlukan extra opsional. Semua yang diimpornya tersedia
dalam instalasi dasar.

```bash
pip install libreyolo
```

## Prediksi

Bobot diunduh dari Hugging Face saat pertama kali digunakan dan disimpan dalam
cache lokal.

<code-tabs name="predict" />

Objek `Results` yang dikembalikan sama seperti yang dikembalikan setiap family,
sehingga mengganti detector hanya memerlukan perubahan satu baris. Memuat
checkpoint tanpa argumen `task` mengembalikan mask instance karena segmentasi
adalah task default family ini. `result.masks` kemudian memuat mask tersebut
bersama box. Meneruskan `task="detect"` memuat bobot yang sama tanpa head mask
dan hanya mengembalikan box. `conf` dan `iou` menetapkan ambang batas confidence
dan NMS. Mask R-CNN mempertahankan langkah NMS upstream, tidak seperti detector
berbasis query. Lihat [prediksi](/docs/predict) untuk sumber, streaming, dan
penanganan hasil.

## Varian

Ada satu backbone, ResNet-50 dengan feature pyramid, yang memakai builder Mask
R-CNN v2 dari torchvision. Checkpoint yang dipublikasikan berlisensi
BSD-3-Clause dan melayani kedua task dalam family ini, sehingga tidak ada ukuran
yang perlu dipilih.

## Validasi

`val()` mengembalikan dictionary berisi key `metrics/`. Untuk task segmentasi
default checkpoint ini, key polos `metrics/mAP50-95` memuat skor mask, sedangkan
run yang sama melaporkan box dengan akhiran `(B)` sehingga keduanya tersedia
dalam satu pass.

<code-tabs name="val" />

## Ekspor

<export-matrix />

Mask R-CNN hanya dapat diekspor ke ONNX dengan ukuran batch 1. Graph hasil
ekspor mempertahankan langkah resize dan penempelan mask upstream di dalamnya,
sehingga LibreYOLO memaksa `dynamic=True` apa pun nilai yang diteruskan agar
graph tetap valid untuk sumber yang tidak berbentuk persegi. Berkas `.onnx`
hasil ekspor dapat dimuat kembali melalui `LibreYOLO()` berdasarkan akhiran
berkasnya dan mengembalikan `Results` yang sama.

<code-tabs name="export" />

## Checkpoint

Semua berkas bobot yang dipublikasikan untuk family ini. Satu checkpoint di
bawah tercantum pada deteksi, tetapi berkas yang sama juga dimuat untuk
segmentasi. Jangan teruskan argumen `task` agar mask dikembalikan secara default.

<checkpoint-table />

## Lisensi

<provenance-box>

Mask R-CNN dibangun sebagai subclass wrapper Faster R-CNN milik LibreYOLO.
Model ini memakai sumber torchvision dan lisensi BSD-3-Clause yang sama, serta
menambahkan predictor mask dan head RoI mask dari commit yang di-port yang sama.

</provenance-box>
