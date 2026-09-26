---
title: LingBot-Vision
families:
  - lingbotvision
seo_title: 'LingBot-Vision: segmentasi semantik di LibreYOLO'
description: >-
  Gunakan LingBot-Vision di LibreYOLO untuk segmentasi semantik pada backbone
  ViT berlisensi Apache-2.0. Instal, prediksi, latih, validasi, dan ekspor
  ukuran s/b/l.
lead: >-
  LingBot-Vision adalah family backbone vision transformer self-supervised yang
  dilatih dengan masked modeling berpusat pada batas untuk persepsi spasial
  padat dan dirilis oleh Robbyant. LibreYOLO memasangkan backbone dengan head
  padat dan mendukungnya untuk satu task, yaitu segmentasi semantik.
keywords:
  - LingBot-Vision
  - segmentasi semantik Python
  - vision transformer
  - self-supervised pretraining
  - boundary modeling
  - Robbyant
  - dense prediction
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape, mask.classes)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreLingBotVisions-sem.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python (linear probe)
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Backbone dibekukan secara default agar sesuai dengan protokol evaluasi
        upstream:

        # hanya head padat 1x1 yang dilatih.

        model = LibreYOLO("LibreLingBotVisions-sem.pt")

        model.train(data="my-dataset.yaml", epochs=20, imgsz=512, batch=16)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreLingBotVisions-sem.pt data=my-dataset.yaml \
          epochs=20 imgsz=512 batch=16
    - label: Fine-tuning penuh
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        model.train(
            data="my-dataset.yaml", epochs=20, imgsz=512, batch=16,
            freeze_backbone=False,
        )
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreLingBotVisions-sem.pt data=my-dataset.yaml \
          epochs=20 device=0,1 batch=32
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreLingBotVisions-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreLingBotVisions-sem.pt")
        model.export(format="onnx", imgsz=512)
        model.export(format="coreai", imgsz=512)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreLingBotVisions-sem.pt format=onnx imgsz=512
    - label: Gunakan berkas hasil ekspor
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Factory merutekan berdasarkan akhiran berkas, sehingga artefak hasil
        ekspor dimuat

        # seperti checkpoint lain dan mengembalikan objek Results yang sama.

        model = LibreYOLO("LibreLingBotVisions-sem.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.semantic_mask.data.shape)
source_hash: c47b33fdc6fa1139
---

## Instalasi

LingBot-Vision tidak memerlukan extra opsional. Semua yang diimpornya tersedia
dalam instalasi dasar.

```bash
pip install libreyolo
```

## Prediksi

Bobot diunduh dari Hugging Face saat pertama kali digunakan dan disimpan dalam
cache lokal.

<code-tabs name="predict" />

`result.semantic_mask` memuat peta kelas padat. `.data` adalah tensor ID kelas
`(H, W)` pada ukuran gambar asli, sedangkan `.classes` mencantumkan ID kelas yang
benar-benar ada. `result.boxes` adalah `None` karena tidak ada deteksi per
instance. `conf` dan `iou` diterima demi paritas API, tetapi tidak mengubah
output karena model mengembalikan satu kelas per piksel, bukan deteksi yang
perlu difilter. Lihat [prediksi](/docs/predict) untuk sumber, streaming, dan
penanganan hasil.

## Varian

Ada tiga ukuran yang dipublikasikan, s, b, dan l, yang didistilasi dari teacher
ViT-g/16 dengan 1.1 miliar parameter. Teacher itu sendiri, ukuran `g`, dapat
dimuat dan menjalani fine-tuning di LibreYOLO, tetapi LibreYOLO tidak menghosting
checkpoint `g` miliknya sendiri.

<checkpoint-table />

## Pelatihan

`train()` melakukan fine-tuning pada checkpoint yang dipublikasikan. Resep
default adalah linear probe dari laporan upstream: backbone ViT dibekukan dan
hanya head padat 1x1 yang dilatih, sesuai cara bobot yang dihosting LibreYOLO di
atas dibuat. Teruskan `freeze_backbone=False` untuk melakukan fine-tuning pada
seluruh jaringan dan turunkan `lr0` sesuai kebutuhan.

<code-tabs name="train" />

Lihat [pelatihan](/docs/train) untuk dataset, augmentasi, multi-GPU, dan logger.

## Validasi

`val()` mengembalikan dictionary key `metrics/`, yaitu mIoU dan akurasi piksel
yang diukur terhadap dataset apa pun dalam format yang digunakan saat pelatihan.

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

<provenance-box>

Rilis upstream mendokumentasikan bahwa ViT-nya dibangun berdasarkan arsitektur
DINOv2/DINOv3 yang dipublikasikan Meta AI. Robbyant mendistribusikan
implementasinya dengan lisensi Apache-2.0, dan port LibreYOLO ini hanya dibuat
dari repository Robbyant, tidak pernah dari kode DINOv2 atau DINOv3 milik Meta.

</provenance-box>

## Sitasi

<citation-block />
