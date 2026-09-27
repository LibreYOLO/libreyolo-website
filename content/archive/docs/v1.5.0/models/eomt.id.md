---
title: EoMT
families:
  - eomt
seo_title: 'EoMT: prediksi segmentasi semantik, instance dan panoptik'
description: >-
  Pakai EoMT di LibreYOLO untuk segmentasi semantik, instance dan panoptik di
  atas vision transformer DINOv2 biasa, tanpa perlu decoder. Berlisensi MIT.
lead: >-
  Jaringan segmentasi yang dibangun di atas vision transformer biasa tanpa pixel
  decoder khusus: query terlatih tambahan yang disisipkan ke encoder itu
  sendirilah yang memprediksi mask. LibreYOLO mendukungnya untuk segmentasi
  semantik, instance dan panoptik.
keywords:
  - EoMT
  - encoder-only mask transformer
  - DINOv2
  - panoptic segmentation
  - instance segmentation
  - segmentasi panoptik python
last_verified: 1.5.0
snippets:
  predict:
    - label: Semantik
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreEoMTl-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape)   # (H, W) id kelas
        print(mask.classes)      # id kelas yang ada di gambar, terurut
    - label: Segmentasi instance
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Sufiks -seg pada nama berkas memilih task instance, jadi argumen
        # task tidak diperlukan di sini.
        model = LibreYOLO("LibreEoMTl-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.boxes.xyxy)
        print(result.masks.data.shape)
    - label: Panoptik
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreEoMTl-panoptic.pt")
        result = model(SAMPLE_IMAGE, save=True)

        pan = result.panoptic
        print(pan.data.shape)       # (H, W) id segmen
        print(pan.segments_info)    # [{"id": ..., "category_id": ...}, ...]
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreEoMTl-sem.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Semantik
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: Segmentasi instance
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95(M)"])   # mask
        print(metrics["metrics/mAP50-95(B)"])   # box
    - label: Panoptik
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-panoptic.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/PQ"])
        print(metrics["metrics/SQ"], metrics["metrics/RQ"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreEoMTl-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-sem.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreEoMTl-sem.pt format=onnx
        libreyolo export model=LibreEoMTl-sem.pt format=tensorrt half=True
    - label: Memakai berkas hasil ekspor
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory memilih rute berdasarkan sufiks berkas, jadi artefak hasil
        # ekspor dimuat seperti checkpoint biasa dan mengembalikan objek
        # Results yang sama.
        model = LibreYOLO("LibreEoMTl-sem.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
source_hash: 64b2da642999f150
---

## Instalasi

EoMT tidak memerlukan extra opsional. Semua yang diimpornya sudah ada di
instalasi dasar.

```bash
pip install libreyolo
```

## Prediksi

Bobot diunduh dari Hugging Face saat pertama kali dipakai dan disimpan di cache
lokal. Sufiks task pada nama berkas (`-sem`, `-seg`, `-panoptic`) menentukan
task, dan `LibreYOLO()` menyimpulkannya dari nama berkas itu sehingga argumen
`task=` tidak diperlukan.

<code-tabs name="predict" />

Segmentasi semantik mengisi `result.semantic_mask`, yaitu array `(H, W)` berisi
id kelas di `.data`. Segmentasi instance mengisi `result.boxes` dan
`result.masks`, bentuk yang sama seperti yang dikembalikan family segmentasi
lain. Segmentasi panoptik mengisi `result.panoptic`: peta id segmen `(H, W)` di
`.data`, ditambah `.segments_info`, yaitu daftar dict `{"id", "category_id"}`,
satu untuk setiap segmen. `conf` menyaring pemilihan query; `iou` tidak
berpengaruh pada task semantik, karena task itu melakukan argmax per piksel
tanpa langkah NMS. Lihat [prediksi](/docs/predict) untuk sumber, streaming dan
penanganan hasil.

## Varian

Tiga ukuran encoder, s/b/l, semuanya bertumpu pada DINOv2. Checkpoint semantik
dilatih pada ADE20K di 512 px; checkpoint instance dan panoptik dilatih pada
COCO di 640 px, ditambah satu checkpoint instance kedua yang dilatih di 1280 px.
Upstream hanya merilis bobot segmentasi instance DINOv2 pada ukuran l; s dan b
dipublikasikan untuk semantik dan panoptik saja. Varian EoMT yang bertumpu pada
DINOv3 ada di upstream, tetapi tidak disertakan di sini karena bergantung pada
bobot DINOv3 non-komersial yang aksesnya dibatasi.

LibreYOLO tidak melatih EoMT: `train()` memunculkan `NotImplementedError` untuk
family ini, dan [tingkat dukungan](/docs/models) di atas menandainya sebagai
inferensi saja.

## Validasi

`val()` memilih jalur sesuai task. Semantik mengembalikan `metrics/mIoU` dan
`metrics/pixel_accuracy`. Segmentasi instance mengembalikan kunci mAP mask dan
box yang sama seperti family segmentasi lain. Panoptik mengembalikan Panoptic
Quality sebagai `metrics/PQ`, yang dipecah menjadi `metrics/SQ` (segmentation
quality) dan `metrics/RQ` (recognition quality), ditambah `metrics/PQ_things`
dan `metrics/PQ_stuff`.

<code-tabs name="val" />

## Ekspor

<export-matrix />

Saat ini hanya task semantik yang bisa diekspor: segmentasi instance dan
panoptik yang memanggil `export()` akan mendapat `NotImplementedError`, karena
keluaran query-mask keduanya belum punya kontrak ekspor untuk runtime. Artefak
semantik hasil ekspor dimuat kembali lewat `LibreYOLO()` berdasarkan sufiks
berkasnya, jadi berkas `.onnx` atau `.engine` berperilaku seperti checkpoint dan
mengembalikan `Results` yang sama.

<code-tabs name="export" />

## Checkpoint

Semua berkas bobot yang dipublikasikan untuk family ini.

<checkpoint-table />

## Lisensi

<provenance-box></provenance-box>

## Sitasi

<citation-block />
