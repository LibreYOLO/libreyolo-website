---
title: ViT
families:
  - vit
seo_title: 'ViT: jalankan pengklasifikasi Vision Transformer klasik di LibreYOLO'
description: >-
  Prediksi, validasi, dan ekspor pengklasifikasi ViT dengan LibreYOLO. Bobot
  AugReg Apache-2.0; fine-tuning belum didukung.
lead: >-
  Vision Transformer klasik: transformer murni yang diterapkan pada patch gambar
  berukuran tetap, dengan token kelas yang dipelajari dan tanpa konvolusi.
  LibreYOLO menyediakan empat ukuran pretrained AugReg untuk klasifikasi gambar.
keywords:
  - ViT
  - Vision Transformer
  - AugReg
  - klasifikasi gambar
  - transformer classifier
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreViTti-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        probs = result.probs
        print(probs.top1, probs.top1conf)
        print(probs.top5, probs.top5conf)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreViTti-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreViTti-cls.pt")

        # data adalah root direktori dengan split folder kelas train/ dan val/
        # (tata letak ImageFolder), bukan YAML dataset.
        metrics = model.val(data="imagenet-1k/")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreViTti-cls.pt data=imagenet-1k/
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreViTti-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreViTti-cls.pt format=onnx
        libreyolo export model=LibreViTti-cls.pt format=tensorrt half=True
    - label: Gunakan berkas hasil ekspor
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Factory merutekan berdasarkan sufiks berkas, sehingga artefak hasil
        ekspor

        # dimuat seperti checkpoint lain dan mengembalikan objek Results yang
        sama.

        model = LibreYOLO("LibreViTti-cls.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.probs.top1)
source_hash: f63e98454913765a
---

## Instalasi

ViT tidak memerlukan komponen tambahan opsional. Semua impornya tersedia dalam instalasi dasar.

```bash
pip install libreyolo
```

## Prediksi

Bobot diunduh dari Hugging Face saat pertama kali digunakan dan disimpan dalam cache lokal.

<code-tabs name="predict" />

Pengklasifikasi mengembalikan `result.probs`, bukan `result.boxes`: `top1` dan `top5`
memberikan indeks kelas, sedangkan `top1conf` dan `top5conf` memberikan confidence-nya.
Prapemrosesan mengubah ukuran dan melakukan center crop ke input tetap 224px dengan resep
evaluasi AugReg milik timm: interpolasi bikubik pada fraksi crop 0.9. Lihat
[prediksi](/docs/predict) untuk sumber, streaming, dan penanganan hasil.

## Varian

Empat ukuran, tiny hingga large, yang memakai satu graph patch-16 tetap pada 224px dan
berbeda dalam lebar embedding serta kedalaman transformer. LibreYOLO menyediakan family ini
khusus untuk inferensi: prediksi, validasi top-1/top-5 bergaya ImageNet, dan ekspor didukung,
sedangkan resep fine-tuning AugReg tidak diimplementasikan.

## Validasi

`val()` berjalan terhadap split bergaya ImageFolder (direktori dengan subfolder `train/`
dan `val/`, satu folder per kelas) dan mengembalikan akurasi top-1 dan top-5.

<code-tabs name="val" />

## Ekspor

<export-matrix />

Artefak hasil ekspor dimuat kembali melalui `LibreYOLO()` berdasarkan sufiks berkasnya,
sehingga berkas `.onnx` atau `.engine` berperilaku seperti checkpoint dan mengembalikan
`Results` yang sama. [Ekspor](/docs/export) mencantumkan argumen yang diterima setiap format
beserta komponen tambahan yang disediakan beberapa format.

<code-tabs name="export" />

## Checkpoint

Setiap berkas bobot yang dipublikasikan untuk family ini.

<checkpoint-table />

## Lisensi

<provenance-box></provenance-box>

## Sitasi

<citation-block />
