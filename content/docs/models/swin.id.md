---
title: Swin Transformer
families:
  - swin
seo_title: 'Swin Transformer: klasifikasikan gambar dengan LibreSwin dari LibreYOLO'
description: >-
  Prediksi, validasi, dan ekspor pengklasifikasi Swin Transformer dengan
  LibreYOLO. Bobot MIT; fine-tuning belum didukung.
lead: >-
  Swin Transformer V1: vision transformer hierarkis yang menghitung attention di
  dalam window lokal bergeser, bukan di seluruh gambar. LibreYOLO menyediakan
  empat ukuran untuk klasifikasi gambar.
keywords:
  - Swin Transformer
  - hierarchical vision transformer
  - shifted window attention
  - klasifikasi gambar
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSwint-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        probs = result.probs
        print(probs.top1, probs.top1conf)
        print(probs.top5, probs.top5conf)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreSwint-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSwint-cls.pt")

        # data adalah root direktori dengan split folder kelas train/ dan val/
        # (tata letak ImageFolder), bukan YAML dataset.
        metrics = model.val(data="imagenet-1k/")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSwint-cls.pt data=imagenet-1k/
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSwint-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreSwint-cls.pt format=onnx
        libreyolo export model=LibreSwint-cls.pt format=tensorrt half=True
    - label: Gunakan berkas hasil ekspor
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Factory merutekan berdasarkan sufiks berkas, sehingga artefak hasil
        ekspor

        # dimuat seperti checkpoint lain dan mengembalikan objek Results yang
        sama.

        model = LibreYOLO("LibreSwint-cls.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.probs.top1)
source_hash: faa6bbacae62d88e
---

## Instalasi

Swin tidak memerlukan komponen tambahan opsional. Semua impornya tersedia dalam instalasi dasar.

```bash
pip install libreyolo
```

## Prediksi

Bobot diunduh dari Hugging Face saat pertama kali digunakan dan disimpan dalam cache lokal.

<code-tabs name="predict" />

Pengklasifikasi mengembalikan `result.probs`, bukan `result.boxes`: `top1` dan `top5`
memberikan indeks kelas, sedangkan `top1conf` dan `top5conf` memberikan confidence-nya.
Setiap ukuran ditetapkan pada input 224px karena tahap attention akhir dibuat untuk resolusi
tersebut; prediksi, validasi, dan ekspor semuanya memunculkan error jika `imgsz` berbeda
diteruskan. Lihat [prediksi](/docs/predict) untuk sumber, streaming, dan penanganan hasil.

## Varian

Empat ukuran, tiny hingga large, dibuat dari tower shifted-window yang sama dan berbeda
dalam lebar embedding serta kedalaman tahap. Large di-pretrain pada ImageNet-22k dan
dilakukan fine-tuning pada ImageNet-1k; tiga lainnya dilatih langsung pada ImageNet-1k.
LibreYOLO menyediakan family ini khusus untuk inferensi: prediksi, validasi top-1/top-5
bergaya ImageNet, dan ekspor didukung, sedangkan resep pelatihan ImageNet upstream tidak
diimplementasikan.

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
