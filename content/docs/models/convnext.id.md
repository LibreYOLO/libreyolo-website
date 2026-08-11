---
title: ConvNeXt
families:
  - convnext
seo_title: 'ConvNeXt: latih, validasi dan ekspor di bawah Apache-2.0'
description: >-
  Pakai ConvNeXt di LibreYOLO untuk klasifikasi gambar. Instalasi, prediksi,
  fine-tuning dengan LoRA, validasi dan ekspor LibreConvNeXt tiny/small/base.
lead: >-
  ConvNeXt adalah classifier gambar yang dibangun sepenuhnya dari konvolusi
  standar, dimodernisasi blok demi blok dari ResNet menuju pilihan desain vision
  transformer. LibreYOLO mendukungnya untuk satu task: klasifikasi.
keywords:
  - ConvNeXt
  - ConvNeXt tiny
  - klasifikasi gambar python
  - image classification
  - ImageNet classifier
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.probs.top1, result.probs.top1conf)
        print(result.probs.top5)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreConvNeXtt-cls.pt source=cat.jpg save=True
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        model.train(data="imagenette160", epochs=5)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreConvNeXtt-cls.pt data=imagenette160 epochs=5
    - label: LoRA
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        model.train(data="imagenette160", epochs=5, lora=True)
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreConvNeXtt-cls.pt data=imagenette160 \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        metrics = model.val(data="imagenette160")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreConvNeXtt-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreConvNeXtt-cls.pt format=onnx
        libreyolo export model=LibreConvNeXtt-cls.pt format=tensorrt half=True
    - label: Memakai berkas hasil ekspor
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory memilih rute berdasarkan sufiks berkas, jadi artefak hasil
        # ekspor dimuat seperti checkpoint biasa dan mengembalikan objek
        # Results yang sama.
        model = LibreYOLO("LibreConvNeXtt-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
source_hash: 1682cc69cf2925e6
---

## Instalasi

ConvNeXt tidak memerlukan extra opsional. Semua yang diimpornya sudah ada di
instalasi dasar.

```bash
pip install libreyolo
```

Fine-tuning adapter dengan `lora=True` adalah pengecualiannya, dan memerlukan
extra `lora`.

```bash
pip install "libreyolo[lora]"
```

## Prediksi

Bobot diunduh dari Hugging Face saat pertama kali dipakai dan disimpan di cache
lokal.

<code-tabs name="predict" />

Objek `Results` yang dikembalikan sama dengan yang dikembalikan setiap family,
jadi mengganti model dengan model lain cukup satu baris perubahan. Classifier
tidak membawa box atau mask: `result.probs` menyimpan prediksi untuk seluruh
gambar, dengan `top1`, `top5`, `top1conf` dan `top5conf`. `conf`, `iou` dan
`max_det` tetap diterima demi paritas API tetapi tidak berpengaruh, karena pada
satu vektor probabilitas tidak ada yang perlu disaring dengan ambang batas atau
ditekan. Lihat [prediksi](/docs/predict) untuk sumber, streaming dan penanganan
hasil.

## Varian

Tiga ukuran, tiny/small/base, semuanya dilatih dan dievaluasi dengan cara yang
sama, jadi memilih salah satunya hanyalah pertukaran antara jumlah parameter dan
akurasi. Task-nya tetap: setiap ukuran hanya mencakup klasifikasi. Nama berkas
bobot selalu berakhiran `-cls.pt` di semua ukuran, dan sufiks itulah yang dibaca
factory untuk mengarahkan ke family ini; argumen `task=` tidak diperlukan.

## Pelatihan

Fine-tuning dimulai dari backbone ImageNet yang dipublikasikan dan membangun
ulang lapisan classifier terakhir sesuai jumlah kelas dataset target secara
otomatis.

<code-tabs name="train" />

Tanpa pengaturan tambahan, trainer menjalankan 100 epoch pada `lr0=1e-3` dengan
AdamW, batch 64 dan early stopping setelah 50 epoch tanpa perbaikan. `data`
menerima root dataset (`train/` dan `val/`, satu folder per kelas), nama pendek
yang dikenal seperti `imagenette160`, atau URL `.zip`. Blok ConvNeXt memuat MLP
`nn.Linear` yang dibutuhkan LoRA, jadi `lora=True` didukung di sini, dan
menyisipkan adapter ke MLP di dalam blok alih-alih melakukan fine-tuning pada
seluruh backbone.

Lihat [pelatihan](/docs/train) untuk dataset, augmentasi, multi-GPU dan
logger.

## Validasi

`val()` mengembalikan dictionary berisi kunci `metrics/`. Untuk klasifikasi,
isinya akurasi top-1 dan top-5 pada split validasi.

<code-tabs name="val" />

## Ekspor

<export-matrix />

Artefak hasil ekspor dimuat kembali lewat `LibreYOLO()` berdasarkan sufiks
berkasnya, jadi berkas `.onnx` atau `.engine` berperilaku seperti checkpoint dan
mengembalikan `Results` yang sama. [Ekspor](/docs/export) memuat daftar argumen
yang diterima setiap format dan tambahan yang dipakai beberapa di antaranya.

<code-tabs name="export" />

## Checkpoint

Semua berkas bobot yang dipublikasikan untuk family ini.

<checkpoint-table />

## Lisensi

<provenance-box>

Hanya ConvNeXt V1 yang disertakan di family ini. Checkpoint pretrained kecil
milik ConvNeXt-V2 berlisensi CC-BY-NC 4.0 dan sengaja tidak disertakan, karena
bobot non-komersial tidak bisa didistribusikan ulang di dalam library
MIT/komersial.

</provenance-box>

## Sitasi

<citation-block />
