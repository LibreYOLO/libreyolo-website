---
title: EfficientNetV2
families:
  - efficientnetv2
seo_title: 'EfficientNetV2: latih, validasi dan ekspor di bawah Apache-2.0'
description: >-
  Pakai EfficientNetV2 di LibreYOLO untuk klasifikasi gambar. Instalasi,
  prediksi, fine-tuning, validasi dan ekspor LibreEfficientNetV2 b0 sampai b3.
lead: >-
  EfficientNetV2 adalah classifier gambar yang kedalaman, lebar dan pilihan blok
  tiap tahapnya ditemukan lewat neural architecture search, dioptimalkan
  sekaligus untuk akurasi dan kecepatan pelatihan, bukan untuk akurasi saja.
  LibreYOLO mendukungnya untuk satu task: klasifikasi.
keywords:
  - EfficientNetV2
  - EfficientNetV2-b0
  - klasifikasi gambar python
  - image classification
  - neural architecture search
  - MBConv
  - ImageNet classifier
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreEfficientNetV2b0-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.probs.top1, result.probs.top1conf)
        print(result.probs.top5)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreEfficientNetV2b0-cls.pt source=cat.jpg
        save=True
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEfficientNetV2b0-cls.pt")
        model.train(data="imagenette160", epochs=5)
    - label: CLI
      language: bash
      code: >
        libreyolo train model=LibreEfficientNetV2b0-cls.pt data=imagenette160
        epochs=5
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreEfficientNetV2b0-cls.pt data=imagenette160 \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEfficientNetV2b0-cls.pt")
        metrics = model.val(data="imagenette160")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreEfficientNetV2b0-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEfficientNetV2b0-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreEfficientNetV2b0-cls.pt format=onnx

        libreyolo export model=LibreEfficientNetV2b0-cls.pt format=tensorrt
        half=True
    - label: Memakai berkas hasil ekspor
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory memilih rute berdasarkan sufiks berkas, jadi artefak hasil
        # ekspor dimuat seperti checkpoint biasa dan mengembalikan objek
        # Results yang sama.
        model = LibreYOLO("LibreEfficientNetV2b0-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1)
source_hash: ad3ff140aad824bd
---

## Instalasi

EfficientNetV2 tidak memerlukan extra opsional. Semua yang diimpornya sudah ada
di instalasi dasar.

```bash
pip install libreyolo
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

Empat ukuran, b0 sampai b3, masing-masing dievaluasi pada resolusi dan rasio
crop-nya sendiri, bukan memakai satu ukuran input yang sama untuk seluruh
family. Memilih ukuran hanyalah pertukaran langsung antara jumlah parameter dan
akurasi. Task-nya tetap: setiap ukuran hanya mencakup klasifikasi. Nama berkas
bobot selalu berakhiran `-cls.pt` di semua ukuran, dan sufiks itulah yang dibaca
factory untuk mengarahkan ke family ini; argumen `task=` tidak diperlukan.

## Pelatihan

Fine-tuning dimulai dari backbone ImageNet yang dipublikasikan dan membangun
ulang lapisan classifier terakhir sesuai jumlah kelas dataset target secara
otomatis. Secara bawaan `imgsz` mengikuti resolusi evaluasi milik ukuran itu
sendiri, kecuali disetel secara eksplisit.

<code-tabs name="train" />

Tanpa pengaturan tambahan, trainer menjalankan 100 epoch pada `lr0=1e-3` dengan
AdamW, batch 64 dan early stopping setelah 50 epoch tanpa perbaikan. `data`
menerima root dataset (`train/` dan `val/`, satu folder per kelas), nama pendek
yang dikenal seperti `imagenette160`, atau URL `.zip`. `lora=True` tidak
didukung di sini; memberikannya akan memunculkan error, karena LoRA di LibreYOLO
menyasar komponen transformer yang memiliki lapisan `nn.Linear`, sedangkan blok
MBConv di family ini tidak punya.

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

<provenance-box></provenance-box>
