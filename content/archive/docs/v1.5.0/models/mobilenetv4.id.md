---
title: MobileNetV4
families:
  - mobilenetv4
seo_title: 'MobileNetV4: pelatihan, validasi, dan ekspor berlisensi Apache-2.0'
description: >-
  Gunakan MobileNetV4 di LibreYOLO untuk klasifikasi gambar. Instal, prediksi,
  lakukan fine-tuning, validasi, dan ekspor LibreMobileNetV4 small/medium/large.
lead: >-
  MobileNetV4 adalah classifier gambar untuk perangkat mobile dan edge yang
  memakai blok Universal Inverted Bottleneck guna menyatukan beberapa desain
  blok mobile terdahulu ke satu struktur yang dapat dicari. LibreYOLO
  mendukungnya untuk satu task: klasifikasi.
keywords:
  - MobileNetV4
  - MobileNetV4 conv
  - klasifikasi gambar Python
  - mobile inference
  - edge classifier
  - ImageNet classifier
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMobileNetV4s-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.probs.top1, result.probs.top1conf)
        print(result.probs.top5)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreMobileNetV4s-cls.pt source=cat.jpg
        save=True
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMobileNetV4s-cls.pt")
        model.train(data="imagenette160", epochs=5)
    - label: CLI
      language: bash
      code: >
        libreyolo train model=LibreMobileNetV4s-cls.pt data=imagenette160
        epochs=5
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreMobileNetV4s-cls.pt data=imagenette160 \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMobileNetV4s-cls.pt")
        metrics = model.val(data="imagenette160")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreMobileNetV4s-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMobileNetV4s-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreMobileNetV4s-cls.pt format=onnx

        libreyolo export model=LibreMobileNetV4s-cls.pt format=tensorrt
        half=True
    - label: Gunakan berkas hasil ekspor
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Factory merutekan berdasarkan akhiran berkas, sehingga artefak hasil
        ekspor dimuat

        # seperti checkpoint lain dan mengembalikan objek Results yang sama.

        model = LibreYOLO("LibreMobileNetV4s-cls.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.probs.top1)
source_hash: 4a9a1b392ffb136d
---

## Instalasi

MobileNetV4 tidak memerlukan extra opsional. Semua yang diimpornya tersedia
dalam instalasi dasar.

```bash
pip install libreyolo
```

## Prediksi

Bobot diunduh dari Hugging Face saat pertama kali digunakan dan disimpan dalam
cache lokal.

<code-tabs name="predict" />

Objek `Results` yang dikembalikan sama seperti yang dikembalikan setiap family,
sehingga mengganti model hanya memerlukan perubahan satu baris. Classifier tidak
memiliki box atau mask. `result.probs` memuat prediksi seluruh gambar dengan
`top1`, `top5`, `top1conf`, dan `top5conf`. `conf`, `iou`, dan `max_det`
diterima demi paritas API, tetapi tidak berpengaruh karena tidak ada yang perlu
diberi ambang batas atau di-suppress pada satu vektor probabilitas. Lihat
[prediksi](/docs/predict) untuk sumber, streaming, dan penanganan hasil.

## Varian

Ada tiga ukuran, small/medium/large, semuanya hanya berbasis konvolusi. Family
ini tidak menyertakan varian hybrid yang menambahkan attention Mobile MQA.
Pemilihan ukuran merupakan pertukaran langsung antara jumlah parameter dan
akurasi. Task-nya tetap: setiap ukuran hanya mencakup klasifikasi. Nama berkas
bobot berakhiran `-cls.pt` pada setiap ukuran, dan akhiran itulah yang dibaca
factory untuk merutekan ke family ini. Argumen `task=` tidak diperlukan.

## Pelatihan

Fine-tuning dimulai dari backbone ImageNet yang dipublikasikan dan membangun
ulang lapisan classifier terakhir secara otomatis sesuai jumlah kelas dataset
target.

<code-tabs name="train" />

Jika dibiarkan, trainer berjalan selama 100 epoch pada `lr0=1e-3` dengan AdamW,
batch 64, dan early stopping setelah 50 epoch tanpa peningkatan. `data` menerima
root dataset (`train/` dan `val/`, satu folder per kelas), nama pendek yang
dikenal seperti `imagenette160`, atau URL `.zip`. `lora=True` tidak didukung di
sini. Meneruskannya memunculkan error karena LoRA di LibreYOLO menargetkan
komponen transformer dengan lapisan `nn.Linear`, sedangkan blok UIB family ini
tidak memilikinya.

Lihat [pelatihan](/docs/train) untuk dataset, augmentasi, multi-GPU, dan logger.

## Validasi

`val()` mengembalikan dictionary berisi key `metrics/`. Untuk klasifikasi, ini
adalah akurasi top-1 dan top-5 pada split validasi.

<code-tabs name="val" />

## Ekspor

<export-matrix />

Artefak hasil ekspor dapat dimuat kembali melalui `LibreYOLO()` berdasarkan
akhiran berkasnya, sehingga berkas `.onnx` atau `.engine` berperilaku seperti
checkpoint dan mengembalikan `Results` yang sama. [Ekspor](/docs/export)
mencantumkan argumen yang diterima setiap format serta extra yang ditambahkan
oleh beberapa format.

<code-tabs name="export" />

## Checkpoint

Semua berkas bobot yang dipublikasikan untuk family ini.

<checkpoint-table />

## Lisensi

<provenance-box></provenance-box>
