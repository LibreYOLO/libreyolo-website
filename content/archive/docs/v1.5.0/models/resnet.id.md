---
title: ResNet
families:
  - resnet
seo_title: 'ResNet: latih, validasi, dan ekspor di bawah Apache-2.0'
description: >-
  Gunakan ResNet di LibreYOLO untuk klasifikasi gambar. Instal, prediksi,
  lakukan fine-tuning, validasi, dan ekspor LibreResNet18/34/50/101.
lead: >-
  ResNet adalah pengklasifikasi gambar yang dibangun dari residual block dan
  skip connection, sehingga jaringan dapat menambahkan jauh lebih banyak lapisan
  tanpa kehilangan akurasi seperti pada tumpukan konvolusional biasa yang dalam.
  LibreYOLO mendukungnya untuk satu task: klasifikasi.
keywords:
  - ResNet
  - ResNet50
  - klasifikasi gambar
  - residual learning
  - deep residual network
  - classifier ImageNet
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreResNet50-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.probs.top1, result.probs.top1conf)
        print(result.probs.top5)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreResNet50-cls.pt source=cat.jpg save=True
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreResNet50-cls.pt")
        model.train(data="imagenette160", epochs=5)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreResNet50-cls.pt data=imagenette160 epochs=5
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreResNet50-cls.pt data=imagenette160 \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreResNet50-cls.pt")
        metrics = model.val(data="imagenette160")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreResNet50-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreResNet50-cls.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreResNet50-cls.pt format=onnx
        libreyolo export model=LibreResNet50-cls.pt format=tensorrt half=True
    - label: Gunakan berkas hasil ekspor
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Factory merutekan berdasarkan sufiks berkas, sehingga artefak hasil
        ekspor

        # dimuat seperti checkpoint lain dan mengembalikan objek Results yang
        sama.

        model = LibreYOLO("LibreResNet50-cls.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.probs.top1)
source_hash: e2f46c73716af1b7
---

## Instalasi

ResNet tidak memerlukan komponen tambahan opsional. Semua impornya tersedia dalam instalasi dasar.

```bash
pip install libreyolo
```

## Prediksi

Bobot diunduh dari Hugging Face saat pertama kali digunakan dan disimpan dalam cache lokal.

<code-tabs name="predict" />

Objek `Results` yang dikembalikan sama dengan yang dikembalikan setiap family, jadi mengganti
model hanya memerlukan perubahan satu baris. Pengklasifikasi tidak memiliki kotak atau mask:
`result.probs` menyimpan prediksi seluruh gambar dengan `top1`, `top5`,
`top1conf`, dan `top5conf`. `conf`, `iou`, dan `max_det` diterima demi kesetaraan API
tetapi tidak berpengaruh karena tidak ada yang perlu diberi ambang batas atau ditekan pada
satu vektor probabilitas. Lihat [prediksi](/docs/predict) untuk sumber,
streaming, dan penanganan hasil.

## Varian

Empat kedalaman, semuanya dilatih dan dievaluasi dengan cara yang sama, sehingga pemilihannya
merupakan pertukaran langsung antara jumlah parameter dan akurasi. Task-nya tetap: setiap
ukuran hanya mencakup klasifikasi. Nama berkas bobot berakhir dengan `-cls.pt` pada setiap
ukuran, dan sufiks itulah yang dibaca factory untuk merutekan ke family ini; argumen
`task=` tidak diperlukan.

## Pelatihan

Fine-tuning dimulai dari backbone ImageNet yang dipublikasikan dan secara otomatis membangun
ulang lapisan pengklasifikasi akhir agar sesuai dengan jumlah kelas dataset target.

<code-tabs name="train" />

Jika dibiarkan, pelatih berjalan selama 100 epoch pada `lr0=1e-3` dengan AdamW, batch
64, dan early stopping setelah 50 epoch tanpa peningkatan. `data` menerima root dataset
(`train/` dan `val/`, satu folder per kelas), nama pendek yang dikenal seperti
`imagenette160`, atau URL `.zip`. `lora=True` tidak didukung di sini; meneruskannya akan
memunculkan error karena LoRA di LibreYOLO menargetkan komponen transformer dengan lapisan
`nn.Linear`, sedangkan ResNet tidak memilikinya.

Lihat [pelatihan](/docs/train) untuk dataset, augmentasi, multi-GPU, dan logger.

## Validasi

`val()` mengembalikan dictionary dengan key `metrics/`. Untuk klasifikasi, key tersebut
mencakup akurasi top-1 dan top-5 pada split validasi.

<code-tabs name="val" />

## Ekspor

<export-matrix />

Artefak hasil ekspor dimuat kembali melalui `LibreYOLO()` berdasarkan sufiks berkasnya,
sehingga berkas `.onnx` atau `.engine` berperilaku seperti checkpoint dan mengembalikan
`Results` yang sama. [Ekspor](/docs/export) mencantumkan argumen yang diterima setiap
format beserta komponen tambahan yang disediakan beberapa format.

<code-tabs name="export" />

## Checkpoint

Setiap berkas bobot yang dipublikasikan untuk family ini.

<checkpoint-table />

## Lisensi

<provenance-box></provenance-box>

## Sitasi

<citation-block />
