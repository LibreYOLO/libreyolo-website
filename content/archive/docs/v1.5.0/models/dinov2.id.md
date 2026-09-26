---
title: DINOv2
families:
  - dinov2
seo_title: 'DINOv2 di LibreYOLO: segmentasi semantik, klasifikasi, dan embedding'
description: >-
  Gunakan DINOv2 di LibreYOLO untuk segmentasi semantik, klasifikasi, dan
  embedding seluruh gambar pada backbone DINOv2-with-Registers. Seluruhnya
  berlisensi Apache-2.0.
lead: >-
  DINOv2 adalah vision transformer self-supervised yang dilatih oleh Meta AI
  untuk menghasilkan fitur gambar serbaguna tanpa label. LibreYOLO membungkus
  backbone DINOv2-with-Registers untuk tiga task: segmentasi semantik,
  klasifikasi, dan embedding seluruh gambar.
keywords:
  - DINOv2
  - DINOv2 with registers
  - self-supervised learning
  - vision transformer
  - segmentasi semantik
  - image embedding
  - ekstraksi fitur
  - Meta AI
last_verified: 1.5.0
snippets:
  predict:
    - label: Semantik
      language: python
      code: >
        from libreyolo import SAMPLE_IMAGE

        from libreyolo.models.dinov2.model import LibreDINOv2


        # Tidak ada checkpoint yang dihosting LibreYOLO untuk family ini. Kode
        ini

        # mengunduh backbone DINOv2-with-Registers-small berlisensi Apache-2.0
        dari

        # organisasi Hugging Face milik Meta. Head padat dimulai dengan

        # inisialisasi acak sampai model dilatih (lihat Pelatihan di bawah).

        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)

        result = model(SAMPLE_IMAGE)


        mask = result.semantic_mask

        print(mask.data.shape, mask.classes)
    - label: Klasifikasi
      language: python
      code: |
        from libreyolo import SAMPLE_IMAGE
        from libreyolo.models.dinov2.model import LibreDINOv2

        # nb_classes= adalah jumlah kelas dataset Anda. Head linear dimulai
        # dengan inisialisasi acak sampai model dilatih.
        model = LibreDINOv2(size="s", task="classify", nb_classes=10)
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1, result.probs.top1conf)
    - label: Embedding
      language: python
      code: |
        from libreyolo import SAMPLE_IMAGE
        from libreyolo.models.dinov2.model import LibreDINOv2

        # Melewati setiap head task. Backbone saja sudah cukup, sehingga
        # fine-tuning tidak diperlukan agar berguna.
        model = LibreDINOv2(size="s", task="embed")
        result = model(SAMPLE_IMAGE)

        print(result.embeddings.data.shape)   # (1, D), dinormalisasi L2
    - label: Embedding satu batch
      language: python
      code: >
        from libreyolo.models.dinov2.model import LibreDINOv2


        model = LibreDINOv2(size="s", task="embed")


        # Wrapper praktis: menjalankan predict() dan menumpuk setiap baris
        menjadi satu

        # tensor (N, D).

        features = model.embed(["a.jpg", "b.jpg", "c.jpg"])

        print(features.shape)
  train:
    - label: Semantik
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        model.train(data="my-dataset.yaml", epochs=100, batch_size=4, lr=1e-4)
    - label: Klasifikasi
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="classify", nb_classes=10)
        model.train(data="my-dataset.yaml", epochs=100, batch_size=4, lr=1e-4)
    - label: Multi-GPU
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        model.train(
            data="my-dataset.yaml", epochs=100, batch_size=4, lr=1e-4,
            device="0,1",
        )
  val:
    - label: Semantik
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: Klasifikasi
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="classify", nb_classes=10)
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
  export:
    - label: Semantik
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="semantic", nb_classes=19)
        model.export(format="onnx")
    - label: Klasifikasi
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="classify", nb_classes=10)
        model.export(format="onnx")
    - label: Embedding
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="embed")
        model.export(format="tflite")
    - label: Gunakan berkas hasil ekspor
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Factory merutekan berdasarkan akhiran berkas, sehingga artefak hasil
        ekspor dimuat

        # seperti checkpoint lain dan mengembalikan objek Results yang sama.
        Ekspor

        # menamai berkas berdasarkan task, dalam contoh ini
        LibreDINOv2s-sem.onnx.

        model = LibreYOLO("LibreDINOv2s-sem.onnx")

        result = model(SAMPLE_IMAGE)
source_hash: 4256e0a0398e5aaf
---

## Instalasi

LibreDINOv2 hanya terdaftar saat `transformers` terpasang. Ini adalah dependency
opsional yang juga diperlukan RF-DETR untuk backbone DINOv2, sehingga extra yang
diperlukan pun sama.

```bash
pip install "libreyolo[rfdetr]"
```

## Prediksi

LibreYOLO tidak memublikasikan checkpoint LibreDINOv2. Buat wrapper secara
langsung alih-alih memuat berkas: `model_path=None` (default) mengunduh backbone
`facebook/dinov2-with-registers-small` berlisensi Apache-2.0 milik Meta dari
Hugging Face saat pertama kali digunakan. `task=` memilih apa yang dijalankan
di atasnya.

<code-tabs name="predict" />

`task="semantic"` dan `task="classify"` menambahkan head padat atau linear di
atas backbone. Head tersebut diinisialisasi secara acak dan hanya berguna
setelah dilatih (lihat [Pelatihan](#train)). `task="embed"` melewati setiap
head dan mengembalikan token CLS terakhir backbone yang telah dinormalisasi
sebagai satu baris seluruh gambar dalam `result.embeddings`, sehingga tidak
memerlukan pelatihan sama sekali. `result.boxes` selalu `None`: tidak satu pun
dari ketiga task menghasilkan deteksi per instance. Lihat
[prediksi](/docs/predict) untuk sumber, streaming, dan penanganan hasil.

## Varian

`size` memilih lebar projector bergaya RF-DETR yang dilapiskan di atas backbone,
bukan backbone itu sendiri. Setiap ukuran memakai encoder DINOv2-S (small) yang
sama. Segmentasi semantik berjalan pada grid patch persegi native DINOv2,
sedangkan klasifikasi dan embedding berjalan pada resolusi klasifikasi lebih
kecil yang digunakan untuk melatih linear probe.

## Pelatihan

`task="semantic"` dan `task="classify"` dapat dilatih. `task="embed"` tidak
memiliki head yang bergantung pada kelas untuk di-fit dan memunculkan
`NotImplementedError` jika `train()` dipanggil padanya.

<code-tabs name="train" />

Argumen kata kunci utama di sini adalah `batch_size` dan `lr`, bukan `batch`
dan `lr0` yang digunakan sebagian besar family lain. `batch` dan `lr0` tetap
diterima dan dipetakan ke argumen tersebut, tetapi meneruskan keduanya
memunculkan error konflik. `output_dir=` (default `"runs/train"`) menggantikan
`project=`/`name=` sebagai cara utama untuk menempatkan sebuah run, meskipun
`project=`/`name=` tetap dapat diteruskan secara langsung. Lihat
[pelatihan](/docs/train) untuk dataset, augmentasi, multi-GPU, dan logger.

## Validasi

`val()` mengembalikan dictionary key `metrics/`: mIoU dan akurasi piksel untuk
`task="semantic"`, serta akurasi top-1 dan top-5 untuk `task="classify"`.
`task="embed"` tidak memiliki ground truth sebagai pembanding skor dan
memunculkan `NotImplementedError` jika `val()` dipanggil padanya.

<code-tabs name="val" />

## Ekspor

<export-matrix />

Setiap task mendukung subset format yang berbeda, seperti ditampilkan di atas.
Artefak hasil ekspor dapat dimuat kembali melalui `LibreYOLO()` berdasarkan
akhiran berkasnya, sehingga berkas `.onnx` atau `.engine` berperilaku seperti
checkpoint dan mengembalikan `Results` yang sama. [Ekspor](/docs/export)
mencantumkan argumen yang diterima oleh setiap format.

<code-tabs name="export" />

## Lisensi

<provenance-box>

Baris "Weights" di atas menyebutkan lisensi yang berlaku, yaitu Apache-2.0,
tetapi tidak ada apa pun yang benar-benar dipublikasikan ulang di organisasi
Hugging Face LibreYOLO untuk family ini. LibreYOLO tidak menghosting checkpoint
LibreDINOv2 miliknya sendiri. Yang diunduh oleh
`LibreDINOv2(model_path=None)` adalah repository
`facebook/dinov2-with-registers-small` milik Meta, tanpa perubahan.

</provenance-box>

## Sitasi

<citation-block />
