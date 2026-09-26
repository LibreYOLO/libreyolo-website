---
title: Segmentasi semantik
seo_title: Segmentasi semantik di LibreYOLO
description: >-
  Beri label kelas pada setiap piksel di LibreYOLO: family yang melayani task,
  format mask rapat, serta pemanggilan prediksi, pelatihan, validasi, dan
  ekspor.
lead: >-
  Segmentasi semantik menetapkan kelas pada setiap piksel gambar dan tidak
  membedakan instance dari kelas yang sama. Kunci task-nya adalah semantic.
keywords:
  - segmentasi semantik Python
  - klasifikasi piksel
  - dense prediction
  - melatih model segmentasi
  - mIoU
  - library segmentasi MIT
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Akhiran -sem pada nama berkas memilih task, sehingga argumen task
        # tidak diperlukan.
        model = LibreYOLO("LibreSegformerb0-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape)   # (H, W) id kelas pada kanvas asli
        print(mask.classes)      # id kelas terurut yang ada, mengabaikan 255
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreSegformerb0-sem.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Satu kelas pada satu waktu
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        result = LibreYOLO("LibreSegformerb0-sem.pt")(SAMPLE_IMAGE)
        mask = result.semantic_mask

        for class_id in mask.classes:
            pixels = mask.class_mask(class_id)   # boolean (H, W)
            print(result.names[class_id], int(pixels.sum()))
    - label: 'Family lain, pemanggilan sama'
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePIDNets-sem.pt")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        model.train(data="my-dataset.yaml", epochs=160, imgsz=512, batch=8)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreSegformerb0-sem.pt data=my-dataset.yaml \
          epochs=160 imgsz=512 batch=8
    - label: Pada ADE20K
      language: bash
      code: |
        # ade20k.yaml memuat skrip unduhan tertanam untuk arsip sekitar 1 GB,
        # sehingga memerlukan izin eksplisit kecuali datanya sudah lokal.
        libreyolo train model=LibreSegformerb0-sem.pt data=ade20k.yaml \
          epochs=160 imgsz=512 batch=8 allow_download_scripts=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")

        # val() mengembalikan dict biasa, bukan objek.
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSegformerb0-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        model.export(format="onnx", imgsz=512)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreSegformerb0-sem.pt format=onnx imgsz=512
    - label: Gunakan berkas hasil ekspor
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory merutekan berdasarkan akhiran berkas, sehingga artefak hasil
        # ekspor dimuat seperti checkpoint dan mengembalikan objek Results yang sama.
        model = LibreYOLO("LibreSegformerb0-sem.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.semantic_mask.data.shape)
source_hash: 44b92d8ba6062f04
---

## Definisi

Segmentasi semantik memberi label piksel, bukan objek. Setiap piksel menerima satu
id kelas, dan dua mobil yang bersentuhan dalam gambar menjadi satu region kelas
mobil tanpa batas di antaranya. Menghitung instance adalah
[segmentasi instance](/docs/tasks/instance-segmentation); memberi label setiap
piksel sekaligus memisahkan instance adalah
[segmentasi panoptik](/docs/tasks/panoptic-segmentation).

`semantic` adalah kunci task kanonis, dan akhiran `-sem` pada nama berkas checkpoint
memilihnya, sehingga `task=` tidak diperlukan saat memuat bobot terbitan.

`predict()` mengisi `result.semantic_mask`. `.data` adalah peta kelas integer
`(H, W)` pada kanvas gambar asli, `.classes` mencantumkan id yang ada dalam urutan
terurut, dan `.class_mask(id)` mengembalikan pilihan boolean `(H, W)` untuk satu
kelas. Nilai `255` adalah label ignore: nilai ini tidak pernah menjadi kelas,
dikecualikan dari loss dan metrik, serta tidak disertakan dalam `.classes`.

## Model

Tiga family dapat berlatih dan memprediksi:
[SegFormer](/docs/models/segformer), [LingBot-Vision](/docs/models/lingbot-vision),
dan [DINOv2](/docs/models/dinov2). SegFormer serta LingBot-Vision berjalan dengan
paket dasar dan menyertakan bobot terbitan. DINOv2 memerlukan
`pip install "libreyolo[rfdetr]"` dan tidak memiliki checkpoint yang disediakan oleh
LibreYOLO: model ini memuat backbone upstream dan dense head-nya dimulai dengan
inisialisasi acak, sehingga menjadi titik awal pelatihan, bukan predictor siap pakai.

Empat lainnya dapat memprediksi, memvalidasi, dan mengekspor, tetapi `train()`-nya
memunculkan `NotImplementedError`: [FCN](/docs/models/fcn),
[DeepLabv3](/docs/models/deeplabv3), [PIDNet](/docs/models/pidnet), dan
[EoMT](/docs/models/eomt).

Kumpulan kelas berbeda berdasarkan checkpoint, bukan family. Bobot terbitan
berasal dari dataset dengan ruang label yang sangat berbeda, termasuk 150 kelas
ADE20K dibanding 19 kelas Cityscapes, sehingga `names` milik checkpoint
menentukan apa yang dapat diberi label. Dua checkpoint hanya dapat dibandingkan
jika dilatih pada kumpulan yang sama.

## Prediksi

Bobot diunduh dari Hugging Face saat pertama kali digunakan dan disimpan dalam
cache lokal.

<code-tabs name="predict" />

Peta merupakan argmax per piksel, sehingga tidak ada langkah NMS dan `iou` tidak
pernah berpengaruh. `conf` serta `max_det` diterima demi paritas API dan tidak
berpengaruh pada SegFormer, PIDNet, dan dense predictor lain; EoMT adalah
pengecualian, karena `conf` memfilter pemilihan query. Lihat
[prediksi](/docs/predict) untuk sumber, streaming, dan penanganan hasil.

## Format dataset

Setiap gambar dipasangkan dengan mask satu channel yang rapat, bukan berkas label
`.txt`, yang ditemukan dengan mengganti `images` menjadi direktori mask dalam
path gambar.

```text
dataset/
  data.yaml
  images/
    train/000001.jpg
    val/000101.jpg
  masks/
    train/000001.png
    val/000101.png
```

Mask adalah gambar satu channel lossless, biasanya PNG, dan PNG palette-mode
dibaca sebagai indeks palette. Setiap nilai piksel adalah id kelas dalam
`0..nc-1`, nilai `255` berarti ignore, dan resolusi mask harus sama dengan
resolusi gambar pasangannya.

YAML menambahkan dua kunci di atas kontrak bersama:

```yaml
path: dataset
train: images/train
val: images/val
masks_dir: masks
nc: 19
names:
  0: road
  1: sidewalk
```

`masks_dir` adalah nama direktori pengganti `images`, dengan default `masks`.
`label_mapping` adalah pemetaan opsional `{source_id: train_id}` yang diterapkan
pada nilai piksel mask saat pemuatan. Dengan cara ini, dataset bernomor 1 hingga
150 menjadi 0 hingga 149; nilai sumber yang tidak dipetakan menjadi ignore, dan
setiap train id harus berada dalam `0..nc-1`.

Menghilangkan `masks_dir` mengalihkan loader ke fallback: mask dirasterisasi saat
pemuatan dari label poligon yang ditemukan melalui konvensi `images` ke `labels`
biasa, dan kelas `background` ditambahkan setelah kelas objek, sehingga `nc`
bertambah satu.

Loader kanonis adalah `libreyolo.data.SemanticDataset`.

## Pelatihan

<code-tabs name="train" />

`imgsz` dibatasi di sini dengan cara yang tidak berlaku pada detector. Setiap
family mendeklarasikan pembagi yang harus menjadi kelipatan input, sesuai grid
patch atau output stride, dan pelatihan maupun validasi memunculkan `ValueError`
sebelum proses dimulai jika `imgsz` tidak habis dibagi. Pembaginya adalah 32 untuk
SegFormer, 16 untuk LingBot-Vision dan EoMT, 14 untuk DINOv2, serta 8 untuk FCN
dan PIDNet. Lihat [pelatihan](/docs/train) untuk dataset, augmentasi, multi-GPU,
dan logger.

## Validasi

`val()` mengembalikan dictionary biasa berisi kunci `metrics/`, yang dihitung pada
split bernama `val` dalam YAML dataset.

<code-tabs name="val" />

`metrics/mIoU` adalah mean intersection over union: untuk setiap kelas, overlap
antara piksel prediksi dan ground truth dibagi union-nya, lalu dirata-ratakan
pada semua kelas. Nilai ini menjadi angka utama dan digunakan untuk memilih epoch
terbaik selama pelatihan. `metrics/pixel_accuracy` adalah bagian piksel yang
diberi kelas benar, yang dapat dibesar-besarkan oleh kelas background besar,
sehingga mIoU menjadi angka untuk dibandingkan. Piksel bertanda `255` tidak
dihitung pada keduanya. Dictionary juga memuat `fitness`, salinan nilai mIoU.

## Ekspor

<code-tabs name="export" />

Artefak hasil ekspor dimuat kembali melalui `LibreYOLO()` berdasarkan akhiran
filenya, sehingga berkas `.onnx` atau `.engine` berperilaku seperti checkpoint dan
mengembalikan `Results` yang sama. Cakupan format berbeda per family; matriks pada
setiap halaman model dibuat dari kumpulan tervalidasi, bukan diketik manual.
Lihat [ekspor dan deployment](/docs/export) untuk format, extra, dan batasannya.

