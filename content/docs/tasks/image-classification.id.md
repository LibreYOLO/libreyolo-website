---
title: Klasifikasi gambar
seo_title: Klasifikasi gambar di LibreYOLO
description: >-
  Memberi label pada seluruh gambar di LibreYOLO: keluarga yang melayani task,
  tata letak ImageFolder dataset, dan panggilan predict, train, validate, dan
  ekspor.
lead: >-
  Klasifikasi gambar menetapkan satu distribusi label pada seluruh gambar dan
  tidak menempatkan apapun di dalamnya. Kunci task adalah classify.
keywords:
  - klasifikasi gambar python
  - melatih pengklasifikasi gambar
  - ImageFolder dataset
  - akurasi top-1
  - klasifikasi zero-shot
  - perpustakaan klasifikasi MIT
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Sufiks -cls pada nama berkas memilih task, sehingga tidak perlu argumen task.
        # argumen diperlukan.
        model = LibreYOLO("LibreResNet50-cls.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.names[result.probs.top1], float(result.probs.top1conf))
        print(result.probs.top5)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreResNet50-cls.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Seluruh distribusi
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        result = LibreYOLO("LibreResNet50-cls.pt")(SAMPLE_IMAGE)
        probs = result.probs

        # .data adalah vektor penuh (C,); top5/top5conf adalah tampilan yang diurutkan.
        print(probs.data.shape)
        for index, score in zip(probs.top5, probs.top5conf):
            print(result.names[index], float(score))
    - label: 'Zero-shot, tanpa pelatihan'
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # CLIP menilai gambar terhadap prompt teks, sehingga set label ditentukan
        # saat pemanggilan, bukan tertanam dalam checkpoint.
        model = LibreYOLO("LibreCLIPb32-cls.pt")
        model.set_classes(["a person jumping", "an empty street", "a parked car"])
        result = model(SAMPLE_IMAGE)

        print(model.names[result.probs.top1], float(result.probs.top1conf))
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # imagenette160 adalah nama dataset yang dikenal dan diunduh pada penggunaan pertama.
        # Lewatkan direktori dengan pembagian train/ untuk data Anda sendiri.
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

        # val() mengembalikan dict biasa, bukan objek.
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
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreResNet50-cls.pt format=onnx
    - label: Gunakan berkas yang diekspor
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Pabrik mengarahkan berdasarkan akhiran berkas, sehingga artifak yang diekspor memuat
        # seperti checkpoint dan mengembalikan objek Results yang sama.
        model = LibreYOLO("LibreResNet50-cls.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.probs.top1, result.probs.top1conf)
source_hash: 836bea76cd2cdf92
---

## Definisi

Klasifikasi gambar menghasilkan satu skor per kelas untuk seluruh gambar dan tidak ada
koordinat sama sekali. Ini menjawab apa yang ada dalam gambar, bukan di mana, yang
apa yang memisahkannya dari [object detection](/docs/tasks/object-detection).

`classify` adalah kunci task kanonik, dan sufiks `-cls` dalam checkpoint
nama berkas memilihnya. Sufiks itu diperlukan dan bukan opsional pada
keluarga klasifikasi, jadi `LibreResNet50.pt` tidak dibaca sebagai pengklasifikasi dan
hanya `LibreResNet50-cls.pt` saja.

`predict()` mengisi `result.probs` dan meninggalkan `boxes` kosong. `.data` adalah
vektor skor penuh, `.top1` indeks dari skor tertinggi dan `.top1conf`nya
nilai, `.top5` lima indeks tertinggi dalam urutan menurun dan `.top5conf`
skor mereka. Indeks menunjuk ke `result.names`. Memotong sebuah objek `Results`
tidak pernah memotong `probs`, karena vektor tersebut milik gambar, bukan milik
satu baris.

## Model

Lima keluarga baik melatih maupun memprediksi: [ResNet](/docs/models/resnet),
[ConvNeXt](/docs/models/convnext), [MobileNetV4](/docs/models/mobilenetv4),
[EfficientNetV2](/docs/models/efficientnetv2) dan
[DINOv2](/docs/models/dinov2). Empat pertama berjalan pada paket dasar dan dikirim
bobot yang diterbitkan. DINOv2 membutuhkan `pip install "libreyolo[rfdetr]"` dan tidak memiliki
LibreYOLO-menghosting checkpoint: ini memuat backbone hulu secara acak
menginisisialisasi head linier, jadi itu adalah titik awal fine-tuning daripada
prediktor siap.

Lima lagi memprediksi, memvalidasi, dan mengekspor, tetapi `train()` mereka meningkat
`NotImplementedError`: [ViT](/docs/models/vit), [Swin](/docs/models/swin),
[VGG](/docs/models/vgg), [AlexNet](/docs/models/alexnet) dan
[DeiT](/docs/models/deit).

[CLIP](/docs/models/clip) dan [SigLIP2](/docs/models/siglip2) mengklasifikasikan tanpa
set label tetap. Mereka menilai gambar terhadap teks prompt, sehingga
`set_classes()` mendefinisikan kelas pada saat pemanggilan dan tidak ada langkah pelatihan
untuk set label baru sama sekali. Keduanya juga melayani `embed` task.

## Prediksi

Bobot diunduh dari Hugging Face saat pertama digunakan dan disimpan di cache secara lokal.

<code-tabs name="predict" />

`conf`, `iou` dan `max_det` tidak memiliki efek di sini: tidak ada kandidat untuk
disaring atau ditekan, hanya ada satu distribusi. Lihat
[prediksi](/docs/predict) untuk sumber, streaming dan penanganan hasil.

## Format dataset

Klasifikasi menggunakan pohon direktori, bukan berkas label dan bukan YAML. `data` adalah
akar dataset.

```text
dataset/
  train/
    tench/000001.jpg
    parachute/000002.jpg
  val/
    tench/000101.jpg
    parachute/000102.jpg
```

`train/` diperlukan untuk pelatihan dan ini mendefinisikan pemetaan kelas-ke-indeks dengan
nama folder yang diurutkan, sehingga folder pertama secara alfabet menjadi kelas 0. `val/`
diperlukan untuk validasi. Pemisahan `test/` mungkin ada dan default
perintah train dan validate jangan menggunakannya. Setiap pembagian selain `train` harus
mengandung nama folder kelas yang sama dengan set kelas yang diharapkan, yang adalah apa
menyebabkan ketidaksesuaian gagal dengan keras daripada dinilai sebagai prediksi yang salah. The
ekstensi gambar yang diterima adalah `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tif`
dan `.tiff`.

`data` menerima tiga hal: jalur ke direktori yang berisi split `train/`,
URL `.zip`, atau salah satu nama dataset yang dikenal, `imagenette160` dan `smoke10`,
yang diunduh dan disimpan dalam cache saat penggunaan pertama.

Loader kanonik adalah `libreyolo.data.classify_dataset`.

## Latih

<code-tabs name="train" />

Tidak ada `nc` yang perlu dideklarasikan: jumlah kelas berasal dari nama folder di bawah
`train/`, dan lapisan linier terakhir dibangun kembali untuk menyesuaikannya sementara backbone
dipindahkan tanpa perubahan. Lihat [pelatihan](/docs/train) untuk dataset, augmentasi,
multi-GPU dan logger.

## Validasi

`val()` mengembalikan kamus sederhana dari kunci `metrics/`, dihitung di atas `val/`
pembagian dari akar dataset.

<code-tabs name="val" />

`metrics/accuracy_top1` adalah bagian dari gambar-gambar yang kelas dengan skor tertinggi adalah
kelas yang benar, dan ini adalah angka utama, yang digunakan pelatihan untuk memilih
epoch terbaik. `metrics/accuracy_top5` adalah bagian yang kelas aslinya muncul
di mana saja dalam lima kelas dengan skor tertinggi, yang berarti lebih sedikit sebanyak kelas
yang dimiliki dataset. Kamus juga memuat `fitness`, salinan dari nilai top-1.


## Ekspor

<code-tabs name="export" />

Sebuah artefak yang diekspor dapat dimuat kembali melalui `LibreYOLO()` berdasarkan akhiran berkas-nya, jadi sebuah
berkas `.onnx` atau `.engine` berperilaku seperti checkpoint dan mengembalikan
`Results` yang sama. Cakupan format berbeda menurut family; matriks di setiap halaman model adalah
dihasilkan dari set yang tervalidasi daripada diketik secara manual. Lihat
[ekspor dan deploy](/docs/export) untuk format, tambahan mereka dan
batasannya.


