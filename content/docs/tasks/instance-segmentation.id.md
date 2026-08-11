---
title: Segmentasi instance
seo_title: Segmentasi instance di LibreYOLO
description: >-
  Memisahkan objek individual di LibreYOLO: keluarga yang melayani task, format
  label poligon, dan panggilan predict, train, validate, dan ekspor.
lead: >-
  Segmentasi instance menempatkan setiap instance objek dan mengembalikan mask
  per-piksel untuk masing-masing, bersama dengan kotak, kelas dan skor yang
  dikembalikan oleh detektor. Kunci task adalah segment.
keywords:
  - segmentasi instance python
  - prediksi mask objek
  - pelatihan model segmentasi
  - label poligon
  - pustaka segmentasi MIT
  - mask mAP
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Akhiran -seg pada nama berkas memilih head mask, sehingga tidak diperlukan task
        # argumen.
        model = LibreYOLO("LibreDFINEn-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)   # (N, H, W), satu mask per deteksi
        print(result.boxes.xyxy.shape)   # (N, 4), N baris yang sama
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreDFINEn-seg.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Kontur mask
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDFINEn-seg.pt")
        result = model(SAMPLE_IMAGE)

        # .xy adalah daftar kontur (P, 2) dalam piksel, .xyn normalisasi yang sama.
        for name, contour in zip(result.boxes.cls, result.masks.xy):
            print(result.names[int(name)], contour.shape)
    - label: 'family lain, panggilan yang sama'
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRTMDets-seg.pt")
        result = model(SAMPLE_IMAGE)

        print(result.masks.data.shape)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Berlanjut dari bobot segmentasi yang diterbitkan, mask head termasuk.
        # data harus menunjuk ke dataset yang labelnya membawa poligon.
        model = LibreYOLO("LibreDFINEn-seg.pt")
        model.train(data="my-dataset.yaml", epochs=50, imgsz=640, batch=8, lr0=2e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreDFINEn-seg.pt data=my-dataset.yaml \
          epochs=50 imgsz=640 batch=8 lr0=2e-4
    - label: Dari bobot deteksi
      language: bash
      code: |
        # Bobot deteksi tidak membawa mask head, jadi ini adalah transfer eksplisit
        # : head memulai tanpa pelatihan. Meminta task=segment adalah
        # apa yang memberinya wewenang.
        libreyolo train model=LibreDFINEn.pt data=my-dataset.yaml \
          task=segment epochs=50 imgsz=640
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])       # mask
        print(metrics["metrics/mAP50-95(M)"])    # mask, eksplisit
        print(metrics["metrics/mAP50-95(B)"])    # kotak
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDFINEn-seg.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn-seg.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDFINEn-seg.pt format=onnx imgsz=640
    - label: Gunakan berkas yang diekspor
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Pabrik mengarahkan berdasarkan sufiks berkas, jadi artefak yang diekspor dimuat
        # seperti checkpoint dan mengembalikan objek Results yang sama.
        model = LibreYOLO("LibreDFINEn-seg.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.masks.data.shape)
source_hash: 33e331eac0f9b0af
---

## Definisi

Segmentasi instance adalah deteksi ditambah bentuk. Setiap instance objek masih mendapatkan
sebuah kotak, sebuah kelas, dan sebuah skor, dan juga mendapatkan mask biner yang menutupi piksel
yang termasuk ke dalamnya. Masker dapat tumpang tindih, dan piksel yang tidak termasuk ke objek manapun
dibiarkan tidak ditetapkan, yang membedakan task dari
[segmentasi semantik](/docs/tasks/semantic-segmentation) dan
[segmentasi panoptik](/docs/tasks/panoptic-segmentation).

`segment` adalah kunci task kanonik, dan akhiran `-seg` dalam nama berkas checkpoint
memilihnya, sehingga `task=` tidak diperlukan saat memuat bobot yang dipublikasikan.

`predict()` mengisi `result.masks` bersama `result.boxes`. `.data` adalah
tumpukan `(N, H, W)` pada kanvas gambar asli, sejajar baris dengan kotak-kotak,
sehingga mask `i` milik kotak `i`. `.xy` mengubah setiap mask menjadi kontur terluar terbesar sebagai
array piksel `(P, 2)`, dan `.xyn` memberikan kontur yang sama dalam keadaan normalisasi.


## Model

Empat keluarga baik melatih maupun memprediksi mask: [RF-DETR](/docs/models/rf-detr),
[EdgeCrafter](/docs/models/edgecrafter), [D-FINE](/docs/models/d-fine) dan
[RTMDet](/docs/models/rtmdet). RF-DETR membutuhkan tambahan sendiri,
`pip install "libreyolo[rfdetr]"`; ketiga lainnya berjalan pada paket dasar.

[Mask R-CNN](/docs/models/mask-rcnn) memprediksi, memvalidasi, dan mengekspor mask, tetapi
`train()`-nya menimbulkan `NotImplementedError`.

[EoMT](/docs/models/eomt) memprediksi dan memvalidasi mask dan juga tidak dapat melatih,
dan ekspornya lebih sempit lagi: `export()` hanya menerima task semantik, dan
menaikkan `NotImplementedError` untuk `segment` dan `panoptic`, karena
kontrak runtime query-mask yang kedua diperlukan belum didefinisikan. Gunakan EoMT untuk
mask instance di Python, bukan melalui grafik yang diekspor.

Grup terpisah mengekstrak segmentasi dari prompt daripada daftar kelas: klik,
kotak atau frase memilih objek, dan model mengembalikan masknya.
[SAM](/docs/models/sam), [SAM 2](/docs/models/sam-2),
[SAM 3](/docs/models/sam-3), [MobileSAM](/docs/models/mobilesam),
[EdgeTAM](/docs/models/edgetam) dan [PicoSAM3](/docs/models/picosam3) bekerja seperti ini
, begitu juga dengan [SenseNova-Vision](/docs/models/sensenova-vision), yang
segmentasinya dirujuk: itu mengambil frase yang menamai satu objek. Mereka memuat
melalui pabrik mereka sendiri dan tambahan, dan setiap halaman model membawa panggilan
yang tepat.

## Prediksi

Bobot diunduh dari Hugging Face saat penggunaan pertama dan disimpan dalam cache secara lokal.

<code-tabs name="predict" />

`conf` dan `max_det` membentuk output dengan cara yang sama seperti mereka melakukan deteksi, dan
topeng disaring bersama kotak yang mereka miliki. Lihat
[prediksi](/docs/predict) untuk sumber, streaming, dan penanganan hasil.

## Format dataset

Tata letaknya adalah tata letak deteksi: satu berkas label `.txt` per gambar, ditemukan dengan
menukar `images` dengan `labels` dalam jalur gambar dan mengubah ekstensi.

```text
dataset/
  data.yaml
  images/
    train/000001.jpg
    val/000101.jpg
  labels/
    train/000001.txt
    val/000101.txt
```

Yang berubah adalah barisnya. Sebuah segmen adalah indeks kelas diikuti oleh poligon datar:

```text
<class_id> <x1> <y1> ... <xN> <yN>
```

Setidaknya tiga titik, sehingga jumlah koordinat setelah indeks kelas genap
dan setidaknya enam, dan poligonnya harus tidak terdeformasi. Koordinatnya adalah
float dalam `[0, 1]` relatif terhadap lebar dan tinggi gambar asli. Sebuah baris deteksi lima
bidang juga diterima dalam dataset segmentasi dan dibaca sebagai
segmen persegi panjang, yang membuat dataset hanya kotak dapat dimuat tanpa
proses konversi.

YAML adalah YAML deteksi:

```yaml
path: dataset
train: images/train
val: images/val
names:
  0: person
  1: bicycle
```

JSON COCO asli juga berfungsi: tambahkan pemetaan `annotations` dari nama split ke
Berkas JSON, dan jalur split memberikan akar gambar.

## Kereta

<code-tabs name="train" />

Pelatihan berlanjut dari `-seg` checkpoint yang diterbitkan secara default. Dimulai dari
bobot deteksi dimungkinkan tetapi merupakan transfer yang disengaja: bobot-bobot itu membawa
tidak ada mask head, jadi dimulai tanpa pelatihan, dan melewati `task=segment` itu apa
mengotorisasi pertukaran. Lihat [pelatihan](/docs/train) untuk dataset, augmentasi,
multi-GPU dan pencatat.

## Validasi

`val()` mengembalikan kamus biasa dari kunci `metrics/`. Kotak dan mask adalah
dinilai secara terpisah, keduanya dengan evaluasi COCO, dan nomor mask adalah
utama.

<code-tabs name="val" />

Kunci tanpa akhiran memegang mask results: `metrics/mAP50-95`, `metrics/mAP50`,
`metrics/mAP75`, kemudian `metrics/mAP_small`, `metrics/mAP_medium` dan
`metrics/mAP_large` menurut area objek, dan `metrics/AR1`, `metrics/AR10`,
`metrics/AR100`, `metrics/AR_small`, `metrics/AR_medium`, `metrics/AR_large`
untuk rata-rata recall. `metrics/AR_max_det` dan `metrics/max_det` merekam
deteksi kap yang digunakan saat menjalankan.

Empat angka juga diterbitkan dengan akhiran eksplisit, `(M)` untuk mask dan
`(B)` untuk kotak, sehingga perbandingan tidak pernah bergantung pada nomor mana yang family
diputuskan untuk dipanggil sebagai utama: `metrics/mAP50-95(M)` dan `metrics/mAP50-95(B)`,
`metrics/mAP50(M)` dan `metrics/mAP50(B)`, `metrics/precision(M)` dan
`metrics/precision(B)`, `metrics/recall(M)` dan `metrics/recall(B)`. Ada
tidak ada `metrics/precision` atau `metrics/recall` tanpa akhiran pada task ini.

Baca kunci presisi dan recall dengan hati-hati. Mereka disimpan untuk kompatibilitas
mundur dan merupakan alias, bukan titik operasi: `metrics/precision(M)`
memiliki nilai yang sama dengan `metrics/mAP50-95(M)`, dan `metrics/recall(M)` memiliki
nilai yang sama dengan mask AR pada 100 deteksi, dengan `(B)` berperilaku sama untuk kotak.
Memetakan sepasang dari mereka melaporkan satu angka dua kali.

## Ekspor

<code-tabs name="export" />

Artefak yang diekspor dimuat kembali melalui `LibreYOLO()` berdasarkan akhiran filenya, sehingga
berkas `.onnx` atau `.engine` berperilaku seperti checkpoint dan mengembalikan
`Results` yang sama. Cakupan segmentasi lebih sempit daripada cakupan deteksi pada
sama dengan family. Matriks di setiap halaman model dihasilkan dari set yang tervalidasi
dan menyebutkan alasan mengapa target tidak tersedia. Lihat
[ekspor dan deploy](/docs/export) untuk format, tambahan mereka dan batasan
mereka.


