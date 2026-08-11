---
title: BiRefNet
families:
  - birefnet
seo_title: 'BiRefNet: penghapusan background dan matting di LibreYOLO'
description: >-
  Gunakan BiRefNet di LibreYOLO untuk penghapusan background dan dichotomous
  image segmentation. Pasang, prediksi, validasi dan ekspor checkpoint
  general-nya.
lead: >-
  Jaringan bilateral-reference yang memprediksi alpha matte lembut untuk
  memisahkan subjek dari background-nya. LibreYOLO menyediakan inferensi dan
  validasi untuk task matte BiRefNet.
keywords:
  - BiRefNet
  - hapus background gambar
  - background removal
  - dichotomous image segmentation
  - alpha matte
  - image matting python
  - cutout png transparan
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        result = model(SAMPLE_IMAGE, save=True)

        matte = result.matte
        print(matte.array.shape, matte.array.dtype)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreBiRefNetl-matte.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Cutout
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        result = model(SAMPLE_IMAGE)

        # RGBA (H, W, 4) uint8: RGB sumber plus matte sebagai kanal alpha.
        rgba = result.cutout()
        result.save("subject.png")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreBiRefNetl-matte.pt")

        # Sebuah direktori berisi images/ dan direktori matte yang terdeteksi
        # otomatis (mattes/, matte/, gt/, masks/, mask/ atau alpha/) juga bisa
        # dipakai sebagai ganti YAML dataset.
        metrics = model.val(data="my-matte-dataset/")

        print(metrics["metrics/MAE"])
        print(metrics["metrics/Smeasure"])
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        model.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreBiRefNetl-matte.pt format=onnx
    - label: Memakai berkas hasil ekspor
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory memilih rute berdasarkan sufiks berkas, jadi artefak hasil
        # ekspor dimuat seperti checkpoint biasa dan mengembalikan objek
        # Results yang sama.
        model = LibreYOLO("LibreBiRefNetl-matte.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.matte.array.shape)
source_hash: 1af1bd7f4f905081
---

## Instalasi

BiRefNet tidak membutuhkan extra opsional. Semua yang diimpornya sudah ada di
instalasi dasar.

```bash
pip install libreyolo
```

## Prediksi

Bobot diunduh dari Hugging Face saat pertama kali dipakai dan disimpan di cache
lokal.

<code-tabs name="predict" />

Hasil matte tidak membawa box; `result.matte` adalah array float32 padat
berbentuk `(H, W)` dengan nilai di `[0, 1]`, 1 berarti sepenuhnya foreground dan
0 sepenuhnya background. Berbeda dengan mask biner, matte lembut mempertahankan
detail tepi anti-aliased seperti rambut dan bulu. `result.cutout()` menggabungkan
gambar sumber dengan kanal alpha itu menjadi array RGBA, dan `result.save(path)`
(atau `save=True` pada panggilan predict) langsung menulisnya ke PNG berlatar
transparan. Model berjalan pada kanvas native tetap 1024x1024; resolusi lain
tidak didukung, karena tabel posisi relatif milik backbone Swin terikat pada
ukuran itu, dan ketidakcocokan akan menginterpolasinya dengan buruk alih-alih
memunculkan error. Lihat [prediksi](/docs/predict) untuk sumber, streaming dan
penanganan hasil.

## Varian

Satu checkpoint yang dipublikasikan, `l`, yaitu model BiRefNet-general tier
Swin-L sekaligus default kualitas di upstream. Kode family ini juga mendukung
tier lite Swin-T, `t`, tetapi konversi LibreYOLO untuknya belum dipublikasikan.

## Validasi

`val()` melaporkan dua metrik atas folder berisi pasangan gambar/matte, keduanya
berada di `[0, 1]` dan tidak bergantung pada resolusi: MAE, yaitu mean absolute
error terhadap alpha ground truth (makin rendah makin baik), dan S-measure (Fan
et al., ICCV 2017), sebuah kemiripan struktural yang menghargai terjaganya bentuk
dan lubang pada subjek, hal yang luput bila hanya mengandalkan MAE per piksel
(makin tinggi makin baik). Validasi menjalankan `predict` milik model itu
sendiri, jadi preprocessing yang dipakai persis sama dengan milik family ini.

<code-tabs name="val" />

Validasi hanya melakukan inferensi; fine-tuning adalah kelanjutan yang
terdokumentasi, bukan fitur yang sudah tersedia (lihat Prediksi untuk batasan
resolusi persis yang akan diwarisi trainer mana pun di masa depan).

## Ekspor

<export-matrix />

Artefak hasil ekspor dimuat kembali lewat `LibreYOLO()` berdasarkan sufiks
berkasnya, jadi berkas `.onnx` berperilaku seperti checkpoint dan mengembalikan
`Results` yang sama. TorchScript adalah jalur yang sudah tervalidasi; konversi
ONNX berjalan tetapi belum melewati standar paritas yang sama.
[Ekspor](/docs/export) memuat daftar argumen yang diterima setiap format serta
tambahan yang dibawa beberapa di antaranya.

<code-tabs name="export" />

## Checkpoint

Semua berkas bobot yang dipublikasikan untuk family ini.

<checkpoint-table />

## Lisensi

<provenance-box></provenance-box>

## Sitasi

<citation-block />
